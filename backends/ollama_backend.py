"""
Simple async Ollama streaming backend.

Provides:
- async def check_ollama_available(base_url: str) -> bool
- async def stream_ollama_generate(job_id: str, model: str, prompt: str,
                                   max_tokens: int, temperature: float,
                                   num_ctx: int, base_url: str,
                                   on_token: Callable[[str, float], Awaitable[None]])
    -> Dict[str, Any]

Uses httpx.AsyncClient to stream NDJSON from Ollama.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Awaitable, Callable, Dict

import httpx

# tune as needed
OLLAMA_TIMEOUT = 5.0  # seconds for tags check

# Token buffering configuration for smoother WebSocket streaming
# Buffer small fragments and flush when buffer reaches N chars, contains newline,
# or time threshold elapses.
STREAM_FLUSH_CHARS = int(os.getenv("BENCH_STREAM_FLUSH_CHARS", "64"))
STREAM_FLUSH_SECONDS = float(os.getenv("BENCH_STREAM_FLUSH_SECONDS", "0.10"))


async def check_ollama_available(base_url: str) -> bool:
    """Return True if Ollama HTTP API responds to /api/tags quickly."""
    try:
        url = base_url.rstrip("/") + "/api/tags"
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            r = await client.get(url)
            return r.status_code == 200
    except Exception:
        return False


async def stream_ollama_generate(
    job_id: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    num_ctx: int,
    base_url: str,
    on_token: Callable[[str, float], Awaitable[None]],
) -> Dict[str, Any]:
    """
    Send a streaming generate request to Ollama and call on_token(text, timestamp_s)
    for each incremental text chunk. Returns final metrics dict.

    This expects Ollama to stream NDJSON lines (each a JSON object).

    Token buffering: small fragments are buffered and flushed when:
    - Buffer reaches STREAM_FLUSH_CHARS characters, OR
    - Buffer contains a newline, OR
    - STREAM_FLUSH_SECONDS have elapsed since last flush, OR
    - Stream is done.
    """
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": float(temperature),
            "num_ctx": int(num_ctx),
            "num_predict": int(max_tokens),  # Ollama uses num_predict, not max_tokens
        },
    }

    gen_tokens = 0
    t0 = time.perf_counter()
    t_first = None

    # Token buffering state
    buf = ""
    last_flush_time = t0

    async def flush_buffer() -> None:
        """Flush accumulated buffer to on_token callback."""
        nonlocal buf, last_flush_time, gen_tokens, t_first
        if not buf:
            return
        now = time.perf_counter()
        if t_first is None:
            t_first = now
        await on_token(buf, now)
        gen_tokens += len(buf.split())
        buf = ""
        last_flush_time = now

    # Use a long timeout for streaming; httpx will keep connection open
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                # iterate lines (NDJSON)
                async for raw_line in resp.aiter_lines():
                    if raw_line is None:
                        continue
                    line = raw_line.strip()
                    if not line:
                        continue
                    # Ollama may send non-json control lines; guard parse
                    try:
                        obj = json.loads(line)
                    except Exception:
                        # ignore non-json line
                        continue

                    # Ollama NDJSON streaming shapes can vary.
                    # Many messages contain {"response": "<text chunk>", ...}
                    # We'll look for keys likely to contain text.
                    text = None
                    if "response" in obj and isinstance(obj["response"], str):
                        text = obj["response"]
                    elif "token" in obj and isinstance(obj["token"], str):
                        text = obj["token"]
                    elif "content" in obj and isinstance(obj["content"], str):
                        text = obj["content"]

                    if text:
                        # Record t_first on very first text received
                        if t_first is None:
                            t_first = time.perf_counter()
                        # Accumulate into buffer
                        buf += text

                        # Decide whether to flush
                        now = time.perf_counter()
                        should_flush = (
                            len(buf) >= STREAM_FLUSH_CHARS
                            or "\n" in buf
                            or (now - last_flush_time) >= STREAM_FLUSH_SECONDS
                        )
                        if should_flush:
                            await flush_buffer()

                    # detect done markers if Ollama includes them
                    if obj.get("done") is True or obj.get("type") == "done":
                        break

            # Final flush for any remaining buffered text
            await flush_buffer()

        except Exception as exc:
            # Flush any remaining buffer before raising
            await flush_buffer()
            raise

    tend = time.perf_counter()
    ttft_ms = (t_first - t0) * 1000.0 if t_first else None
    total_ms = (tend - t0) * 1000.0
    gen_tps = gen_tokens / (tend - t_first) if (t_first and tend > t_first) else None

    result = {
        "ttft_ms": ttft_ms,
        "gen_tokens": gen_tokens,
        "gen_tokens_per_s": gen_tps,
        "total_ms": total_ms,
        "model": model,
        "engine": "ollama",
    }
    return result

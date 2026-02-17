"""
Async vLLM streaming backend using the OpenAI-compatible API.

Provides:
- async def check_vllm_available(base_url: str) -> bool
- async def get_vllm_models(base_url: str) -> List[str]
- async def stream_vllm_generate(job_id: str, model: str, prompt: str,
                                  max_tokens: int, temperature: float,
                                  num_ctx: int, base_url: str,
                                  on_token: Callable[[str, float], Awaitable[None]])
    -> Dict[str, Any]

Uses httpx.AsyncClient to stream SSE from vLLM's /v1/completions endpoint.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Awaitable, Callable, Dict, List

import httpx

# Logger for this module
log = logging.getLogger("vllm_backend")

# Default timeout for health checks
VLLM_TIMEOUT = 5.0  # seconds

# Token buffering (same env vars as Ollama backend for consistency)
STREAM_FLUSH_CHARS = int(os.getenv("BENCH_STREAM_FLUSH_CHARS", "64"))
STREAM_FLUSH_SECONDS = float(os.getenv("BENCH_STREAM_FLUSH_SECONDS", "0.10"))


async def check_vllm_available(base_url: str) -> bool:
    """Return True if vLLM HTTP API responds to /health quickly."""
    try:
        url = base_url.rstrip("/") + "/health"
        async with httpx.AsyncClient(timeout=VLLM_TIMEOUT) as client:
            r = await client.get(url)
            available = r.status_code == 200
            if not available:
                log.warning("vLLM health check failed: HTTP %d from %s", r.status_code, url)
            return available
    except Exception as e:
        log.warning("vLLM unreachable at %s: %s", base_url, e)
        return False


async def get_vllm_models(base_url: str) -> List[str]:
    """
    Fetch list of available model names from vLLM's OpenAI-compatible /v1/models endpoint.
    Returns empty list if vLLM is unreachable or on error.
    """
    try:
        url = base_url.rstrip("/") + "/v1/models"
        async with httpx.AsyncClient(timeout=VLLM_TIMEOUT) as client:
            r = await client.get(url)
            if r.status_code != 200:
                log.warning("Failed to fetch vLLM models: HTTP %d from %s", r.status_code, url)
                return []
            data = r.json()
            models = []
            for entry in data.get("data", []):
                model_id = entry.get("id")
                if model_id:
                    models.append(model_id)
            log.debug("vLLM models available: %s", models)
            return models
    except Exception as e:
        log.warning("Failed to fetch vLLM models from %s: %s", base_url, e)
        return []


async def stream_vllm_generate(
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
    Send a streaming completions request to vLLM via the OpenAI-compatible
    /v1/completions endpoint and call on_token(text, timestamp_s) for each
    incremental text chunk.

    Returns the same metrics dict shape as stream_ollama_generate so the
    caller can treat backends interchangeably.

    Token buffering: small fragments are buffered and flushed when:
    - Buffer reaches STREAM_FLUSH_CHARS characters, OR
    - Buffer contains a newline, OR
    - STREAM_FLUSH_SECONDS have elapsed since last flush, OR
    - Stream is done.
    """
    url = base_url.rstrip("/") + "/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": float(temperature),
        "stream": True,
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

    log.debug("Starting vLLM stream for job %s, model=%s", job_id, model)
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                log.debug("vLLM stream connected for job %s", job_id)

                async for raw_line in resp.aiter_lines():
                    if raw_line is None:
                        continue
                    line = raw_line.strip()
                    if not line:
                        continue

                    # SSE format: "data: {...}" or "data: [DONE]"
                    if line.startswith("data: "):
                        line = line[len("data: "):]

                    if line == "[DONE]":
                        break

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Extract text from OpenAI completions SSE chunk
                    text = ""
                    finish_reason = None
                    for choice in chunk.get("choices", []):
                        text += choice.get("text", "")
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]

                    if text:
                        if t_first is None:
                            t_first = time.perf_counter()
                        buf += text

                        now = time.perf_counter()
                        should_flush = (
                            len(buf) >= STREAM_FLUSH_CHARS
                            or "\n" in buf
                            or (now - last_flush_time) >= STREAM_FLUSH_SECONDS
                        )
                        if should_flush:
                            await flush_buffer()

                    if finish_reason is not None:
                        break

            # Final flush for any remaining buffered text
            await flush_buffer()
            log.debug("vLLM stream completed for job %s", job_id)

        except httpx.HTTPStatusError as exc:
            log.error(
                "vLLM HTTP error for job %s: status=%d, url=%s",
                job_id, exc.response.status_code, url,
            )
            await flush_buffer()
            raise
        except httpx.TimeoutException as exc:
            log.error("vLLM timeout for job %s: %s", job_id, exc)
            await flush_buffer()
            raise
        except Exception as exc:
            log.error("vLLM stream error for job %s: %s", job_id, exc)
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
        "engine": "vllm",
    }
    log.info(
        "vLLM job %s completed: tokens=%d, ttft=%.1fms, total=%.1fms",
        job_id, gen_tokens, ttft_ms or 0, total_ms,
    )
    return result

"""
vLLM backend — shared streaming client.

Mirrors backends/ollama_backend.py but targets vLLM's OpenAI-compatible API.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Awaitable, Callable, Dict

import httpx

log = logging.getLogger("vllm_backend")

VLLM_TIMEOUT = 5.0

STREAM_FLUSH_CHARS = int(os.getenv("BENCH_STREAM_FLUSH_CHARS", "64"))
STREAM_FLUSH_SECONDS = float(os.getenv("BENCH_STREAM_FLUSH_SECONDS", "0.10"))


async def check_vllm_available(base_url: str) -> bool:
    """Return True if vLLM responds to health or model list."""
    try:
        async with httpx.AsyncClient(timeout=VLLM_TIMEOUT) as client:
            for path in ("/health", "/v1/models"):
                url = base_url.rstrip("/") + path
                try:
                    r = await client.get(url)
                    if r.status_code == 200:
                        return True
                except httpx.RequestError:
                    continue
        return False
    except Exception as e:
        log.warning("vLLM unreachable at %s: %s", base_url, e)
        return False


async def get_vllm_models(base_url: str) -> list[str]:
    """Fetch model names from vLLM /v1/models."""
    try:
        url = base_url.rstrip("/") + "/v1/models"
        async with httpx.AsyncClient(timeout=VLLM_TIMEOUT) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return []
            data = r.json()
            return [m["id"] for m in data.get("data", []) if m.get("id")]
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
    """Stream completions from vLLM and return metrics dict."""
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
    buf = ""
    last_flush_time = t0

    async def flush_buffer() -> None:
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

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for raw_line in resp.aiter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if not line:
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            obj = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                    else:
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                    text = None
                    choices = obj.get("choices", [])
                    if choices:
                        choice = choices[0]
                        if "text" in choice:
                            text = choice["text"]
                        elif "delta" in choice and "content" in choice["delta"]:
                            text = choice["delta"]["content"]

                    if text:
                        if t_first is None:
                            t_first = time.perf_counter()
                        buf += text
                        now = time.perf_counter()
                        if (
                            len(buf) >= STREAM_FLUSH_CHARS
                            or "\n" in buf
                            or (now - last_flush_time) >= STREAM_FLUSH_SECONDS
                        ):
                            await flush_buffer()

                    if choices and choices[0].get("finish_reason"):
                        break

            await flush_buffer()

        except Exception:
            await flush_buffer()
            raise

    tend = time.perf_counter()
    ttft_ms = (t_first - t0) * 1000.0 if t_first else None
    total_ms = (tend - t0) * 1000.0
    gen_tps = gen_tokens / (tend - t_first) if (t_first and tend > t_first) else None

    return {
        "ttft_ms": ttft_ms,
        "gen_tokens": gen_tokens,
        "gen_tokens_per_s": gen_tps,
        "total_ms": total_ms,
        "model": model,
        "engine": "vllm",
    }

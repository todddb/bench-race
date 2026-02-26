"""
vLLM backend wrapper for bench-race agent.

Provides:
- async def check_vllm_available(base_url: str) -> bool
- async def get_vllm_models(base_url: str) -> list[str]
- async def stream_vllm_generate(
      job_id, model, prompt, max_tokens, temperature, num_ctx,
      base_url, on_token
  ) -> Dict[str, Any]

Uses the OpenAI-compatible API that vLLM exposes:
  POST /v1/completions  (streaming)
  GET  /v1/models
  GET  /health
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Awaitable, Callable, Dict

import httpx

log = logging.getLogger("vllm_backend")

VLLM_TIMEOUT = 5.0  # seconds for health/model list

# Token buffering (same logic as ollama_backend)
STREAM_FLUSH_CHARS = int(os.getenv("BENCH_STREAM_FLUSH_CHARS", "64"))
STREAM_FLUSH_SECONDS = float(os.getenv("BENCH_STREAM_FLUSH_SECONDS", "0.10"))


async def check_vllm_available(base_url: str) -> bool:
    """Return True if vLLM's health or models endpoint responds."""
    try:
        async with httpx.AsyncClient(timeout=VLLM_TIMEOUT) as client:
            # Try /health first (vLLM native), then /v1/models (OpenAI compat)
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
    """Fetch model names from vLLM's OpenAI-compatible /v1/models endpoint."""
    try:
        url = base_url.rstrip("/") + "/v1/models"
        async with httpx.AsyncClient(timeout=VLLM_TIMEOUT) as client:
            r = await client.get(url)
            if r.status_code != 200:
                log.warning("Failed to fetch vLLM models: HTTP %d from %s", r.status_code, url)
                return []
            data = r.json()
            # OpenAI format: {"data": [{"id": "model-name", ...}, ...]}
            models = data.get("data", [])
            return [m.get("id", "") for m in models if m.get("id")]
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
    Send a streaming completions request to vLLM's OpenAI-compatible API
    and call on_token(text, timestamp) for each text chunk.
    Returns final metrics dict matching LLMResult schema.

    Uses POST /v1/completions with stream=true, which returns SSE lines
    in the format: data: {"choices": [{"text": "...", ...}], ...}
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
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            obj = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                    else:
                        # Try raw JSON line
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                    # Extract text from OpenAI completions response
                    text = None
                    choices = obj.get("choices", [])
                    if choices:
                        choice = choices[0]
                        # Completions API: {"choices": [{"text": "..."}]}
                        if "text" in choice:
                            text = choice["text"]
                        # Chat API fallback: {"choices": [{"delta": {"content": "..."}}]}
                        elif "delta" in choice and "content" in choice["delta"]:
                            text = choice["delta"]["content"]

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

                    # Check for finish_reason
                    if choices and choices[0].get("finish_reason"):
                        break

            await flush_buffer()
            log.debug("vLLM stream completed for job %s", job_id)

        except httpx.HTTPStatusError as exc:
            log.error(
                "vLLM HTTP error for job %s: status=%d, url=%s",
                job_id,
                exc.response.status_code,
                url,
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
        job_id,
        gen_tokens,
        ttft_ms or 0,
        total_ms,
    )
    return result

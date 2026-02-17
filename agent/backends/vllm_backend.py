from __future__ import annotations

import json
import time
from typing import Awaitable, Callable, Dict, List, Optional

import httpx


DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000"


async def check_vllm_available(base_url: str, timeout_s: float = 2.0) -> bool:
    """Check whether a vLLM server is reachable via its health endpoint."""
    url = base_url.rstrip("/") + "/health"
    timeout = httpx.Timeout(timeout_s)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            return resp.status_code == 200
    except httpx.RequestError:
        return False


async def get_vllm_models(base_url: str, timeout_s: float = 5.0) -> List[str]:
    """Fetch the list of models served by vLLM via the OpenAI-compatible /v1/models endpoint."""
    url = base_url.rstrip("/") + "/v1/models"
    timeout = httpx.Timeout(timeout_s)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return []
            data = resp.json()
            models = []
            for entry in data.get("data", []):
                model_id = entry.get("id")
                if model_id:
                    models.append(model_id)
            return models
    except (httpx.RequestError, json.JSONDecodeError, KeyError):
        return []


async def stream_vllm_generate(
    *,
    job_id: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    num_ctx: int,
    base_url: str,
    on_token: Callable[[str, float], Awaitable[None]],
) -> dict:
    """
    Stream token generation from a vLLM server using the OpenAI-compatible
    /v1/completions endpoint with ``stream=true``.

    Returns the same metrics dict shape as ``stream_ollama_generate`` so the
    caller can treat backends interchangeably.
    """
    url = base_url.rstrip("/") + "/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    t0 = time.perf_counter()
    t_first: Optional[float] = None
    t_end: Optional[float] = None
    gen_tokens = 0

    timeout = httpx.Timeout(connect=5.0, read=None, write=60.0, pool=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()

            async for line in resp.aiter_lines():
                if not line:
                    continue
                # SSE format: "data: {...}" or "data: [DONE]"
                if line.startswith("data: "):
                    line = line[len("data: "):]

                if line.strip() == "[DONE]":
                    t_end = time.perf_counter()
                    break

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Extract text from OpenAI completions SSE chunk
                choices = chunk.get("choices", [])
                text = ""
                finish_reason = None
                for choice in choices:
                    text += choice.get("text", "")
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

                if text:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    gen_tokens += 1  # vLLM streams one token at a time
                    await on_token(text, now)

                if finish_reason is not None:
                    t_end = time.perf_counter()
                    break

    if t_end is None:
        t_end = time.perf_counter()

    ttft_ms = (t_first - t0) * 1000.0 if t_first is not None else None
    total_ms = (t_end - t0) * 1000.0
    gen_tokens_per_s = (
        gen_tokens / (t_end - t_first)
        if t_first is not None and t_end > t_first
        else None
    )

    return {
        "ttft_ms": ttft_ms,
        "gen_tokens": gen_tokens,
        "gen_tokens_per_s": gen_tokens_per_s,
        "total_ms": total_ms,
        "model": model,
        "engine": "vllm",
    }

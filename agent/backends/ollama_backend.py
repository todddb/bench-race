from __future__ import annotations

import json
import time
from typing import Awaitable, Callable, Optional

import httpx


async def check_ollama_available(base_url: str, timeout_s: float = 0.8) -> bool:
    url = base_url.rstrip("/") + "/api/tags"
    timeout = httpx.Timeout(timeout_s)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            return resp.status_code == 200
    except httpx.RequestError:
        return False


async def stream_ollama_generate(
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
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }

    t0 = time.perf_counter()
    t_first: Optional[float] = None
    t_end: Optional[float] = None
    gen_tokens = 0

    timeout = httpx.Timeout(connect=5.0, read=None, write=60.0, pool=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload, stream=True)
        resp.raise_for_status()

        async for line in resp.aiter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                chunk = {"response": line}

            text = ""
            if isinstance(chunk, dict):
                if isinstance(chunk.get("response"), str):
                    text = chunk["response"]
                elif isinstance(chunk.get("token"), str):
                    text = chunk["token"]
                elif isinstance(chunk.get("delta"), str):
                    text = chunk["delta"]

            if text:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                gen_tokens += len(text.split())
                await on_token(text, now)

            if isinstance(chunk, dict) and chunk.get("done") is True:
                t_end = time.perf_counter()
                break

    if t_end is None:
        t_end = time.perf_counter()

    ttft_ms = (t_first - t0) * 1000.0 if t_first is not None else None
    total_ms = (t_end - t0) * 1000.0
    gen_tokens_per_s = (
        gen_tokens / (t_end - t_first) if t_first is not None and t_end > t_first else None
    )

    return {
        "ttft_ms": ttft_ms,
        "gen_tokens": gen_tokens,
        "gen_tokens_per_s": gen_tokens_per_s,
        "total_ms": total_ms,
        "model": model,
        "engine": "ollama",
    }

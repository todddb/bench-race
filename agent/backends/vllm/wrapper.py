from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
import psutil
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel

app = FastAPI(title="bench-race vLLM wrapper")

ENGINE = "vllm"
OPENAI_BASE = os.getenv("VLLM_OPENAI_BASE", "http://127.0.0.1:8001")
CURRENT_MODEL = os.getenv("VLLM_MODEL", "")


class StartRequest(BaseModel):
    model_id: str
    args: Dict[str, Any] = {}


class InferRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False
    model: Optional[str] = None


class SwitchRequest(BaseModel):
    model_id: str


async def _models() -> List[Dict[str, Any]]:
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(f"{OPENAI_BASE}/v1/models")
        resp.raise_for_status()
        payload = resp.json()
    models = []
    for m in payload.get("data", []):
        mid = m.get("id", "unknown")
        models.append({
            "id": mid,
            "size": "unknown",
            "quant": "fp16",
            "arch": "transformer",
            "equivalence_group": mid.split("/")[-1],
            "supported_backends": ["vllm"],
        })
    return models


@app.get("/health")
async def health():
    status = "ok"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.get(f"{OPENAI_BASE}/health")
    except Exception:
        status = "degraded"
    vm = psutil.virtual_memory()
    return {
        "status": status,
        "engine": ENGINE,
        "model": CURRENT_MODEL,
        "mem": {"ram_pct": vm.percent, "vram_pct": 0.0},
    }


@app.get("/models")
async def models():
    return {"models": await _models()}


@app.post("/start")
async def start(req: StartRequest):
    global CURRENT_MODEL
    CURRENT_MODEL = req.model_id
    return {"started": True, "pid": os.getpid()}


@app.post("/stop")
async def stop():
    return {"stopped": True}


@app.post("/model/switch")
async def model_switch(req: SwitchRequest):
    global CURRENT_MODEL
    t0 = time.time()
    CURRENT_MODEL = req.model_id
    return {"ok": True, "load_time": round(time.time() - t0, 3)}


@app.post("/infer")
async def infer(req: InferRequest):
    model = req.model or CURRENT_MODEL
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{OPENAI_BASE}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
    text = data["choices"][0]["message"]["content"]
    return {"text": text, "tokens": len(text.split()), "meta": {"engine": ENGINE}}


async def _stream_tokens(prompt: str, model: str, params: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(params.get("max_tokens", 256)),
        "temperature": float(params.get("temperature", 0.7)),
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{OPENAI_BASE}/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            idx = 0
            full = ""
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    token = chunk["choices"][0]["delta"].get("content", "")
                except Exception:
                    continue
                if not token:
                    continue
                full += token
                yield {"type": "token", "token": token, "text": full, "token_idx": idx}
                idx += 1
    yield {"type": "done", "text": full, "token_idx": idx}


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    try:
        msg = await ws.receive_json()
        model = msg.get("model") or CURRENT_MODEL
        prompt = msg.get("prompt", "")
        params = msg.get("params", {})
        async for frame in _stream_tokens(prompt, model, params):
            await ws.send_json(frame)
    except Exception as exc:
        await ws.send_json({"type": "error", "token": "", "text": str(exc), "token_idx": 0})
    finally:
        await ws.close()


@app.get("/capabilities")
async def capabilities():
    return {"streaming": True, "supports_model_switch": True, "max_batch": 1}

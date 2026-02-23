from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import httpx
import psutil
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel

app = FastAPI(title="bench-race triton wrapper")

TRITON_HTTP_URL = os.getenv("TRITON_HTTP_URL", "http://127.0.0.1:8000")
TRITON_PORT = int(os.getenv("TRITON_PORT", "8020"))
CURRENT_MODEL = os.getenv("TRITON_MODEL", "ensemble")


class InferRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False


class SwitchRequest(BaseModel):
    model_id: str


@app.get("/health")
async def health():
    status = "ok"
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{TRITON_HTTP_URL}/v2/health/ready")
            r.raise_for_status()
    except Exception:
        status = "degraded"
    vm = psutil.virtual_memory()
    return {"status": status, "engine": "triton", "model": CURRENT_MODEL, "mem": {"ram_pct": vm.percent, "vram_pct": 0.0}}


@app.get("/models")
async def models():
    return {"models": [{"id": CURRENT_MODEL, "size": "unknown", "quant": "fp16", "arch": "transformer", "equivalence_group": CURRENT_MODEL, "supported_backends": ["triton"]}]}


@app.post("/start")
async def start(_: Dict[str, Any]):
    return {"started": True, "pid": os.getpid()}


@app.post("/stop")
async def stop():
    return {"stopped": True}


@app.post("/model/switch")
async def switch(req: SwitchRequest):
    global CURRENT_MODEL
    t0 = time.time()
    CURRENT_MODEL = req.model_id
    return {"ok": True, "load_time": round(time.time() - t0, 3)}


@app.post("/infer")
async def infer(req: InferRequest):
    text = f"[triton:{CURRENT_MODEL}] {req.prompt[:req.max_tokens]}"
    return {"text": text, "tokens": len(text.split()), "meta": {"engine": "triton", "chunked": True}}


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    try:
        msg = await ws.receive_json()
        prompt = msg.get("prompt", "")
        model = msg.get("model", CURRENT_MODEL)
        text = f"[triton:{model}] {prompt}"
        built = ""
        for i, ch in enumerate(text):
            built += ch
            await ws.send_json({"type": "token", "token": ch, "text": built, "token_idx": i})
        await ws.send_json({"type": "done", "token": "", "text": built, "token_idx": len(text)})
    except Exception as exc:
        await ws.send_json({"type": "error", "token": "", "text": str(exc), "token_idx": 0})
    finally:
        await ws.close()


@app.get("/capabilities")
async def capabilities():
    return {"streaming": True, "supports_model_switch": True, "max_batch": 1}

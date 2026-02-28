from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .adapters import MLXAdapter, TRTAdapter
from .service_manager import ServiceManager

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "backends.json"
REPO_BACKENDS_PATH = Path(__file__).resolve().parents[3] / "config" / "backends.yaml"


def _load_repo_backend_defaults() -> Dict[str, Any]:
    if not REPO_BACKENDS_PATH.exists():
        return {}
    try:
        with REPO_BACKENDS_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def load_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    repo_defaults = _load_repo_backend_defaults()
    mlx_host = (repo_defaults.get("mlx") or {}).get("host", "127.0.0.1")
    mlx_port = (repo_defaults.get("mlx") or {}).get("port", 8321)
    trt_port = (repo_defaults.get("trtllm") or {}).get("port", 8000)

    cfg["backends"]["mlx"]["base_url"] = cfg["backends"]["mlx"].get("base_url") or f"http://{mlx_host}:{mlx_port}"
    cfg["backends"]["trt"]["base_url"] = cfg["backends"]["trt"].get("base_url") or f"http://127.0.0.1:{trt_port}"

    cfg["backends"]["mlx"]["base_url"] = os.getenv("WRAPPER_MLX_BASE_URL", cfg["backends"]["mlx"]["base_url"])
    cfg["backends"]["trt"]["base_url"] = os.getenv("WRAPPER_TRT_BASE_URL", cfg["backends"]["trt"]["base_url"])
    return cfg


def route_model_to_backend(model_id: str | None, cfg: Dict[str, Any]) -> str:
    if not model_id:
        return cfg.get("default_backend", "mlx")
    for mapping in cfg.get("mappings", []):
        if model_id.startswith(mapping.get("prefix", "")):
            return mapping["backend"]
    return cfg.get("default_backend", "mlx")


def normalize_infer_response(model_id: str, backend: str, response: Dict[str, Any]) -> Dict[str, Any]:
    if "choices" in response:
        choices = response.get("choices") or [{"text": "", "index": 0, "finish_reason": None}]
        usage = response.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return {
            "id": response.get("id", str(uuid.uuid4())),
            "model": model_id,
            "engine": backend,
            "choices": choices,
            "usage": usage,
            "raw": response,
        }
    text = response.get("text", "")
    tokens = int(response.get("tokens", 0))
    return {
        "id": str(uuid.uuid4()),
        "model": model_id,
        "engine": backend,
        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": tokens, "total_tokens": tokens},
        "raw": response,
    }


cfg = load_config()
service_manager = ServiceManager()
adapters = {
    "mlx": MLXAdapter(base_url=cfg["backends"]["mlx"]["base_url"]),
    "trt": TRTAdapter(
        base_url=cfg["backends"]["trt"]["base_url"],
        run_script=cfg["backends"]["trt"].get("run_script", "agent/backends/trtllm_run.sh"),
        service_manager=service_manager,
    ),
}

active_state: Dict[str, Any] = {"backend": None, "model": None}

app = FastAPI(title="bench-race unified LLM wrapper", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/health")
async def health() -> Dict[str, Any]:
    if active_state["backend"]:
        backend = active_state["backend"]
        try:
            data = await adapters[backend].health()
            return {
                "status": data.get("status", "ok"),
                "engine": backend,
                "model": active_state.get("model") or data.get("model"),
                "mem": data.get("mem", {}),
            }
        except Exception as exc:
            return {"status": "degraded", "engine": backend, "model": active_state.get("model"), "mem": {}, "error": str(exc)}

    statuses: Dict[str, Any] = {}
    for name, adapter in adapters.items():
        try:
            statuses[name] = await adapter.health()
        except Exception as exc:
            statuses[name] = {"status": "down", "error": str(exc)}
    return {"status": "ok", "engine": None, "model": None, "mem": {}, "backends": statuses}


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    models = []
    for name, adapter in adapters.items():
        try:
            for m in await adapter.list_models():
                if isinstance(m, str):
                    models.append({"id": m, "backend": name})
                else:
                    m.setdefault("backend", name)
                    models.append(m)
        except Exception:
            continue
    return {"models": models}


@app.post("/v1/models/start")
async def start_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    model_id = payload.get("model_id")
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    backend = route_model_to_backend(model_id, cfg)
    pre = service_manager.start_backend(backend, model_id=model_id)
    result = await adapters[backend].start_model(model_id, payload.get("args") or {})
    active_state.update({"backend": backend, "model": model_id})
    return {"ok": True, "backend": backend, "model_id": model_id, "lifecycle": pre, "result": result}


@app.post("/v1/models/switch")
async def switch_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    model_id = payload.get("model_id")
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    backend = route_model_to_backend(model_id, cfg)
    pre = service_manager.switch_backend(backend, model_id)
    result = await adapters[backend].switch_model(model_id)
    active_state.update({"backend": backend, "model": model_id})
    return {"ok": True, "backend": backend, "model_id": model_id, "lifecycle": pre, "result": result}


@app.post("/v1/models/stop")
async def stop_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    backend = payload.get("backend") or active_state.get("backend")
    if not backend:
        raise HTTPException(status_code=400, detail="backend not provided and no active backend")
    lifecycle = service_manager.stop_backend(backend)
    if active_state.get("backend") == backend:
        active_state.update({"backend": None, "model": None})
    return {"ok": lifecycle.get("ok", False), "backend": backend, "lifecycle": lifecycle}


@app.post("/v1/infer")
async def infer(payload: Dict[str, Any]) -> JSONResponse:
    model_id = payload.get("model") or payload.get("model_id")
    backend = payload.get("backend") or route_model_to_backend(model_id, cfg)
    if backend not in adapters:
        raise HTTPException(status_code=400, detail=f"Unsupported backend: {backend}")
    response = await adapters[backend].infer(model_id or "", payload)
    return JSONResponse(normalize_infer_response(model_id or "", backend, response))


@app.post("/v1/infer/stream")
async def infer_stream(payload: Dict[str, Any], request: Request) -> StreamingResponse:
    model_id = payload.get("model") or payload.get("model_id")
    backend = payload.get("backend") or route_model_to_backend(model_id, cfg)
    if backend not in adapters:
        raise HTTPException(status_code=400, detail=f"Unsupported backend: {backend}")

    async def event_stream() -> AsyncGenerator[bytes, None]:
        async for chunk in adapters[backend].infer_stream(model_id or "", payload):
            if await request.is_disconnected():
                break
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run("agent.backends.wrapper.app:app", host="0.0.0.0", port=9010, reload=False)

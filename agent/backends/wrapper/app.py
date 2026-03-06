from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .adapters import MLXAdapter, TRTAdapter
from .logging_config import configure_logging
from .service_manager import ServiceManager

configure_logging()
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "backends.json"
REPO_BACKENDS_PATH = Path(__file__).resolve().parents[3] / "config" / "backends.yaml"
WRAPPER_PORT = int(os.getenv("WRAPPER_PORT", "9002"))


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

logger.info("wrapper_starting", extra={"port": WRAPPER_PORT})
app = FastAPI(title="bench-race unified LLM wrapper", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_exception", extra={"endpoint": request.url.path, "error": str(exc)})
    return JSONResponse(status_code=500, content={"error": "internal_server_error"})


@app.get("/v1/health")
async def health() -> Dict[str, Any]:
    logger.debug("health_check", extra={"endpoint": "/v1/health"})
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
            logger.exception("health_check_failed", extra={"endpoint": "/v1/health", "backend": backend, "error": str(exc)})
            return {"status": "degraded", "engine": backend, "model": active_state.get("model"), "mem": {}, "error": str(exc)}

    statuses: Dict[str, Any] = {}
    for name, adapter in adapters.items():
        try:
            statuses[name] = await adapter.health()
        except Exception as exc:
            logger.exception("backend_health_failed", extra={"endpoint": "/v1/health", "backend": name, "error": str(exc)})
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
        except Exception as exc:
            logger.exception("list_models_backend_failed", extra={"backend": name, "error": str(exc), "endpoint": "/v1/models"})
            continue
    return {"models": models}


@app.post("/v1/models/start")
async def start_model(payload: Dict[str, Any]) -> JSONResponse:
    model_id = payload.get("model_id")
    if not model_id:
        return JSONResponse(status_code=400, content={"error": "model_id is required"})

    backend = route_model_to_backend(model_id, cfg)
    logger.info("model_start_requested", extra={"endpoint": "/v1/models/start", "model_id": model_id, "backend": backend})

    try:
        pre = service_manager.start_backend(backend, model_id=model_id)
        if not pre.get("ok", False):
            raise RuntimeError(pre.get("stderr") or pre.get("stdout") or "backend lifecycle start failed")

        result = await adapters[backend].start_model(model_id, payload.get("args") or {})
        active_state.update({"backend": backend, "model": model_id})

        logger.info("model_start_success", extra={"endpoint": "/v1/models/start", "model_id": model_id, "backend": backend})
        return JSONResponse({"ok": True, "backend": backend, "model_id": model_id, "lifecycle": pre, "result": result})

    except Exception as exc:
        logger.exception("model_start_failed", extra={"endpoint": "/v1/models/start", "model_id": model_id, "backend": backend, "error": str(exc)})
        return JSONResponse(status_code=500, content={"error": "model_start_failed", "detail": str(exc)})


@app.post("/v1/models/switch")
async def switch_model(payload: Dict[str, Any]) -> JSONResponse:
    model_id = payload.get("model_id")
    if not model_id:
        return JSONResponse(status_code=400, content={"error": "model_id is required"})

    backend = route_model_to_backend(model_id, cfg)
    try:
        pre = service_manager.switch_backend(backend, model_id)
        if not pre.get("ok", False):
            raise RuntimeError(pre.get("stderr") or pre.get("stdout") or "backend switch failed")
        result = await adapters[backend].switch_model(model_id)
        active_state.update({"backend": backend, "model": model_id})
        return JSONResponse({"ok": True, "backend": backend, "model_id": model_id, "lifecycle": pre, "result": result})
    except Exception as exc:
        logger.exception("model_switch_failed", extra={"endpoint": "/v1/models/switch", "model_id": model_id, "backend": backend, "error": str(exc)})
        return JSONResponse(status_code=500, content={"error": "model_switch_failed", "detail": str(exc)})


@app.post("/v1/models/stop")
async def stop_model(payload: Dict[str, Any]) -> JSONResponse:
    backend = payload.get("backend") or active_state.get("backend")
    if not backend:
        return JSONResponse(status_code=400, content={"error": "backend not provided and no active backend"})

    logger.info("model_stop_requested", extra={"endpoint": "/v1/models/stop", "backend": backend, "model_id": active_state.get("model")})
    try:
        lifecycle = service_manager.stop_backend(backend)
        if active_state.get("backend") == backend:
            active_state.update({"backend": None, "model": None})
        return JSONResponse({"ok": lifecycle.get("ok", False), "backend": backend, "lifecycle": lifecycle})
    except Exception as exc:
        logger.exception("model_stop_failed", extra={"endpoint": "/v1/models/stop", "backend": backend, "error": str(exc)})
        return JSONResponse(status_code=500, content={"error": "model_stop_failed", "detail": str(exc)})


@app.post("/v1/infer")
async def infer(payload: Dict[str, Any]) -> JSONResponse:
    model_id = payload.get("model") or payload.get("model_id")
    backend = payload.get("backend") or route_model_to_backend(model_id, cfg)
    if backend not in adapters:
        return JSONResponse(status_code=400, content={"error": f"Unsupported backend: {backend}"})
    logger.info("inference_request", extra={"endpoint": "/v1/infer", "model_id": active_state.get("model") or model_id, "backend": backend})
    try:
        response = await adapters[backend].infer(model_id or "", payload)
        return JSONResponse(normalize_infer_response(model_id or "", backend, response))
    except Exception as exc:
        logger.exception("inference_failed", extra={"endpoint": "/v1/infer", "model_id": model_id, "backend": backend, "error": str(exc)})
        return JSONResponse(status_code=500, content={"error": "inference_failed", "detail": str(exc)})


@app.post("/v1/infer/stream")
async def infer_stream(payload: Dict[str, Any], request: Request) -> StreamingResponse:
    model_id = payload.get("model") or payload.get("model_id")
    backend = payload.get("backend") or route_model_to_backend(model_id, cfg)
    if backend not in adapters:
        raise HTTPException(status_code=400, detail=f"Unsupported backend: {backend}")

    async def event_stream() -> AsyncGenerator[bytes, None]:
        try:
            async for chunk in adapters[backend].infer_stream(model_id or "", payload):
                if await request.is_disconnected():
                    break
                yield chunk
        except Exception as exc:
            logger.exception("infer_stream_failed", extra={"endpoint": "/v1/infer/stream", "backend": backend, "model_id": model_id, "error": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/plain")


def _ollama_model_entry(model_id: str, backend: str) -> Dict[str, Any]:
    return {
        "name": model_id,
        "model": model_id,
        "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "size": 0,
        "digest": "",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": backend,
            "families": [backend],
            "parameter_size": "",
            "quantization_level": "",
        },
    }


@app.get("/api/tags")
async def ollama_tags() -> Dict[str, Any]:
    models: List[Dict[str, Any]] = []
    for name, adapter in adapters.items():
        try:
            for m in await adapter.list_models():
                mid = m["id"] if isinstance(m, dict) else str(m)
                models.append(_ollama_model_entry(mid, name))
        except Exception as exc:
            logger.exception("ollama_tags_backend_failed", extra={"endpoint": "/api/tags", "backend": name, "error": str(exc)})
            continue
    return {"models": models}


@app.post("/api/generate")
async def ollama_generate(payload: Dict[str, Any], request: Request) -> StreamingResponse:
    model_id = payload.get("model", "")
    prompt = payload.get("prompt", "")
    stream = payload.get("stream", True)
    backend = route_model_to_backend(model_id, cfg)

    if backend not in adapters:
        return JSONResponse(status_code=400, content={"error": f"Unsupported backend: {backend}"})

    if not active_state.get("backend"):
        return JSONResponse(status_code=503, content={"error": "No managed engine running. Start a model first."})

    infer_payload = {
        "prompt": prompt,
        "max_tokens": int(payload.get("options", {}).get("num_predict", 256)),
        "temperature": float(payload.get("options", {}).get("temperature", 0.7)),
    }
    logger.info("inference_request", extra={"endpoint": "/api/generate", "model_id": active_state.get("model") or model_id, "backend": backend})

    if not stream:
        try:
            response = await adapters[backend].infer(model_id, {**infer_payload, "stream": False})
            text = response.get("text", "")
            tokens = int(response.get("tokens", 0))
            return JSONResponse({
                "model": model_id,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "response": text,
                "done": True,
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": 0,
                "eval_count": tokens,
                "eval_duration": 0,
            })
        except Exception as exc:
            logger.exception("ollama_generate_failed", extra={"endpoint": "/api/generate", "model_id": model_id, "backend": backend, "error": str(exc)})
            return JSONResponse(status_code=500, content={"error": "inference_failed", "detail": str(exc)})

    async def _stream() -> AsyncGenerator[bytes, None]:
        try:
            async for chunk in adapters[backend].infer_stream(model_id, infer_payload):
                if await request.is_disconnected():
                    break
                text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
                frame = json.dumps({
                    "model": model_id,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "response": text,
                    "done": False,
                })
                yield (frame + "\n").encode("utf-8")
        except Exception as exc:
            logger.exception("ollama_generate_stream_failed", extra={"endpoint": "/api/generate", "model_id": model_id, "backend": backend, "error": str(exc)})
        done_frame = json.dumps({
            "model": model_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "response": "",
            "done": True,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "eval_count": 0,
            "eval_duration": 0,
        })
        yield (done_frame + "\n").encode("utf-8")

    return StreamingResponse(_stream(), media_type="application/x-ndjson")


def run() -> None:
    uvicorn.run("agent.backends.wrapper.app:app", host="0.0.0.0", port=WRAPPER_PORT, reload=False)


if __name__ == "__main__":
    run()

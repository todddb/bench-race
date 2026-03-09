from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

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
    trt_port = (repo_defaults.get("trtllm") or {}).get("port", 8000)

    cfg["backends"]["trt"]["base_url"] = cfg["backends"]["trt"].get("base_url") or f"http://127.0.0.1:{trt_port}"

    cfg["backends"]["trt"]["base_url"] = os.getenv("WRAPPER_TRT_BASE_URL", cfg["backends"]["trt"]["base_url"])
    return cfg


def route_model_to_backend(model_id: str | None, cfg: Dict[str, Any]) -> str:
    """Deprecated: Retained only for inference endpoints during transition.
    All model start/switch requests now require an explicit backend from Central."""
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
    "mlx": MLXAdapter(),
    "trt": TRTAdapter(
        base_url=cfg["backends"]["trt"]["base_url"],
        run_script=cfg["backends"]["trt"].get("run_script", "agent/backends/trtllm_run.sh"),
        service_manager=service_manager,
    ),
}

active_state: Dict[str, Any] = {"backend": None, "model": None}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


def _public_backend_name(backend: str | None) -> str | None:
    if backend == "trt":
        return "trtllm"
    return backend

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
    backend = active_state.get("backend")
    model = active_state.get("model")

    if not backend:
        return {"status": "ok", "engine": None, "model": None, "mem": {}}

    adapter = adapters.get(backend)
    public_backend = _public_backend_name(backend)
    if adapter is None:
        logger.error("health_check_unknown_backend", extra={"endpoint": "/v1/health", "backend": backend})
        return {"status": "down", "engine": public_backend, "model": model, "mem": {}}

    try:
        data = await adapter.health()
        return {
            "status": "ok" if model else "down",
            "engine": public_backend,
            "model": model,
            "mem": data.get("mem", {}) if isinstance(data, dict) else {},
        }
    except Exception as exc:
        logger.exception("health_check_failed", extra={"endpoint": "/v1/health", "backend": backend, "error": str(exc)})
        return {"status": "down", "engine": public_backend, "model": model, "mem": {}}


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    models = []
    for name, adapter in adapters.items():
        try:
            for m in await adapter.list_models():
                if isinstance(m, str):
                    models.append({"id": m, "backend": _public_backend_name(name) or name})
                else:
                    m.setdefault("backend", _public_backend_name(name) or name)
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

    # Central/Agent must provide explicit backend — no heuristic routing
    explicit_backend = (payload.get("backend") or "").strip().lower()
    _backend_map = {"mlx": "mlx", "trtllm": "trt", "ollama": "ollama"}
    if explicit_backend not in _backend_map:
        return JSONResponse(status_code=400, content={
            "error": f"Explicit backend required ('ollama', 'mlx' or 'trtllm'), got: '{explicit_backend}'"
        })
    backend = _backend_map[explicit_backend]

    logger.info("model_start_requested", extra={"endpoint": "/v1/models/start", "model_id": model_id, "backend": explicit_backend})

    try:
        current_backend = active_state.get("backend")
        if current_backend and current_backend != backend:
            stopped = await service_manager.ensure_backend_stopped(current_backend)
            if not stopped.get("ok", False):
                raise RuntimeError(stopped.get("stderr") or stopped.get("stdout") or "failed to stop inactive backend")
            prev_adapter = adapters.get(current_backend)
            if prev_adapter is not None and hasattr(prev_adapter, "stop_model"):
                await prev_adapter.stop_model()

        pre = await service_manager.ensure_backend_running(backend, model_id=model_id)
        if not pre.get("ok", False):
            raise RuntimeError(pre.get("stderr") or pre.get("stdout") or "backend lifecycle start failed")

        adapter = adapters.get(backend)
        result: Dict[str, Any] = {"ok": True, "backend": backend, "model": model_id}
        if adapter is not None:
            result = await adapter.start_model(model_id, payload.get("args") or {})

        active_state.update({"backend": backend, "model": model_id})

        logger.info("model_start_success", extra={"endpoint": "/v1/models/start", "model_id": model_id, "backend": backend})
        return JSONResponse({"ok": True, "backend": explicit_backend, "model_id": model_id, "lifecycle": pre, "result": result})

    except Exception as exc:
        logger.exception("model_start_failed", extra={"endpoint": "/v1/models/start", "model_id": model_id, "backend": explicit_backend, "error": str(exc)})
        return JSONResponse(status_code=500, content={"error": "model_start_failed", "detail": str(exc)})


@app.post("/v1/models/switch")
async def switch_model(payload: Dict[str, Any]) -> JSONResponse:
    model_id = payload.get("model_id")
    if not model_id:
        return JSONResponse(status_code=400, content={"error": "model_id is required"})

    explicit_backend = (payload.get("backend") or "").strip().lower()
    _backend_map = {"mlx": "mlx", "trtllm": "trt", "ollama": "ollama"}
    if explicit_backend not in _backend_map:
        return JSONResponse(status_code=400, content={
            "error": f"Explicit backend required ('ollama', 'mlx' or 'trtllm'), got: '{explicit_backend}'"
        })
    backend = _backend_map[explicit_backend]
    try:
        current_backend = active_state.get("backend")
        if current_backend and current_backend != backend:
            stopped = await service_manager.ensure_backend_stopped(current_backend)
            if not stopped.get("ok", False):
                raise RuntimeError(stopped.get("stderr") or stopped.get("stdout") or "failed to stop inactive backend")
            prev_adapter = adapters.get(current_backend)
            if prev_adapter is not None and hasattr(prev_adapter, "stop_model"):
                await prev_adapter.stop_model()

        pre = await service_manager.ensure_backend_running(backend, model_id)
        if not pre.get("ok", False):
            raise RuntimeError(pre.get("stderr") or pre.get("stdout") or "backend switch failed")
        adapter = adapters.get(backend)
        result: Dict[str, Any] = {"ok": True, "backend": backend, "model": model_id}
        if adapter is not None:
            result = await adapter.switch_model(model_id)
        active_state.update({"backend": backend, "model": model_id})
        return JSONResponse({"ok": True, "backend": explicit_backend, "model_id": model_id, "lifecycle": pre, "result": result})
    except Exception as exc:
        logger.exception("model_switch_failed", extra={"endpoint": "/v1/models/switch", "model_id": model_id, "backend": explicit_backend, "error": str(exc)})
        return JSONResponse(status_code=500, content={"error": "model_switch_failed", "detail": str(exc)})


@app.post("/v1/models/stop")
async def stop_model(payload: Dict[str, Any]) -> JSONResponse:
    backend = payload.get("backend") or active_state.get("backend")
    if backend == "trtllm":
        backend = "trt"
    if not backend:
        return JSONResponse(status_code=400, content={"error": "backend not provided and no active backend"})

    logger.info("model_stop_requested", extra={"endpoint": "/v1/models/stop", "backend": backend, "model_id": active_state.get("model")})
    try:
        lifecycle = await service_manager.ensure_backend_stopped(backend)
        adapter = adapters.get(backend)
        if adapter is not None and hasattr(adapter, "stop_model"):
            await adapter.stop_model()
        if active_state.get("backend") == backend:
            active_state.update({"backend": None, "model": None})
        return JSONResponse({"ok": lifecycle.get("ok", False), "backend": backend, "lifecycle": lifecycle})
    except Exception as exc:
        logger.exception("model_stop_failed", extra={"endpoint": "/v1/models/stop", "backend": backend, "error": str(exc)})
        return JSONResponse(status_code=500, content={"error": "model_stop_failed", "detail": str(exc)})


@app.post("/v1/infer")
async def infer(payload: Dict[str, Any]) -> JSONResponse:
    model_id = payload.get("model") or payload.get("model_id")
    backend = payload.get("backend") or active_state.get("backend")
    try:
        normalized = await infer_internal(model_id=model_id or "", payload=payload, backend=backend, endpoint="/v1/infer")
        return JSONResponse(normalized)
    except Exception as exc:
        if isinstance(exc, HTTPException):
            return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
        logger.exception("inference_failed", extra={"endpoint": "/v1/infer", "model_id": model_id, "backend": backend, "error": str(exc)})
        return JSONResponse(status_code=500, content={"error": "inference_failed", "detail": str(exc)})


async def infer_internal(model_id: str, payload: Dict[str, Any], backend: Optional[str], endpoint: str) -> Dict[str, Any]:
    resolved_backend = backend
    if resolved_backend == "trtllm":
        resolved_backend = "trt"
    if not resolved_backend:
        raise HTTPException(status_code=400, detail="No backend specified and no active backend. Start a model first.")

    logger.info("inference_request", extra={"endpoint": endpoint, "model_id": active_state.get("model") or model_id, "backend": resolved_backend})

    if resolved_backend == "ollama":
        ollama_model = model_id or active_state.get("model")
        if not ollama_model:
            raise HTTPException(status_code=400, detail="model is required for ollama inference")
        ollama_payload = {
            "model": ollama_model,
            "prompt": payload.get("prompt") or payload.get("inputs") or "",
            "stream": False,
            "options": {
                "temperature": float(payload.get("temperature", 0.7)),
                "num_predict": int(payload.get("max_tokens", 256)),
            },
        }
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(f"{service_manager.ollama_base_url}/api/generate", json=ollama_payload)
            resp.raise_for_status()
            data = resp.json()
        response = {
            "text": data.get("response", ""),
            "tokens": int(data.get("eval_count") or 0),
        }
        return normalize_infer_response(ollama_model, "ollama", response)

    if resolved_backend not in adapters:
        raise HTTPException(status_code=400, detail=f"Unsupported backend: {resolved_backend}")

    response = await adapters[resolved_backend].infer(model_id or "", payload)
    return normalize_infer_response(model_id or "", _public_backend_name(resolved_backend) or resolved_backend, response)


def _build_chat_prompt(messages: List[ChatMessage]) -> str:
    prompt_parts: List[str] = []
    for msg in messages:
        if msg.role == "system":
            prompt_parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            prompt_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}")
    return "\n".join(prompt_parts) + "\nAssistant:"


def _build_chat_infer_payload(payload: ChatCompletionRequest, prompt: str) -> Dict[str, Any]:
    infer_payload: Dict[str, Any] = {
        "model": payload.model,
        "prompt": prompt,
    }
    if payload.max_tokens is not None:
        infer_payload["max_tokens"] = payload.max_tokens
    if payload.temperature is not None:
        infer_payload["temperature"] = payload.temperature
    return infer_payload


@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, request: Request):
    prompt = _build_chat_prompt(payload.messages)
    infer_payload = _build_chat_infer_payload(payload, prompt)

    if payload.stream:
        backend = active_state.get("backend")
        if backend == "trtllm":
            backend = "trt"
        if not backend:
            raise HTTPException(status_code=400, detail="No backend specified and no active backend. Start a model first.")
        if backend not in adapters:
            raise HTTPException(status_code=400, detail=f"Unsupported backend: {backend}")

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        model_id = payload.model

        async def sse_generator() -> AsyncGenerator[str, None]:
            try:
                async for chunk in adapters[backend].infer_stream(model_id, infer_payload):
                    if await request.is_disconnected():
                        break
                    raw = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
                    if raw.startswith("data: "):
                        # TRT native SSE frame — extract token text
                        payload_str = raw[6:].strip()
                        if payload_str == "[DONE]":
                            break
                        try:
                            parsed = json.loads(payload_str)
                        except json.JSONDecodeError:
                            continue
                        choice = parsed.get("choices", [{}])[0]
                        token_text = ""
                        # Chat-style streaming format
                        if "delta" in choice:
                            token_text = choice["delta"].get("content", "") or ""
                        # Text-completion streaming format
                        elif "text" in choice:
                            token_text = choice.get("text", "") or ""
                        # Skip frames that contain no token content
                        if not token_text:
                            continue
                    else:
                        # Raw token text (MLX synthetic streaming)
                        token_text = raw

                    frame = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token_text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(frame)}\n\n"
            except Exception as exc:
                logger.exception("chat_completions_stream_failed", extra={
                    "endpoint": "/v1/chat/completions",
                    "model_id": model_id,
                    "backend": backend,
                    "error": str(exc),
                })
                return

            # Final chunk with finish_reason
            final_frame = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_frame)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    # Non-streaming path
    normalized = await infer_internal(
        model_id=payload.model,
        payload=infer_payload,
        backend=active_state.get("backend"),
        endpoint="/v1/chat/completions",
    )

    choices = normalized.get("choices") or []
    first_choice = choices[0] if choices else {}
    text_output = first_choice.get("text") or first_choice.get("message", {}).get("content") or ""
    finish_reason = first_choice.get("finish_reason") or "stop"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text_output,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
    }


@app.post("/v1/infer/stream")
async def infer_stream(payload: Dict[str, Any], request: Request) -> StreamingResponse:
    model_id = payload.get("model") or payload.get("model_id")
    backend = payload.get("backend") or active_state.get("backend")
    if not backend:
        raise HTTPException(status_code=400, detail="No backend specified and no active backend. Start a model first.")
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
                models.append(_ollama_model_entry(mid, _public_backend_name(name) or name))
        except Exception as exc:
            logger.exception("ollama_tags_backend_failed", extra={"endpoint": "/api/tags", "backend": name, "error": str(exc)})
            continue
    return {"models": models}


@app.post("/api/generate")
async def ollama_generate(payload: Dict[str, Any], request: Request) -> StreamingResponse:
    model_id = payload.get("model", "")
    prompt = payload.get("prompt", "")
    stream = payload.get("stream", True)
    backend = active_state.get("backend")
    if not backend:
        return JSONResponse(status_code=503, content={"error": "No managed engine running. Start a model first."})

    if backend not in adapters:
        return JSONResponse(status_code=400, content={"error": f"Unsupported backend: {backend}"})

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

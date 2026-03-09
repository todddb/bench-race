"""
MLX backend wrapper for bench-race.

A small FastAPI application that provides a standardised API (the Backend
Contract) around Apple MLX on macOS.  It exposes health, model listing,
start/stop, model switching, synchronous inference, streaming via WebSocket,
and a capabilities endpoint.

Run directly:
    python -m uvicorn agent.backends.mlx.server:app --host 127.0.0.1 --port 8321

Or via the lifecycle helper:
    ./scripts/agent start-mlx
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

MLX_MODELS_DIR = os.getenv("MLX_MODELS_DIR")

MLX_MODELS_ROOT = Path(__file__).resolve().parents[2] / "models" / "mlx"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("MLX_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("mlx_backend")

# ---------------------------------------------------------------------------
# Lazy MLX imports — only loaded when actually needed so the module can still
# be imported on non-macOS systems (e.g. for linting / tests).
# ---------------------------------------------------------------------------
_mlx_lm = None
_mx = None


def _ensure_mlx():
    """Import mlx / mlx_lm on first use."""
    global _mlx_lm, _mx
    if _mlx_lm is None:
        try:
            import mlx.core as mx  # type: ignore[import-untyped]
            import mlx_lm  # type: ignore[import-untyped]

            _mx = mx
            _mlx_lm = mlx_lm
        except ImportError as exc:
            log.error(
                "Failed to import mlx / mlx_lm.  Make sure you are on macOS "
                "with Apple Silicon and the venv has been set up via "
                "scripts/install_macos_mlx.sh: %s",
                exc,
            )
            raise


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

class _State:
    """Mutable singleton holding the currently-loaded model."""

    model: Any = None
    tokenizer: Any = None
    model_id: str = ""
    pid: Optional[int] = None


_state = _State()

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    model_id: str
    args: Dict[str, Any] = Field(default_factory=dict)


class SwitchRequest(BaseModel):
    model_id: str


class InferRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False


class StreamStart(BaseModel):
    model: str = ""
    prompt: str
    params: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mem_stats() -> Dict[str, float]:
    """Return approximate RAM/VRAM usage percentages."""
    try:
        import psutil  # type: ignore[import-untyped]
        ram_pct = psutil.virtual_memory().percent
    except Exception:
        ram_pct = -1.0

    # On Apple Silicon, VRAM is unified with system RAM.
    # mlx.core.metal.get_active_memory() gives current Metal allocation.
    vram_pct = 0.0
    try:
        _ensure_mlx()
        active = _mx.metal.get_active_memory()  # bytes
        peak = _mx.metal.get_peak_memory()  # bytes
        # Express as pct of peak (rough proxy)
        if peak > 0:
            vram_pct = round(active / peak * 100, 1)
    except Exception:
        pass
    return {"ram_pct": round(ram_pct, 1), "vram_pct": vram_pct}


def _load_model(model_id: str) -> float:
    """Load *model_id* via mlx_lm and store in global state.  Returns load
    time in seconds."""
    _ensure_mlx()
    t0 = time.perf_counter()
    log.info("Loading model %s …", model_id)
    model_path = MLX_MODELS_ROOT / model_id
    if not model_path.exists():
        raise FileNotFoundError(
            f"Local MLX model not found at {model_path}"
        )
    model, tokenizer = _mlx_lm.load(str(model_path))
    elapsed = time.perf_counter() - t0
    _state.model = model
    _state.tokenizer = tokenizer
    _state.model_id = model_id
    log.info("Model %s loaded in %.2fs", model_id, elapsed)
    return elapsed


def _generate_sync(prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
    """Run a non-streaming generation and return result dict."""
    _ensure_mlx()
    if _state.model is None:
        raise RuntimeError("No model loaded. POST /start first.")

    t0 = time.perf_counter()
    response = _mlx_lm.generate(
        _state.model,
        _state.tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    t_end = time.perf_counter()
    total_ms = (t_end - t0) * 1000.0
    tokens = len(_state.tokenizer.encode(response))
    return {
        "text": response,
        "tokens": tokens,
        "meta": {
            "model": _state.model_id,
            "engine": "mlx",
            "total_ms": round(total_ms, 2),
            "tokens_per_s": round(tokens / (total_ms / 1000.0), 2) if total_ms > 0 else 0,
        },
    }


async def _stream_generate(
    prompt: str,
    max_tokens: int,
    temperature: float,
    websocket: WebSocket,
):
    """Stream tokens over *websocket* as JSON text frames."""
    _ensure_mlx()
    if _state.model is None:
        await websocket.send_text(
            json.dumps({"type": "error", "error": "No model loaded. POST /start first."})
        )
        return

    token_idx = 0
    full_text = ""
    stats: Dict[str, Any] = {}
    t0 = time.perf_counter()

    try:
        for token_obj in _mlx_lm.stream_generate(
            _state.model,
            _state.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
        ):
            # mlx_lm.stream_generate yields objects with a .text attribute
            # (the incrementally-decoded string so far) and optionally
            # .token (the token id).
            if hasattr(token_obj, "text"):
                new_text = token_obj.text[len(full_text):]
                full_text = token_obj.text
            else:
                new_text = str(token_obj)
                full_text += new_text

            if new_text:
                frame = {
                    "type": "token",
                    "token": new_text,
                    "text": full_text,
                    "token_idx": token_idx,
                }
                await websocket.send_text(json.dumps(frame))
                token_idx += 1
                # Yield control so the event loop can flush the frame
                await asyncio.sleep(0)

        t_end = time.perf_counter()
        total_ms = (t_end - t0) * 1000.0
        stats = {
            "total_ms": round(total_ms, 2),
            "tokens": token_idx,
            "tokens_per_s": round(token_idx / (total_ms / 1000.0), 2) if total_ms > 0 else 0,
        }
    except Exception as exc:
        log.error("Streaming error: %s", exc)
        await websocket.send_text(
            json.dumps({"type": "error", "error": str(exc)})
        )
        return

    # Final "done" frame
    done_frame = {"type": "done", "text": full_text, "stats": stats}
    await websocket.send_text(json.dumps(done_frame))


# ---------------------------------------------------------------------------
# Available models helper (scan HuggingFace cache or local dirs)
# ---------------------------------------------------------------------------

def _discover_models() -> List[Dict[str, Any]]:
    """Return a list of locally-available MLX model dicts.

    Scans the HuggingFace Hub cache and an optional local ``MLX_MODELS_DIR``
    directory for model directories.
    """
    models: List[Dict[str, Any]] = []
    seen: set[str] = set()

    # 1) Check explicit local model directory
    local_dir = os.getenv("MLX_MODELS_DIR", "")
    if local_dir and Path(local_dir).is_dir():
        for entry in sorted(Path(local_dir).iterdir()):
            if entry.is_dir() and (entry / "config.json").exists():
                mid = entry.name
                if mid not in seen:
                    seen.add(mid)
                    models.append({
                        "id": mid,
                        "size": None,
                        "quant": _guess_quant(entry),
                        "arch": None,
                        "equivalence_group": mid,
                        "supported_backends": ["mlx"],
                    })

    # 2) Scan HuggingFace cache
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.is_dir():
        for entry in sorted(hf_cache.iterdir()):
            if entry.is_dir() and entry.name.startswith("models--"):
                mid = entry.name.replace("models--", "").replace("--", "/")
                if mid not in seen:
                    seen.add(mid)
                    models.append({
                        "id": mid,
                        "size": None,
                        "quant": None,
                        "arch": None,
                        "equivalence_group": mid,
                        "supported_backends": ["mlx"],
                    })

    return models


def _discover_models_v1() -> List[Dict[str, str]]:
    """Return OpenAI-compatible model list entries for /v1/models."""
    models: List[Dict[str, str]] = []
    seen: set[str] = set()

    if _state.model_id:
        seen.add(_state.model_id)
        models.append(
            {
                "id": _state.model_id,
                "object": "model",
                "owned_by": "mlx",
            }
        )

    if MLX_MODELS_DIR:
        base = Path(MLX_MODELS_DIR)
        if base.is_dir():
            with os.scandir(base) as entries:
                for entry in sorted(entries, key=lambda e: e.name):
                    if not entry.is_dir() or entry.name.startswith("."):
                        continue
                    if entry.name in seen:
                        continue
                    seen.add(entry.name)
                    models.append(
                        {
                            "id": entry.name,
                            "object": "model",
                            "owned_by": "mlx",
                        }
                    )

    return models


def _guess_quant(model_dir: Path) -> Optional[str]:
    """Try to guess quantisation from directory name or config."""
    name = model_dir.name.lower()
    for q in ("4bit", "8bit", "fp16", "bf16", "q4", "q8"):
        if q in name:
            return q
    return None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="MLX Backend", version="0.1.0")


@app.get("/health")
async def health():
    return {
        "status": "ok" if _state.model is not None else "idle",
        "engine": "mlx",
        "model": _state.model_id or None,
        "mem": _mem_stats(),
    }


@app.get("/models")
async def list_models():
    return {"models": _discover_models()}


@app.get("/v1/models")
async def list_models_v1():
    models = await asyncio.to_thread(_discover_models_v1)
    return {"object": "list", "data": models}


@app.post("/start")
async def start_model(req: StartRequest):
    try:
        load_time = _load_model(req.model_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _state.pid = os.getpid()
    return {"started": True, "pid": _state.pid, "load_time": round(load_time, 3)}


@app.post("/stop")
async def stop_model():
    _state.model = None
    _state.tokenizer = None
    prev = _state.model_id
    _state.model_id = ""
    log.info("Model %s unloaded", prev)
    return {"stopped": True}


@app.post("/model/switch")
async def switch_model(req: SwitchRequest):
    # Unload current
    _state.model = None
    _state.tokenizer = None
    load_time = _load_model(req.model_id)
    return {"ok": True, "load_time": round(load_time, 3)}


@app.post("/infer")
async def infer(req: InferRequest):
    result = await asyncio.to_thread(
        _generate_sync, req.prompt, req.max_tokens, req.temperature
    )
    return result


@app.websocket("/stream")
async def stream_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        raw = await websocket.receive_text()
        data = json.loads(raw)
        start = StreamStart(**data)

        model_id = start.model or _state.model_id
        # If a different model is requested, switch
        if model_id and model_id != _state.model_id:
            try:
                _load_model(model_id)
            except Exception as exc:
                await websocket.send_text(
                    json.dumps({"type": "error", "error": f"Failed to load model: {exc}"})
                )
                await websocket.close()
                return

        max_tokens = start.params.get("max_tokens", 256)
        temperature = start.params.get("temperature", 0.7)

        await _stream_generate(
            prompt=start.prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            websocket=websocket,
        )
    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
    except json.JSONDecodeError as exc:
        log.error("Invalid JSON on WebSocket: %s", exc)
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "error": "Invalid JSON"})
            )
        except Exception:
            pass
    except Exception as exc:
        log.error("WebSocket handler error: %s", exc)
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "error": str(exc)})
            )
        except Exception:
            pass


@app.get("/capabilities")
async def capabilities():
    return {
        "streaming": True,
        "supports_model_switch": True,
        "max_batch": 1,
    }


# ---------------------------------------------------------------------------
# Graceful shutdown on SIGTERM / SIGINT
# ---------------------------------------------------------------------------

def _handle_signal(sig, _frame):
    log.info("Received signal %s — shutting down", signal.Signals(sig).name)
    sys.exit(0)


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("MLX_HOST", "127.0.0.1")
    port = int(os.getenv("MLX_PORT", "8321"))
    uvicorn.run(app, host=host, port=port, log_level="info")

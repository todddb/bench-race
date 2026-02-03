from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import yaml
import httpx
import websockets
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------
# Path and import robustness
# ---------------------------
AGENT_DIR = Path(__file__).resolve().parent            # .../bench-race/agent
ROOT_DIR = AGENT_DIR.parent                            # .../bench-race
CONFIG_PATH = AGENT_DIR / "config" / "agent.yaml"

# Make repo root importable so `shared` works
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# now import shared pydantic schemas
from shared.schemas import Capabilities, Event, JobStartResponse, LLMRequest  # type: ignore
from backends.ollama_backend import check_ollama_available, stream_ollama_generate, get_ollama_models

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bench-agent")

# ---------------------------
# Load config
# ---------------------------
DEPRECATED_MODEL_KEYS = ("llm_models", "whisper_models", "sdxl_profiles")


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if any(key in config for key in DEPRECATED_MODEL_KEYS):
        log.warning("Model lists in agent.yaml are deprecated; central controls required models now.")
        for key in DEPRECATED_MODEL_KEYS:
            config.pop(key, None)
    return config


CFG = load_config()

# ---------------------------
# FastAPI app + websockets
# ---------------------------
app = FastAPI(title="bench-race agent")

# Manage connected websocket clients
# Each entry: client_id -> WebSocket
WS_CLIENTS: Dict[str, WebSocket] = {}

# Track running jobs (job_id -> asyncio.Task)
RUNNING_JOBS: Dict[str, asyncio.Task] = {}
SYNC_TASKS: Dict[str, asyncio.Task] = {}
CHECKPOINT_SYNC_STATUS: Dict[str, Any] = {"active": False, "results": []}


class SyncRequest(BaseModel):
    llm: List[str] = Field(default_factory=list)
    whisper: List[str] = Field(default_factory=list)
    sdxl_profiles: List[str] = Field(default_factory=list)


class ComfyTxt2ImgRequest(BaseModel):
    run_id: str
    prompt: str
    checkpoint: str
    seed: int
    steps: int = 30
    width: int = 1024
    height: int = 1024
    num_images: int = 1
    repeat: int = 1


class ComfySyncRequest(BaseModel):
    checkpoints: List[str] = Field(default_factory=list)
    central_base_url: Optional[str] = None


class ComfyCheckpointItem(BaseModel):
    name: str
    url: str
    resolved_url: Optional[str] = None
    size_bytes: Optional[int] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None


class ComfyCheckpointSyncRequest(BaseModel):
    items: List[ComfyCheckpointItem] = Field(default_factory=list)


# ---------------------------
# Helpers: websocket broadcast
# ---------------------------
async def _broadcast_event(event: Event):
    """Send event to all connected websocket clients (best-effort)."""
    to_remove: List[str] = []
    payload = event.model_dump()
    txt = json.dumps(payload)
    for cid, ws in list(WS_CLIENTS.items()):
        try:
            await ws.send_text(txt)
        except Exception as exc:
            log.warning("WebSocket send failed for %s: %s", cid, exc)
            to_remove.append(cid)
    for cid in to_remove:
        WS_CLIENTS.pop(cid, None)


def _comfy_config() -> Dict[str, Any]:
    return CFG.get("comfyui", {}) if isinstance(CFG, dict) else {}


def _comfy_base_url() -> str:
    comfy = _comfy_config()
    base = comfy.get("base_url")
    if base:
        return base.rstrip("/")
    port = comfy.get("port", 8188)
    host = comfy.get("host", "127.0.0.1")
    return f"http://{host}:{port}"


def _comfy_checkpoints_dir() -> Path:
    comfy = _comfy_config()
    path = comfy.get("checkpoints_dir") or ""
    if path:
        return Path(path)
    return Path.home() / "comfyui" / "models" / "checkpoints"


def _list_checkpoints() -> List[str]:
    path = _comfy_checkpoints_dir()
    if not path.exists():
        return []
    return sorted([p.name for p in path.iterdir() if p.is_file()])


def _list_checkpoint_items() -> List[Dict[str, Any]]:
    path = _comfy_checkpoints_dir()
    if not path.exists():
        return []
    items = []
    for entry in path.iterdir():
        if not entry.is_file():
            continue
        try:
            stat = entry.stat()
        except OSError:
            continue
        items.append(
            {
                "name": entry.name,
                "size_bytes": stat.st_size,
                "path": str(entry),
                "mtime": int(stat.st_mtime),
            }
        )
    return sorted(items, key=lambda item: item["name"])


def _comfy_debug_enabled() -> bool:
    """Check if ComfyUI debug logging is enabled (via env var or config)."""
    # Check environment variable first
    env_debug = os.getenv("COMFY_DEBUG", "").strip()
    if env_debug in ("1", "true", "True", "TRUE", "yes", "Yes", "YES"):
        return True
    # Check config
    comfy = _comfy_config()
    return comfy.get("debug", False)


def _comfy_debug_dir() -> Path:
    """Get the debug directory for ComfyUI payloads."""
    debug_dir = AGENT_DIR / "logs" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def _rotate_debug_files(debug_dir: Path, pattern: str, max_files: int = 50):
    """Keep only the most recent N debug files matching the pattern."""
    files = sorted(debug_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for old_file in files[max_files:]:
        try:
            old_file.unlink()
        except Exception as e:
            log.warning("Failed to delete old debug file %s: %s", old_file, e)


def _save_comfy_debug_payload(workflow: Dict[str, Any], run_id: str) -> Optional[str]:
    """Save ComfyUI workflow payload to debug file with rotation. Returns file path or None."""
    if not _comfy_debug_enabled():
        return None

    try:
        debug_dir = _comfy_debug_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"comfy_prompt_{run_id}_{timestamp}.json"
        filepath = debug_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(workflow, f, indent=2)

        # Rotate old files
        _rotate_debug_files(debug_dir, "comfy_prompt_*.json", max_files=50)

        return str(filepath)
    except Exception as e:
        log.error("Failed to save debug payload: %s", e)
        return None


def _summarize_comfy_payload(workflow: Dict[str, Any]) -> str:
    """Create a summary of the ComfyUI workflow payload for logging."""
    try:
        top_level_keys = list(workflow.keys())
        node_count = len(workflow) if isinstance(workflow, dict) else 0
        return f"keys={top_level_keys}, node_count={node_count}"
    except Exception:
        return "unable to summarize"


def _validate_comfy_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean ComfyUI workflow to ensure only valid nodes are included.

    ComfyUI requires that the prompt graph contains ONLY node IDs, where each node
    is a dict with a "class_type" property. Any other keys (like "outputs", metadata, etc.)
    will cause a 400 invalid_prompt error.

    Returns a cleaned workflow with only valid nodes.
    """
    if not isinstance(workflow, dict):
        return workflow

    cleaned = {}
    stripped_keys = []

    for key, value in workflow.items():
        # Valid nodes must be dicts with a "class_type" property
        if isinstance(value, dict) and "class_type" in value:
            cleaned[key] = value
        else:
            stripped_keys.append(key)

    if stripped_keys:
        log.warning(
            "Stripped non-node keys from ComfyUI workflow before sending: %s. "
            "These keys do not have 'class_type' and would cause invalid_prompt error.",
            stripped_keys
        )

    return cleaned


def _should_skip_checkpoint(path: Path, size_bytes: Optional[int]) -> bool:
    if not path.exists():
        return False
    if size_bytes is None:
        return True
    try:
        return path.stat().st_size == size_bytes
    except OSError:
        return False


async def _download_checkpoint_file(
    item: ComfyCheckpointItem,
    checkpoints_dir: Path,
    client: httpx.AsyncClient,
    progress_cb: Optional[Any] = None,
) -> Dict[str, Any]:
    url = item.resolved_url or item.url
    target_path = checkpoints_dir / item.name
    tmp_path = checkpoints_dir / f"{item.name}.part"
    if _should_skip_checkpoint(target_path, item.size_bytes):
        return {"name": item.name, "status": "skipped", "error": None}

    headers: Dict[str, str] = {}
    mode = "wb"
    existing_size = 0
    if tmp_path.exists():
        try:
            existing_size = tmp_path.stat().st_size
        except OSError:
            existing_size = 0
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"

    timeout = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=60.0)
    bytes_downloaded = existing_size
    last_emit = 0.0
    last_percent = None

    async with client.stream("GET", url, headers=headers, timeout=timeout) as resp:
        if resp.status_code >= 400:
            return {"name": item.name, "status": "error", "error": f"HTTP {resp.status_code}"}
        if resp.status_code == 200 and headers.get("Range"):
            tmp_path.unlink(missing_ok=True)
            bytes_downloaded = 0
            mode = "wb"

        with open(tmp_path, mode) as f:
            async for chunk in resp.aiter_bytes(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                bytes_downloaded += len(chunk)
                if progress_cb:
                    total = item.size_bytes
                    percent = None
                    if total:
                        percent = min(100.0, (bytes_downloaded / total) * 100.0)
                    now = time.time()
                    if percent is not None and (last_percent is None or percent - last_percent >= 1 or percent >= 100):
                        last_percent = percent
                    if now - last_emit >= 1 or (percent is not None and percent >= 100):
                        last_emit = now
                        await progress_cb(
                            {
                                "name": item.name,
                                "percent": percent,
                                "bytes_downloaded": bytes_downloaded,
                                "bytes_total": total,
                                "message": "Downloading",
                            }
                        )

    tmp_path.replace(target_path)

    if item.size_bytes is not None:
        try:
            final_size = target_path.stat().st_size
        except OSError:
            final_size = None
        if final_size is None or final_size != item.size_bytes:
            target_path.unlink(missing_ok=True)
            return {"name": item.name, "status": "error", "error": "Downloaded size mismatch"}

    meta = {
        "resolved_url": item.resolved_url or item.url,
        "etag": item.etag,
        "last_modified": item.last_modified,
        "size_bytes": item.size_bytes,
    }
    meta_path = checkpoints_dir / f"{item.name}.meta.json"
    try:
        meta_path.write_text(json.dumps(meta, indent=2))
    except OSError:
        pass

    return {"name": item.name, "status": "downloaded", "error": None}


# ComfyUI workflow output node metadata (stored separately from workflow graph)
# This is NOT sent to ComfyUI - it's metadata for our internal use
COMFY_OUTPUT_NODE = "7"  # SaveImage node ID
COMFY_OUTPUT_INDEX = 0   # Output slot index


def _build_comfy_workflow(
    prompt: str,
    checkpoint: str,
    seed: int,
    steps: int,
    width: int,
    height: int,
) -> Dict[str, Any]:
    """Build ComfyUI workflow graph with ONLY valid nodes (no metadata keys)."""
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["1", 1]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "", "clip": ["1", 1]},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": 7,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": "bench_race"},
        },
    }


async def _fetch_comfy_image(client: httpx.AsyncClient, base_url: str, filename: str) -> Optional[bytes]:
    try:
        resp = await client.get(f"{base_url}/view", params={"filename": filename, "type": "output"})
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


# ---------------------------
# HTTP endpoints
# ---------------------------
@app.get("/health")
async def health():
    return {"ok": True, "machine_id": CFG.get("machine_id"), "label": CFG.get("label")}


@app.get("/capabilities")
async def capabilities():
    base_url = CFG.get("ollama_base_url", "http://127.0.0.1:11434")

    # Check Ollama reachability and get available models
    ollama_reachable = await check_ollama_available(base_url)
    ollama_models = []
    if ollama_reachable:
        ollama_models = await get_ollama_models(base_url)

    cap = Capabilities(
        machine_id=CFG.get("machine_id"),
        label=CFG.get("label"),
        tests=CFG.get("tests", ["llm_generate"]),
        llm_models=[],
        whisper_models=[],
        sdxl_profiles=[],
        accelerator_type=CFG.get("accelerator_type"),
        accelerator_memory_gb=CFG.get("accelerator_memory_gb"),
        system_memory_gb=CFG.get("system_memory_gb"),
        gpu_name=CFG.get("gpu_name"),
        ollama_reachable=ollama_reachable,
        ollama_models=ollama_models,
    )
    return JSONResponse(cap.model_dump())


@app.get("/api/comfy/health")
async def comfy_health():
    comfy = _comfy_config()
    if not comfy.get("enabled", True):
        return {"running": False, "machine_id": CFG.get("machine_id"), "label": CFG.get("label"), "checkpoints": []}
    base_url = _comfy_base_url()
    running = False
    version = None
    error = None
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{base_url}/system_stats")
            if resp.status_code == 200:
                running = True
                data = resp.json() or {}
                version = data.get("version")
            else:
                error = f"HTTP {resp.status_code}"
    except Exception as exc:
        error = str(exc)
    return {
        "running": running,
        "version": version,
        "checkpoints": _list_checkpoints(),
        "machine_id": CFG.get("machine_id"),
        "label": CFG.get("label"),
        "error": error,
    }


@app.get("/api/comfy/checkpoints")
async def comfy_checkpoints():
    return {"items": _list_checkpoint_items()}


@app.post("/api/comfy/txt2img")
async def comfy_txt2img(req: ComfyTxt2ImgRequest):
    job_id = str(uuid.uuid4())
    if job_id in RUNNING_JOBS:
        raise HTTPException(status_code=409, detail="job already running")
    task = asyncio.create_task(_job_runner_comfy(job_id, req))
    RUNNING_JOBS[job_id] = task

    def _on_done(t: asyncio.Task):
        RUNNING_JOBS.pop(job_id, None)
        if t.exception():
            log.exception("Comfy job %s terminated with exception", job_id)

    task.add_done_callback(_on_done)
    return {"accepted": True, "agent_job_id": job_id}


@app.post("/api/comfy/sync")
async def comfy_sync(req: ComfySyncRequest):
    checkpoints_dir = _comfy_checkpoints_dir()
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    base_url = req.central_base_url or CFG.get("central_base_url") or ""
    if not base_url:
        return {"error": "central_base_url not configured"}, 400
    missing = []
    for checkpoint in req.checkpoints:
        path = checkpoints_dir / checkpoint
        if path.exists():
            continue
        missing.append(checkpoint)
        try:
            url = base_url.rstrip("/") + f"/api/comfy/checkpoints/{checkpoint}"
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=120.0, write=60.0, pool=60.0)) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                path.write_bytes(resp.content)
        except Exception as exc:
            return {"error": f"Failed to download {checkpoint}: {exc}"}, 500
    return {"ok": True, "downloaded": missing}


@app.get("/api/comfy/sync_status")
async def comfy_sync_status():
    return CHECKPOINT_SYNC_STATUS


# ---------------------------
# Background job: LLM streaming
# ---------------------------
async def _run_mock_stream(job_id: str, model: str, prompt: str, max_tokens: int, temperature: float, num_ctx: int, fallback_reason: str = "unknown"):
    """
    Fallback streaming for development when Ollama isn't available.
    Produces a few token chunks with small sleeps to simulate behavior.

    Args:
        fallback_reason: Why mock was used ("ollama_unreachable", "missing_model", "stream_error")
    """
    ttft = None
    gen_tokens = 0
    start_time = time.perf_counter()
    t_first = None

    # Simple deterministic split: break prompt into sentences or words
    chunks = []
    if len(prompt) < 200:
        # split by phrase length
        words = prompt.split()
        chunk_size = max(1, len(words) // 6)
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i : i + chunk_size]))
    else:
        # long prompt -> sample substrings
        for i in range(0, min(len(prompt), 300), 40):
            chunks.append(prompt[i : i + 40])

    # Add a few generated-sounding chunks
    fake_responses = [
        "Here are the main tradeoffs: unified memory simplifies large model usage.",
        "Discrete VRAM can provide higher per-GPU throughput for small models.",
        "Unified memory reduces the need for explicit sharding and can reduce engineering complexity.",
        "However it may have higher latencies in some workloads.",
        "In practice, measure both on your workload and pick what's simpler and faster."
    ]
    chunks.extend(fake_responses)

    for chunk in chunks:
        await asyncio.sleep(0.35)  # simulate chunk delay
        now = time.perf_counter()
        if ttft is None:
            ttft = (now - start_time) * 1000.0
            t_first = now
        toks = len(chunk.split())
        gen_tokens += toks
        ev = Event(job_id=job_id, type="llm_token", payload={"text": chunk, "timestamp_s": now})
        await _broadcast_event(ev)

    end_time = time.perf_counter()
    total_ms = (end_time - start_time) * 1000.0
    gen_tps = gen_tokens / (end_time - t_first) if t_first and end_time > t_first else None
    result = {
        "ttft_ms": ttft,
        "gen_tokens": gen_tokens,
        "gen_tokens_per_s": gen_tps,
        "total_ms": total_ms,
        "model": model,
        "engine": "mock",
        "fallback_reason": fallback_reason,
    }
    return result


async def _job_runner_llm(job_id: str, req: LLMRequest):
    """
    Background runner for llm_generate jobs.
    Tries Ollama streaming first; falls back to mock streaming on failure.
    Emits events while running and a final 'job_done' event with metrics.
    """
    model = req.model
    prompt = req.prompt
    max_tokens = req.max_tokens
    temperature = req.temperature
    num_ctx = req.num_ctx

    log.info("Starting job %s model=%s", job_id, model)
    base_url = CFG.get("ollama_base_url", "http://127.0.0.1:11434")
    try:
        available = await check_ollama_available(base_url)
        if not available:
            log.warning("Ollama unreachable; falling back to mock backend for job %s", job_id)
            result = await _run_mock_stream(job_id, model, prompt, max_tokens, temperature, num_ctx, fallback_reason="ollama_unreachable")
        else:
            # Check if model is available on Ollama
            ollama_models = await get_ollama_models(base_url)
            if model not in ollama_models:
                log.warning("Model %s not found on Ollama (available: %s); falling back to mock for job %s", model, ollama_models, job_id)
                result = await _run_mock_stream(job_id, model, prompt, max_tokens, temperature, num_ctx, fallback_reason="missing_model")
            else:
                log.info("Using Ollama backend for job %s", job_id)
                async def _on_token(text: str, timestamp_s: float) -> None:
                    ev = Event(job_id=job_id, type="llm_token", payload={"text": text, "timestamp_s": timestamp_s})
                    await _broadcast_event(ev)

                result = await stream_ollama_generate(
                    job_id=job_id,
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    num_ctx=num_ctx,
                    base_url=base_url,
                    on_token=_on_token,
                )
    except Exception as e:
        log.warning("Ollama stream failed (%s); falling back to mock stream", e)
        result = await _run_mock_stream(job_id, model, prompt, max_tokens, temperature, num_ctx, fallback_reason="stream_error")

    # Emit final metrics event
    ev = Event(job_id=job_id, type="job_done", payload=result)
    await _broadcast_event(ev)
    log.info("Job %s done. result=%s", job_id, result)


async def _job_runner_comfy(job_id: str, req: ComfyTxt2ImgRequest):
    base_url = _comfy_base_url()
    checkpoints_dir = _comfy_checkpoints_dir()
    if not checkpoints_dir.exists():
        await _broadcast_event(
            Event(
                job_id=job_id,
                type="image_error",
                payload={"run_id": req.run_id, "message": f"Checkpoints dir missing: {checkpoints_dir}"},
            )
        )
        return
    if req.checkpoint not in _list_checkpoints():
        await _broadcast_event(
            Event(
                job_id=job_id,
                type="image_error",
                payload={"run_id": req.run_id, "message": "Checkpoint not synced to agent"},
            )
        )
        return

    submit_time = time.perf_counter()
    total_images = []
    last_preview_step = -1
    current_step = 0
    started_at = None

    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=60.0, write=60.0, pool=60.0)) as client:
        image_count = req.repeat * req.num_images
        for repeat_index in range(image_count):
            client_id = str(uuid.uuid4())
            seed_offset = req.seed + repeat_index
            workflow = _build_comfy_workflow(
                req.prompt,
                req.checkpoint,
                seed_offset,
                req.steps,
                req.width,
                req.height,
            )

            # Validate and clean workflow (strip any non-node keys)
            cleaned_workflow = _validate_comfy_workflow(workflow)

            # Save debug payload if enabled (save the cleaned version that will be sent)
            debug_file = _save_comfy_debug_payload(cleaned_workflow, req.run_id)
            if debug_file:
                log.info("Saved ComfyUI payload to debug file: %s", debug_file)

            try:
                resp = await client.post(
                    f"{base_url}/prompt",
                    json={"prompt": cleaned_workflow, "client_id": client_id},
                )
                resp.raise_for_status()
                prompt_id = resp.json().get("prompt_id")
            except httpx.HTTPStatusError as exc:
                # Enhanced error logging for HTTP errors
                status_code = exc.response.status_code
                try:
                    response_body = exc.response.text[:8192]  # First 8KB
                except Exception:
                    response_body = "<unable to read response body>"

                payload_summary = _summarize_comfy_payload(cleaned_workflow)

                # Log comprehensive diagnostics
                log.error(
                    "ComfyUI POST /prompt failed: status=%d, response_body=%s, payload_summary=%s, debug_file=%s",
                    status_code,
                    response_body,
                    payload_summary,
                    debug_file or "not saved",
                )

                # Include full error details in the broadcast message
                error_msg = f"ComfyUI rejected prompt (HTTP {status_code}): {response_body}"
                if debug_file:
                    error_msg += f"\nDebug payload saved to: {debug_file}"

                await _broadcast_event(
                    Event(
                        job_id=job_id,
                        type="image_error",
                        payload={"run_id": req.run_id, "message": error_msg},
                    )
                )
                return
            except Exception as exc:
                # Handle other errors (connection errors, timeouts, etc.)
                payload_summary = _summarize_comfy_payload(cleaned_workflow)
                log.error(
                    "ComfyUI POST /prompt failed: exception=%s, payload_summary=%s, debug_file=%s",
                    exc,
                    payload_summary,
                    debug_file or "not saved",
                )

                error_msg = f"Failed to submit prompt to ComfyUI: {exc}"
                if debug_file:
                    error_msg += f"\nDebug payload saved to: {debug_file}"

                await _broadcast_event(
                    Event(
                        job_id=job_id,
                        type="image_error",
                        payload={"run_id": req.run_id, "message": error_msg},
                    )
                )
                return

            ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + f"/ws?clientId={client_id}"

            try:
                async with websockets.connect(ws_url) as ws:
                    async for msg in ws:
                        if isinstance(msg, bytes):
                            if started_at is None:
                                started_at = time.perf_counter()
                            if current_step % 4 == 0 and current_step != last_preview_step:
                                last_preview_step = current_step
                                try:
                                    image = Image.open(io.BytesIO(msg))
                                    buf = io.BytesIO()
                                    image.convert("RGB").save(buf, format="JPEG", quality=80)
                                    preview_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                                    await _broadcast_event(
                                        Event(
                                            job_id=job_id,
                                            type="image_preview",
                                            payload={
                                                "run_id": req.run_id,
                                                "step": current_step,
                                                "total_steps": req.steps,
                                                "filename": f"preview_{repeat_index + 1}.jpg",
                                                "image_b64": preview_b64,
                                            },
                                        )
                                    )
                                except Exception:
                                    continue
                            continue

                        try:
                            data = json.loads(msg)
                        except Exception:
                            continue
                        msg_type = data.get("type")
                        if msg_type == "execution_start":
                            started_at = time.perf_counter()
                            queue_latency_ms = (started_at - submit_time) * 1000.0
                            await _broadcast_event(
                                Event(
                                    job_id=job_id,
                                    type="image_started",
                                    payload={
                                        "run_id": req.run_id,
                                        "queue_latency_ms": queue_latency_ms,
                                        "started_at": time.time(),
                                    },
                                )
                            )
                        elif msg_type == "progress":
                            progress_data = data.get("data") or {}
                            current_step = int(progress_data.get("value", current_step))
                            await _broadcast_event(
                                Event(
                                    job_id=job_id,
                                    type="image_progress",
                                    payload={
                                        "run_id": req.run_id,
                                        "step": current_step,
                                        "total_steps": int(progress_data.get("max", req.steps)),
                                    },
                                )
                            )
                        elif msg_type in {"execution_success", "execution_error"}:
                            break
            except Exception as exc:
                await _broadcast_event(
                    Event(
                        job_id=job_id,
                        type="image_error",
                        payload={"run_id": req.run_id, "message": f"ComfyUI stream error: {exc}"},
                    )
                )
                return

            try:
                history_resp = await client.get(f"{base_url}/history/{prompt_id}")
                history_resp.raise_for_status()
                history = history_resp.json() or {}
                images = []
                for outputs in history.values():
                    for node in (outputs.get("outputs") or {}).values():
                        for image_info in node.get("images", []):
                            filename = image_info.get("filename")
                            if not filename:
                                continue
                            image_bytes = await _fetch_comfy_image(client, base_url, filename)
                            if not image_bytes:
                                continue
                            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                            images.append(
                                {
                                    "filename": filename,
                                    "image_b64": image_b64,
                                }
                            )
                total_images.extend(images)
            except Exception as exc:
                await _broadcast_event(
                    Event(
                        job_id=job_id,
                        type="image_error",
                        payload={"run_id": req.run_id, "message": f"Failed to fetch images: {exc}"},
                    )
                )
                return

    end_time = time.perf_counter()
    if started_at is None:
        started_at = submit_time
    queue_latency_ms = (started_at - submit_time) * 1000.0
    gen_time_ms = (end_time - started_at) * 1000.0
    total_ms = (end_time - submit_time) * 1000.0
    await _broadcast_event(
        Event(
            job_id=job_id,
            type="image_complete",
            payload={
                "run_id": req.run_id,
                "queue_latency_ms": queue_latency_ms,
                "gen_time_ms": gen_time_ms,
                "total_ms": total_ms,
                "steps": req.steps,
                "resolution": f"{req.width}x{req.height}",
                "seed": req.seed,
                "checkpoint": req.checkpoint,
                "num_images": req.num_images,
                "images": total_images,
            },
        )
    )


async def _broadcast_sync_event(sync_id: str, event_type: str, payload: Dict) -> None:
    ev = Event(job_id=sync_id, type=event_type, payload=payload)
    await _broadcast_event(ev)


async def _broadcast_checkpoint_sync_event(sync_id: str, event_type: str, payload: Dict) -> None:
    ev = Event(job_id=sync_id, type=event_type, payload=payload)
    await _broadcast_event(ev)


async def _pull_ollama_model(sync_id: str, model: str, base_url: str) -> None:
    url = base_url.rstrip("/") + "/api/pull"
    payload = {"name": model, "stream": True}
    timeout = httpx.Timeout(connect=5.0, read=None, write=60.0, pool=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    data = {"status": line}
                status = data.get("status") or "pulling"
                completed = data.get("completed")
                total = data.get("total")
                percent: Optional[float] = None
                if completed is not None and total:
                    try:
                        percent = min(100.0, (float(completed) / float(total)) * 100.0)
                    except (TypeError, ValueError, ZeroDivisionError):
                        percent = None
                await _broadcast_sync_event(
                    sync_id,
                    "sync_progress",
                    {
                        "model": model,
                        "phase": "pulling",
                        "percent": percent,
                        "bytes_downloaded": completed,
                        "bytes_total": total,
                        "message": status,
                    },
                )
                if status == "success":
                    break


async def _sync_models(sync_id: str, req: SyncRequest) -> None:
    base_url = CFG.get("ollama_base_url", "http://127.0.0.1:11434")
    try:
        await _broadcast_sync_event(
            sync_id,
            "sync_started",
            {
                "models": {
                    "llm": req.llm,
                    "whisper": req.whisper,
                    "sdxl_profiles": req.sdxl_profiles,
                }
            },
        )

        for model in req.llm:
            await _broadcast_sync_event(
                sync_id,
                "sync_progress",
                {"model": model, "phase": "queued", "percent": None, "message": "Queued"},
            )
            await _pull_ollama_model(sync_id, model, base_url)

        for model in req.whisper:
            await _broadcast_sync_event(
                sync_id,
                "sync_progress",
                {"model": model, "phase": "complete", "percent": 100, "message": "Whisper sync not configured"},
            )

        for profile in req.sdxl_profiles:
            await _broadcast_sync_event(
                sync_id,
                "sync_progress",
                {"model": profile, "phase": "complete", "percent": 100, "message": "SDXL sync not configured"},
            )

        await _broadcast_sync_event(sync_id, "sync_done", {"message": "Sync complete"})
    except Exception as exc:
        await _broadcast_sync_event(sync_id, "sync_error", {"message": str(exc)})


async def _sync_image_checkpoints(sync_id: str, items: List[ComfyCheckpointItem]) -> None:
    checkpoints_dir = _comfy_checkpoints_dir()
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_SYNC_STATUS["active"] = True
    CHECKPOINT_SYNC_STATUS["results"] = []
    try:
        await _broadcast_checkpoint_sync_event(
            sync_id,
            "image_checkpoint_sync_start",
            {"items": [item.name for item in items]},
        )

        async def _progress(payload: Dict[str, Any]) -> None:
            CHECKPOINT_SYNC_STATUS["last_progress"] = payload
            await _broadcast_checkpoint_sync_event(sync_id, "image_checkpoint_sync_progress", payload)

        results = []
        async with httpx.AsyncClient() as client:
            for item in items:
                await _broadcast_checkpoint_sync_event(
                    sync_id,
                    "image_checkpoint_sync_progress",
                    {"name": item.name, "percent": None, "message": "Queued"},
                )
                result = await _download_checkpoint_file(item, checkpoints_dir, client, progress_cb=_progress)
                results.append(result)
                await _broadcast_checkpoint_sync_event(
                    sync_id,
                    "image_checkpoint_sync_progress",
                    {
                        "name": item.name,
                        "percent": 100 if result["status"] == "downloaded" else None,
                        "message": result["status"].capitalize(),
                        "error": result.get("error"),
                    },
                )

        CHECKPOINT_SYNC_STATUS["results"] = results
        CHECKPOINT_SYNC_STATUS["active"] = False
        await _broadcast_checkpoint_sync_event(
            sync_id,
            "image_checkpoint_sync_done",
            {"results": results},
        )
    except Exception as exc:
        CHECKPOINT_SYNC_STATUS["active"] = False
        await _broadcast_checkpoint_sync_event(
            sync_id,
            "image_checkpoint_sync_done",
            {"results": [], "error": str(exc)},
        )


# ---------------------------
# Jobs endpoint
# ---------------------------
@app.post("/jobs", response_model=JobStartResponse)
async def start_job(req: LLMRequest):
    """
    Start a job and run it in the background. Returns job_id immediately.
    Streaming events are sent to connected WebSocket clients.
    """
    job_id = str(uuid.uuid4())

    # If a job_id already running, reject
    # (not likely, but defensive)
    if job_id in RUNNING_JOBS:
        raise HTTPException(status_code=409, detail="job already running")

    # Launch background runner
    task = asyncio.create_task(_job_runner_llm(job_id, req))
    RUNNING_JOBS[job_id] = task

    # When task finishes, remove it from RUNNING_JOBS
    def _on_done(t: asyncio.Task):
        RUNNING_JOBS.pop(job_id, None)
        if t.exception():
            log.exception("Job %s terminated with exception", job_id)

    task.add_done_callback(_on_done)

    return JobStartResponse(job_id=job_id)


@app.post("/models/sync")
async def sync_models(req: SyncRequest):
    sync_id = str(uuid.uuid4())
    task = asyncio.create_task(_sync_models(sync_id, req))
    SYNC_TASKS[sync_id] = task

    def _on_done(t: asyncio.Task):
        SYNC_TASKS.pop(sync_id, None)
        if t.exception():
            log.exception("Sync %s terminated with exception", sync_id)

    task.add_done_callback(_on_done)
    return {"sync_id": sync_id}


@app.post("/api/comfy/sync_checkpoints")
async def comfy_sync_checkpoints(req: ComfyCheckpointSyncRequest):
    if not req.items:
        return {"error": "No checkpoint items provided"}, 400
    sync_id = str(uuid.uuid4())
    task = asyncio.create_task(_sync_image_checkpoints(sync_id, req.items))
    SYNC_TASKS[sync_id] = task

    def _on_done(t: asyncio.Task):
        SYNC_TASKS.pop(sync_id, None)
        if t.exception():
            log.exception("Checkpoint sync %s terminated with exception", sync_id)

    task.add_done_callback(_on_done)
    return {"sync_id": sync_id}


# ---------------------------
# WebSocket endpoint
# ---------------------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    """
    Simple pubsub: any connected client receives all events.
    The central server should connect here to receive token streams and job_done events.
    """
    await ws.accept()
    client_id = str(uuid.uuid4())
    WS_CLIENTS[client_id] = ws
    log.info("WS client connected: %s (total=%d)", client_id, len(WS_CLIENTS))
    try:
        while True:
            # keep the connection alive by reading; the central may send pings/commands later
            data = await ws.receive_text()
            # we ignore received messages for now; could implement control messages
            log.debug("WS recv from %s: %s", client_id, data)
    except WebSocketDisconnect:
        log.info("WS client disconnected: %s", client_id)
        WS_CLIENTS.pop(client_id, None)
    except Exception as exc:
        log.exception("WS error for %s: %s", client_id, exc)
        WS_CLIENTS.pop(client_id, None)


# ---------------------------
# Graceful shutdown
# ---------------------------
@app.on_event("shutdown")
async def _shutdown():
    log.info("Shutting down: cancelling %d running jobs", len(RUNNING_JOBS))
    for job_id, t in list(RUNNING_JOBS.items()):
        t.cancel()
    for sync_id, t in list(SYNC_TASKS.items()):
        t.cancel()
    # close websockets
    for cid, ws in list(WS_CLIENTS.items()):
        try:
            await ws.close()
        except Exception:
            pass
    WS_CLIENTS.clear()

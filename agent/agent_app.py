from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import yaml
import httpx
import websockets
import subprocess
import signal
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# ---------------------------
# Path and import robustness
# ---------------------------
AGENT_DIR = Path(__file__).resolve().parent            # .../bench-race/agent
ROOT_DIR = AGENT_DIR.parent                            # .../bench-race
CONFIG_PATH = AGENT_DIR / "config" / "agent.yaml"
OUTPUT_IMAGES_DIR = AGENT_DIR / "output_images"        # Image storage for HTTP serving

# Make repo root importable so `shared` works
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# now import shared pydantic schemas
from shared.schemas import Capabilities, Event, JobStartResponse, LLMRequest  # type: ignore
from backends.ollama_backend import check_ollama_available, stream_ollama_generate, get_ollama_models
from backends.vllm_backend import check_vllm_available, stream_vllm_generate, get_vllm_models

# Import agent-specific modules
from agent.logging_utils import init_logging, get_logger, HOSTNAME
from agent.middleware import ws_logger
from agent.http_client import comfyui_client, ollama_client
from agent.http_logging_asgi import HTTPLoggingASGIMiddleware
from agent.comfy_ws import (
    wait_for_prompt as comfy_wait_for_prompt,
    check_queue_status as comfy_check_queue_status,
    extract_images_from_history,
    ProgressEvent,
    ComfyWSResult,
)
from agent.errors import classify_comfy_error
from agent.startup_checks import run_comfyui_cuda_probe, force_cpu_enabled
from agent.hardware_discovery import discover_hardware
from agent.runtime_sampler import RuntimeSampler, RuntimeSamplerConfig
from agent.reset_helpers import (
    run_subprocess_with_capture,
    StreamingProcessLogger,
    poll_http_health,
    OLLAMA_START_TIMEOUT_S,
    COMFYUI_START_TIMEOUT_S,
    HEALTH_POLL_INTERVAL_S,
)

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bench-agent")  # Keep for backwards compatibility

# Structured logger (will be initialized after config is loaded)
slog = None

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

# Initialize structured logging
machine_id = CFG.get("machine_id", "unknown")
slog = init_logging(agent_id=machine_id)
COMFYUI_PREFLIGHT: Dict[str, Optional[bool]] = {"comfyui_gpu_ok": None, "comfyui_cpu_ok": None}
HARDWARE_INFO: Dict[str, Any] = {}
RUNTIME_SAMPLER: Optional[RuntimeSampler] = None
RUNTIME_SAMPLER_TASK: Optional[asyncio.Task] = None

# ---------------------------
# FastAPI app + websockets
# ---------------------------
app = FastAPI(title="bench-race agent")

# Add ASGI-safe HTTP logging middleware
app.add_middleware(
    HTTPLoggingASGIMiddleware,
    agent_id=machine_id,
    hostname=HOSTNAME,
)


@app.on_event("startup")
async def _startup_checks():
    install_dir = _comfy_install_dir()
    result = run_comfyui_cuda_probe(install_dir)
    if result:
        COMFYUI_PREFLIGHT["comfyui_gpu_ok"] = result.comfyui_gpu_ok
        COMFYUI_PREFLIGHT["comfyui_cpu_ok"] = result.comfyui_cpu_ok
        if result.error:
            slog.warning("startup_cuda_incompatible", error=result.error)
    _initialize_hardware_info()
    _initialize_runtime_sampler()


@app.on_event("shutdown")
async def _shutdown_cleanup():
    _stop_runtime_sampler()

# Manage connected websocket clients
# Each entry: client_id -> WebSocket
WS_CLIENTS: Dict[str, WebSocket] = {}

# Track running jobs (job_id -> asyncio.Task)
RUNNING_JOBS: Dict[str, asyncio.Task] = {}
SYNC_TASKS: Dict[str, asyncio.Task] = {}
CHECKPOINT_SYNC_STATUS: Dict[str, Any] = {"active": False, "results": []}

# Per-job status tracking: job_id -> status dict
# Allows central to poll job state without relying solely on websockets
JOB_STATUS: Dict[str, Dict[str, Any]] = {}


class SyncRequest(BaseModel):
    llm: List[str] = Field(default_factory=list)
    whisper: List[str] = Field(default_factory=list)
    sdxl_profiles: List[str] = Field(default_factory=list)


class ComfyTxt2ImgRequest(BaseModel):
    run_id: str
    prompt: str
    checkpoint: str
    seed: int
    steps: int = 10
    width: int = 512
    height: int = 512
    num_images: int = 1
    repeat: int = 1
    sampler: str = "DPM++ 2M Karras"


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


class ComputeRequest(BaseModel):
    algorithm: str = "segmented_sieve"
    n: int
    threads: int = 1
    repeat_index: int = 1
    stream_first_k: int = 0
    progress_interval_s: float = 1.0
    job_id: Optional[str] = None


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


async def _broadcast_payload(payload: Dict[str, Any]):
    to_remove: List[str] = []
    txt = json.dumps(payload)
    for cid, ws in list(WS_CLIENTS.items()):
        try:
            await ws.send_text(txt)
        except Exception as exc:
            log.warning("WebSocket send failed for %s: %s", cid, exc)
            to_remove.append(cid)
    for cid in to_remove:
        WS_CLIENTS.pop(cid, None)


# ---------------------------
# Image Storage Helpers
# ---------------------------
def _ensure_output_images_dir():
    """Ensure output_images directory exists."""
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _save_image_to_disk(image_bytes: bytes, run_id: str, filename: str) -> str:
    """
    Save image bytes to disk and return image_id.

    Args:
        image_bytes: Raw image bytes
        run_id: Run identifier
        filename: Original filename (e.g., "preview_1.jpg" or "output_001.png")

    Returns:
        image_id: Unique identifier for fetching the image via HTTP
    """
    _ensure_output_images_dir()

    # Create run-specific directory
    run_dir = OUTPUT_IMAGES_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique image_id (uuid)
    image_id = str(uuid.uuid4())

    # Determine file extension from filename
    ext = Path(filename).suffix or ".png"

    # Save to disk with image_id as filename
    image_path = run_dir / f"{image_id}{ext}"
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    slog.debug("image_saved_to_disk", image_id=image_id, path=str(image_path), size_bytes=len(image_bytes))
    return image_id


def _cleanup_old_images(max_runs: int = 10, current_run_id: Optional[str] = None):
    """
    Clean up old image runs, keeping only the most recent max_runs.

    Args:
        max_runs: Maximum number of run directories to keep (overridden by BENCH_IMAGE_RETENTION_RUNS env var)
        current_run_id: Current run_id to protect from deletion
    """
    if not OUTPUT_IMAGES_DIR.exists():
        return

    # Read retention configuration from environment
    max_runs = int(os.environ.get("BENCH_IMAGE_RETENTION_RUNS", max_runs))
    retention_minutes = int(os.environ.get("BENCH_IMAGE_RETENTION_MINUTES", 15))

    current_time = time.time()
    protection_threshold = current_time - (retention_minutes * 60)

    # Get all run directories sorted by modification time (newest first)
    run_dirs = sorted(
        [d for d in OUTPUT_IMAGES_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True
    )

    # Track what we're keeping vs removing
    kept_count = 0
    removed_count = 0

    # Remove old runs beyond max_runs, with safety protections
    for old_dir in run_dirs:
        run_id = old_dir.name
        mtime = old_dir.stat().st_mtime

        # Safety checks: never delete active or recently modified runs
        is_current = current_run_id is not None and run_id == current_run_id
        is_recent = mtime >= protection_threshold
        should_keep_by_limit = kept_count < max_runs

        if is_current:
            slog.debug("image_cleanup_skip", run_id=run_id, reason="current_run")
            kept_count += 1
            continue

        if is_recent:
            minutes_ago = (current_time - mtime) / 60
            slog.debug("image_cleanup_skip", run_id=run_id, reason="recently_modified", minutes_ago=f"{minutes_ago:.1f}")
            kept_count += 1
            continue

        if should_keep_by_limit:
            kept_count += 1
            continue

        # Safe to delete
        try:
            # Get directory size for logging
            total_size = sum(f.stat().st_size for f in old_dir.rglob('*') if f.is_file())
            shutil.rmtree(old_dir)
            removed_count += 1
            slog.info(
                "image_cleanup",
                run_id=run_id,
                path=str(old_dir),
                size_bytes=total_size,
                reason="retention_limit"
            )
        except Exception as e:
            slog.warning("image_cleanup_failed", run_id=run_id, error=str(e))

    if removed_count > 0:
        slog.info(
            "image_cleanup_summary",
            removed=removed_count,
            kept=kept_count,
            total=len(run_dirs),
            max_runs=max_runs,
            retention_minutes=retention_minutes
        )


def _format_count(value: int) -> str:
    return f"{value:,}"


async def _emit_compute_line(job_id: str, line: str) -> None:
    await _broadcast_event(Event(job_id=job_id, type="compute_line", payload={"line": line}))


async def _simple_sieve(n: int) -> bytearray:
    is_prime = bytearray(b"\x01") * (n + 1)
    if n >= 0:
        is_prime[0:2] = b"\x00\x00"
    limit = int(math.isqrt(n))
    for p in range(2, limit + 1):
        if is_prime[p]:
            start = p * p
            is_prime[start:n + 1:p] = b"\x00" * ((n - start) // p + 1)
    return is_prime


async def _run_segmented_sieve(
    n: int,
    stream_first_k: int,
    progress_interval_s: float,
    emit_line,
) -> int:
    sqrt_n = int(math.isqrt(n))
    base_flags = await _simple_sieve(sqrt_n)
    base_primes = [i for i in range(2, sqrt_n + 1) if base_flags[i]]
    segment_size = max(sqrt_n, 1_000_000)
    count = 0
    start_time = time.perf_counter()
    last_progress = start_time

    low = 2
    while low <= n:
        high = min(low + segment_size - 1, n)
        size = high - low + 1
        segment = bytearray(b"\x01") * size
        for p in base_primes:
            start = max(p * p, ((low + p - 1) // p) * p)
            if start > high:
                continue
            for multiple in range(start, high + 1, p):
                segment[multiple - low] = 0
        for idx, is_prime in enumerate(segment):
            if is_prime:
                count += 1
        now = time.perf_counter()
        if progress_interval_s > 0 and now - last_progress >= progress_interval_s:
            pct = min(100.0, (high / n) * 100.0)
            elapsed = now - start_time
            await emit_line(
                f"Progress: {pct:.0f}% | primes so far: {_format_count(count)} | elapsed: {elapsed:.1f}s"
            )
            last_progress = now
        low = high + 1
        await asyncio.sleep(0)
    return count


async def _run_simple_sieve(
    n: int,
    stream_first_k: int,
    progress_interval_s: float,
    emit_line,
) -> int:
    flags = await _simple_sieve(n)
    count = 0
    start_time = time.perf_counter()
    last_progress = start_time
    for i in range(2, n + 1):
        if flags[i]:
            count += 1
        if progress_interval_s > 0:
            now = time.perf_counter()
            if now - last_progress >= progress_interval_s:
                pct = min(100.0, (i / n) * 100.0)
                elapsed = now - start_time
                await emit_line(
                    f"Progress: {pct:.0f}% | primes so far: {_format_count(count)} | elapsed: {elapsed:.1f}s"
                )
                last_progress = now
        if i % 10000 == 0:
            await asyncio.sleep(0)
    return count


async def _run_trial_division(
    n: int,
    stream_first_k: int,
    progress_interval_s: float,
    emit_line,
) -> int:
    primes: List[int] = []
    count = 0
    start_time = time.perf_counter()
    last_progress = start_time

    if n >= 2:
        primes.append(2)
        count = 1

    candidate = 3
    while candidate <= n:
        is_prime = True
        limit = int(math.isqrt(candidate))
        for p in primes:
            if p > limit:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
            count += 1
        if progress_interval_s > 0:
            now = time.perf_counter()
            if now - last_progress >= progress_interval_s:
                pct = min(100.0, (candidate / n) * 100.0)
                elapsed = now - start_time
                await emit_line(
                    f"Progress: {pct:.0f}% | primes so far: {_format_count(count)} | elapsed: {elapsed:.1f}s"
                )
                last_progress = now
        if candidate % 10001 == 1:
            await asyncio.sleep(0)
        candidate += 2
    return count


async def _run_compute(job_id: str, req: ComputeRequest) -> Dict[str, Any]:
    algorithm = req.algorithm
    n = int(req.n)
    threads_requested = int(req.threads)
    progress_interval_s = max(0.1, float(req.progress_interval_s))
    algorithm_id = algorithm

    await _emit_compute_line(job_id, f"Compute: Count primes ≤ {n:,}")
    if algorithm == "trial_division":
        await _emit_compute_line(job_id, "Trial division is intentionally slow and for demo/education.")
    threads_used = 1
    if threads_requested > 1:
        await _emit_compute_line(job_id, "threads>1 not implemented yet; running single-threaded")

    start_time = time.perf_counter()
    if algorithm == "segmented_sieve":
        primes_found = await _run_segmented_sieve(n, 0, progress_interval_s, lambda line: _emit_compute_line(job_id, line))
    elif algorithm == "simple_sieve":
        primes_found = await _run_simple_sieve(n, 0, progress_interval_s, lambda line: _emit_compute_line(job_id, line))
    elif algorithm == "trial_division":
        primes_found = await _run_trial_division(n, 0, progress_interval_s, lambda line: _emit_compute_line(job_id, line))
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    elapsed_s = time.perf_counter() - start_time
    elapsed_ms = int(elapsed_s * 1000)
    primes_per_sec = primes_found / elapsed_s if elapsed_s > 0 else None

    await _emit_compute_line(job_id, f"Algorithm: {algorithm_id}")
    await _emit_compute_line(job_id, f"N: {_format_count(n)}")
    await _emit_compute_line(job_id, f"Primes: {_format_count(primes_found)}")
    if primes_per_sec is not None:
        await _emit_compute_line(job_id, f"Rate: {primes_per_sec:.1f} primes/s")
    else:
        await _emit_compute_line(job_id, "Rate: n/a")
    await _emit_compute_line(job_id, f"Elapsed: {elapsed_s:.2f} s")

    return {
        "ok": True,
        "algorithm": algorithm,
        "n": n,
        "threads_requested": threads_requested,
        "threads_used": threads_used,
        "primes_found": primes_found,
        "elapsed_ms": elapsed_ms,
        "primes_per_sec": primes_per_sec,
        "repeat_index": req.repeat_index,
    }


def _comfy_config() -> Dict[str, Any]:
    return CFG.get("comfyui", {}) if isinstance(CFG, dict) else {}


def _comfy_install_dir() -> Path:
    comfy = _comfy_config()
    path = comfy.get("install_dir") or os.getenv("COMFYUI_DIR", "")
    if path:
        return Path(path)
    return ROOT_DIR / "agent" / "third_party" / "comfyui"


def _load_machine_override() -> Dict[str, Any]:
    override_path = os.getenv("MACHINES_YAML_PATH", "")
    if not override_path:
        override_path = str(ROOT_DIR / "central" / "config" / "machines.yaml")
    path = Path(override_path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        machines = config.get("machines", [])
        return next((m for m in machines if m.get("machine_id") == CFG.get("machine_id")), {}) or {}
    except Exception as exc:
        slog.warning("hardware_override_load_failed", error=str(exc))
        return {}


def _merge_hardware_overrides(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key in ("cpu_cores", "cpu_physical_cores", "total_system_ram_bytes"):
        if override.get(key) is not None:
            merged[key] = override.get(key)
    gpu_override = override.get("gpu") or {}
    if gpu_override:
        merged["gpu_name"] = gpu_override.get("name", merged.get("gpu_name"))
        merged["gpu_vram_bytes"] = gpu_override.get("vram_bytes", merged.get("gpu_vram_bytes"))
        merged["gpu_type"] = gpu_override.get("type", merged.get("gpu_type"))
        merged["cuda_compute"] = gpu_override.get("cuda_compute", merged.get("cuda_compute"))
    return merged


def _initialize_hardware_info():
    global HARDWARE_INFO
    detected = discover_hardware()
    override = _load_machine_override()
    HARDWARE_INFO = _merge_hardware_overrides(detected, override)


def _initialize_runtime_sampler():
    global RUNTIME_SAMPLER, RUNTIME_SAMPLER_TASK
    sampler_cfg = CFG.get("runtime_sampler", {}) if isinstance(CFG, dict) else {}
    if sampler_cfg is None:
        return
    enabled = sampler_cfg.get("enabled", True)
    if not enabled:
        return
    config = RuntimeSamplerConfig(
        interval_s=float(sampler_cfg.get("interval_s", 1.0)),
        buffer_len=int(sampler_cfg.get("buffer_len", 120)),
    )
    RUNTIME_SAMPLER = RuntimeSampler(config)
    RUNTIME_SAMPLER.start()
    RUNTIME_SAMPLER_TASK = asyncio.create_task(_push_runtime_metrics())


def _stop_runtime_sampler():
    global RUNTIME_SAMPLER, RUNTIME_SAMPLER_TASK
    if RUNTIME_SAMPLER:
        RUNTIME_SAMPLER.stop()
        RUNTIME_SAMPLER = None
    if RUNTIME_SAMPLER_TASK:
        RUNTIME_SAMPLER_TASK.cancel()
        RUNTIME_SAMPLER_TASK = None


async def _push_runtime_metrics():
    while True:
        if not RUNTIME_SAMPLER:
            await asyncio.sleep(1)
            continue
        payload = {
            "job_id": "runtime_metrics",
            "type": "runtime_metrics_update",
            "payload": RUNTIME_SAMPLER.snapshot(),
        }
        await _broadcast_payload(payload)
        await asyncio.sleep(RUNTIME_SAMPLER.interval_s)


def _comfy_base_url() -> str:
    comfy = _comfy_config()
    base = comfy.get("base_url")
    if base:
        return base.rstrip("/")
    port = comfy.get("port", 8188)
    host = comfy.get("host", "127.0.0.1")
    return f"http://{host}:{port}"


def _comfy_checkpoints_dir() -> Path:
    """Returns the path where ComfyUI scans for checkpoints."""
    comfy = _comfy_config()
    path = comfy.get("checkpoints_dir") or ""
    if path:
        return Path(path)
    # Default to repo-relative embedded ComfyUI
    return ROOT_DIR / "agent" / "third_party" / "comfyui" / "models" / "checkpoints"


def _comfy_cache_dir() -> Path:
    """Returns the path where agent caches downloaded checkpoints before symlinking."""
    comfy = _comfy_config()
    path = comfy.get("cache_path") or ""
    if path:
        return Path(path)
    # Default to repo-relative cache
    return ROOT_DIR / "agent" / "model_cache" / "comfyui"


def _comfy_pidfile() -> Path:
    return ROOT_DIR / "run" / "comfyui.pid"


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _comfy_cpu_fallback_config() -> Dict[str, Any]:
    comfy = _comfy_config()
    cpu_fallback_on = comfy.get("cpu_fallback_on") or ["cuda_unsupported_arch"]
    allow_cpu_fallback = bool(comfy.get("allow_cpu_fallback", False) or force_cpu_enabled())
    return {"allow": allow_cpu_fallback, "on": cpu_fallback_on}


def _comfy_cpu_fallback_allowed(category: str) -> bool:
    config = _comfy_cpu_fallback_config()
    return config["allow"] and category in config["on"]


async def _restart_comfyui_cpu() -> bool:
    """Restart ComfyUI in CPU mode if it was started via the bench-race launcher."""
    pidfile = _comfy_pidfile()
    if not pidfile.exists():
        return False
    try:
        pid = int(pidfile.read_text().strip())
    except Exception:
        return False
    if not _pid_is_running(pid):
        return False

    install_dir = _comfy_install_dir()
    python_path = install_dir / ".venv" / "bin" / "python"
    if not python_path.exists():
        return False

    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        return False

    for _ in range(30):
        if not _pid_is_running(pid):
            break
        await asyncio.sleep(0.2)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    host = _comfy_config().get("host", "127.0.0.1")
    port = _comfy_config().get("port", 8188)
    try:
        proc = subprocess.Popen(
            [str(python_path), "main.py", "--listen", host, "--port", str(port)],
            cwd=str(install_dir),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        return False

    try:
        pidfile.write_text(str(proc.pid))
    except Exception:
        pass

    base_url = _comfy_base_url()
    async with httpx.AsyncClient(timeout=2.0) as client:
        for _ in range(20):
            try:
                resp = await client.get(f"{base_url}/system_stats")
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.5)
    return False


def _format_comfy_error(error_msg: str, history_status: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    classified = classify_comfy_error(error_msg)
    details = {
        "category": classified.get("category"),
        "remediation": classified.get("action"),
    }
    if history_status:
        details.update(
            {
                "raw_exception_type": history_status.get("exception_type"),
                "raw_exception_message": history_status.get("exception_message"),
                "node_type": history_status.get("node_type"),
                "node_id": history_status.get("node_id"),
            }
        )
    return {
        "short": classified.get("short") or "ComfyUI execution error.",
        "details": details,
        "category": classified.get("category") or "unknown",
    }


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
    cache_dir: Path,
    client: httpx.AsyncClient,
    progress_cb: Optional[Any] = None,
) -> Dict[str, Any]:
    """Downloads checkpoint to cache directory."""
    url = item.resolved_url or item.url
    target_path = cache_dir / item.name
    tmp_path = cache_dir / f"{item.name}.part"
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
    meta_path = cache_dir / f"{item.name}.meta.json"
    try:
        meta_path.write_text(json.dumps(meta, indent=2))
    except OSError:
        pass

    return {"name": item.name, "status": "downloaded", "error": None}


def _ensure_comfyui_checkpoint_visible(filename: str, cache_dir: Path, comfy_dir: Path) -> Dict[str, Any]:
    """
    Ensures a checkpoint in the cache is visible to ComfyUI by creating a symlink or copy.

    Args:
        filename: The checkpoint filename
        cache_dir: Where the checkpoint is cached
        comfy_dir: Where ComfyUI scans for checkpoints

    Returns:
        Dict with status and error (if any)
    """
    cache_path = cache_dir / filename
    comfy_path = comfy_dir / filename

    if not cache_path.exists():
        return {"status": "error", "error": f"Checkpoint not in cache: {cache_path}"}

    # Create ComfyUI checkpoints dir if missing
    comfy_dir.mkdir(parents=True, exist_ok=True)

    # If already exists and is valid, skip
    if comfy_path.exists():
        if comfy_path.is_symlink():
            if comfy_path.resolve() == cache_path.resolve():
                return {"status": "ok", "method": "symlink_exists"}
        try:
            # Verify it's the same file (by size)
            if comfy_path.stat().st_size == cache_path.stat().st_size:
                return {"status": "ok", "method": "file_exists"}
        except OSError:
            pass
        # Remove stale link/file
        comfy_path.unlink(missing_ok=True)

    # Try symlink first (preferred)
    try:
        comfy_path.symlink_to(cache_path)
        log.info(f"Created symlink: {comfy_path} -> {cache_path}")
        return {"status": "ok", "method": "symlink"}
    except (OSError, NotImplementedError) as e:
        log.warning(f"Symlink failed ({e}), falling back to copy")

    # Fallback to copy
    try:
        import shutil
        shutil.copy2(cache_path, comfy_path)
        log.info(f"Copied checkpoint: {cache_path} -> {comfy_path}")
        return {"status": "ok", "method": "copy"}
    except OSError as e:
        return {"status": "error", "error": f"Failed to copy: {e}"}


async def _verify_checkpoint_in_object_info(filename: str) -> Dict[str, Any]:
    """
    Verifies that ComfyUI sees the checkpoint in its object_info.

    Args:
        filename: The checkpoint filename to verify

    Returns:
        Dict with 'visible' (bool) and diagnostic info
    """
    base_url = _comfy_base_url()
    comfy_dir = _comfy_checkpoints_dir()

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url}/object_info")
            if resp.status_code != 200:
                return {
                    "visible": False,
                    "error": f"Failed to fetch object_info: HTTP {resp.status_code}",
                }

            obj_info = resp.json()
            # Navigate to CheckpointLoaderSimple input options
            checkpoint_loader = obj_info.get("CheckpointLoaderSimple", {})
            input_spec = checkpoint_loader.get("input", {})
            required = input_spec.get("required", {})
            ckpt_name_spec = required.get("ckpt_name", [])

            # ckpt_name_spec is typically: [["file1.safetensors", "file2.safetensors", ...]]
            available_checkpoints = []
            if isinstance(ckpt_name_spec, list) and len(ckpt_name_spec) > 0:
                available_checkpoints = ckpt_name_spec[0] if isinstance(ckpt_name_spec[0], list) else []

            visible = filename in available_checkpoints

            if not visible:
                # Provide diagnostic information
                log.warning(
                    f"ComfyUI does not see checkpoint '{filename}'. "
                    f"Available: {available_checkpoints}. "
                    f"Expected path: {comfy_dir / filename}"
                )
                # List actual files in directory for diagnosis
                try:
                    actual_files = [p.name for p in comfy_dir.iterdir() if p.is_file()] if comfy_dir.exists() else []
                except OSError:
                    actual_files = []

                return {
                    "visible": False,
                    "available_checkpoints": available_checkpoints,
                    "expected_path": str(comfy_dir / filename),
                    "actual_files_in_dir": actual_files,
                    "comfy_dir": str(comfy_dir),
                }

            return {"visible": True}

    except Exception as exc:
        log.exception("Failed to verify checkpoint in object_info")
        return {"visible": False, "error": str(exc)}


# ComfyUI workflow output node metadata (stored separately from workflow graph)
# This is NOT sent to ComfyUI - it's metadata for our internal use
COMFY_OUTPUT_NODE = "7"  # SaveImage node ID
COMFY_OUTPUT_INDEX = 0   # Output slot index

# Mapping of UI sampler labels to ComfyUI sampler_name + scheduler pairs
SAMPLER_MAP: Dict[str, Dict[str, str]] = {
    "Euler":                {"sampler_name": "euler",       "scheduler": "normal"},
    "Euler a":              {"sampler_name": "euler_ancestral", "scheduler": "normal"},
    "DPM++ 2M":             {"sampler_name": "dpmpp_2m",    "scheduler": "normal"},
    "DPM++ 2M Karras":      {"sampler_name": "dpmpp_2m",    "scheduler": "karras"},
    "DPM++ SDE Karras":     {"sampler_name": "dpmpp_sde",   "scheduler": "karras"},
}
DEFAULT_SAMPLER_KEY = "DPM++ 2M Karras"


def _resolve_sampler(sampler_label: str) -> Dict[str, str]:
    """Resolve a UI sampler label to ComfyUI sampler_name + scheduler.

    Strips any '(recommended)' suffix and falls back to the default if unknown.
    """
    clean = sampler_label.strip()
    # Strip "(recommended)" or similar parenthetical suffixes
    if "(" in clean:
        clean = clean[:clean.index("(")].strip()
    result = SAMPLER_MAP.get(clean)
    if result:
        return result
    log.warning("Unknown sampler '%s', falling back to '%s'", sampler_label, DEFAULT_SAMPLER_KEY)
    return SAMPLER_MAP[DEFAULT_SAMPLER_KEY]


def _build_comfy_workflow(
    prompt: str,
    checkpoint: str,
    seed: int,
    steps: int,
    width: int,
    height: int,
    sampler: str = "DPM++ 2M Karras",
) -> Dict[str, Any]:
    """Build ComfyUI workflow graph with ONLY valid nodes (no metadata keys)."""
    sampler_cfg = _resolve_sampler(sampler)
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
                "sampler_name": sampler_cfg["sampler_name"],
                "scheduler": sampler_cfg["scheduler"],
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


async def _fetch_comfy_image(
    client: httpx.AsyncClient,
    base_url: str,
    filename: str,
    subfolder: str = "",
    img_type: str = "output",
) -> Optional[bytes]:
    """
    Fetch an image from ComfyUI's /view endpoint.

    Args:
        client: httpx.AsyncClient for making requests
        base_url: ComfyUI base URL
        filename: Image filename from history
        subfolder: Subfolder (if any) from history
        img_type: Type of image (usually 'output', sometimes 'input' or 'temp')

    Returns:
        Image bytes or None if fetch failed
    """
    try:
        params = {"filename": filename, "type": img_type}
        if subfolder:
            params["subfolder"] = subfolder
        resp = await client.get(f"{base_url}/view", params=params)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        log.warning(f"Failed to fetch image {filename}: {e}")
        return None


# ---------------------------
# HTTP endpoints
# ---------------------------
@app.get("/health")
async def health():
    return {"ok": True, "machine_id": CFG.get("machine_id"), "label": CFG.get("label")}


# ---------------------------
# Backend lifecycle endpoints
# ---------------------------

# Track which inference backend is currently active
_ACTIVE_BACKEND: Optional[str] = None  # "ollama", "vllm", "comfyui", or None
_BACKEND_KEEP_WARM: bool = False
_BACKEND_IDLE_SECS: int = int(os.environ.get("AGENT_BACKEND_IDLE_SECS", "0"))
_BACKEND_IDLE_TASK: Optional[asyncio.Task] = None


class BackendSelectRequest(BaseModel):
    backend: str  # "ollama", "vllm", "comfyui"
    model: Optional[str] = None
    keep_warm: bool = False


class BackendStatusResponse(BaseModel):
    active_backend: Optional[str] = None
    backends: Dict[str, Any] = Field(default_factory=dict)
    keep_warm: bool = False


async def _check_backend_health(backend: str) -> Dict[str, Any]:
    """Check if a backend is healthy and return status info."""
    vllm_cfg = CFG.get("vllm", {})
    vllm_base = f"http://{vllm_cfg.get('host', '127.0.0.1')}:{vllm_cfg.get('port', 8000)}"
    ollama_base = CFG.get("ollama", {}).get("base_url", "http://127.0.0.1:11434")
    comfy_host = CFG.get("comfyui", {}).get("host", "127.0.0.1")
    comfy_port = CFG.get("comfyui", {}).get("port", 8188)

    if backend == "ollama":
        healthy = await check_ollama_available(ollama_base)
        return {"name": "ollama", "healthy": healthy, "url": ollama_base}
    elif backend == "vllm":
        from backends.vllm_backend import check_vllm_available
        healthy = await check_vllm_available(vllm_base)
        return {"name": "vllm", "healthy": healthy, "url": vllm_base}
    elif backend == "comfyui":
        try:
            url = f"http://{comfy_host}:{comfy_port}/system_stats"
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(url)
                healthy = r.status_code == 200
        except Exception:
            healthy = False
        return {"name": "comfyui", "healthy": healthy, "url": f"http://{comfy_host}:{comfy_port}"}
    return {"name": backend, "healthy": False, "error": "unknown backend"}


async def _run_agent_script(command: str, *args: str) -> Dict[str, Any]:
    """Run a scripts/agent subcommand and return result."""
    agent_script = ROOT_DIR / "scripts" / "agent"
    if not agent_script.exists():
        return {"ok": False, "error": f"scripts/agent not found at {agent_script}"}

    cmd = [str(agent_script), command] + list(args)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(ROOT_DIR),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": stdout.decode(errors="replace")[-2000:],
            "stderr": stderr.decode(errors="replace")[-2000:],
        }
    except asyncio.TimeoutError:
        return {"ok": False, "error": "Timed out after 600s"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def _idle_timeout_watcher():
    """Background task that stops the backend after inactivity."""
    global _ACTIVE_BACKEND
    if _BACKEND_IDLE_SECS <= 0:
        return
    await asyncio.sleep(_BACKEND_IDLE_SECS)
    if _ACTIVE_BACKEND and not _BACKEND_KEEP_WARM:
        slog.info("backend_idle_timeout", backend=_ACTIVE_BACKEND, idle_secs=_BACKEND_IDLE_SECS)
        await _run_agent_script("stop-backend", _ACTIVE_BACKEND)
        _ACTIVE_BACKEND = None


def _reset_idle_timer():
    """Reset the idle timeout timer."""
    global _BACKEND_IDLE_TASK
    if _BACKEND_IDLE_TASK and not _BACKEND_IDLE_TASK.done():
        _BACKEND_IDLE_TASK.cancel()
    if _BACKEND_IDLE_SECS > 0 and not _BACKEND_KEEP_WARM:
        _BACKEND_IDLE_TASK = asyncio.create_task(_idle_timeout_watcher())


@app.post("/api/backend/select")
async def select_backend(req: BackendSelectRequest):
    """
    Select and start an inference backend.
    Stops any conflicting backend first (only-one-active-LLM-backend policy).
    """
    global _ACTIVE_BACKEND, _BACKEND_KEEP_WARM

    backend = req.backend.lower()
    if backend not in ("ollama", "vllm", "comfyui"):
        raise HTTPException(status_code=400, detail=f"Unknown backend: {backend}")

    slog.info("backend_select", backend=backend, model=req.model, keep_warm=req.keep_warm)
    _BACKEND_KEEP_WARM = req.keep_warm

    # Build args for start-backend
    args = [backend]
    if req.model:
        args.append(req.model)

    result = await _run_agent_script("start-backend", *args)

    if result.get("ok"):
        _ACTIVE_BACKEND = backend
        _reset_idle_timer()

        # Wait for health
        health_info = await _check_backend_health(backend)
        return {
            "ok": True,
            "backend": backend,
            "state": "ready" if health_info.get("healthy") else "starting",
            "health": health_info,
        }
    else:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "backend": backend,
                "state": "error",
                "error": result.get("stderr", result.get("error", "Unknown error")),
            },
        )


@app.post("/api/backend/stop")
async def stop_backend_endpoint(backend: Optional[str] = None):
    """Stop a specific backend or the active one."""
    global _ACTIVE_BACKEND

    target = backend or _ACTIVE_BACKEND
    if not target:
        return {"ok": True, "message": "No active backend to stop"}

    slog.info("backend_stop", backend=target)
    result = await _run_agent_script("stop-backend", target)

    if _ACTIVE_BACKEND == target:
        _ACTIVE_BACKEND = None

    return {
        "ok": result.get("ok", False),
        "backend": target,
        "state": "stopped",
    }


@app.get("/api/backend/status")
async def backend_status():
    """Return status of all backends."""
    statuses = {}
    for b in ("ollama", "vllm", "comfyui"):
        try:
            statuses[b] = await _check_backend_health(b)
        except Exception as e:
            statuses[b] = {"name": b, "healthy": False, "error": str(e)}

    return BackendStatusResponse(
        active_backend=_ACTIVE_BACKEND,
        backends=statuses,
        keep_warm=_BACKEND_KEEP_WARM,
    )


async def _stop_ollama() -> dict:
    """Stop Ollama service with detailed logging. Platform-specific implementation."""
    import platform
    system = platform.system()
    t_start = time.time()

    try:
        cmd = None
        if system == "Darwin":
            # macOS: Use brew services
            cmd = ["brew", "services", "stop", "ollama"]
        elif system == "Linux":
            # Linux: Try systemctl (try both system and user)
            cmd = ["sudo", "-n", "systemctl", "stop", "ollama"]
        else:
            slog.warning("reset_stop_ollama_unsupported", platform=system)
            return {"stopped": False, "error": f"Unsupported platform: {system}"}

        # Run with capture
        result = run_subprocess_with_capture(cmd, "ollama_stop", timeout=30)

        # If Linux systemctl failed, try user systemctl
        if system == "Linux" and result["returncode"] != 0:
            cmd_user = ["systemctl", "--user", "stop", "ollama"]
            result = run_subprocess_with_capture(cmd_user, "ollama_stop_user", timeout=30)

        success = result["returncode"] == 0
        ms = (time.time() - t_start) * 1000

        slog.info("reset_stop_ollama",
                  platform=system.lower(),
                  success=success,
                  returncode=result["returncode"],
                  log_file=result["log_file"])

        return {
            "stopped": success,
            "stop_command": result["command"],
            "stop_stdout_tail": result["stdout_tail"],
            "stop_stderr_tail": result["stderr_tail"],
            "stop_log_file": result["log_file"],
            "error": result.get("friendly_error"),
        }
    except Exception as e:
        ms = (time.time() - t_start) * 1000
        slog.error("reset_stop_ollama_error", error=str(e))
        return {"stopped": False, "error": str(e)}


async def _start_ollama() -> dict:
    """Start Ollama service with detailed logging. Platform-specific implementation."""
    import platform
    system = platform.system()
    t_start = time.time()

    try:
        cmd = None
        if system == "Darwin":
            # macOS: Use brew services
            cmd = ["brew", "services", "start", "ollama"]
        elif system == "Linux":
            # Linux: Try systemctl (try both system and user)
            cmd = ["sudo", "-n", "systemctl", "start", "ollama"]
        else:
            slog.warning("reset_start_ollama_unsupported", platform=system)
            return {"started": False, "error": f"Unsupported platform: {system}"}

        # Run with capture
        result = run_subprocess_with_capture(cmd, "ollama_start", timeout=30)

        # If Linux systemctl failed, try user systemctl
        if system == "Linux" and result["returncode"] != 0:
            cmd_user = ["systemctl", "--user", "start", "ollama"]
            result = run_subprocess_with_capture(cmd_user, "ollama_start_user", timeout=30)

        success = result["returncode"] == 0
        ms = (time.time() - t_start) * 1000

        slog.info("reset_start_ollama",
                  platform=system.lower(),
                  success=success,
                  returncode=result["returncode"],
                  log_file=result["log_file"])

        return {
            "started": success,
            "start_command": result["command"],
            "start_stdout_tail": result["stdout_tail"],
            "start_stderr_tail": result["stderr_tail"],
            "start_log_file": result["log_file"],
            "error": result.get("friendly_error"),
        }
    except Exception as e:
        ms = (time.time() - t_start) * 1000
        slog.error("reset_start_ollama_error", error=str(e))
        return {"started": False, "error": str(e)}


async def _check_ollama_health(timeout_s: float = None) -> dict:
    """
    Wait for Ollama to become healthy using /api/tags endpoint.

    Args:
        timeout_s: Timeout in seconds (uses OLLAMA_START_TIMEOUT_S if None)
    """
    if timeout_s is None:
        timeout_s = OLLAMA_START_TIMEOUT_S

    base_url = CFG.get("ollama_base_url", "http://127.0.0.1:11434")
    health_url = f"{base_url}/api/tags"

    slog.info("reset_checking_ollama_health", url=health_url, timeout_s=timeout_s)

    # Use the new polling helper
    healthy, time_to_ready_ms, error = await poll_http_health(
        health_url,
        timeout_s=timeout_s,
        poll_interval_s=HEALTH_POLL_INTERVAL_S,
        expected_status=200
    )

    if healthy:
        slog.info("reset_ollama_healthy", time_to_ready_ms=int(time_to_ready_ms))
        return {
            "healthy": True,
            "time_to_ready_ms": int(time_to_ready_ms),
        }
    else:
        slog.warning("reset_ollama_unhealthy", timeout_s=timeout_s, error=error)
        return {
            "healthy": False,
            "time_to_ready_ms": None,
            "error": error,
        }


async def _stop_comfyui() -> dict:
    """Stop ComfyUI process with detailed logging."""
    t_start = time.time()
    notes = []

    try:
        pidfile = ROOT_DIR / "comfy.pid"
        if not pidfile.exists():
            slog.info("reset_stop_comfyui", status="no_pidfile")
            notes.append("No ComfyUI PID file found")
            return {"stopped": True, "notes": notes}

        try:
            pid = int(pidfile.read_text().strip())

            # Check if process is actually running
            if not _pid_is_running(pid):
                pidfile.unlink(missing_ok=True)
                notes.append(f"ComfyUI process {pid} not running, cleaned up PID file")
                slog.info("reset_stop_comfyui", status="not_running", pid=pid)
                return {"stopped": True, "notes": notes}

            # Send SIGTERM
            os.kill(pid, signal.SIGTERM)
            slog.info("reset_stop_comfyui", action="sent_sigterm", pid=pid)

            # Wait for process to exit (up to 5 seconds)
            for i in range(50):
                if not _pid_is_running(pid):
                    pidfile.unlink(missing_ok=True)
                    ms = (time.time() - t_start) * 1000
                    slog.info("reset_stop_comfyui", success=True, pid=pid, ms=int(ms))
                    return {"stopped": True, "pid": pid}
                await asyncio.sleep(0.1)

            # Process still running after timeout
            notes.append(f"ComfyUI process {pid} did not exit after SIGTERM (5s timeout)")
            slog.warning("reset_stop_comfyui_timeout", pid=pid)
            return {"stopped": False, "pid": pid, "notes": notes}

        except ProcessLookupError:
            pidfile.unlink(missing_ok=True)
            notes.append("ComfyUI process not found")
            slog.info("reset_stop_comfyui", status="process_not_found")
            return {"stopped": True, "notes": notes}
        except Exception as e:
            ms = (time.time() - t_start) * 1000
            slog.error("reset_stop_comfyui_error", error=str(e))
            return {"stopped": False, "error": str(e)}
    except Exception as e:
        ms = (time.time() - t_start) * 1000
        slog.error("reset_stop_comfyui_error", error=str(e))
        return {"stopped": False, "error": str(e)}


async def _start_comfyui() -> dict:
    """Start ComfyUI process with log capture."""
    from agent.reset_helpers import RESET_LOG_DIR, get_timestamp
    t_start = time.time()
    notes = []

    try:
        comfy = _comfy_config()
        if not comfy.get("enabled", True):
            notes.append("ComfyUI disabled in config")
            return {"started": False, "notes": notes}

        # Check if already running
        pidfile = ROOT_DIR / "comfy.pid"
        if pidfile.exists():
            try:
                pid = int(pidfile.read_text().strip())
                if _pid_is_running(pid):
                    slog.info("reset_start_comfyui", status="already_running", pid=pid)
                    notes.append(f"ComfyUI already running (PID {pid})")
                    return {"started": True, "pid": pid, "notes": notes}
            except Exception:
                pass

        # Start ComfyUI using existing logic
        install_dir = _comfy_install_dir()
        python_path = install_dir / ".venv" / "bin" / "python"
        if not python_path.exists():
            slog.warning("reset_start_comfyui", status="no_venv")
            return {"started": False, "error": "ComfyUI venv not found"}

        env = os.environ.copy()
        host = comfy.get("host", "127.0.0.1")
        port = comfy.get("port", 8188)

        # Create log file for ComfyUI output
        timestamp = get_timestamp()
        log_file = RESET_LOG_DIR / f"comfyui_start_{timestamp}.log"
        log_handle = open(log_file, "w")

        cmd = [str(python_path), "main.py", "--listen", host, "--port", str(port)]

        proc = subprocess.Popen(
            cmd,
            cwd=str(install_dir),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
        )

        pidfile.write_text(str(proc.pid))
        ms = (time.time() - t_start) * 1000
        slog.info("reset_start_comfyui", success=True, pid=proc.pid, ms=int(ms), log_file=str(log_file))

        return {
            "started": True,
            "pid": proc.pid,
            "start_command": cmd,
            "start_log_file": str(log_file.absolute()),
            "start_stdout_tail": f"ComfyUI started, logs being written to {log_file}",
            "start_stderr_tail": "",
        }
    except Exception as e:
        ms = (time.time() - t_start) * 1000
        slog.error("reset_start_comfyui_error", error=str(e))
        return {"started": False, "error": str(e)}


async def _check_comfyui_health(timeout_s: float = None) -> dict:
    """
    Wait for ComfyUI to become healthy.

    Args:
        timeout_s: Timeout in seconds (uses COMFYUI_START_TIMEOUT_S if None)
    """
    if timeout_s is None:
        timeout_s = COMFYUI_START_TIMEOUT_S

    base_url = _comfy_base_url()
    health_url = f"{base_url}/system_stats"

    slog.info("reset_checking_comfyui_health", url=health_url, timeout_s=timeout_s)

    # Use the new polling helper
    healthy, time_to_ready_ms, error = await poll_http_health(
        health_url,
        timeout_s=timeout_s,
        poll_interval_s=HEALTH_POLL_INTERVAL_S,
        expected_status=200
    )

    if healthy:
        slog.info("reset_comfyui_healthy", time_to_ready_ms=int(time_to_ready_ms))
        return {
            "healthy": True,
            "time_to_ready_ms": int(time_to_ready_ms),
        }
    else:
        slog.warning("reset_comfyui_unhealthy", timeout_s=timeout_s, error=error)
        return {
            "healthy": False,
            "time_to_ready_ms": None,
            "error": error,
        }


@app.post("/api/reset")
async def reset_agent():
    """
    Reset agent by restarting Ollama and ComfyUI services with detailed diagnostics.

    Returns detailed JSON with:
    - Full command output logs and log file paths
    - Time-to-ready metrics for each service
    - Detailed notes about what happened
    - Always returns HTTP 200 with diagnostics (never error codes)

    Configuration via environment variables:
    - OLLAMA_START_TIMEOUT_S (default 120)
    - COMFYUI_START_TIMEOUT_S (default 60)
    - HEALTH_POLL_INTERVAL_S (default 1.5)
    - RESET_LOG_DIR (default ./logs)
    """
    t_reset_start = time.time()
    slog.info("reset_agent_start")
    notes = []

    # Step 1: Stop Ollama
    slog.info("reset: stopping ollama")
    ollama_stop = await _stop_ollama()

    # Step 2: Start Ollama
    slog.info("reset: starting ollama")
    ollama_start = await _start_ollama()

    # Step 3: Wait for Ollama health (configurable timeout)
    slog.info("reset: checking ollama health")
    ollama_health = await _check_ollama_health()  # Uses OLLAMA_START_TIMEOUT_S

    if not ollama_health.get("healthy"):
        notes.append(f"Ollama unhealthy after {OLLAMA_START_TIMEOUT_S}s")

    # Step 4: Stop ComfyUI
    slog.info("reset: stopping comfyui")
    comfyui_stop = await _stop_comfyui()
    if comfyui_stop.get("notes"):
        notes.extend(comfyui_stop["notes"])

    # Step 5: Start ComfyUI
    slog.info("reset: starting comfyui")
    comfyui_start = await _start_comfyui()
    if comfyui_start.get("notes"):
        notes.extend(comfyui_start["notes"])

    # Step 6: Wait for ComfyUI health (configurable timeout)
    slog.info("reset: checking comfyui health")
    comfyui_health = await _check_comfyui_health()  # Uses COMFYUI_START_TIMEOUT_S

    if not comfyui_health.get("healthy"):
        notes.append(f"ComfyUI unhealthy after {COMFYUI_START_TIMEOUT_S}s")

    # Calculate total duration
    duration_ms = int((time.time() - t_reset_start) * 1000)

    # Build detailed response
    final_ollama_healthy = ollama_health.get("healthy", False)
    final_comfyui_healthy = comfyui_health.get("healthy", False)
    ok = final_ollama_healthy and final_comfyui_healthy
    warnings = bool(notes)

    result = {
        "ok": ok,
        "warnings": warnings,
        "duration_ms": duration_ms,
        "ollama": {
            "stopped": ollama_stop.get("stopped", False),
            "stop_stdout_tail": ollama_stop.get("stop_stdout_tail", ""),
            "stop_stderr_tail": ollama_stop.get("stop_stderr_tail", ""),
            "stop_log_file": ollama_stop.get("stop_log_file"),
            "start_command": ollama_start.get("start_command", []),
            "pid": ollama_start.get("pid"),
            "start_stdout_tail": ollama_start.get("start_stdout_tail", ""),
            "start_stderr_tail": ollama_start.get("start_stderr_tail", ""),
            "start_log_file": ollama_start.get("start_log_file"),
            "healthy": final_ollama_healthy,
            "time_to_ready_ms": ollama_health.get("time_to_ready_ms"),
        },
        "comfyui": {
            "stopped": comfyui_stop.get("stopped", False),
            "pid": comfyui_start.get("pid"),
            "start_command": comfyui_start.get("start_command", []),
            "start_stdout_tail": comfyui_start.get("start_stdout_tail", ""),
            "start_stderr_tail": comfyui_start.get("start_stderr_tail", ""),
            "start_log_file": comfyui_start.get("start_log_file"),
            "healthy": final_comfyui_healthy,
            "time_to_ready_ms": comfyui_health.get("time_to_ready_ms"),
        },
        "notes": notes if notes else [],
    }

    slog.info("reset_agent_complete", ok=ok, duration_ms=duration_ms)
    return result


@app.get("/capabilities")
async def capabilities():
    base_url = CFG.get("ollama_base_url", "http://127.0.0.1:11434")

    # Check Ollama reachability and get available models
    ollama_reachable = await check_ollama_available(base_url)
    ollama_models = []
    if ollama_reachable:
        ollama_models = await get_ollama_models(base_url)

    # Check vLLM reachability and get available models
    vllm_cfg = CFG.get("vllm", {})
    vllm_base = f"http://{vllm_cfg.get('host', '127.0.0.1')}:{vllm_cfg.get('port', 8000)}"
    vllm_reachable = await check_vllm_available(vllm_base)
    vllm_models = []
    if vllm_reachable:
        vllm_models = await get_vllm_models(vllm_base)

    cap = Capabilities(
        machine_id=CFG.get("machine_id"),
        label=CFG.get("label"),
        tests=CFG.get("tests", ["llm_generate"]),
        llm_models=[],
        whisper_models=[],
        sdxl_profiles=[],
        accelerator_type=CFG.get("accelerator_type")
        or ("cuda" if HARDWARE_INFO.get("gpu_type") == "discrete" else "metal" if HARDWARE_INFO.get("gpu_type") else None),
        accelerator_memory_gb=CFG.get("accelerator_memory_gb")
        or (
            (HARDWARE_INFO.get("gpu_vram_bytes") or 0) / (1024**3)
            if HARDWARE_INFO.get("gpu_vram_bytes")
            else None
        ),
        system_memory_gb=CFG.get("system_memory_gb")
        or (
            (HARDWARE_INFO.get("total_system_ram_bytes") or 0) / (1024**3)
            if HARDWARE_INFO.get("total_system_ram_bytes")
            else None
        ),
        gpu_name=CFG.get("gpu_name") or HARDWARE_INFO.get("gpu_name"),
        gpu_vram_bytes=HARDWARE_INFO.get("gpu_vram_bytes"),
        gpu_type=HARDWARE_INFO.get("gpu_type"),
        cuda_compute=HARDWARE_INFO.get("cuda_compute"),
        cpu_cores=HARDWARE_INFO.get("cpu_cores"),
        cpu_physical_cores=HARDWARE_INFO.get("cpu_physical_cores"),
        total_system_ram_bytes=HARDWARE_INFO.get("total_system_ram_bytes"),
        ollama_reachable=ollama_reachable,
        ollama_models=ollama_models,
        comfyui_gpu_ok=COMFYUI_PREFLIGHT.get("comfyui_gpu_ok"),
        comfyui_cpu_ok=COMFYUI_PREFLIGHT.get("comfyui_cpu_ok"),
    )
    cap_dict = cap.model_dump()
    # Add extended backend info (not in base schema but useful for UI)
    cap_dict["vllm_reachable"] = vllm_reachable
    cap_dict["vllm_models"] = vllm_models
    cap_dict["active_backend"] = _ACTIVE_BACKEND
    return JSONResponse(cap_dict)


@app.get("/api/agent/runtime_metrics")
async def agent_runtime_metrics():
    if not RUNTIME_SAMPLER:
        return JSONResponse({"available": False})
    return JSONResponse(RUNTIME_SAMPLER.snapshot())


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


@app.get("/debug/logging")
async def debug_logging():
    """
    Debug endpoint to test logging configuration and emit sample logs.
    """
    import os
    from agent.logging_utils import LOG_LEVEL, LOG_JSON, LOG_HTTP_BODY, LOG_HTTP_MAXLEN, HOSTNAME, AGENT_ID

    # Emit sample logs at each level
    slog.debug("debug_test", test="This is a DEBUG level log", timestamp=time.time())
    slog.info("info_test", test="This is an INFO level log", timestamp=time.time())
    slog.warning("warning_test", test="This is a WARNING level log", timestamp=time.time())
    slog.error("error_test", test="This is an ERROR level log", timestamp=time.time())

    return {
        "status": "ok",
        "message": "Sample logs emitted at all levels",
        "config": {
            "LOG_LEVEL": LOG_LEVEL,
            "LOG_JSON": LOG_JSON,
            "LOG_HTTP_BODY": LOG_HTTP_BODY,
            "LOG_HTTP_MAXLEN": LOG_HTTP_MAXLEN,
            "HOSTNAME": HOSTNAME,
            "AGENT_ID": AGENT_ID,
        },
        "environment": {
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "not set"),
            "LOG_JSON": os.getenv("LOG_JSON", "not set"),
            "LOG_HTTP_BODY": os.getenv("LOG_HTTP_BODY", "not set"),
            "LOG_HTTP_MAXLEN": os.getenv("LOG_HTTP_MAXLEN", "not set"),
        }
    }


@app.post("/api/comfy/txt2img")
async def comfy_txt2img(req: ComfyTxt2ImgRequest):
    job_id = str(uuid.uuid4())

    # Clean up old images to maintain retention policy
    # Protect current run_id from deletion
    _cleanup_old_images(max_runs=10, current_run_id=req.run_id)

    # Log job received
    slog.info(
        "job_received",
        job_id=job_id,
        job_type="image",
        run_id=req.run_id,
        checkpoint=req.checkpoint,
        seed=req.seed,
        steps=req.steps,
        resolution=f"{req.width}x{req.height}",
        num_images=req.num_images,
        repeat=req.repeat,
    )

    if job_id in RUNNING_JOBS:
        raise HTTPException(status_code=409, detail="job already running")

    # Initialize job status tracking
    created_at_ms = time.time() * 1000.0
    JOB_STATUS[job_id] = {
        "status": "pending",
        "progress": {"current_step": 0, "total_steps": req.steps, "percent": 0},
        "queue_latency_ms": None,
        "gen_time_ms": None,
        "total_ms": None,
        "total_time_s": None,
        "images": [],
        "image_filenames": [],
        "error": None,
        "created_at": created_at_ms / 1000.0,
        "created_at_ms": created_at_ms,
        "updated_at": created_at_ms / 1000.0,
        "updated_at_ms": created_at_ms,
        "started_at_ms": None,
        "first_progress_at_ms": None,
        "completed_at_ms": None,
        "run_id": req.run_id,
        "checkpoint": req.checkpoint,
        "resolution": f"{req.width}x{req.height}",
        "seed": req.seed,
        "steps": req.steps,
        "sampler": req.sampler,
        "num_images": req.num_images,
        "repeat": req.repeat,
    }

    task = asyncio.create_task(_job_runner_comfy(job_id, req))
    RUNNING_JOBS[job_id] = task

    def _on_done(t: asyncio.Task):
        RUNNING_JOBS.pop(job_id, None)
        if t.exception():
            slog.error("job_failed", job_id=job_id, error="Task exception", stack_trace=str(t.exception()))
            log.exception("Comfy job %s terminated with exception", job_id)
            # Update job status on task exception
            if job_id in JOB_STATUS:
                JOB_STATUS[job_id]["status"] = "error"
                JOB_STATUS[job_id]["error"] = str(t.exception())
                JOB_STATUS[job_id]["updated_at"] = time.time()

    task.add_done_callback(_on_done)
    return {"accepted": True, "agent_job_id": job_id}


@app.post("/api/image/start")
async def image_start(req: ComfyTxt2ImgRequest):
    """Alias for /api/comfy/txt2img that also returns job_id field."""
    return await comfy_txt2img(req)


@app.get("/api/image/job_status")
async def image_job_status(job_id: str = ""):
    """Return status of a specific image generation job."""
    if not job_id:
        raise HTTPException(status_code=400, detail="job_id parameter required")
    status = JOB_STATUS.get(job_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    return JSONResponse(status)


@app.get("/api/image/jobs")
async def image_jobs(active: bool = False):
    """List image generation jobs. If active=true, only return non-terminal jobs."""
    jobs = []
    for jid, status in JOB_STATUS.items():
        if active and status.get("status") in ("complete", "error", "timeout"):
            continue
        jobs.append({"job_id": jid, **status})
    return JSONResponse({"jobs": jobs})


@app.get("/api/image/result/{image_id}")
async def get_image_result(image_id: str):
    """
    Serve image bytes by image_id.

    Args:
        image_id: UUID of the image to fetch

    Returns:
        FileResponse with image bytes and appropriate Content-Type

    Raises:
        HTTPException: 404 if image not found, 400 if invalid image_id
    """
    # Validate image_id is a valid UUID to prevent path traversal
    try:
        uuid.UUID(image_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image_id format")

    # Search for the image in all run directories
    if not OUTPUT_IMAGES_DIR.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    # Find the image file
    image_path = None
    for run_dir in OUTPUT_IMAGES_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        # Check for image with any extension
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            potential_path = run_dir / f"{image_id}{ext}"
            if potential_path.exists():
                image_path = potential_path
                break
        if image_path:
            break

    if not image_path or not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    # Determine content type from extension
    ext = image_path.suffix.lower()
    content_type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    content_type = content_type_map.get(ext, "image/png")

    return FileResponse(
        path=str(image_path),
        media_type=content_type,
        headers={
            "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
        }
    )


@app.post("/api/comfy/sync")
async def comfy_sync(req: ComfySyncRequest):
    cache_dir = _comfy_cache_dir()
    comfy_dir = _comfy_checkpoints_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    comfy_dir.mkdir(parents=True, exist_ok=True)
    base_url = req.central_base_url or CFG.get("central_base_url") or ""
    if not base_url:
        return {"error": "central_base_url not configured"}, 400
    missing = []
    for checkpoint in req.checkpoints:
        # Check if already in ComfyUI dir (skip if exists)
        comfy_path = comfy_dir / checkpoint
        cache_path = cache_dir / checkpoint
        if comfy_path.exists():
            continue
        missing.append(checkpoint)
        try:
            # Step 1: Download to cache
            url = base_url.rstrip("/") + f"/api/comfy/checkpoints/{checkpoint}"
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=120.0, write=60.0, pool=60.0)) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                cache_path.write_bytes(resp.content)

            # Step 2: Ensure visible in ComfyUI
            visibility = _ensure_comfyui_checkpoint_visible(checkpoint, cache_dir, comfy_dir)
            if visibility["status"] == "error":
                return {"error": f"Failed to make {checkpoint} visible: {visibility.get('error')}"}, 500
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

    # Log job received (already logged in POST /jobs)
    # Log job accepted
    slog.info(
        "job_accepted",
        job_id=job_id,
        job_type="llm",
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        num_ctx=num_ctx,
        prompt_length=len(prompt),
    )

    log.info("Starting job %s model=%s", job_id, model)

    # Determine which LLM backend to use based on active backend selection
    ollama_base_url = CFG.get("ollama_base_url", "http://127.0.0.1:11434")
    vllm_cfg = CFG.get("vllm", {})
    vllm_base_url = f"http://{vllm_cfg.get('host', '127.0.0.1')}:{vllm_cfg.get('port', 8000)}"

    # If vLLM is the active backend, try it first
    use_vllm = _ACTIVE_BACKEND == "vllm"

    backend_selected = None
    try:
        if use_vllm:
            # Try vLLM first
            vllm_available = await check_vllm_available(vllm_base_url)
            if vllm_available:
                backend_selected = "vllm"
                slog.info("job_backend_selected", job_id=job_id, backend="vllm", reason="active_backend")
                log.info("Using vLLM backend for job %s", job_id)

                slog.info("job_started", job_id=job_id, backend=backend_selected)

                async def _on_token_vllm(text: str, timestamp_s: float) -> None:
                    ev = Event(job_id=job_id, type="llm_token", payload={"text": text, "timestamp_s": timestamp_s})
                    await _broadcast_event(ev)

                result = await stream_vllm_generate(
                    job_id=job_id,
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    num_ctx=num_ctx,
                    base_url=vllm_base_url,
                    on_token=_on_token_vllm,
                )
                # Reset idle timer on activity
                _reset_idle_timer()
            else:
                backend_selected = "mock"
                slog.info("job_backend_selected", job_id=job_id, backend="mock", reason="vllm_unreachable")
                log.warning("vLLM unreachable; falling back to mock backend for job %s", job_id)
                slog.info("job_started", job_id=job_id, backend=backend_selected)
                result = await _run_mock_stream(job_id, model, prompt, max_tokens, temperature, num_ctx, fallback_reason="vllm_unreachable")
        else:
            # Default: try Ollama
            available = await check_ollama_available(ollama_base_url)
            if not available:
                backend_selected = "mock"
                slog.info(
                    "job_backend_selected",
                    job_id=job_id,
                    backend=backend_selected,
                    reason="ollama_unreachable",
                )
                log.warning("Ollama unreachable; falling back to mock backend for job %s", job_id)

                slog.info("job_started", job_id=job_id, backend=backend_selected)
                result = await _run_mock_stream(job_id, model, prompt, max_tokens, temperature, num_ctx, fallback_reason="ollama_unreachable")
            else:
                # Check if model is available on Ollama
                ollama_models = await get_ollama_models(ollama_base_url)
                if model not in ollama_models:
                    backend_selected = "mock"
                    slog.info(
                        "job_backend_selected",
                        job_id=job_id,
                        backend=backend_selected,
                        reason="missing_model",
                        available_models=ollama_models,
                    )
                    log.warning("Model %s not found on Ollama (available: %s); falling back to mock for job %s", model, ollama_models, job_id)

                    slog.info("job_started", job_id=job_id, backend=backend_selected)
                    result = await _run_mock_stream(job_id, model, prompt, max_tokens, temperature, num_ctx, fallback_reason="missing_model")
                else:
                    backend_selected = "ollama"
                    slog.info(
                        "job_backend_selected",
                        job_id=job_id,
                        backend=backend_selected,
                        reason="model_available",
                    )
                    log.info("Using Ollama backend for job %s", job_id)

                    slog.info("job_started", job_id=job_id, backend=backend_selected)

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
                        base_url=ollama_base_url,
                        on_token=_on_token,
                    )
                    # Reset idle timer on activity
                    _reset_idle_timer()
    except Exception as e:
        backend_selected = "mock"
        slog.info(
            "job_backend_selected",
            job_id=job_id,
            backend=backend_selected,
            reason="stream_error",
            error=str(e),
        )
        log.warning("LLM stream failed (%s); falling back to mock stream", e)

        slog.info("job_started", job_id=job_id, backend=backend_selected)
        result = await _run_mock_stream(job_id, model, prompt, max_tokens, temperature, num_ctx, fallback_reason="stream_error")

    # Log job completion
    slog.info(
        "job_completed",
        job_id=job_id,
        job_type="llm",
        tokens_generated=result.get("tokens_generated", 0),
        duration_ms=result.get("total_ms", 0),
    )

    # Emit final metrics event
    ev = Event(job_id=job_id, type="job_done", payload=result)
    await _broadcast_event(ev)
    log.info("Job %s done. result=%s", job_id, result)


async def _job_runner_comfy(job_id: str, req: ComfyTxt2ImgRequest):
    base_url = _comfy_base_url()
    checkpoints_dir = _comfy_checkpoints_dir()

    def _update_job_status(updates: Dict[str, Any]) -> None:
        """Update JOB_STATUS for this job_id."""
        if job_id in JOB_STATUS:
            JOB_STATUS[job_id].update(updates)
            updated_at_ms = time.time() * 1000.0
            JOB_STATUS[job_id]["updated_at"] = updated_at_ms / 1000.0
            JOB_STATUS[job_id]["updated_at_ms"] = updated_at_ms

    def _remediation_hint(remediation: List[str]) -> Optional[str]:
        return remediation[0] if remediation else None

    async def _emit_image_error(error_msg: str, formatted: Dict[str, Any]) -> None:
        """Broadcast image_error event and update JOB_STATUS."""
        remediation = formatted["details"].get("remediation", [])
        _update_job_status({
            "status": "error",
            "error": error_msg,
            "error_category": formatted["details"].get("category", "unknown"),
            "remediation_hint": _remediation_hint(remediation),
        })
        await _broadcast_event(
            Event(
                job_id=job_id,
                type="image_error",
                payload={
                    "run_id": req.run_id,
                    "message": error_msg,
                    "remediation": remediation,
                    "remediation_hint": _remediation_hint(remediation),
                    "category": formatted["details"].get("category", "unknown"),
                },
            )
        )

    # Log job accepted
    slog.info(
        "job_accepted",
        job_id=job_id,
        job_type="image",
        checkpoint=req.checkpoint,
        seed=req.seed,
        steps=req.steps,
        resolution=f"{req.width}x{req.height}",
        num_images=req.num_images,
        repeat=req.repeat,
    )

    # Validate checkpoints directory
    if not checkpoints_dir.exists():
        error_msg = f"Checkpoints dir missing: {checkpoints_dir}"
        formatted = _format_comfy_error(error_msg)
        slog.error(
            "job_failed",
            job_id=job_id,
            error=formatted["short"],
            remediation=formatted["details"]["remediation"],
            category=formatted["details"]["category"],
            raw_error=error_msg,
        )
        await _emit_image_error(formatted["short"], formatted)
        return

    # Validate checkpoint availability
    if req.checkpoint not in _list_checkpoints():
        error_msg = "Checkpoint not found"
        formatted = _format_comfy_error(error_msg)
        slog.error(
            "job_failed",
            job_id=job_id,
            error=formatted["short"],
            checkpoint=req.checkpoint,
            remediation=formatted["details"]["remediation"],
            category=formatted["details"]["category"],
            raw_error=error_msg,
        )
        await _emit_image_error(formatted["short"], formatted)
        return

    # Log checkpoint resolved
    slog.info(
        "job_checkpoint_resolved",
        job_id=job_id,
        checkpoint=req.checkpoint,
        checkpoints_dir=str(checkpoints_dir),
    )

    # Log backend selected
    slog.info(
        "job_backend_selected",
        job_id=job_id,
        backend="comfyui",
        reason="image_job",
    )

    # Log job started
    slog.info("job_started", job_id=job_id, backend="comfyui")
    _update_job_status({"status": "pending"})

    submit_time = time.perf_counter()
    total_images = []
    last_preview_step = -1
    current_step = 0
    max_step_observed = 0  # Track maximum step for zero-step detection
    first_step_logged = False  # Track if we've logged job_first_step
    started_at = None
    started_at_ms = None
    fallback_attempted = False
    fallback_used = False

    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=60.0, write=60.0, pool=60.0)) as client:
        image_count = req.repeat * req.num_images
        for repeat_index in range(image_count):
            attempt = 0
            while True:
                client_id = str(uuid.uuid4())
                seed_offset = req.seed + repeat_index
                workflow = _build_comfy_workflow(
                    req.prompt,
                    req.checkpoint,
                    seed_offset,
                    req.steps,
                    req.width,
                    req.height,
                    sampler=req.sampler,
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
                    prompt_data = resp.json()
                    prompt_id = prompt_data.get("prompt_id")

                    # Log successful prompt submission
                    slog.info(
                        "job_prompt_sent",
                        job_id=job_id,
                        prompt_id=prompt_id,
                        client_id=client_id,
                        image_index=repeat_index + 1,
                        total_images=image_count,
                    )
                except httpx.HTTPStatusError as exc:
                    # Enhanced error logging for HTTP errors
                    status_code = exc.response.status_code
                    try:
                        response_body = exc.response.text[:8192]  # First 8KB
                    except Exception:
                        response_body = "<unable to read response body>"

                    payload_summary = _summarize_comfy_payload(cleaned_workflow)

                    # Log comprehensive diagnostics
                    error_msg = f"ComfyUI rejected prompt (HTTP {status_code})"
                    formatted = _format_comfy_error(error_msg)
                    slog.error(
                        "job_failed",
                        job_id=job_id,
                        error=formatted["short"],
                        status_code=status_code,
                        response_body=response_body[:500],  # Truncate for logging
                        payload_summary=payload_summary,
                        debug_file=debug_file or "not saved",
                        remediation=formatted["details"]["remediation"],
                        category=formatted["details"]["category"],
                        raw_error=error_msg,
                    )
                    log.error(
                        "ComfyUI POST /prompt failed: status=%d, response_body=%s, payload_summary=%s, debug_file=%s",
                        status_code,
                        response_body,
                        payload_summary,
                        debug_file or "not saved",
                    )

                    # Include full error details in the broadcast message
                    full_error_msg = formatted["short"]
                    await _emit_image_error(full_error_msg, formatted)
                    return
                except Exception as exc:
                    # Handle other errors (connection errors, timeouts, etc.)
                    payload_summary = _summarize_comfy_payload(cleaned_workflow)

                    error_msg = f"Failed to submit prompt to ComfyUI: {exc}"
                    formatted = _format_comfy_error(error_msg)
                    slog.error(
                        "job_failed",
                        job_id=job_id,
                        error=formatted["short"],
                        payload_summary=payload_summary,
                        debug_file=debug_file or "not saved",
                        stack_trace=str(exc),
                        remediation=formatted["details"]["remediation"],
                        category=formatted["details"]["category"],
                        raw_error=error_msg,
                    )
                    log.error(
                        "ComfyUI POST /prompt failed: exception=%s, payload_summary=%s, debug_file=%s",
                        exc,
                        payload_summary,
                        debug_file or "not saved",
                    )

                    full_error_msg = formatted["short"]
                    await _emit_image_error(full_error_msg, formatted)
                    return

                # Define progress callback to track steps and broadcast events.
                # This callback is invoked directly (as an async coroutine) by
                # the ComfyUI websocket tracker, so updates stream live.
                async def on_progress_async(progress: ProgressEvent):
                    nonlocal current_step, max_step_observed, first_step_logged, started_at, started_at_ms
                    current_step = progress.current_step
                    total_steps_ws = progress.total_steps

                    # Refine started_at timing on first real progress tick
                    if not first_step_logged and current_step >= 1:
                        started_at = time.perf_counter()
                        started_at_ms = time.time() * 1000.0
                        queue_latency_ms = (started_at - submit_time) * 1000.0
                        _update_job_status({
                            "status": "running",
                            "queue_latency_ms": queue_latency_ms,
                            "started_at_ms": started_at_ms,
                            "first_progress_at_ms": started_at_ms,
                        })

                    # Track maximum step observed
                    if current_step > max_step_observed:
                        max_step_observed = current_step

                    # Update job status with progress
                    pct = round((current_step / total_steps_ws) * 100, 1) if total_steps_ws > 0 else 0
                    _update_job_status({
                        "status": "running",
                        "progress": {
                            "current_step": current_step,
                            "total_steps": total_steps_ws,
                            "percent": pct,
                        },
                    })

                    # Log first step (IMPORTANT for detecting "0 steps" issue)
                    if current_step >= 1 and not first_step_logged:
                        first_step_logged = True
                        slog.info(
                            "job_first_step",
                            job_id=job_id,
                            step=current_step,
                            total_steps=total_steps_ws,
                        )

                    # Log progress periodically (every 2 steps, or step 1 and final step)
                    if current_step == 1 or current_step == total_steps_ws or current_step % 2 == 0:
                        slog.info(
                            "job_progress",
                            job_id=job_id,
                            step=current_step,
                            total_steps=total_steps_ws,
                            percent=round((current_step / total_steps_ws) * 100, 1) if total_steps_ws > 0 else 0,
                        )

                    await _broadcast_event(
                        Event(
                            job_id=job_id,
                            type="image_progress",
                            payload={
                                "run_id": req.run_id,
                                "step": current_step,
                                "total_steps": total_steps_ws,
                            },
                        )
                    )

                # Define preview callback for step images
                async def on_preview_async(image_bytes: bytes, step: int, total: int):
                    nonlocal last_preview_step
                    if step % 4 == 0 and step != last_preview_step:
                        last_preview_step = step
                        try:
                            # Convert to JPEG for preview
                            image = Image.open(io.BytesIO(image_bytes))
                            buf = io.BytesIO()
                            image.convert("RGB").save(buf, format="JPEG", quality=80)
                            preview_bytes = buf.getvalue()

                            # Save image to disk and get image_id
                            filename = f"preview_{repeat_index + 1}.jpg"
                            image_id = _save_image_to_disk(preview_bytes, req.run_id, filename)

                            # Only include preview_b64 if small enough (< 150KB)
                            preview_b64 = None
                            MAX_PREVIEW_SIZE = 150 * 1024  # 150KB
                            if len(preview_bytes) < MAX_PREVIEW_SIZE:
                                preview_b64 = base64.b64encode(preview_bytes).decode("utf-8")

                            # Prepare payload with image_id (and optional small preview_b64)
                            payload = {
                                "run_id": req.run_id,
                                "step": step,
                                "total_steps": total or req.steps,
                                "filename": filename,
                                "image_id": image_id,
                                "content_type": "image/jpeg",
                                "size_bytes": len(preview_bytes),
                            }
                            if preview_b64:
                                payload["preview_b64"] = preview_b64

                            await _broadcast_event(
                                Event(
                                    job_id=job_id,
                                    type="image_preview",
                                    payload=payload,
                                )
                            )
                        except Exception as e:
                            log.debug(f"Preview processing error: {e}")

                # Emit image_started immediately after prompt submission
                # so the UI shows activity even for cached/instant runs.
                started_at = time.perf_counter()
                started_at_ms = time.time() * 1000.0
                queue_latency_ms = (started_at - submit_time) * 1000.0
                _update_job_status({
                    "status": "running",
                    "queue_latency_ms": queue_latency_ms,
                    "started_at_ms": started_at_ms,
                    "first_progress_at_ms": started_at_ms,
                })
                await _broadcast_event(
                    Event(
                        job_id=job_id,
                        type="image_started",
                        payload={
                            "run_id": req.run_id,
                            "queue_latency_ms": queue_latency_ms,
                            "started_at": started_at_ms / 1000.0,
                            "started_at_ms": started_at_ms,
                            "first_progress_at_ms": started_at_ms,
                        },
                    )
                )

                # Wait for completion using proper websocket tracking with prompt_id filtering.
                # Pass async callbacks directly so progress/preview events are
                # broadcast to the UI in real time instead of being buffered.
                slog.info(
                    "job_waiting_completion",
                    job_id=job_id,
                    prompt_id=prompt_id,
                    timeout_seconds=300.0,
                )

                ws_result: ComfyWSResult = await comfy_wait_for_prompt(
                    prompt_id=prompt_id,
                    client_id=client_id,
                    base_url=base_url,
                    http_client=client,
                    timeout_seconds=300.0,  # 5 minutes timeout
                    on_progress=on_progress_async,
                    on_preview=on_preview_async,
                    structured_logger=slog,
                )

                # Update max_step from tracker result
                if ws_result.max_step > max_step_observed:
                    max_step_observed = ws_result.max_step

                # Check for errors
                if ws_result.timed_out:
                    error_msg = "Job timed out"
                    timeout_total_ms = (time.perf_counter() - submit_time) * 1000.0
                    slog.error(
                        "job_timeout",
                        job_id=job_id,
                        prompt_id=prompt_id,
                        timeout_seconds=300.0,
                    )
                    _update_job_status({
                        "status": "timeout",
                        "error": error_msg,
                        "total_ms": timeout_total_ms,
                        "total_time_s": timeout_total_ms / 1000.0,
                        "image_filenames": [],
                        "error_category": "timeout",
                        "remediation_hint": "Check ComfyUI responsiveness and retry.",
                    })
                    await _broadcast_event(
                        Event(
                            job_id=job_id,
                            type="image_error",
                            payload={
                                "run_id": req.run_id,
                                "message": error_msg,
                                "remediation": ["Check ComfyUI responsiveness and retry."],
                                "remediation_hint": "Check ComfyUI responsiveness and retry.",
                                "category": "timeout",
                                "total_time_s": timeout_total_ms / 1000.0,
                            },
                        )
                    )
                    await _broadcast_event(
                        Event(
                            job_id=job_id,
                            type="job_timeout",
                            payload={
                                "run_id": req.run_id,
                                "message": error_msg,
                                "category": "timeout",
                                "total_time_s": timeout_total_ms / 1000.0,
                            },
                        )
                    )
                    return

                if ws_result.error:
                    formatted = _format_comfy_error(ws_result.error, ws_result.history_status)
                    if (
                        not fallback_attempted
                        and _comfy_cpu_fallback_allowed(formatted["category"])
                        and await _restart_comfyui_cpu()
                    ):
                        fallback_attempted = True
                        fallback_used = True
                        attempt += 1
                        slog.warning(
                            "job_cpu_fallback_retry",
                            job_id=job_id,
                            prompt_id=prompt_id,
                            attempt=attempt,
                        )
                        continue

                    # Get queue status for better diagnostics
                    queue_status = await comfy_check_queue_status(prompt_id, base_url, client)

                    slog.error(
                        "job_failed",
                        job_id=job_id,
                        error=formatted["short"],
                        prompt_id=prompt_id,
                        max_step=ws_result.max_step,
                        events_seen=ws_result.events_seen,
                        execution_time_ms=ws_result.execution_time_ms,
                        history_status=ws_result.history_status,
                        queue_status=queue_status,
                        remediation=formatted["details"]["remediation"],
                        category=formatted["details"]["category"],
                        raw_error=ws_result.error,
                        raw_exception_type=formatted["details"].get("raw_exception_type"),
                        raw_exception_message=formatted["details"].get("raw_exception_message"),
                        node_type=formatted["details"].get("node_type"),
                        node_id=formatted["details"].get("node_id"),
                    )
                    await _emit_image_error(formatted["short"], formatted)
                    return

                if not ws_result.completed:
                    # Not an error but also not completed - check queue and history for debugging
                    queue_status = await comfy_check_queue_status(prompt_id, base_url, client)

                    slog.warning(
                        "job_incomplete",
                        job_id=job_id,
                        prompt_id=prompt_id,
                        events_seen=ws_result.events_seen,
                        max_step=ws_result.max_step,
                        history_status=ws_result.history_status,
                        queue_status=queue_status,
                    )
                    await _broadcast_event(
                        Event(
                            job_id=job_id,
                            type="image_error",
                            payload={
                                "run_id": req.run_id,
                                "message": "Job did not complete - no outputs found",
                                "remediation": [],
                                "remediation_hint": None,
                                "category": "unknown",
                            },
                        )
                    )
                    _update_job_status({
                        "status": "error",
                        "error": "Job did not complete - no outputs found",
                        "error_category": "unknown",
                        "remediation_hint": None,
                    })
                    return

                # Fetch history and extract images
                try:
                    history_resp = await client.get(f"{base_url}/history/{prompt_id}")
                    history_resp.raise_for_status()
                    history = history_resp.json() or {}

                    # Use helper to extract image info
                    image_infos = extract_images_from_history(history, prompt_id)

                    if not image_infos:
                        # No images found even though completion said success
                        queue_status = await comfy_check_queue_status(prompt_id, base_url, client)
                        slog.warning(
                            "job_suspicious_zero_images",
                            job_id=job_id,
                            prompt_id=prompt_id,
                            history_output_keys=list(history.get(prompt_id, {}).get("outputs", {}).keys()),
                            queue_status=queue_status,
                            warning="Job completed but no images found in history outputs",
                        )
                        await _broadcast_event(
                            Event(
                                job_id=job_id,
                                type="image_error",
                                payload={
                                    "run_id": req.run_id,
                                    "message": "Job completed but no images found in outputs",
                                    "remediation": [],
                                    "remediation_hint": None,
                                    "category": "unknown",
                                },
                            )
                        )
                        _update_job_status({
                            "status": "error",
                            "error": "Job completed but no images found in outputs",
                            "error_category": "unknown",
                            "remediation_hint": None,
                        })
                        return

                    # Download each image and save to disk with image_id
                    images = []
                    for img_info in image_infos:
                        filename = img_info["filename"]
                        subfolder = img_info.get("subfolder", "")
                        img_type = img_info.get("type", "output")

                        image_bytes = await _fetch_comfy_image(
                            client, base_url, filename,
                            subfolder=subfolder, img_type=img_type
                        )
                        if not image_bytes:
                            log.warning(f"Failed to fetch image: {filename}")
                            continue

                        # Save image to disk and get image_id
                        image_id = _save_image_to_disk(image_bytes, req.run_id, filename)

                        # Determine content type from filename
                        ext = Path(filename).suffix.lower()
                        content_type_map = {
                            ".png": "image/png",
                            ".jpg": "image/jpeg",
                            ".jpeg": "image/jpeg",
                            ".webp": "image/webp",
                        }
                        content_type = content_type_map.get(ext, "image/png")

                        # Build image info with image_id (no full image_b64)
                        img_data = {
                            "filename": filename,
                            "image_id": image_id,
                            "content_type": content_type,
                            "size_bytes": len(image_bytes),
                        }

                        # Optional: Include small preview_b64 if image is tiny (< 150KB)
                        MAX_INLINE_SIZE = 150 * 1024  # 150KB
                        if len(image_bytes) < MAX_INLINE_SIZE:
                            img_data["preview_b64"] = base64.b64encode(image_bytes).decode("utf-8")

                        images.append(img_data)
                    total_images.extend(images)

                    slog.info(
                        "job_images_fetched",
                        job_id=job_id,
                        prompt_id=prompt_id,
                        num_images=len(images),
                    )

                except Exception as exc:
                    formatted = _format_comfy_error(f"Failed to fetch images: {exc}")
                    slog.error(
                        "job_failed",
                        job_id=job_id,
                        error=formatted["short"],
                        prompt_id=prompt_id,
                        remediation=formatted["details"]["remediation"],
                        category=formatted["details"]["category"],
                        raw_error=str(exc),
                    )
                    await _emit_image_error(formatted["short"], formatted)
                    return
                break

    end_time = time.perf_counter()
    completed_at_ms = time.time() * 1000.0
    if started_at is None:
        started_at = submit_time
    if started_at_ms is None:
        started_at_ms = completed_at_ms
    queue_latency_ms = (started_at - submit_time) * 1000.0
    gen_time_ms = (end_time - started_at) * 1000.0
    total_ms = (end_time - submit_time) * 1000.0
    total_time_s = total_ms / 1000.0

    # CRITICAL: Detect suspicious "zero steps" completion
    if max_step_observed == 0:
        slog.warning(
            "job_suspicious_zero_steps",
            job_id=job_id,
            max_step_observed=max_step_observed,
            expected_steps=req.steps,
            started_at_is_none=started_at is None,
            total_images=len(total_images),
            warning="Image job completed with 0 steps observed - possible progress tracking issue",
        )
        # Add warning to the payload
        payload_warning = "WARNING: Job completed with 0 sampler steps observed. This may indicate a progress tracking issue."
    else:
        payload_warning = None

    # Log successful completion
    slog.info(
        "job_completed",
        job_id=job_id,
        job_type="image",
        max_step_observed=max_step_observed,
        expected_steps=req.steps,
        queue_latency_ms=round(queue_latency_ms, 2),
        gen_time_ms=round(gen_time_ms, 2),
        total_ms=round(total_ms, 2),
        num_images=len(total_images),
        warning=payload_warning,
        completed_with_fallback=fallback_used,
    )

    payload = {
        "run_id": req.run_id,
        "queue_latency_ms": queue_latency_ms,
        "gen_time_ms": gen_time_ms,
        "total_ms": total_ms,
        "total_time_s": total_time_s,
        "completed_at_ms": completed_at_ms,
        "first_progress_at_ms": started_at_ms,
        "started_at_ms": started_at_ms,
        "steps": req.steps,
        "resolution": f"{req.width}x{req.height}",
        "seed": req.seed,
        "checkpoint": req.checkpoint,
        "sampler": req.sampler,
        "num_images": req.num_images,
        "images": total_images,
        "image_filenames": [img.get("filename") for img in total_images if isinstance(img, dict)],
    }
    if fallback_used:
        payload["completed_with_fallback"] = True
    if payload_warning:
        payload["warning"] = payload_warning

    # Update JOB_STATUS on completion
    _update_job_status({
        "status": "complete",
        "queue_latency_ms": queue_latency_ms,
        "gen_time_ms": gen_time_ms,
        "total_ms": total_ms,
        "total_time_s": total_time_s,
        "images": [img.get("filename", f"image_{i}.png") for i, img in enumerate(total_images)],
        "image_filenames": [img.get("filename", f"image_{i}.png") for i, img in enumerate(total_images)],
        "progress": {"current_step": req.steps, "total_steps": req.steps, "percent": 100},
        "completed_at_ms": completed_at_ms,
        "sampler": req.sampler,
    })

    await _broadcast_event(
        Event(
            job_id=job_id,
            type="image_complete",
            payload=payload,
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
    cache_dir = _comfy_cache_dir()
    comfy_dir = _comfy_checkpoints_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    comfy_dir.mkdir(parents=True, exist_ok=True)
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
                # Step 1: Download to cache
                result = await _download_checkpoint_file(item, cache_dir, client, progress_cb=_progress)

                # Step 2: Ensure visible in ComfyUI (symlink/copy)
                if result["status"] in ("downloaded", "skipped"):
                    visibility = _ensure_comfyui_checkpoint_visible(item.name, cache_dir, comfy_dir)
                    if visibility["status"] == "error":
                        result["status"] = "error"
                        result["error"] = visibility.get("error", "Failed to make checkpoint visible")

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

        # Step 3: Verify ComfyUI sees the checkpoints (optional verification)
        verification_results = []
        for item in items:
            verification = await _verify_checkpoint_in_object_info(item.name)
            verification_results.append({
                "name": item.name,
                "visible": verification.get("visible", False),
                "diagnostics": verification if not verification.get("visible") else None,
            })
            if not verification.get("visible"):
                log.warning(f"Post-sync verification: {item.name} not visible in ComfyUI object_info")

        CHECKPOINT_SYNC_STATUS["results"] = results
        CHECKPOINT_SYNC_STATUS["verification"] = verification_results
        CHECKPOINT_SYNC_STATUS["active"] = False
        await _broadcast_checkpoint_sync_event(
            sync_id,
            "image_checkpoint_sync_done",
            {"results": results, "verification": verification_results},
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
@app.post("/api/compute")
async def run_compute(req: ComputeRequest):
    job_id = req.job_id or str(uuid.uuid4())
    try:
        result = await _run_compute(job_id, req)
    except Exception as exc:
        error_payload = {
            "ok": False,
            "error": str(exc),
            "algorithm": req.algorithm,
            "n": req.n,
            "threads_requested": req.threads,
            "threads_used": 1,
            "repeat_index": req.repeat_index,
        }
        await _broadcast_event(Event(job_id=job_id, type="compute_done", payload=error_payload))
        return JSONResponse(error_payload, status_code=500)
    await _broadcast_event(Event(job_id=job_id, type="compute_done", payload=result))
    return result


@app.post("/jobs", response_model=JobStartResponse)
async def start_job(req: LLMRequest):
    """
    Start a job and run it in the background. Returns job_id immediately.
    Streaming events are sent to connected WebSocket clients.
    """
    job_id = str(uuid.uuid4())

    # Log job received
    slog.info(
        "job_received",
        job_id=job_id,
        job_type="llm",
        model=req.model,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        num_ctx=req.num_ctx,
        prompt_length=len(req.prompt),
    )

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
            slog.error("job_failed", job_id=job_id, error="Task exception", stack_trace=str(t.exception()))
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

    # Log WebSocket connection
    ws_logger.log_connect(ws, client_id)
    log.info("WS client connected: %s (total=%d)", client_id, len(WS_CLIENTS))

    try:
        while True:
            # keep the connection alive by reading; the central may send pings/commands later
            data = await ws.receive_text()
            # we ignore received messages for now; could implement control messages
            log.debug("WS recv from %s: %s", client_id, data)
    except WebSocketDisconnect:
        ws_logger.log_disconnect(client_id, "normal")
        log.info("WS client disconnected: %s", client_id)
        WS_CLIENTS.pop(client_id, None)
    except Exception as exc:
        ws_logger.log_error(client_id, str(exc))
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

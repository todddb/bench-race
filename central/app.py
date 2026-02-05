# central/app.py
from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import io
import json
import logging
import random
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import uuid

import os
import requests
import yaml
import websockets
from flask import Flask, Response, jsonify, render_template, request, send_file
from flask_socketio import SocketIO, emit


# -----------------------------------------------------------------------------
# Paths (robust regardless of cwd)
# -----------------------------------------------------------------------------
CENTRAL_DIR = Path(__file__).resolve().parent          # .../bench-race/central
ROOT_DIR = CENTRAL_DIR.parent                          # .../bench-race
CONFIG_PATH = CENTRAL_DIR / "config" / "machines.yaml"
MODEL_POLICY_PATH = CENTRAL_DIR / "config" / "model_policy.yaml"
COMFY_SETTINGS_PATH = CENTRAL_DIR / "config" / "comfyui.yaml"
MACHINE_STATE_PATH = CENTRAL_DIR / "config" / "machine_state.json"
DEFAULT_CHECKPOINT_URL = (
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/"
    "sd_xl_base_1.0.safetensors"
)
CHECKPOINT_VALIDATION_TTL_S = 600
CHECKPOINT_VALIDATION_CACHE: Dict[str, Dict[str, Any]] = {}

import sys
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from central.services import controller as service_controller
from central.fit_util import compute_model_fit_score, get_model_size_bytes

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bench-central")

# -----------------------------------------------------------------------------
# Flask + SocketIO
# Use threading mode to avoid eventlet/asyncio incompatibilities.
# -----------------------------------------------------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = "dev"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# -----------------------------------------------------------------------------
# Run persistence
# -----------------------------------------------------------------------------
RUNS_DIR = CENTRAL_DIR / "runs"
RUN_HISTORY_LIMIT = 200
RUNS_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOCK = threading.Lock()
RUN_CACHE: Dict[str, Dict[str, Any]] = {}
JOB_RUN_MAP: Dict[str, Dict[str, str]] = {}
ACTIVE_RUNS: Dict[str, Dict[str, Any]] = {}  # run_id -> {start_time, type, machine_ids}
RUN_ID_PATTERN = re.compile(r"^\d{8}-\d{6}_[a-f0-9]{6}$")
SAMPLE_PROMPT_RATE_LIMIT_S = 10
SAMPLE_PROMPT_TIMEOUT_S = 20
_sample_prompt_last_request: Dict[str, float] = {}
_sample_job_lock = threading.Lock()
_sample_job_events: Dict[str, threading.Event] = {}
_sample_job_buffers: Dict[str, List[str]] = {}
MODEL_POLICY_RATE_LIMIT_S = 5
_model_policy_last_request: Dict[str, float] = {}
MODEL_POLICY_LOCK = threading.Lock()
RUNTIME_METRICS: Dict[str, Dict[str, Any]] = {}
MACHINE_STATE_LOCK = threading.Lock()

# Agent health tracking for hysteresis-based offline detection
AGENT_HEALTH_STATE: Dict[str, Dict[str, Any]] = {}
# Structure: {
#   machine_id: {
#       "last_success_ts": float,
#       "consecutive_failures": int,
#       "last_error": str,
#       "current_status": str,  # "ready", "degraded", "offline"
#   }
# }
OFFLINE_THRESHOLD_S = 12  # Mark offline after 12s without success
CONSECUTIVE_FAILURE_THRESHOLD = 3  # Mark offline after 3 consecutive failures


def _current_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT_DIR), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        sha = result.stdout.strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


def _new_run_id(ts: datetime) -> str:
    return f"{ts.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _run_path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.json"


def _valid_run_id(run_id: str) -> bool:
    return bool(RUN_ID_PATTERN.match(run_id))


def _write_run_record(record: Dict[str, Any]) -> None:
    path = _run_path(record["run_id"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def _load_run_record(run_id: str) -> Optional[Dict[str, Any]]:
    path = _run_path(run_id)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _prune_old_runs() -> None:
    files = sorted(RUNS_DIR.glob("*.json"))
    if len(files) <= RUN_HISTORY_LIMIT:
        return
    for path in files[: len(files) - RUN_HISTORY_LIMIT]:
        try:
            path.unlink()
        except OSError:
            log.warning("Failed to delete old run file %s", path)


def _prompt_preview(prompt: str, limit: int = 120) -> str:
    preview = " ".join((prompt or "").strip().split())
    if len(preview) > limit:
        return preview[: limit - 3] + "..."
    return preview


def _run_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    machines = record.get("machines") or []
    has_mock = any(m.get("engine") == "mock" for m in machines)
    run_type = record.get("type") or "inference"
    return {
        "run_id": record.get("run_id"),
        "timestamp": record.get("timestamp"),
        "model": record.get("model"),
        "type": run_type,
        "prompt_preview": _prompt_preview(record.get("prompt_text", "")),
        "has_mock": has_mock,
    }


def _mark_run_active(run_id: str, run_type: str, machine_ids: List[str]) -> None:
    """Mark a run as active and emit run_start event to all connected clients."""
    with RUN_LOCK:
        ACTIVE_RUNS[run_id] = {
            "start_time": time.time(),
            "type": run_type,
            "machine_ids": machine_ids,
        }
    socketio.emit("run_lifecycle", {
        "type": "run_start",
        "run_id": run_id,
        "run_type": run_type,
        "machine_ids": machine_ids,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    log.info("Run started: %s (type=%s, machines=%d)", run_id, run_type, len(machine_ids))


def _mark_run_complete(run_id: str) -> None:
    """Mark a run as complete and emit run_end event if it was active."""
    machine_ids: List[str] = []
    with RUN_LOCK:
        if run_id not in ACTIVE_RUNS:
            return
        machine_ids = list(ACTIVE_RUNS[run_id].get("machine_ids") or [])
        del ACTIVE_RUNS[run_id]
    socketio.emit("run_lifecycle", {
        "type": "run_end",
        "run_id": run_id,
        "machine_ids": machine_ids,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    log.info("Run completed: %s", run_id)


def _check_run_complete(run_id: str) -> None:
    """Check if all jobs for a run are complete and mark it accordingly."""
    machine_ids: List[str] = []
    with RUN_LOCK:
        if run_id not in ACTIVE_RUNS:
            return
        machine_ids = list(ACTIVE_RUNS[run_id].get("machine_ids") or [])
        # Check if there are any remaining jobs for this run
        has_pending_jobs = any(
            info["run_id"] == run_id for info in JOB_RUN_MAP.values()
        )
        if not has_pending_jobs:
            del ACTIVE_RUNS[run_id]
    if not has_pending_jobs:
        socketio.emit("run_lifecycle", {
            "type": "run_end",
            "run_id": run_id,
            "machine_ids": machine_ids,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        log.info("Run completed: %s", run_id)


def _get_active_runs() -> Dict[str, Any]:
    """Get info about currently active runs."""
    with RUN_LOCK:
        run_ids = list(ACTIVE_RUNS.keys())
        active_runs = {
            run_id: {
                "machine_ids": list(info.get("machine_ids") or []),
                "type": info.get("type"),
            }
            for run_id, info in ACTIVE_RUNS.items()
        }
        active_machine_ids = sorted({
            machine_id
            for info in ACTIVE_RUNS.values()
            for machine_id in (info.get("machine_ids") or [])
        })
        return {
            "is_active": len(run_ids) > 0,
            "run_ids": run_ids,
            "count": len(run_ids),
            "active_machine_ids": active_machine_ids,
            "runs": active_runs,
        }


def _default_comfy_settings() -> Dict[str, Any]:
    return {
        "base_url": "",
        "models_path": "",  # Deprecated, use comfyui_models_path
        "central_cache_path": str(CENTRAL_DIR / "model_cache" / "comfyui"),
        "agent_cache_path": str(ROOT_DIR / "agent" / "model_cache" / "comfyui"),
        "comfyui_models_path": str(ROOT_DIR / "agent" / "third_party" / "comfyui" / "models" / "checkpoints"),
        "checkpoint_urls": [DEFAULT_CHECKPOINT_URL],
        "comfyui_checkpoints": [DEFAULT_CHECKPOINT_URL],
    }


def _load_comfy_settings() -> Dict[str, Any]:
    if not COMFY_SETTINGS_PATH.exists():
        # Auto-copy example config if missing
        example_path = COMFY_SETTINGS_PATH.parent / "comfyui.example.yaml"
        if example_path.exists():
            COMFY_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(example_path, COMFY_SETTINGS_PATH)
            log.warning(
                f"Created {COMFY_SETTINGS_PATH.name} from example. "
                "Please review and customize checkpoint URLs if needed."
            )
        else:
            return _default_comfy_settings()
    with open(COMFY_SETTINGS_PATH, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    comfy = payload.get("comfyui") if isinstance(payload, dict) else {}
    settings = _default_comfy_settings()
    if isinstance(comfy, dict):
        settings.update({k: v for k, v in comfy.items() if v is not None})
    checkpoint_urls = settings.get("comfyui_checkpoints") or settings.get("checkpoint_urls") or []
    if not checkpoint_urls:
        checkpoint_urls = [DEFAULT_CHECKPOINT_URL]
    settings["comfyui_checkpoints"] = checkpoint_urls
    settings["checkpoint_urls"] = checkpoint_urls
    return settings


def _save_comfy_settings(settings: Dict[str, Any]) -> None:
    COMFY_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"comfyui": settings}
    with open(COMFY_SETTINGS_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _comfy_cache_dir(settings: Optional[Dict[str, Any]] = None) -> Path:
    conf = settings or _load_comfy_settings()
    path = Path(conf.get("central_cache_path") or (CENTRAL_DIR / "model_cache" / "comfyui"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checkpoint_filename_from_url(url: str) -> str:
    """Extract filename from URL (without query params)."""
    name = url.split("?")[0].rstrip("/").split("/")[-1]
    return name or url


def _generate_label_from_filename(filename: str) -> str:
    """Generate human-friendly label from checkpoint filename.

    Examples:
        sd_xl_base_1.0.safetensors -> sdxl-base-1.0
        flux-dev.safetensors -> flux-dev
        my_model_v2.safetensors -> my-model-v2
    """
    # Remove .safetensors extension
    name = filename.replace(".safetensors", "")
    # Replace underscores with hyphens and normalize
    name = name.replace("_", "-")
    # Collapse multiple hyphens
    name = re.sub(r"-+", "-", name)
    # Remove leading/trailing hyphens
    name = name.strip("-")
    return name or filename


def _checkpoint_entry_from_line(line: str) -> Optional[Dict[str, str]]:
    """Parse checkpoint config line into entry dict.

    Supports formats:
    - <url>                          # Plain URL (backwards compatible)
    - <url> | <sha256>              # URL with hash (backwards compatible)
    - <url> | <label>               # URL with custom label
    - <label> | <url>               # Label first, then URL
    - <url> | <label> | <sha256>    # URL, label, and hash

    Returns dict with keys: url, label (optional), sha256 (optional), filename
    """
    if not line:
        return None
    parts = [part.strip() for part in line.split("|") if part.strip()]
    if not parts:
        return None

    # Determine which part is the URL (must start with http:// or https://)
    url_idx = None
    for idx, part in enumerate(parts):
        if part.startswith(("http://", "https://")):
            url_idx = idx
            break

    if url_idx is None:
        # No valid URL found, treat first part as URL for backwards compatibility
        url_idx = 0

    entry = {"url": parts[url_idx]}
    filename = _checkpoint_filename_from_url(entry["url"])
    entry["filename"] = filename

    # Parse remaining parts
    for idx, part in enumerate(parts):
        if idx == url_idx:
            continue
        # Check if it's a SHA256 hash (64 hex chars)
        if len(part) == 64 and all(c in "0123456789abcdefABCDEF" for c in part):
            entry["sha256"] = part
        else:
            # Treat as custom label
            entry["label"] = part

    # Generate default label if not provided
    if "label" not in entry:
        entry["label"] = _generate_label_from_filename(filename)

    return entry


def _checkpoint_entries(settings: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    conf = settings or _load_comfy_settings()
    urls = conf.get("comfyui_checkpoints") or conf.get("checkpoint_urls") or []
    entries = []
    for line in urls:
        entry = _checkpoint_entry_from_line(line)
        if entry:
            entries.append(entry)
    return entries


def _extract_content_length(headers: Dict[str, str]) -> Optional[int]:
    if not headers:
        return None
    content_range = headers.get("Content-Range") or headers.get("content-range")
    if content_range:
        match = re.search(r"/(\d+)", content_range)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    content_length = headers.get("Content-Length") or headers.get("content-length")
    if content_length:
        try:
            return int(content_length)
        except ValueError:
            return None
    return None


def _format_http_error(status: int, reason: Optional[str]) -> str:
    if reason:
        return f"HTTP {status}: {reason}"
    return f"HTTP {status}"


def _validate_checkpoint_url(url: str, filename: str, force: bool = False) -> Dict[str, Any]:
    """Validate checkpoint URL and metadata.

    Args:
        url: The checkpoint download URL
        filename: The intended checkpoint filename (must end with .safetensors)
        force: Force revalidation even if cached

    Returns:
        Dict with validation result including valid, error, size_bytes, etag, etc.
    """
    now = time.time()
    cached = CHECKPOINT_VALIDATION_CACHE.get(url)
    if cached and not force and now - cached["checked_at"] < CHECKPOINT_VALIDATION_TTL_S:
        return cached["result"]

    result = {
        "url": url,
        "resolved_url": None,
        "valid": False,
        "error": None,
        "status": None,
        "size_bytes": None,
        "etag": None,
        "last_modified": None,
    }

    if not url or not url.startswith(("http://", "https://")):
        result["error"] = "URL must start with http:// or https://"
    elif not filename.lower().endswith(".safetensors"):
        result["error"] = "Checkpoint filename must end with .safetensors"
    else:
        response = None
        try:
            response = requests.head(url, allow_redirects=True, timeout=10)
            if response.status_code in (405, 501):
                response = None
        except requests.RequestException:
            response = None

        if response is None:
            try:
                response = requests.get(
                    url,
                    allow_redirects=True,
                    headers={"Range": "bytes=0-0"},
                    stream=True,
                    timeout=(10, 60),
                )
            except requests.RequestException as exc:
                result["error"] = f"Request failed: {exc}"
                response = None

        if response is not None:
            result["status"] = response.status_code
            result["resolved_url"] = response.url
            if response.status_code >= 400:
                result["error"] = _format_http_error(response.status_code, response.reason)
            else:
                # Check Content-Disposition header for filename validation
                content_disp = response.headers.get("Content-Disposition") or response.headers.get(
                    "content-disposition"
                )
                filename_from_header = None
                if content_disp:
                    match_star = re.search(r"filename\*\s*=\s*[^']*''([^;]+)", content_disp, flags=re.IGNORECASE)
                    match_plain = re.search(
                        r'filename\s*=\s*"?(?P<name>[^";]+)"?', content_disp, flags=re.IGNORECASE
                    )
                    if match_star:
                        filename_from_header = match_star.group(1)
                    elif match_plain:
                        filename_from_header = match_plain.group("name")

                # Validate Content-Disposition filename if present
                # (allows HF/Xet signed URLs whose path doesn't end with .safetensors)
                if filename_from_header and not filename_from_header.lower().endswith(".safetensors"):
                    result["error"] = f"Content-Disposition filename must end with .safetensors (got: {filename_from_header})"
                else:
                    result["valid"] = True
                    result["etag"] = response.headers.get("ETag") or response.headers.get("etag")
                    result["last_modified"] = response.headers.get("Last-Modified") or response.headers.get(
                        "last-modified"
                    )
                    result["size_bytes"] = _extract_content_length(response.headers)
            response.close()

    CHECKPOINT_VALIDATION_CACHE[url] = {"result": result, "checked_at": now}
    return result


def _checkpoint_catalog(settings: Optional[Dict[str, Any]] = None, force: bool = False) -> List[Dict[str, Any]]:
    """Build checkpoint catalog with id, label, and filename.

    Returns list of checkpoint objects with:
    - id: SHA256 hash of URL (stable internal identifier)
    - label: Human-friendly name shown in UI
    - filename: Exact filename expected by ComfyUI
    - url, resolved_url, size_bytes, etag, last_modified, valid, error, status
    """
    entries = _checkpoint_entries(settings)
    items = []
    for entry in entries:
        url = entry["url"]
        filename = entry["filename"]
        label = entry["label"]

        # Generate stable ID from URL (SHA256 hash)
        checkpoint_id = hashlib.sha256(url.encode("utf-8")).hexdigest()

        validation = _validate_checkpoint_url(url, filename, force=force)
        items.append(
            {
                "id": checkpoint_id,
                "label": label,
                "filename": filename,
                "url": url,
                "resolved_url": validation.get("resolved_url"),
                "size_bytes": validation.get("size_bytes"),
                "etag": validation.get("etag"),
                "last_modified": validation.get("last_modified"),
                "valid": validation.get("valid", False),
                "error": validation.get("error"),
                "status": validation.get("status"),
                # Keep "name" for backwards compatibility (deprecated)
                "name": filename,
            }
        )
    return items


def _find_checkpoint_by_id(checkpoint_id: str, settings: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Find checkpoint in catalog by ID.

    Args:
        checkpoint_id: The checkpoint ID (SHA256 hash of URL) or filename (for backwards compat)
        settings: Optional settings dict

    Returns:
        Checkpoint object with id, label, filename, etc., or None if not found
    """
    catalog = _checkpoint_catalog(settings)
    # Try to find by ID first
    checkpoint = next((item for item in catalog if item["id"] == checkpoint_id), None)
    # Fall back to filename match for backwards compatibility
    if not checkpoint:
        checkpoint = next((item for item in catalog if item["filename"] == checkpoint_id), None)
    # Fall back to name match for backwards compatibility
    if not checkpoint:
        checkpoint = next((item for item in catalog if item.get("name") == checkpoint_id), None)
    return checkpoint


def _list_cached_checkpoints(settings: Optional[Dict[str, Any]] = None) -> List[str]:
    cache_dir = _comfy_cache_dir(settings)
    if not cache_dir.exists():
        return []
    return sorted([p.name for p in cache_dir.iterdir() if p.is_file()])


def _ensure_checkpoint_cached(checkpoint_name: str, settings: Optional[Dict[str, Any]] = None) -> Optional[str]:
    if not checkpoint_name:
        return "Missing checkpoint name."
    conf = settings or _load_comfy_settings()
    cache_dir = _comfy_cache_dir(conf)
    target_path = cache_dir / checkpoint_name
    if target_path.exists():
        return None

    entries = _checkpoint_entries(conf)
    url_entry = next(
        (entry for entry in entries if _checkpoint_filename_from_url(entry["url"]) == checkpoint_name),
        None,
    )
    if not url_entry:
        return f"Checkpoint URL not configured for {checkpoint_name}."

    url = url_entry["url"]
    sha256 = url_entry.get("sha256")
    try:
        with requests.get(url, stream=True, timeout=20) as resp:
            resp.raise_for_status()
            hasher = hashlib.sha256() if sha256 else None
            with open(target_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    if hasher:
                        hasher.update(chunk)
        if hasher and sha256:
            digest = hasher.hexdigest()
            if digest.lower() != sha256.lower():
                target_path.unlink(missing_ok=True)
                return f"Checksum mismatch for {checkpoint_name}."
    except Exception as exc:
        if target_path.exists():
            target_path.unlink(missing_ok=True)
        return f"Failed to download checkpoint: {exc}"
    return None


def _run_images_dir(run_id: str) -> Path:
    path = RUNS_DIR / run_id / "images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_image_payload(run_id: str, machine_id: str, filename: str, b64_data: str) -> Optional[str]:
    if not b64_data:
        return None
    try:
        binary = base64.b64decode(b64_data)
    except Exception:
        return None
    base_dir = _run_images_dir(run_id) / machine_id
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / filename
    try:
        with open(path, "wb") as f:
            f.write(binary)
    except OSError:
        return None
    images_root = _run_images_dir(run_id)
    return str(path.relative_to(images_root))


def _machine_model_fit(machine: Dict[str, Any], model: str, num_ctx: int) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{machine['agent_base_url'].rstrip('/')}/capabilities", timeout=2)
        r.raise_for_status()
        cap = r.json()
        memory_label = "RAM"
        accel_type = cap.get("accelerator_type")
        if accel_type == "cuda":
            memory_label = "VRAM"
        elif accel_type == "metal":
            memory_label = "Unified"
        fit = _compute_fit(machine, cap, model, num_ctx)
        return {**fit, "memory_label": memory_label}
    except Exception:
        return None


def _update_run_from_event(event: Dict[str, Any]) -> None:
    event_type = event.get("type")
    job_id = event.get("job_id")
    if not job_id or not event_type:
        return

    if event_type == "job_done":
        with RUN_LOCK:
            job_info = JOB_RUN_MAP.get(job_id)
            if not job_info:
                return
            run_id = job_info["run_id"]
            record = RUN_CACHE.get(run_id) or _load_run_record(run_id)
            if not record:
                return
            machine_id = job_info["machine_id"]
            payload = event.get("payload") or {}
            machine_entry = next(
                (m for m in (record.get("machines") or []) if m.get("machine_id") == machine_id),
                None,
            )
            if not machine_entry:
                machine_entry = {"machine_id": machine_id, "label": job_info.get("label")}
                record.setdefault("machines", []).append(machine_entry)
            machine_entry.update(
                {
                    "status": "complete",
                    "ttft_ms": payload.get("ttft_ms"),
                    "tok_s": payload.get("gen_tokens_per_s"),
                    "total_ms": payload.get("total_ms"),
                    "tokens": payload.get("gen_tokens"),
                    "engine": payload.get("engine"),
                    "model": payload.get("model"),
                    "fallback_reason": payload.get("fallback_reason"),
                }
            )
            record["updated_at"] = datetime.now(timezone.utc).isoformat()
            RUN_CACHE[run_id] = record
            _write_run_record(record)
            JOB_RUN_MAP.pop(job_id, None)
        # Check if this run is now complete (all jobs done)
        _check_run_complete(run_id)
        return

    if event_type == "compute_done":
        with RUN_LOCK:
            job_info = JOB_RUN_MAP.get(job_id)
            if not job_info:
                return
            run_id = job_info["run_id"]
            record = RUN_CACHE.get(run_id) or _load_run_record(run_id)
            if not record:
                return
            machine_id = job_info["machine_id"]
            payload = event.get("payload") or {}
            machine_entry = next(
                (m for m in (record.get("machines") or []) if m.get("machine_id") == machine_id),
                None,
            )
            if not machine_entry:
                machine_entry = {"machine_id": machine_id, "label": job_info.get("label"), "results": []}
                record.setdefault("machines", []).append(machine_entry)
            results = machine_entry.setdefault("results", [])
            repeat_index = payload.get("repeat_index") or job_info.get("repeat_index")
            result_entry = {
                "repeat_index": repeat_index,
                "algorithm": payload.get("algorithm"),
                "n": payload.get("n"),
                "threads_requested": payload.get("threads_requested"),
                "threads_used": payload.get("threads_used"),
                "primes_found": payload.get("primes_found"),
                "elapsed_ms": payload.get("elapsed_ms"),
                "primes_per_sec": payload.get("primes_per_sec"),
                "ok": payload.get("ok", True),
                "error": payload.get("error"),
            }
            existing = next((r for r in results if r.get("repeat_index") == repeat_index), None)
            if existing:
                existing.update(result_entry)
            else:
                results.append(result_entry)
            repeat_total = int(record.get("settings", {}).get("repeats", 1))
            status = "running"
            if payload.get("ok", True) is False:
                status = "error"
            elif len(results) >= repeat_total:
                status = "complete"
            machine_entry.update(
                {
                    "status": status,
                    "algorithm": payload.get("algorithm"),
                    "n": payload.get("n"),
                    "threads_requested": payload.get("threads_requested"),
                    "threads_used": payload.get("threads_used"),
                    "primes_found": payload.get("primes_found"),
                    "elapsed_ms": payload.get("elapsed_ms"),
                    "primes_per_sec": payload.get("primes_per_sec"),
                    "repeat_index": repeat_index,
                    "error": payload.get("error"),
                }
            )
            record["updated_at"] = datetime.now(timezone.utc).isoformat()
            RUN_CACHE[run_id] = record
            _write_run_record(record)
            JOB_RUN_MAP.pop(job_id, None)
        _check_run_complete(run_id)
        return

    if event_type.startswith("image_"):
        with RUN_LOCK:
            job_info = JOB_RUN_MAP.get(job_id)
            if not job_info:
                return
            run_id = job_info["run_id"]
            record = RUN_CACHE.get(run_id) or _load_run_record(run_id)
            if not record:
                return
            machine_id = job_info["machine_id"]
            payload = event.get("payload") or {}
            machine_entry = next(
                (m for m in (record.get("machines") or []) if m.get("machine_id") == machine_id),
                None,
            )
            if not machine_entry:
                machine_entry = {"machine_id": machine_id, "label": job_info.get("label")}
                record.setdefault("machines", []).append(machine_entry)

            if event_type == "image_started":
                machine_entry.update(
                    {
                        "status": "running",
                        "queue_latency_ms": payload.get("queue_latency_ms"),
                        "started_at": payload.get("started_at"),
                    }
                )
            elif event_type == "image_progress":
                machine_entry.update(
                    {
                        "status": "running",
                        "step": payload.get("step"),
                        "total_steps": payload.get("total_steps"),
                    }
                )
            elif event_type == "image_preview":
                image_b64 = payload.get("image_b64")
                filename = payload.get("filename") or "preview.jpg"
                saved_path = _save_image_payload(run_id, machine_id, filename, image_b64)
                if saved_path:
                    machine_entry["preview_path"] = saved_path
                machine_entry.update(
                    {
                        "status": "running",
                        "step": payload.get("step"),
                        "total_steps": payload.get("total_steps"),
                    }
                )
            elif event_type == "image_complete":
                images_payload = payload.get("images") or []
                stored_images = []
                for idx, image in enumerate(images_payload):
                    filename = image.get("filename") or f"image_{idx + 1}.png"
                    saved_path = _save_image_payload(run_id, machine_id, filename, image.get("image_b64") or "")
                    if saved_path:
                        stored_images.append(saved_path)
                machine_entry.update(
                    {
                        "status": "complete",
                        "queue_latency_ms": payload.get("queue_latency_ms"),
                        "gen_time_ms": payload.get("gen_time_ms"),
                        "total_ms": payload.get("total_ms"),
                        "images": stored_images,
                        "steps": payload.get("steps"),
                        "resolution": payload.get("resolution"),
                        "seed": payload.get("seed"),
                        "checkpoint": payload.get("checkpoint"),
                        "num_images": payload.get("num_images"),
                    }
                )
                JOB_RUN_MAP.pop(job_id, None)
            elif event_type == "image_error":
                machine_entry.update(
                    {
                        "status": "error",
                        "error": payload.get("message") or "Generation failed",
                    }
                )
                JOB_RUN_MAP.pop(job_id, None)
            record["updated_at"] = datetime.now(timezone.utc).isoformat()
            RUN_CACHE[run_id] = record
            _write_run_record(record)

        # Check if this run is now complete (image_complete or image_error removes from job map)
        if event_type in ("image_complete", "image_error"):
            _check_run_complete(run_id)
        return


def _record_compute_error(run_id: str, machine_id: str, job_id: str, error: str, repeat_index: Optional[int]) -> None:
    with RUN_LOCK:
        record = RUN_CACHE.get(run_id) or _load_run_record(run_id)
        if not record:
            return
        machine_entry = next(
            (m for m in (record.get("machines") or []) if m.get("machine_id") == machine_id),
            None,
        )
        if not machine_entry:
            machine_entry = {"machine_id": machine_id, "results": []}
            record.setdefault("machines", []).append(machine_entry)
        results = machine_entry.setdefault("results", [])
        results.append(
            {
                "repeat_index": repeat_index,
                "ok": False,
                "error": error,
            }
        )
        machine_entry.update(
            {
                "status": "error",
                "error": error,
                "repeat_index": repeat_index,
            }
        )
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        RUN_CACHE[run_id] = record
        _write_run_record(record)
        JOB_RUN_MAP.pop(job_id, None)
    _check_run_complete(run_id)


def _sample_prompt_rate_limited(remote_addr: str) -> bool:
    now = time.time()
    last = _sample_prompt_last_request.get(remote_addr)
    if last and (now - last) < SAMPLE_PROMPT_RATE_LIMIT_S:
        return True
    _sample_prompt_last_request[remote_addr] = now
    return False


def _track_sample_job(job_id: str) -> threading.Event:
    event = threading.Event()
    with _sample_job_lock:
        _sample_job_events[job_id] = event
        _sample_job_buffers[job_id] = []
    return event


def _finalize_sample_job(job_id: str) -> str:
    with _sample_job_lock:
        chunks = _sample_job_buffers.pop(job_id, [])
        _sample_job_events.pop(job_id, None)
    return "".join(chunks)


def _record_sample_event(event: Dict[str, Any]) -> None:
    job_id = event.get("job_id")
    if not job_id:
        return
    event_type = event.get("type")
    if event_type == "llm_token":
        payload = event.get("payload") or {}
        text = payload.get("text") or ""
        if not text:
            return
        with _sample_job_lock:
            if job_id in _sample_job_buffers:
                _sample_job_buffers[job_id].append(text)
    elif event_type == "job_done":
        with _sample_job_lock:
            ev = _sample_job_events.get(job_id)
        if ev:
            ev.set()


def _wait_for_sample_prompt(job_id: str, timeout_s: int) -> Optional[str]:
    event = _track_sample_job(job_id)
    if not event.wait(timeout_s):
        _finalize_sample_job(job_id)
        return None
    text = _finalize_sample_job(job_id)
    return text or None


def _strip_prompt_fences(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"```(?:[a-zA-Z0-9_-]+)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _heuristic_sample_prompt(seed: str) -> str:
    topics = [
        "distributed cache invalidation strategy",
        "multi-tenant billing audit pipeline",
        "observability for streaming data pipelines",
        "feature-flag rollout with safety checks",
        "LLM-driven code review automation",
    ]
    constraints = [
        "Include edge cases and failure modes.",
        "Provide pseudo-code and a reference implementation.",
        "Describe metrics to validate correctness and performance.",
        "Explain tradeoffs between at least two approaches.",
        "Outline a staged rollout plan with rollback criteria.",
    ]
    topic = random.choice(topics)
    extra = " ".join(random.sample(constraints, k=3))
    return (
        f"Design a detailed plan for {topic}. "
        f"Use the seed {seed} to name the example project and any identifiers. "
        "Your response should include multiple steps: requirements, architecture, implementation details, "
        "data models, and a test strategy. "
        f"{extra} "
        "Provide code snippets (with comments) in one language of your choice, and explain how to handle "
        "edge cases, retries, and data consistency. Finish with a checklist of acceptance criteria."
    )


def _select_agent_for_sample(selected_model: Optional[str]) -> Optional[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for machine in MACHINES:
        try:
            r = requests.get(f"{machine['agent_base_url'].rstrip('/')}/capabilities", timeout=2)
            r.raise_for_status()
            cap = r.json()
            available = _available_llm_models(cap)
            if not available:
                continue
            candidates.append({"machine": machine, "available": available})
        except Exception:
            continue

    if not candidates:
        return None

    if selected_model:
        for candidate in candidates:
            if selected_model in candidate["available"]:
                return {"machine": candidate["machine"], "model": selected_model}

    candidate = candidates[0]
    return {"machine": candidate["machine"], "model": candidate["available"][0]}

# -----------------------------------------------------------------------------
# Load machines config
# -----------------------------------------------------------------------------
def _load_machine_state() -> Dict[str, Dict[str, Any]]:
    """Load machine state overrides (excluded status, etc.) from JSON file."""
    if not MACHINE_STATE_PATH.exists():
        return {}
    try:
        with open(MACHINE_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Failed to load machine state from {MACHINE_STATE_PATH}: {e}")
        return {}


def _save_machine_state(state: Dict[str, Dict[str, Any]]) -> None:
    """Save machine state overrides to JSON file."""
    with MACHINE_STATE_LOCK:
        try:
            # Ensure config directory exists
            MACHINE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(MACHINE_STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log.error(f"Failed to save machine state to {MACHINE_STATE_PATH}: {e}")


def detect_vendor(machine: Dict[str, Any]) -> str:
    """
    Detect vendor from machine metadata.

    Checks (in order):
    1. Explicit 'vendor' field if already set
    2. Machine label (e.g., "MacBook Pro" -> apple)
    3. GPU name (e.g., "NVIDIA RTX" -> nvidia, "Apple M4" -> apple)
    4. Hardware fields (gpu_name, accelerator_type, etc.)

    Returns:
        'apple', 'nvidia', or 'other'
    """
    # If vendor explicitly set, use it
    if "vendor" in machine:
        return machine["vendor"]

    label = machine.get("label", "").lower()
    gpu_name = machine.get("gpu", {}).get("name", "") if isinstance(machine.get("gpu"), dict) else ""
    gpu_name = gpu_name.lower() if gpu_name else ""

    # Check label for vendor hints
    if any(keyword in label for keyword in ["mac", "macbook", "imac", "apple"]):
        return "apple"
    if any(keyword in label for keyword in ["nvidia", "rtx", "gtx", "tesla"]):
        return "nvidia"

    # Check GPU name
    if any(keyword in gpu_name for keyword in ["apple", "m1", "m2", "m3", "m4", "metal"]):
        return "apple"
    if any(keyword in gpu_name for keyword in ["nvidia", "rtx", "gtx", "tesla", "quadro"]):
        return "nvidia"

    # Default to "other"
    return "other"


def load_machines() -> List[Dict[str, Any]]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing central config: {CONFIG_PATH}\n"
            f"Create it at central/config/machines.yaml\n"
            f"Example:\n"
            f"machines:\n"
            f"  - machine_id: local\n"
            f"    label: Local Dev Agent\n"
            f"    agent_base_url: http://127.0.0.1:9001\n"
        )
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    machines = cfg.get("machines") or []

    # Merge in state overrides (excluded status)
    state = _load_machine_state()
    for machine in machines:
        machine_id = machine.get("machine_id")
        if machine_id and machine_id in state:
            # Apply overrides from state file
            machine_overrides = state[machine_id]
            if "excluded" in machine_overrides:
                machine["excluded"] = machine_overrides["excluded"]
        # Ensure excluded field exists (default: False)
        if "excluded" not in machine:
            machine["excluded"] = False

        # Use logo field if present, otherwise detect vendor
        if "logo" in machine:
            machine["vendor"] = machine["logo"]
        else:
            machine["vendor"] = detect_vendor(machine)

    return machines


MACHINES: List[Dict[str, Any]] = load_machines()


def load_model_policy() -> Dict[str, Any]:
    if not MODEL_POLICY_PATH.exists():
        return {"required": {"llm": []}, "optional": {}, "optional_profiles": {}}
    with open(MODEL_POLICY_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


MODEL_POLICY: Dict[str, Any] = load_model_policy()
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")

MODEL_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)b", re.IGNORECASE)


def _required_models() -> Dict[str, List[str]]:
    required = MODEL_POLICY.get("required") or {}
    return {
        "llm": list(required.get("llm") or []),
        "whisper": list(required.get("whisper") or []),
        "sdxl_profiles": list(required.get("sdxl_profiles") or []),
    }


def _write_model_policy(policy: Dict[str, Any]) -> None:
    MODEL_POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    if MODEL_POLICY_PATH.exists():
        backup_path = MODEL_POLICY_PATH.with_name(f"{MODEL_POLICY_PATH.name}.bak.{timestamp}")
        shutil.copy2(MODEL_POLICY_PATH, backup_path)
    temp_path = MODEL_POLICY_PATH.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(f"# Updated {datetime.now(timezone.utc).isoformat()}\n")
        yaml.safe_dump(policy, f, sort_keys=False)
    temp_path.replace(MODEL_POLICY_PATH)


def _model_policy_rate_limited(remote_addr: str) -> bool:
    now = time.time()
    last = _model_policy_last_request.get(remote_addr, 0)
    if now - last < MODEL_POLICY_RATE_LIMIT_S:
        return True
    _model_policy_last_request[remote_addr] = now
    return False


def _missing_models_for_policy(models: List[str]) -> Dict[str, List[str]]:
    missing: Dict[str, List[str]] = {model: [] for model in models}
    for m in MACHINES:
        label = m.get("label") or m.get("machine_id") or "unknown"
        try:
            r = requests.get(f"{m['agent_base_url'].rstrip('/')}/capabilities", timeout=2)
            r.raise_for_status()
            cap = r.json()
            available = _available_llm_models(cap)
        except Exception:
            available = []
        for model in models:
            if model not in available:
                missing[model].append(label)
    return {model: machines for model, machines in missing.items() if machines}


def _available_llm_models(cap: Dict[str, Any]) -> List[str]:
    if cap.get("ollama_models"):
        return list(cap.get("ollama_models") or [])
    return list(cap.get("llm_models") or [])


def estimate_model_size_gb(model_name: str, num_ctx: int) -> Optional[float]:
    match = MODEL_PARAM_RE.search(model_name or "")
    if not match:
        return None
    params_b = float(match.group(1))
    name = model_name.lower()
    if "q4" in name:
        bytes_per_param = 0.5
    elif "q5" in name:
        bytes_per_param = 0.625
    elif "q6" in name:
        bytes_per_param = 0.75
    elif "q8" in name:
        bytes_per_param = 1.0
    else:
        bytes_per_param = 1.0
    overhead_gb = 2.0 + (float(num_ctx) / 4096.0) * 4.0
    return params_b * bytes_per_param + overhead_gb


def classify_model_fit(available_gb: Optional[float], needed_gb: Optional[float]) -> Dict[str, Any]:
    if available_gb is None or needed_gb is None:
        return {"status": "unknown", "needed_gb": needed_gb, "available_gb": available_gb}
    if needed_gb <= 0.75 * available_gb:
        status = "good"
    elif needed_gb <= 1.05 * available_gb:
        status = "average"
    else:
        status = "bad"
    return {"status": status, "needed_gb": needed_gb, "available_gb": available_gb}


def _resolve_model_bytes(model_name: str, num_ctx: int) -> Optional[int]:
    fallback_gb = estimate_model_size_gb(model_name, num_ctx)
    return get_model_size_bytes(model_name, fallback_gb=fallback_gb)


def _resolve_machine_vram_bytes(machine: Dict[str, Any], cap: Dict[str, Any]) -> Optional[int]:
    gpu = machine.get("gpu") or {}
    vram_bytes = gpu.get("vram_bytes") or cap.get("gpu_vram_bytes")
    if vram_bytes:
        return int(vram_bytes)
    accel_memory_gb = cap.get("accelerator_memory_gb")
    if accel_memory_gb:
        return int(float(accel_memory_gb) * 1024**3)
    system_ram_bytes = cap.get("total_system_ram_bytes")
    if system_ram_bytes:
        return int(system_ram_bytes)
    system_memory_gb = cap.get("system_memory_gb")
    if system_memory_gb:
        return int(float(system_memory_gb) * 1024**3)
    return None


def _compute_fit(machine: Dict[str, Any], cap: Dict[str, Any], model: str, num_ctx: int) -> Dict[str, Any]:
    model_bytes = _resolve_model_bytes(model, num_ctx)
    vram_bytes = _resolve_machine_vram_bytes(machine, cap)
    if not model_bytes or not vram_bytes:
        return {"label": "unknown", "fit_score": None, "fit_ratio": None}
    fit = compute_model_fit_score(model_bytes, vram_bytes)
    return {
        **fit,
        "model_bytes": model_bytes,
        "vram_bytes": vram_bytes,
    }

# -----------------------------------------------------------------------------
# Agent WS connectors
# -----------------------------------------------------------------------------
_stop_event = threading.Event()
_ws_threads: Dict[str, threading.Thread] = {}


def _base_to_ws_uri(agent_base_url: str) -> str:
    base = agent_base_url.rstrip("/")
    if base.startswith("https://"):
        return "wss://" + base[len("https://") :] + "/ws"
    if base.startswith("http://"):
        return "ws://" + base[len("http://") :] + "/ws"
    # fallback: assume already ws
    if base.startswith("ws://") or base.startswith("wss://"):
        return base + "/ws"
    return "ws://" + base + "/ws"


async def _agent_ws_loop(machine_id: str, ws_uri: str):
    backoff = 1.0
    while not _stop_event.is_set():
        try:
            log.info("Connecting to agent ws %s -> %s", machine_id, ws_uri)
            async with websockets.connect(ws_uri) as ws:
                log.info("Connected to agent ws %s", machine_id)
                backoff = 1.0
                async for raw in ws:
                    if _stop_event.is_set():
                        break
                    try:
                        evt = json.loads(raw)
                    except Exception:
                        log.warning("Bad JSON from %s: %r", machine_id, raw[:200] if isinstance(raw, str) else raw)
                        continue

                    # attach machine_id and forward to all browsers
                    evt["machine_id"] = machine_id
                    if evt.get("type") == "runtime_metrics_update":
                        payload = evt.get("payload") or {}
                        RUNTIME_METRICS[machine_id] = payload
                    _update_run_from_event(evt)
                    _record_sample_event(evt)
                    socketio.emit("agent_event", evt)

        except Exception as exc:
            if _stop_event.is_set():
                break
            log.warning("Agent ws %s disconnected: %s; retrying in %.1fs", machine_id, exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, 10.0)


def _ws_thread_main(machine_id: str, ws_uri: str):
    # Each connector gets its own event loop in its own thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_agent_ws_loop(machine_id, ws_uri))
    finally:
        try:
            loop.stop()
        except Exception:
            pass
        loop.close()


def start_ws_connectors():
    for m in MACHINES:
        mid = m.get("machine_id")
        base = m.get("agent_base_url")
        if not mid or not base:
            continue

        if mid in _ws_threads and _ws_threads[mid].is_alive():
            continue

        ws_uri = _base_to_ws_uri(base)
        t = threading.Thread(target=_ws_thread_main, args=(mid, ws_uri), daemon=True)
        _ws_threads[mid] = t
        t.start()
        log.info("Started WS connector thread for %s -> %s", mid, ws_uri)


def stop_ws_connectors():
    _stop_event.set()
    # threads are daemon; they'll exit; give a moment
    time.sleep(0.2)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def index():
    # Your templates/index.html should already render panes based on MACHINES
    model_options = _required_models().get("llm") or []
    return render_template("index.html", machines=MACHINES, model_options=model_options)


@app.get("/inference")
def inference():
    model_options = _required_models().get("llm") or []
    return render_template("index.html", machines=MACHINES, model_options=model_options)


@app.get("/compute")
def compute():
    return render_template("compute.html", machines=MACHINES)


@app.get("/image")
def image():
    settings = _load_comfy_settings()
    checkpoint_options = [item["name"] for item in _checkpoint_catalog(settings) if item.get("valid")]
    return render_template("image.html", machines=MACHINES, checkpoint_options=checkpoint_options)


@app.get("/admin")
def admin():
    return render_template("admin.html")


@app.get("/api/machines")
def api_machines():
    machines: List[Dict[str, Any]] = []
    for m in MACHINES:
        machine = dict(m)
        try:
            r = requests.get(f"{m['agent_base_url'].rstrip('/')}/capabilities", timeout=2)
            r.raise_for_status()
            cap = r.json()
            cap["agent_reachable"] = True
            machine["agent_reachable"] = True
            machine["capabilities"] = cap
        except Exception as e:
            machine["agent_reachable"] = False
            machine["capabilities"] = {
                "agent_reachable": False,
                "error": str(e),
            }
        machines.append(machine)
    return jsonify(machines)


@app.patch("/api/machines/<machine_id>")
def api_update_machine(machine_id: str):
    """Update machine settings (excluded status)."""
    payload = request.get_json(force=True) or {}
    excluded = payload.get("excluded")

    # Validate machine_id exists
    machine = next((m for m in MACHINES if m.get("machine_id") == machine_id), None)
    if not machine:
        return jsonify({"error": f"Unknown machine_id: {machine_id}"}), 404

    # Validate excluded is a boolean
    if excluded is not None and not isinstance(excluded, bool):
        return jsonify({"error": "excluded must be a boolean"}), 400

    # Update in-memory state
    if excluded is not None:
        machine["excluded"] = excluded

        # Persist to disk
        state = _load_machine_state()
        if machine_id not in state:
            state[machine_id] = {}
        state[machine_id]["excluded"] = excluded
        _save_machine_state(state)

        log.info(f"Machine {machine_id} excluded status updated to: {excluded}")

    return jsonify({
        "machine_id": machine_id,
        "excluded": machine.get("excluded", False)
    })


@app.post("/api/agents/<machine_id>/reset")
def api_reset_agent(machine_id: str):
    """
    Proxy endpoint to reset an agent by restarting its Ollama and ComfyUI services.
    Forwards the request to the agent's /api/reset endpoint and returns detailed diagnostics.

    Returns:
        - HTTP 200 with agent's full JSON response (includes detailed diagnostics)
        - HTTP 404 if machine_id not found
        - HTTP 502 if agent unreachable
        - HTTP 504 if reset operation times out
    """
    import time
    t_start = time.time()

    # Find the machine
    machine = next((m for m in MACHINES if m.get("machine_id") == machine_id), None)
    if not machine:
        log.warning(f"Reset requested for unknown machine_id: {machine_id}")
        return jsonify({"error": f"Unknown machine_id: {machine_id}", "ok": False}), 404

    agent_base_url = machine.get("agent_base_url", "").rstrip("/")
    if not agent_base_url:
        log.error(f"No agent_base_url configured for {machine_id}")
        return jsonify({"error": f"No agent_base_url configured for {machine_id}", "ok": False}), 500

    log.info(f"central: reset requested for {machine_id} at {agent_base_url}")

    try:
        # Forward to agent's reset endpoint with longer timeout (OLLAMA_START_TIMEOUT_S + COMFYUI_START_TIMEOUT_S + overhead)
        # Default timeouts: 120s + 60s + 60s overhead = 240s (4 minutes)
        response = requests.post(
            f"{agent_base_url}/api/reset",
            timeout=240
        )

        # Agent always returns HTTP 200 with detailed JSON
        result = response.json()
        duration_ms = int((time.time() - t_start) * 1000)

        log.info(f"central: reset completed for {machine_id} in {duration_ms}ms: ok={result.get('ok')}")

        # Forward agent's response exactly as-is
        return jsonify(result), response.status_code

    except requests.exceptions.ConnectionError as e:
        duration_ms = int((time.time() - t_start) * 1000)
        log.error(f"central: reset failed for {machine_id} - agent unreachable: {e}")
        return jsonify({
            "error": "agent unreachable",
            "details": str(e),
            "ok": False,
            "duration_ms": duration_ms
        }), 502

    except requests.exceptions.Timeout:
        duration_ms = int((time.time() - t_start) * 1000)
        log.error(f"central: reset timed out for {machine_id} after {duration_ms}ms")
        return jsonify({
            "error": "Reset operation timed out",
            "ok": False,
            "duration_ms": duration_ms,
            "notes": ["Central proxy timeout (240s). Agent may still be processing reset."]
        }), 504

    except requests.exceptions.RequestException as e:
        duration_ms = int((time.time() - t_start) * 1000)
        log.error(f"central: reset failed for {machine_id}: {e}")
        return jsonify({
            "error": f"Failed to reset agent: {str(e)}",
            "ok": False,
            "duration_ms": duration_ms
        }), 500

    except Exception as e:
        duration_ms = int((time.time() - t_start) * 1000)
        log.error(f"central: reset unexpected error for {machine_id}: {e}")
        return jsonify({
            "error": f"Unexpected error: {str(e)}",
            "ok": False,
            "duration_ms": duration_ms
        }), 500


def _is_machine_running(machine_id: str) -> bool:
    """Check if a machine is currently participating in any active run."""
    with RUN_LOCK:
        for run_data in ACTIVE_RUNS.values():
            if machine_id in run_data.get("machine_ids", []):
                return True
    return False


def _update_agent_health(machine_id: str, success: bool, error: Optional[str] = None, is_running: bool = False) -> str:
    """
    Update agent health state with hysteresis-based offline detection.

    Args:
        machine_id: The agent's machine ID
        success: Whether the health check succeeded
        error: Error message if health check failed
        is_running: Whether the agent is currently running a job

    Returns:
        Current status: "ready", "degraded", or "offline"
    """
    now = time.time()

    # Initialize health state if not exists
    if machine_id not in AGENT_HEALTH_STATE:
        AGENT_HEALTH_STATE[machine_id] = {
            "last_success_ts": now if success else 0,
            "consecutive_failures": 0 if success else 1,
            "last_error": None,
            "current_status": "ready" if success else "degraded",
        }

    state = AGENT_HEALTH_STATE[machine_id]
    old_status = state["current_status"]

    if success:
        # Health check succeeded - reset failure counters
        state["last_success_ts"] = now
        state["consecutive_failures"] = 0
        state["last_error"] = None
        state["current_status"] = "ready"
        new_status = "ready"

        # Log status recovery if transitioning from degraded/offline
        if old_status in ["degraded", "offline"]:
            log.info(
                f"Agent {machine_id} status recovered: {old_status} → {new_status}"
            )
    else:
        # Health check failed - increment failure counter
        state["consecutive_failures"] += 1
        state["last_error"] = error or "unknown error"

        time_since_success = now - state["last_success_ts"] if state["last_success_ts"] > 0 else float("inf")

        # Determine new status based on hysteresis thresholds
        # For agents running jobs, prefer degraded over offline
        if state["consecutive_failures"] >= CONSECUTIVE_FAILURE_THRESHOLD or time_since_success >= OFFLINE_THRESHOLD_S:
            # Mark offline only if not running OR if it's been too long
            if is_running and state["consecutive_failures"] < CONSECUTIVE_FAILURE_THRESHOLD * 2:
                # Agent is running a job - stay degraded unless severely overdue
                new_status = "degraded"
            else:
                new_status = "offline"
        elif state["consecutive_failures"] > 0:
            # Some failures but not enough to mark offline - show degraded
            new_status = "degraded"
        else:
            new_status = "ready"

        state["current_status"] = new_status

        # Log status transitions with diagnostic info
        if old_status != new_status:
            log.warning(
                f"Agent {machine_id} status changed: {old_status} → {new_status} "
                f"(consecutive_failures={state['consecutive_failures']}, "
                f"last_success_age_s={time_since_success:.1f}s, "
                f"last_error='{state['last_error']}')"
            )

    return state["current_status"]


@app.get("/api/capabilities")
def api_capabilities():
    caps = []
    for m in MACHINES:
        machine_id = m.get("machine_id")
        try:
            r = requests.get(f"{m['agent_base_url'].rstrip('/')}/capabilities", timeout=2)
            r.raise_for_status()
            cap = r.json()

            # Update health state - success
            status = _update_agent_health(machine_id, success=True)

            cap["agent_reachable"] = True
            cap["agent_status"] = status  # Add status field
            caps.append(cap)
        except Exception as e:
            # Update health state - failure
            status = _update_agent_health(machine_id, success=False, error=str(e))

            caps.append(
                {
                    "machine_id": machine_id,
                    "label": m.get("label"),
                    "agent_reachable": status != "offline",  # degraded still counts as "reachable"
                    "agent_status": status,
                    "ollama_reachable": None,
                    "ollama_models": [],
                    "llm_models": [],
                    "error": str(e),
                }
            )
    return jsonify(caps)


@app.get("/api/status")
def api_status():
    selected_model = request.args.get("model")
    num_ctx = int(request.args.get("num_ctx", 4096))
    required = _required_models()
    statuses = []
    for m in MACHINES:
        machine_id = m.get("machine_id")
        last_checked = time.time()
        is_running = _is_machine_running(machine_id)

        try:
            r = requests.get(f"{m['agent_base_url'].rstrip('/')}/capabilities", timeout=2)
            r.raise_for_status()
            cap = r.json()

            # Update health state - success
            status = _update_agent_health(machine_id, success=True, is_running=is_running)

            cap["agent_reachable"] = True
            available_llm = _available_llm_models(cap)
            has_selected_model = bool(selected_model and selected_model in available_llm)
            missing_required = {
                "llm": [model for model in required["llm"] if model not in available_llm],
                "whisper": [model for model in required["whisper"] if model not in (cap.get("whisper_models") or [])],
                "sdxl_profiles": [
                    profile for profile in required["sdxl_profiles"] if profile not in (cap.get("sdxl_profiles") or [])
                ],
            }
            memory_label = "RAM"
            if cap.get("accelerator_type") == "cuda":
                memory_label = "VRAM"
            elif cap.get("accelerator_type") == "metal":
                memory_label = "Unified"
            fit = _compute_fit(m, cap, selected_model or "", num_ctx)

            # Add health state info for UI diagnostics
            health_state = AGENT_HEALTH_STATE.get(machine_id, {})

            statuses.append(
                {
                    "machine_id": cap.get("machine_id") or machine_id,
                    "label": cap.get("label") or m.get("label"),
                    "logo": m.get("logo") or m.get("vendor"),
                    "excluded": m.get("excluded", False),
                    "reachable": True,
                    "agent_reachable": True,
                    "agent_status": status,
                    "selected_model": selected_model,
                    "has_selected_model": has_selected_model,
                    "available_llm_models": available_llm,
                    "missing_required": missing_required,
                    "last_checked": last_checked,
                    "error": None,
                    "health_diagnostics": {
                        "consecutive_failures": health_state.get("consecutive_failures", 0),
                        "last_success_age_s": time.time() - health_state.get("last_success_ts", time.time()),
                        "last_error": health_state.get("last_error"),
                    },
                    "capabilities": {
                        "agent_reachable": True,
                        "comfyui_gpu_ok": cap.get("comfyui_gpu_ok"),
                        "comfyui_cpu_ok": cap.get("comfyui_cpu_ok"),
                        "total_system_ram_bytes": cap.get("total_system_ram_bytes"),
                        "system_memory_gb": cap.get("system_memory_gb"),
                    },
                    "model_fit": {
                        **fit,
                        "memory_label": memory_label,
                    },
                    "runtime_metrics": RUNTIME_METRICS.get(cap.get("machine_id") or machine_id),
                }
            )
        except Exception as e:
            # Update health state - failure (run-aware)
            status = _update_agent_health(machine_id, success=False, error=str(e), is_running=is_running)

            # Add health state info for UI diagnostics
            health_state = AGENT_HEALTH_STATE.get(machine_id, {})

            statuses.append(
                {
                    "machine_id": machine_id,
                    "label": m.get("label"),
                    "logo": m.get("logo") or m.get("vendor"),
                    "excluded": m.get("excluded", False),
                    "reachable": status != "offline",  # degraded still counts as "reachable"
                    "agent_reachable": status != "offline",
                    "agent_status": status,
                    "selected_model": selected_model,
                    "has_selected_model": False,
                    "available_llm_models": [],
                    "missing_required": required,
                    "last_checked": last_checked,
                    "error": str(e),
                    "health_diagnostics": {
                        "consecutive_failures": health_state.get("consecutive_failures", 0),
                        "last_success_age_s": time.time() - health_state.get("last_success_ts", time.time()) if health_state.get("last_success_ts", 0) > 0 else float("inf"),
                        "last_error": health_state.get("last_error"),
                    },
                    "capabilities": {"agent_reachable": status != "offline", "comfyui_gpu_ok": None, "comfyui_cpu_ok": None},
                    "model_fit": {
                        "label": "unknown",
                        "fit_score": None,
                        "fit_ratio": None,
                        "memory_label": "RAM",
                    },
                    "runtime_metrics": RUNTIME_METRICS.get(machine_id),
                }
            )
    return jsonify({"model": selected_model, "required": required, "machines": statuses})


@app.get("/api/runtime_metrics")
def api_runtime_metrics():
    return jsonify(RUNTIME_METRICS)


@app.get("/api/image/status")
def api_image_status():
    checkpoint_id = request.args.get("checkpoint") or ""
    # Look up checkpoint by ID to get filename
    checkpoint_filename = ""
    if checkpoint_id:
        checkpoint_obj = _find_checkpoint_by_id(checkpoint_id)
        if checkpoint_obj:
            checkpoint_filename = checkpoint_obj["filename"]
        else:
            # Fall back to treating it as a filename for backwards compatibility
            checkpoint_filename = checkpoint_id

    statuses = []
    for m in MACHINES:
        last_checked = time.time()
        try:
            r = requests.get(f"{m['agent_base_url'].rstrip('/')}/api/comfy/health", timeout=3)
            r.raise_for_status()
            cap = r.json()
            checkpoints = cap.get("checkpoints") or []
            has_checkpoint = bool(checkpoint_filename and checkpoint_filename in checkpoints)
            statuses.append(
                {
                    "machine_id": cap.get("machine_id") or m.get("machine_id"),
                    "label": cap.get("label") or m.get("label"),
                    "logo": m.get("logo") or m.get("vendor"),
                    "excluded": m.get("excluded", False),
                    "reachable": True,
                    "comfy_running": cap.get("running", False),
                    "checkpoints": checkpoints,
                    "has_checkpoint": has_checkpoint,
                    "missing_checkpoint": checkpoint_id if checkpoint_id and not has_checkpoint else None,
                    "last_checked": last_checked,
                    "error": cap.get("error"),
                }
            )
        except Exception as e:
            statuses.append(
                {
                    "machine_id": m.get("machine_id"),
                    "label": m.get("label"),
                    "logo": m.get("logo") or m.get("vendor"),
                    "excluded": m.get("excluded", False),
                    "reachable": False,
                    "comfy_running": False,
                    "checkpoints": [],
                    "has_checkpoint": False,
                    "missing_checkpoint": checkpoint_id or None,
                    "last_checked": last_checked,
                    "error": str(e),
                }
            )
    return jsonify({"checkpoint": checkpoint_id, "machines": statuses})


@app.get("/api/image/checkpoints")
def api_image_checkpoints():
    settings = _load_comfy_settings()
    force = request.args.get("refresh") == "1"
    items = _checkpoint_catalog(settings, force=force)
    return jsonify({"items": items})


@app.get("/api/runs")
def api_runs():
    limit = int(request.args.get("limit", 20))
    files = sorted(RUNS_DIR.glob("*.json"), reverse=True)
    summaries = []
    for path in files[:limit]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                record = json.load(f)
            summaries.append(_run_summary(record))
        except Exception as exc:
            log.warning("Failed to read run %s: %s", path, exc)
    return jsonify(summaries)


@app.get("/api/runs/active")
def api_runs_active():
    """Return info about currently active runs for frontend polling optimization."""
    return jsonify(_get_active_runs())


@app.get("/api/runs/<run_id>")
def api_run_detail(run_id: str):
    record = _load_run_record(run_id)
    if not record:
        return jsonify({"error": "Run not found"}), 404
    return jsonify(record)


@app.delete("/api/runs/<run_id>")
def api_delete_run(run_id: str):
    if not _valid_run_id(run_id):
        return jsonify({"error": "Invalid run id"}), 400
    path = _run_path(run_id)
    if not path.exists():
        return jsonify({"error": "Run not found"}), 404
    try:
        path.unlink()
    except OSError as exc:
        log.warning("Failed to delete run %s: %s", path, exc)
        return jsonify({"error": "Failed to delete run"}), 500
    images_dir = RUNS_DIR / run_id
    if images_dir.exists():
        try:
            shutil.rmtree(images_dir)
        except OSError as exc:
            log.warning("Failed to delete run images %s: %s", run_id, exc)
    with RUN_LOCK:
        RUN_CACHE.pop(run_id, None)
    return jsonify({"ok": True})


@app.get("/api/runs/<run_id>/export.json")
def api_run_export_json(run_id: str):
    record = _load_run_record(run_id)
    if not record:
        return jsonify({"error": "Run not found"}), 404
    payload = json.dumps(record, indent=2)
    return Response(
        payload,
        mimetype="application/json",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.json"'},
    )


@app.get("/api/runs/<run_id>/export.csv")
def api_run_export_csv(run_id: str):
    record = _load_run_record(run_id)
    if not record:
        return jsonify({"error": "Run not found"}), 404
    output = io.StringIO()
    writer = csv.writer(output)
    if record.get("type") == "compute":
        writer.writerow(
            [
                "run_id",
                "timestamp",
                "algorithm",
                "n",
                "machine_id",
                "label",
                "repeat_index",
                "primes_found",
                "elapsed_ms",
                "primes_per_sec",
                "threads_requested",
                "threads_used",
                "ok",
                "error",
            ]
        )
        for machine in record.get("machines") or []:
            results = machine.get("results") or [machine]
            for result in results:
                writer.writerow(
                    [
                        record.get("run_id"),
                        record.get("timestamp"),
                        result.get("algorithm") or record.get("settings", {}).get("algorithm"),
                        result.get("n") or record.get("settings", {}).get("n"),
                        machine.get("machine_id"),
                        machine.get("label"),
                        result.get("repeat_index"),
                        result.get("primes_found"),
                        result.get("elapsed_ms"),
                        result.get("primes_per_sec"),
                        result.get("threads_requested"),
                        result.get("threads_used"),
                        result.get("ok"),
                        result.get("error"),
                    ]
                )
    else:
        writer.writerow(
            [
                "run_id",
                "timestamp",
                "model",
                "machine_id",
                "label",
                "ttft_ms",
                "tok_s",
                "total_ms",
                "tokens",
                "engine",
                "model_fit_label",
            ]
        )
        for machine in record.get("machines") or []:
            model_fit = machine.get("model_fit") or {}
            writer.writerow(
                [
                    record.get("run_id"),
                    record.get("timestamp"),
                    record.get("model"),
                    machine.get("machine_id"),
                    machine.get("label"),
                    machine.get("ttft_ms"),
                    machine.get("tok_s"),
                    machine.get("total_ms"),
                    machine.get("tokens"),
                    machine.get("engine"),
                    model_fit.get("label") or model_fit.get("status"),
                ]
            )
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.csv"'},
    )


@app.get("/api/runs/<run_id>/images/<path:filename>")
def api_run_image(run_id: str, filename: str):
    if not _valid_run_id(run_id):
        return jsonify({"error": "Invalid run id"}), 400
    file_path = RUNS_DIR / run_id / "images" / filename
    if not file_path.exists():
        return jsonify({"error": "Image not found"}), 404
    return send_file(file_path)


@app.get("/api/comfy/checkpoints/<path:checkpoint_name>")
def api_get_checkpoint(checkpoint_name: str):
    settings = _load_comfy_settings()
    cache_dir = _comfy_cache_dir(settings)
    file_path = cache_dir / checkpoint_name
    if not file_path.exists():
        return jsonify({"error": "Checkpoint not found"}), 404
    return send_file(file_path, as_attachment=True)


@app.post("/api/machines/<machine_id>/sync")
def api_sync_models(machine_id: str):
    required = _required_models()
    machine = next((m for m in MACHINES if m.get("machine_id") == machine_id), None)
    if not machine:
        return jsonify({"error": f"Unknown machine_id: {machine_id}"}), 404

    try:
        r = requests.get(f"{machine['agent_base_url'].rstrip('/')}/capabilities", timeout=2)
        r.raise_for_status()
        cap = r.json()
        available_llm = _available_llm_models(cap)
        missing = {
            "llm": [model for model in required["llm"] if model not in available_llm],
            "whisper": [model for model in required["whisper"] if model not in (cap.get("whisper_models") or [])],
            "sdxl_profiles": [
                profile for profile in required["sdxl_profiles"] if profile not in (cap.get("sdxl_profiles") or [])
            ],
        }
        if not any(missing.values()):
            return jsonify({"sync_id": None, "message": "No missing required models"})
        sync_resp = requests.post(
            f"{machine['agent_base_url'].rstrip('/')}/models/sync",
            json=missing,
            timeout=5,
        )
        sync_resp.raise_for_status()
        return jsonify(sync_resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/machines/<machine_id>/sync_image")
def api_sync_image_models(machine_id: str):
    settings = _load_comfy_settings()
    machine = next((m for m in MACHINES if m.get("machine_id") == machine_id), None)
    if not machine:
        return jsonify({"error": f"Unknown machine_id: {machine_id}"}), 404
    try:
        r = requests.get(f"{machine['agent_base_url'].rstrip('/')}/api/comfy/health", timeout=3)
        r.raise_for_status()
        cap = r.json()
        available = cap.get("checkpoints") or []
        required = _list_cached_checkpoints(settings)
        missing = [ckpt for ckpt in required if ckpt not in available]
        if not missing:
            return jsonify({"message": "No missing checkpoints"})
        sync_resp = requests.post(
            f"{machine['agent_base_url'].rstrip('/')}/api/comfy/sync",
            json={
                "checkpoints": missing,
                "central_base_url": request.host_url.rstrip("/"),
            },
            timeout=5,
        )
        sync_resp.raise_for_status()
        return jsonify(sync_resp.json())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/machines/<machine_id>/sync_image_checkpoints")
def api_sync_image_checkpoints(machine_id: str):
    payload = request.get_json(force=True) or {}
    checkpoint_ids = payload.get("checkpoint_names") or []  # Name kept for backwards compat
    if not isinstance(checkpoint_ids, list):
        return jsonify({"error": "checkpoint_names must be a list"}), 400

    settings = _load_comfy_settings()
    machine = next((m for m in MACHINES if m.get("machine_id") == machine_id), None)
    if not machine:
        return jsonify({"error": f"Unknown machine_id: {machine_id}"}), 404

    catalog = _checkpoint_catalog(settings)
    # Build lookup by ID, filename, and name for backwards compatibility
    valid_items_by_id = {item["id"]: item for item in catalog if item.get("valid")}
    valid_items_by_filename = {item["filename"]: item for item in catalog if item.get("valid")}
    valid_items_by_name = {item["name"]: item for item in catalog if item.get("valid")}

    if checkpoint_ids:
        items = []
        missing = []
        for checkpoint_id in checkpoint_ids:
            # Try lookup by ID first, then filename, then name
            item = (
                valid_items_by_id.get(checkpoint_id)
                or valid_items_by_filename.get(checkpoint_id)
                or valid_items_by_name.get(checkpoint_id)
            )
            if item:
                items.append(item)
            else:
                missing.append(checkpoint_id)
        if missing:
            return jsonify({"error": f"Unknown or invalid checkpoints: {', '.join(missing)}"}), 400
    else:
        items = list(valid_items_by_id.values())

    if not items:
        return jsonify({"error": "No valid checkpoints to sync"}), 400

    try:
        sync_resp = requests.post(
            f"{machine['agent_base_url'].rstrip('/')}/api/comfy/sync_checkpoints",
            json={"items": items},
            timeout=5,
        )
        sync_resp.raise_for_status()
        return jsonify(sync_resp.json())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/image/sync_checkpoints")
def api_sync_image_checkpoints_all():
    payload = request.get_json(force=True) or {}
    checkpoint_ids = payload.get("checkpoint_names") or []  # Name kept for backwards compat
    if not isinstance(checkpoint_ids, list):
        return jsonify({"error": "checkpoint_names must be a list"}), 400

    settings = _load_comfy_settings()
    catalog = _checkpoint_catalog(settings)
    # Build lookup by ID, filename, and name for backwards compatibility
    valid_items_by_id = {item["id"]: item for item in catalog if item.get("valid")}
    valid_items_by_filename = {item["filename"]: item for item in catalog if item.get("valid")}
    valid_items_by_name = {item["name"]: item for item in catalog if item.get("valid")}

    if checkpoint_ids:
        items = []
        missing = []
        for checkpoint_id in checkpoint_ids:
            # Try lookup by ID first, then filename, then name
            item = (
                valid_items_by_id.get(checkpoint_id)
                or valid_items_by_filename.get(checkpoint_id)
                or valid_items_by_name.get(checkpoint_id)
            )
            if item:
                items.append(item)
            else:
                missing.append(checkpoint_id)
        if missing:
            return jsonify({"error": f"Unknown or invalid checkpoints: {', '.join(missing)}"}), 400
    else:
        items = list(valid_items_by_id.values())

    if not items:
        return jsonify({"error": "No valid checkpoints to sync"}), 400

    results = []
    for machine in MACHINES:
        machine_id = machine.get("machine_id")
        if not machine_id:
            continue
        try:
            sync_resp = requests.post(
                f"{machine['agent_base_url'].rstrip('/')}/api/comfy/sync_checkpoints",
                json={"items": items},
                timeout=5,
            )
            sync_resp.raise_for_status()
            results.append({"machine_id": machine_id, "ok": True, "response": sync_resp.json()})
        except Exception as exc:
            results.append({"machine_id": machine_id, "ok": False, "error": str(exc)})
    return jsonify({"items": results})


@app.post("/api/start_llm")
def api_start_llm():
    payload = request.get_json(force=True) or {}

    model = payload.get("model", "llama3.1:8b-instruct-q8_0")
    prompt = payload.get("prompt", "Hello!")
    max_tokens = int(payload.get("max_tokens", 256))
    temperature = float(payload.get("temperature", 0.2))
    num_ctx = int(payload.get("num_ctx", 4096))
    repeat = int(payload.get("repeat", 1))

    # Optional: only run on specific machines (for preflight filtering)
    machine_ids = payload.get("machine_ids")  # None means all machines

    run_timestamp = datetime.now(timezone.utc)
    run_id = _new_run_id(run_timestamp)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    run_record: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": run_timestamp.isoformat(),
        "type": "inference",
        "model": model,
        "prompt_text": prompt,
        "prompt_hash": prompt_hash,
        "settings": {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "num_ctx": num_ctx,
            "repeat": repeat,
            "machine_ids": machine_ids,
        },
        "central_git_sha": _current_git_sha(),
        "machines": [],
    }

    results = []
    for m in MACHINES:
        # Skip excluded machines
        if m.get("excluded", False):
            run_record["machines"].append(
                {
                    "machine_id": m.get("machine_id"),
                    "label": m.get("label"),
                    "status": "skipped",
                    "reason": "excluded",
                }
            )
            results.append({
                "machine_id": m.get("machine_id"),
                "skipped": True,
                "reason": "excluded"
            })
            continue

        # Skip machines not in the list (if list is provided)
        if machine_ids is not None and m.get("machine_id") not in machine_ids:
            run_record["machines"].append(
                {
                    "machine_id": m.get("machine_id"),
                    "label": m.get("label"),
                    "status": "skipped",
                    "reason": "not in ready list",
                }
            )
            results.append({
                "machine_id": m.get("machine_id"),
                "skipped": True,
                "reason": "not in ready list"
            })
            continue

        try:
            model_fit = _machine_model_fit(m, model, num_ctx)
            r = requests.post(
                f"{m['agent_base_url'].rstrip('/')}/jobs",
                json={
                    "test_type": "llm_generate",
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "num_ctx": num_ctx,
                    "repeat": repeat,
                    "stream": True,
                },
                timeout=5,
            )
            r.raise_for_status()
            job = r.json()
            results.append({"machine_id": m["machine_id"], "job": job})
            run_record["machines"].append(
                {
                    "machine_id": m.get("machine_id"),
                    "label": m.get("label"),
                    "status": "pending",
                    "job_id": job.get("job_id"),
                    "model_fit": model_fit,
                }
            )
            if job.get("job_id"):
                with RUN_LOCK:
                    JOB_RUN_MAP[job["job_id"]] = {
                        "run_id": run_id,
                        "machine_id": m.get("machine_id"),
                        "label": m.get("label", ""),
                    }
        except Exception as e:
            run_record["machines"].append(
                {
                    "machine_id": m.get("machine_id"),
                    "label": m.get("label"),
                    "status": "error",
                    "error": str(e),
                }
            )
            results.append({"machine_id": m.get("machine_id"), "error": str(e)})

    with RUN_LOCK:
        RUN_CACHE[run_id] = run_record
        _write_run_record(run_record)
        _prune_old_runs()

    # Mark run as active with the machine IDs that successfully started jobs
    active_machine_ids = [r["machine_id"] for r in results if "job" in r]
    if active_machine_ids:
        _mark_run_active(run_id, "inference", active_machine_ids)

    return jsonify({"run_id": run_id, "results": results})


def _dispatch_compute_job(
    machine: Dict[str, Any],
    run_id: str,
    job_id: str,
    payload: Dict[str, Any],
    repeat_index: int,
) -> None:
    try:
        r = requests.post(
            f"{machine['agent_base_url'].rstrip('/')}/api/compute",
            json=payload,
            timeout=600,
        )
        r.raise_for_status()
    except Exception as exc:
        log.warning("Compute job failed to start for %s: %s", machine.get("machine_id"), exc)
        _record_compute_error(run_id, machine.get("machine_id"), job_id, str(exc), repeat_index)


@app.post("/api/compute/run")
def api_compute_run():
    payload = request.get_json(force=True) or {}

    algorithm = payload.get("algorithm", "segmented_sieve")
    n = int(payload.get("n", 50_000_000))
    threads = int(payload.get("threads", 1))
    repeats = int(payload.get("repeats", 1))
    stream_first_k = int(payload.get("stream_first_k", 0))
    progress_interval_s = float(payload.get("progress_interval_s", 1.0))
    machine_ids = payload.get("machine_ids")

    algorithm_labels = {
        "segmented_sieve": "Segmented Sieve",
        "simple_sieve": "Simple Sieve",
        "trial_division": "Trial Division",
    }
    if algorithm not in algorithm_labels:
        return jsonify({"error": "Invalid algorithm"}), 400
    if n < 10:
        return jsonify({"error": "n must be >= 10"}), 400
    if repeats < 1:
        return jsonify({"error": "repeats must be >= 1"}), 400

    threads = max(1, threads)
    stream_first_k = max(0, min(stream_first_k, 5000))
    progress_interval_s = max(0.1, progress_interval_s)

    run_timestamp = datetime.now(timezone.utc)
    run_id = _new_run_id(run_timestamp)
    algorithm_label = algorithm_labels[algorithm]

    run_record: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": run_timestamp.isoformat(),
        "type": "compute",
        "model": algorithm_label,
        "prompt_text": f"Count primes ≤ {n:,} ({algorithm_label})",
        "settings": {
            "algorithm": algorithm,
            "n": n,
            "threads": threads,
            "repeats": repeats,
            "stream_first_k": stream_first_k,
            "progress_interval_s": progress_interval_s,
            "machine_ids": machine_ids,
        },
        "central_git_sha": _current_git_sha(),
        "machines": [],
    }

    results = []
    active_machine_ids = []
    for m in MACHINES:
        if m.get("excluded", False):
            run_record["machines"].append(
                {
                    "machine_id": m.get("machine_id"),
                    "label": m.get("label"),
                    "status": "skipped",
                    "reason": "excluded",
                }
            )
            results.append({
                "machine_id": m.get("machine_id"),
                "skipped": True,
                "reason": "excluded",
            })
            continue

        if machine_ids is not None and m.get("machine_id") not in machine_ids:
            run_record["machines"].append(
                {
                    "machine_id": m.get("machine_id"),
                    "label": m.get("label"),
                    "status": "skipped",
                    "reason": "not in ready list",
                }
            )
            results.append({
                "machine_id": m.get("machine_id"),
                "skipped": True,
                "reason": "not in ready list",
            })
            continue

        active_machine_ids.append(m.get("machine_id"))
        run_record["machines"].append(
            {
                "machine_id": m.get("machine_id"),
                "label": m.get("label"),
                "status": "pending",
                "results": [],
            }
        )
        results.append({"machine_id": m.get("machine_id"), "ok": True})

        for repeat_index in range(1, repeats + 1):
            job_id = str(uuid.uuid4())
            job_payload = {
                "job_id": job_id,
                "algorithm": algorithm,
                "n": n,
                "threads": threads,
                "repeat_index": repeat_index,
                "stream_first_k": stream_first_k,
                "progress_interval_s": progress_interval_s,
            }
            with RUN_LOCK:
                JOB_RUN_MAP[job_id] = {
                    "run_id": run_id,
                    "machine_id": m.get("machine_id"),
                    "label": m.get("label", ""),
                    "repeat_index": repeat_index,
                }
            thread = threading.Thread(
                target=_dispatch_compute_job,
                args=(m, run_id, job_id, job_payload, repeat_index),
                daemon=True,
            )
            thread.start()

    with RUN_LOCK:
        RUN_CACHE[run_id] = run_record
        _write_run_record(run_record)
        _prune_old_runs()

    if active_machine_ids:
        _mark_run_active(run_id, "compute", active_machine_ids)

    return jsonify({"run_id": run_id, "results": results})


@app.post("/api/start_image")
def api_start_image():
    payload = request.get_json(force=True) or {}

    prompt = payload.get("prompt", "")
    checkpoint_id = payload.get("checkpoint", "")
    seed_mode = payload.get("seed_mode", "fixed")
    seed = payload.get("seed")
    steps = int(payload.get("steps", 30))
    width = int(payload.get("width", 1024))
    height = int(payload.get("height", 1024))
    num_images = int(payload.get("num_images", 1))
    repeat = int(payload.get("repeat", 1))

    if seed_mode == "random" or seed is None:
        seed = random.randint(1, 2**31 - 1)

    # Validate checkpoint ID is provided
    if not checkpoint_id:
        return jsonify({"error": "Missing checkpoint"}), 400

    # Look up checkpoint by ID to get filename and label
    checkpoint_obj = _find_checkpoint_by_id(checkpoint_id)
    if not checkpoint_obj:
        return jsonify({"error": f"Checkpoint not found: {checkpoint_id}"}), 400

    checkpoint_filename = checkpoint_obj["filename"]
    checkpoint_label = checkpoint_obj["label"]

    run_timestamp = datetime.now(timezone.utc)
    run_id = _new_run_id(run_timestamp)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    run_record: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": run_timestamp.isoformat(),
        "type": "image",
        "model": checkpoint_label,  # Display label in UI
        "prompt_text": prompt,
        "prompt_hash": prompt_hash,
        "settings": {
            "checkpoint": checkpoint_label,  # Store label for display
            "checkpoint_id": checkpoint_id,  # Store ID for reference
            "checkpoint_filename": checkpoint_filename,  # Store filename for reference
            "seed_mode": seed_mode,
            "seed": seed,
            "steps": steps,
            "width": width,
            "height": height,
            "num_images": num_images,
            "repeat": repeat,
        },
        "central_git_sha": _current_git_sha(),
        "machines": [],
    }

    results = []
    for m in MACHINES:
        # Skip excluded machines
        if m.get("excluded", False):
            run_record["machines"].append(
                {
                    "machine_id": m.get("machine_id"),
                    "label": m.get("label"),
                    "status": "skipped",
                    "reason": "excluded",
                }
            )
            results.append({
                "machine_id": m.get("machine_id"),
                "skipped": True,
                "reason": "excluded"
            })
            continue

        try:
            health = requests.get(
                f"{m['agent_base_url'].rstrip('/')}/api/comfy/health",
                timeout=3,
            )
            health.raise_for_status()
            cap = health.json()
            checkpoints = cap.get("checkpoints") or []
            # Check if checkpoint filename exists on agent
            if checkpoint_filename and checkpoint_filename not in checkpoints:
                run_record["machines"].append(
                    {
                        "machine_id": m.get("machine_id"),
                        "label": m.get("label"),
                        "status": "blocked",
                        "error": "Checkpoint not synced to agent",
                    }
                )
                results.append({"machine_id": m.get("machine_id"), "skipped": True, "reason": "checkpoint_not_synced"})
                continue

            # Send filename to agent (ComfyUI expects exact filename)
            resp = requests.post(
                f"{m['agent_base_url'].rstrip('/')}/api/comfy/txt2img",
                json={
                    "run_id": run_id,
                    "prompt": prompt,
                    "checkpoint": checkpoint_filename,  # Pass filename to agent
                    "seed": seed,
                    "steps": steps,
                    "width": width,
                    "height": height,
                    "num_images": num_images,
                    "repeat": repeat,
                },
                timeout=5,
            )
            resp.raise_for_status()
            job = resp.json()
            agent_job_id = job.get("agent_job_id")
            if agent_job_id:
                JOB_RUN_MAP[agent_job_id] = {
                    "run_id": run_id,
                    "machine_id": m.get("machine_id"),
                    "label": m.get("label"),
                }
            results.append({"machine_id": m.get("machine_id"), "job": job})
            run_record["machines"].append(
                {
                    "machine_id": m.get("machine_id"),
                    "label": m.get("label"),
                    "status": "pending",
                    "job_id": agent_job_id,
                    "checkpoint": checkpoint_label,  # Store label for display
                }
            )
        except Exception as exc:
            run_record["machines"].append(
                {
                    "machine_id": m.get("machine_id"),
                    "label": m.get("label"),
                    "status": "error",
                    "error": str(exc),
                }
            )
            results.append({"machine_id": m.get("machine_id"), "error": str(exc)})

    with RUN_LOCK:
        RUN_CACHE[run_id] = run_record
        _write_run_record(run_record)
        _prune_old_runs()

    # Mark run as active with the machine IDs that successfully started jobs
    active_machine_ids = [r["machine_id"] for r in results if "job" in r]
    if active_machine_ids:
        _mark_run_active(run_id, "image", active_machine_ids)

    return jsonify({"run_id": run_id, "seed": seed, "results": results})


@app.post("/api/generate_sample_prompt")
def api_generate_sample_prompt():
    remote_addr = request.remote_addr or "unknown"
    if _sample_prompt_rate_limited(remote_addr):
        return jsonify({"error": "Rate limited. Try again shortly."}), 429

    payload = request.get_json(silent=True) or {}
    selected_model = payload.get("model") or request.args.get("model")
    seed = uuid.uuid4().hex[:8]
    candidate = _select_agent_for_sample(selected_model)

    if not candidate:
        return jsonify({"prompt": _heuristic_sample_prompt(seed), "fallback": True})

    machine = candidate["machine"]
    model = candidate["model"]
    sample_instruction = (
        "You are a prompt generation assistant. Produce a unique, realistic, and moderately long prompt for a "
        "code/analysis task that will take non-trivial compute to answer. Include several steps and ask the model "
        "to produce code with explanations and edge-case handling. Use a random topic seed: "
        f"{seed}. Output only the prompt text inside triple backticks."
    )

    try:
        r = requests.post(
            f"{machine['agent_base_url'].rstrip('/')}/jobs",
            json={
                "test_type": "llm_generate",
                "model": model,
                "prompt": sample_instruction,
                "max_tokens": 512,
                "temperature": 0.7,
                "num_ctx": 4096,
                "repeat": 1,
                "stream": True,
            },
            timeout=5,
        )
        r.raise_for_status()
        job = r.json()
        job_id = job.get("job_id")
        if not job_id:
            raise ValueError("No job_id returned from agent")
        generated = _wait_for_sample_prompt(job_id, SAMPLE_PROMPT_TIMEOUT_S)
        if not generated:
            raise TimeoutError("No sample prompt returned in time")
        prompt_text = _strip_prompt_fences(generated)
        if not prompt_text:
            raise ValueError("Empty prompt returned from agent")
        return jsonify({"prompt": prompt_text})
    except Exception as exc:
        log.warning("Sample prompt generation failed: %s", exc)
        fallback = _heuristic_sample_prompt(seed)
        return jsonify({"prompt": fallback, "fallback": True, "error": "agent_generation_failed"}), 503


@app.get("/api/settings/model_policy")
def api_get_model_policy():
    required = _required_models()
    return jsonify({"models": required.get("llm") or []})


@app.post("/api/settings/model_policy")
def api_set_model_policy():
    remote_addr = request.remote_addr or "unknown"
    if _model_policy_rate_limited(remote_addr):
        return jsonify({"error": "Rate limited. Try again shortly."}), 429

    payload = request.get_json(silent=True) or {}
    models = payload.get("models")
    if models is None or not isinstance(models, list):
        return jsonify({"error": "Models payload must be a list."}), 400

    cleaned: List[str] = []
    for model in models:
        if not isinstance(model, str):
            return jsonify({"error": "Each model must be a string."}), 400
        name = model.strip()
        if not name:
            continue
        if not MODEL_ID_RE.match(name):
            return jsonify({"error": f"Invalid model name: {name}"}), 400
        cleaned.append(name)

    with MODEL_POLICY_LOCK:
        policy = load_model_policy()
        required = policy.setdefault("required", {})
        required["llm"] = cleaned
        _write_model_policy(policy)
        global MODEL_POLICY
        MODEL_POLICY = policy

    missing = _missing_models_for_policy(cleaned)
    return jsonify({"ok": True, "changed": True, "models": cleaned, "missing": missing})


@app.get("/api/settings/comfy")
def api_get_comfy_settings():
    settings = _load_comfy_settings()
    return jsonify(settings)


@app.post("/api/settings/comfy")
def api_set_comfy_settings():
    payload = request.get_json(force=True) or {}
    settings = _default_comfy_settings()
    settings.update({k: v for k, v in payload.items() if v is not None})
    checkpoint_urls = settings.get("comfyui_checkpoints") or settings.get("checkpoint_urls") or []
    if not checkpoint_urls:
        checkpoint_urls = [DEFAULT_CHECKPOINT_URL]
    settings["checkpoint_urls"] = checkpoint_urls
    settings["comfyui_checkpoints"] = checkpoint_urls
    _save_comfy_settings(settings)
    return jsonify({"ok": True, "settings": settings})


# -----------------------------------------------------------------------------
# Service Control API
# Security: Check for local requests or valid token
# -----------------------------------------------------------------------------
SERVICE_CONTROL_TOKEN = os.environ.get("SERVICE_CONTROL_TOKEN")


def _check_service_auth():
    """
    Check if request is authorized for service control.
    Allows:
    - Local requests (127.0.0.1, localhost, ::1)
    - Requests with valid Authorization header if SERVICE_CONTROL_TOKEN is set
    """
    # Check if request is from localhost
    remote_addr = request.remote_addr or ""
    is_local = remote_addr in ("127.0.0.1", "localhost", "::1", "")

    if is_local:
        return True

    # Check for token auth if configured
    if SERVICE_CONTROL_TOKEN:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if token == SERVICE_CONTROL_TOKEN:
                return True

    return False


@app.get("/api/service/<component>/status")
def api_service_status(component: str):
    """
    Get status of a service component.

    GET /api/service/agent/status
    GET /api/service/central/status

    Returns:
        {
            "component": "agent",
            "running": true,
            "pid": 12345,
            "info": "listening on 127.0.0.1:9001"
        }
    """
    if component not in ("agent", "central"):
        return jsonify({"error": f"Unknown component: {component}"}), 400

    result = service_controller.get_status(component)
    return jsonify(result)


@app.post("/api/service/<component>/<action>")
def api_service_action(component: str, action: str):
    """
    Perform an action on a service component.

    POST /api/service/agent/start
    POST /api/service/agent/stop
    POST /api/service/central/start
    POST /api/service/central/stop

    Returns:
        {
            "component": "agent",
            "action": "start",
            "result": "started",
            "pid": 12345
        }
    """
    if component not in ("agent", "central"):
        return jsonify({"error": f"Unknown component: {component}"}), 400

    if action not in ("start", "stop"):
        return jsonify({"error": f"Unknown action: {action}"}), 400

    # Check authorization
    if not _check_service_auth():
        return jsonify({"error": "Unauthorized. Use local access or provide valid token."}), 403

    result = service_controller.perform_action(component, action)

    # Determine HTTP status code
    if result.get("result") == "error":
        return jsonify(result), 500
    return jsonify(result)


# -----------------------------------------------------------------------------
# Socket.IO events
# -----------------------------------------------------------------------------
@socketio.on("connect")
def on_connect():
    emit("status", {"ok": True})
    emit("machines", MACHINES)


@socketio.on("start_llm")
def on_start_llm(payload):
    # Browser can call via socket instead of fetch. We call the HTTP handler logic.
    with app.test_request_context(json=payload):
        resp = api_start_llm()
    emit("llm_jobs_started", resp.get_json())


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Start WS connectors before serving
    start_ws_connectors()
    try:
        socketio.run(app, host="0.0.0.0", port=8080)
    finally:
        stop_ws_connectors()

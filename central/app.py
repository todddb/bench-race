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

import sys
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from central.services import controller as service_controller

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


def _default_comfy_settings() -> Dict[str, Any]:
    return {
        "base_url": "",
        "models_path": "",
        "central_cache_path": str(CENTRAL_DIR / "model_cache" / "comfyui"),
        "checkpoint_urls": [],
    }


def _load_comfy_settings() -> Dict[str, Any]:
    if not COMFY_SETTINGS_PATH.exists():
        return _default_comfy_settings()
    with open(COMFY_SETTINGS_PATH, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    comfy = payload.get("comfyui") if isinstance(payload, dict) else {}
    settings = _default_comfy_settings()
    if isinstance(comfy, dict):
        settings.update({k: v for k, v in comfy.items() if v is not None})
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


def _checkpoint_entry_from_line(line: str) -> Optional[Dict[str, str]]:
    if not line:
        return None
    parts = [part.strip() for part in line.split("|") if part.strip()]
    if not parts:
        return None
    entry = {"url": parts[0]}
    if len(parts) > 1:
        entry["sha256"] = parts[1]
    return entry


def _checkpoint_filename_from_url(url: str) -> str:
    name = url.split("?")[0].rstrip("/").split("/")[-1]
    return name or url


def _checkpoint_entries(settings: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    conf = settings or _load_comfy_settings()
    urls = conf.get("checkpoint_urls") or []
    entries = []
    for line in urls:
        entry = _checkpoint_entry_from_line(line)
        if entry:
            entries.append(entry)
    return entries


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
        available_memory = cap.get("accelerator_memory_gb") or cap.get("system_memory_gb")
        fit = classify_model_fit(available_memory, estimate_model_size_gb(model, num_ctx))
        memory_label = "RAM"
        if cap.get("accelerator_type") == "cuda":
            memory_label = "VRAM"
        elif cap.get("accelerator_type") == "metal":
            memory_label = "Unified"
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
        return


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


@app.get("/image")
def image():
    settings = _load_comfy_settings()
    checkpoint_options = _list_cached_checkpoints(settings)
    if not checkpoint_options:
        checkpoint_options = [
            _checkpoint_filename_from_url(entry["url"]) for entry in _checkpoint_entries(settings)
        ]
    return render_template("image.html", machines=MACHINES, checkpoint_options=checkpoint_options)


@app.get("/admin")
def admin():
    return render_template("admin.html")


@app.get("/api/machines")
def api_machines():
    return jsonify(MACHINES)


@app.get("/api/capabilities")
def api_capabilities():
    caps = []
    for m in MACHINES:
        try:
            r = requests.get(f"{m['agent_base_url'].rstrip('/')}/capabilities", timeout=2)
            r.raise_for_status()
            cap = r.json()
            cap["agent_reachable"] = True
            caps.append(cap)
        except Exception as e:
            caps.append(
                {
                    "machine_id": m.get("machine_id"),
                    "label": m.get("label"),
                    "agent_reachable": False,
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
        last_checked = time.time()
        try:
            r = requests.get(f"{m['agent_base_url'].rstrip('/')}/capabilities", timeout=2)
            r.raise_for_status()
            cap = r.json()
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
            available_memory = cap.get("accelerator_memory_gb") or cap.get("system_memory_gb")
            fit = classify_model_fit(available_memory, estimate_model_size_gb(selected_model or "", num_ctx))
            memory_label = "RAM"
            if cap.get("accelerator_type") == "cuda":
                memory_label = "VRAM"
            elif cap.get("accelerator_type") == "metal":
                memory_label = "Unified"
            statuses.append(
                {
                    "machine_id": cap.get("machine_id") or m.get("machine_id"),
                    "label": cap.get("label") or m.get("label"),
                    "reachable": True,
                    "selected_model": selected_model,
                    "has_selected_model": has_selected_model,
                    "available_llm_models": available_llm,
                    "missing_required": missing_required,
                    "last_checked": last_checked,
                    "error": None,
                    "model_fit": {
                        **fit,
                        "memory_label": memory_label,
                    },
                }
            )
        except Exception as e:
            statuses.append(
                {
                    "machine_id": m.get("machine_id"),
                    "label": m.get("label"),
                    "reachable": False,
                    "selected_model": selected_model,
                    "has_selected_model": False,
                    "available_llm_models": [],
                    "missing_required": required,
                    "last_checked": last_checked,
                    "error": str(e),
                    "model_fit": {"status": "unknown", "needed_gb": None, "available_gb": None, "memory_label": "RAM"},
                }
            )
    return jsonify({"model": selected_model, "required": required, "machines": statuses})


@app.get("/api/image/status")
def api_image_status():
    checkpoint = request.args.get("checkpoint") or ""
    statuses = []
    for m in MACHINES:
        last_checked = time.time()
        try:
            r = requests.get(f"{m['agent_base_url'].rstrip('/')}/api/comfy/health", timeout=3)
            r.raise_for_status()
            cap = r.json()
            checkpoints = cap.get("checkpoints") or []
            has_checkpoint = bool(checkpoint and checkpoint in checkpoints)
            statuses.append(
                {
                    "machine_id": cap.get("machine_id") or m.get("machine_id"),
                    "label": cap.get("label") or m.get("label"),
                    "reachable": True,
                    "comfy_running": cap.get("running", False),
                    "checkpoints": checkpoints,
                    "has_checkpoint": has_checkpoint,
                    "missing_checkpoint": checkpoint if checkpoint and not has_checkpoint else None,
                    "last_checked": last_checked,
                    "error": cap.get("error"),
                }
            )
        except Exception as e:
            statuses.append(
                {
                    "machine_id": m.get("machine_id"),
                    "label": m.get("label"),
                    "reachable": False,
                    "comfy_running": False,
                    "checkpoints": [],
                    "has_checkpoint": False,
                    "missing_checkpoint": checkpoint or None,
                    "last_checked": last_checked,
                    "error": str(e),
                }
            )
    return jsonify({"checkpoint": checkpoint, "machines": statuses})


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
            "model_fit_status",
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
                model_fit.get("status"),
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

    return jsonify({"run_id": run_id, "results": results})


@app.post("/api/start_image")
def api_start_image():
    payload = request.get_json(force=True) or {}

    prompt = payload.get("prompt", "")
    checkpoint = payload.get("checkpoint", "")
    seed_mode = payload.get("seed_mode", "fixed")
    seed = payload.get("seed")
    steps = int(payload.get("steps", 30))
    width = int(payload.get("width", 1024))
    height = int(payload.get("height", 1024))
    num_images = int(payload.get("num_images", 1))
    repeat = int(payload.get("repeat", 1))

    if seed_mode == "random" or seed is None:
        seed = random.randint(1, 2**31 - 1)

    settings = _load_comfy_settings()
    checkpoint_error = _ensure_checkpoint_cached(checkpoint, settings)
    if checkpoint_error:
        return jsonify({"error": checkpoint_error}), 400

    run_timestamp = datetime.now(timezone.utc)
    run_id = _new_run_id(run_timestamp)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    run_record: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": run_timestamp.isoformat(),
        "type": "image",
        "model": checkpoint,
        "prompt_text": prompt,
        "prompt_hash": prompt_hash,
        "settings": {
            "checkpoint": checkpoint,
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
        try:
            health = requests.get(
                f"{m['agent_base_url'].rstrip('/')}/api/comfy/health",
                timeout=3,
            )
            health.raise_for_status()
            cap = health.json()
            checkpoints = cap.get("checkpoints") or []
            if checkpoint and checkpoint not in checkpoints:
                run_record["machines"].append(
                    {
                        "machine_id": m.get("machine_id"),
                        "label": m.get("label"),
                        "status": "blocked",
                        "error": "Missing checkpoint",
                    }
                )
                results.append({"machine_id": m.get("machine_id"), "skipped": True, "reason": "missing_checkpoint"})
                continue

            resp = requests.post(
                f"{m['agent_base_url'].rstrip('/')}/api/comfy/txt2img",
                json={
                    "run_id": run_id,
                    "prompt": prompt,
                    "checkpoint": checkpoint,
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
                    "checkpoint": checkpoint,
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
    settings["checkpoint_urls"] = settings.get("checkpoint_urls") or []
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

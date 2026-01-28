# central/app.py
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import os
import requests
import yaml
import websockets
from flask import Flask, jsonify, render_template, request, abort
from flask_socketio import SocketIO, emit


# -----------------------------------------------------------------------------
# Paths (robust regardless of cwd)
# -----------------------------------------------------------------------------
CENTRAL_DIR = Path(__file__).resolve().parent          # .../bench-race/central
ROOT_DIR = CENTRAL_DIR.parent                          # .../bench-race
CONFIG_PATH = CENTRAL_DIR / "config" / "machines.yaml"

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
    return render_template("index.html", machines=MACHINES)


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

    results = []
    for m in MACHINES:
        # Skip machines not in the list (if list is provided)
        if machine_ids is not None and m.get("machine_id") not in machine_ids:
            results.append({
                "machine_id": m.get("machine_id"),
                "skipped": True,
                "reason": "not in ready list"
            })
            continue

        try:
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
            results.append({"machine_id": m["machine_id"], "job": r.json()})
        except Exception as e:
            results.append({"machine_id": m.get("machine_id"), "error": str(e)})

    return jsonify(results)


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


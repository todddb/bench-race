# central/app.py
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
import yaml
import websockets
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit

# -----------------------------------------------------------------------------
# Paths (robust regardless of cwd)
# -----------------------------------------------------------------------------
CENTRAL_DIR = Path(__file__).resolve().parent          # .../bench-race/central
ROOT_DIR = CENTRAL_DIR.parent                          # .../bench-race
CONFIG_PATH = CENTRAL_DIR / "config" / "machines.yaml"

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
            caps.append(r.json())
        except Exception as e:
            caps.append(
                {
                    "machine_id": m.get("machine_id"),
                    "label": m.get("label"),
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

    results = []
    for m in MACHINES:
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


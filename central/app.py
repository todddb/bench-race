from __future__ import annotations

import atexit
import asyncio
import json
import logging
import threading
from urllib.parse import urlparse

import requests
import websockets
import yaml
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev"
socketio = SocketIO(app, cors_allowed_origins="*")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

with open("config/machines.yaml", "r", encoding="utf-8") as f:
    MACHINES = yaml.safe_load(f)["machines"]

_ws_stop_event = threading.Event()
_ws_loop: asyncio.AbstractEventLoop | None = None
_ws_thread: threading.Thread | None = None
_ws_tasks: list[asyncio.Task] = []


@app.get("/")
def index():
    return render_template("index.html", machines=MACHINES)


@app.get("/api/machines")
def api_machines():
    return jsonify(MACHINES)


@app.get("/api/capabilities")
def api_capabilities():
    caps = []
    for m in MACHINES:
        try:
            r = requests.get(f"{m['agent_base_url']}/capabilities", timeout=2)
            r.raise_for_status()
            caps.append(r.json())
        except Exception as e:
            caps.append({"machine_id": m["machine_id"], "label": m["label"], "error": str(e)})
    return jsonify(caps)


@app.post("/api/start_llm")
def api_start_llm():
    payload = request.get_json() or {}
    return jsonify(_fan_out_llm(payload))


@socketio.on("connect")
def on_connect():
    emit("status", {"ok": True})


@socketio.on("start_llm")
def on_start_llm(payload):
    results = _fan_out_llm(payload)
    emit("llm_jobs_started", results)


@socketio.on("llm_run")
def on_llm_run(payload):
    """
    Payload example:
    {
      "model": "...",
      "prompt": "...",
      "max_tokens": 256,
      "temperature": 0.2,
      "num_ctx": 4096,
      "repeat": 1
    }
    For now we just fan out the request; streaming will be implemented next.
    """
    results = _fan_out_llm(payload)
    emit("llm_jobs_started", results)


def _fan_out_llm(payload: dict) -> list[dict]:
    results = []
    for m in MACHINES:
        try:
            r = requests.post(
                f"{m['agent_base_url']}/jobs",
                json={
                    "test_type": "llm_generate",
                    "model": payload["model"],
                    "prompt": payload["prompt"],
                    "max_tokens": int(payload.get("max_tokens", 256)),
                    "temperature": float(payload.get("temperature", 0.2)),
                    "num_ctx": int(payload.get("num_ctx", 4096)),
                    "repeat": int(payload.get("repeat", 1)),
                    "stream": True,
                },
                timeout=5,
            )
            r.raise_for_status()
            results.append({"machine_id": m["machine_id"], "job": r.json()})
        except Exception as e:
            results.append({"machine_id": m["machine_id"], "error": str(e)})
    return results


def _build_ws_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc or parsed.path
    path = parsed.path if parsed.netloc else ""
    path = path.rstrip("/")
    ws_path = f"{path}/ws" if path else "/ws"
    return f"{scheme}://{netloc}{ws_path}"


async def _agent_ws_listener(machine: dict) -> None:
    machine_id = machine["machine_id"]
    ws_url = _build_ws_url(machine["agent_base_url"])
    backoff = 1
    while not _ws_stop_event.is_set():
        try:
            logging.info("Connecting to agent %s at %s", machine_id, ws_url)
            async with websockets.connect(ws_url) as websocket:
                logging.info("Connected to agent %s", machine_id)
                backoff = 1
                async for message in websocket:
                    if _ws_stop_event.is_set():
                        break
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        logging.warning("Non-JSON message from %s: %s", machine_id, message)
                        continue
                    if isinstance(data, dict):
                        data["machine_id"] = machine_id
                    else:
                        data = {"machine_id": machine_id, "payload": data}
                    socketio.emit("agent_event", data)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logging.warning("Agent %s connection error: %s", machine_id, exc)
        if _ws_stop_event.is_set():
            break
        logging.info("Reconnecting to agent %s in %s seconds", machine_id, backoff)
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 10)


def _start_ws_tasks() -> None:
    global _ws_loop, _ws_thread, _ws_tasks
    if _ws_thread and _ws_thread.is_alive():
        return
    _ws_loop = asyncio.new_event_loop()

    def runner() -> None:
        global _ws_tasks
        asyncio.set_event_loop(_ws_loop)
        _ws_tasks = [_ws_loop.create_task(_agent_ws_listener(m)) for m in MACHINES]
        try:
            _ws_loop.run_until_complete(asyncio.gather(*_ws_tasks, return_exceptions=True))
        finally:
            _ws_loop.close()

    _ws_thread = threading.Thread(target=runner, daemon=True)
    _ws_thread.start()


def _stop_ws_tasks() -> None:
    global _ws_loop, _ws_thread, _ws_tasks
    _ws_stop_event.set()
    if _ws_loop:
        for task in _ws_tasks:
            _ws_loop.call_soon_threadsafe(task.cancel)
        _ws_loop.call_soon_threadsafe(lambda: None)
    if _ws_thread:
        _ws_thread.join(timeout=5)


if __name__ == "__main__":
    _start_ws_tasks()
    atexit.register(_stop_ws_tasks)
    socketio.run(app, host="0.0.0.0", port=8080)

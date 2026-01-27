from __future__ import annotations

import yaml
import requests
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev"
socketio = SocketIO(app, cors_allowed_origins="*")

with open("config/machines.yaml", "r", encoding="utf-8") as f:
    MACHINES = yaml.safe_load(f)["machines"]


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


@socketio.on("connect")
def on_connect():
    emit("status", {"ok": True})


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
    results = []
    for m in MACHINES:
        try:
            r = requests.post(f"{m['agent_base_url']}/jobs", json={
                "test_type": "llm_generate",
                "model": payload["model"],
                "prompt": payload["prompt"],
                "max_tokens": int(payload.get("max_tokens", 256)),
                "temperature": float(payload.get("temperature", 0.2)),
                "num_ctx": int(payload.get("num_ctx", 4096)),
                "repeat": int(payload.get("repeat", 1)),
                "stream": True,
            }, timeout=5)
            r.raise_for_status()
            results.append({"machine_id": m["machine_id"], "job": r.json()})
        except Exception as e:
            results.append({"machine_id": m["machine_id"], "error": str(e)})

    emit("llm_jobs_started", results)


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8080)

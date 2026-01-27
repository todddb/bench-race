#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/projects/bench-race}"

echo "Scaffolding bench-race at: $ROOT"
mkdir -p "$ROOT"
cd "$ROOT"

# ---------- directories ----------
mkdir -p central/{templates,static/{css,js},config}
mkdir -p agent/{backends,config}
mkdir -p shared
mkdir -p scripts
mkdir -p results/runs
mkdir -p assets/{audio,images,workflows}
mkdir -p docs

# ---------- root files ----------
cat > README.md <<'EOF'
# bench-race

A side-by-side benchmarking demo that compares multiple machines in real time.
- Central UI: Flask + Socket.IO (webpage with 4 panes)
- Per-machine Agent: FastAPI (wraps local engines like Ollama / Whisper / ComfyUI)
- Metrics computed on-agent for fairness (TTFT, tok/s, realtime factor, etc.)

## Quick start (dev on a laptop)
1) Start a local agent:
   cd agent
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn agent_app:app --host 0.0.0.0 --port 9001

2) Start the central UI:
   cd central
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   python app.py

Then open http://localhost:8080
EOF

cat > .gitignore <<'EOF'
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.egg-info/
dist/
build/

# Venvs
.venv/
venv/

# OS
.DS_Store

# Results/artifacts
results/runs/*
!results/runs/.keep
assets/audio/*
!assets/audio/.keep
assets/images/*
!assets/images/.keep

# IDE
.vscode/
.idea/
EOF

touch results/runs/.keep assets/audio/.keep assets/images/.keep

cat > pyproject.toml <<'EOF'
[tool.black]
line-length = 100

[tool.ruff]
line-length = 100
EOF

# ---------- shared schemas ----------
cat > shared/schemas.py <<'EOF'
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ----- Common -----
class MachineInfo(BaseModel):
    machine_id: str
    label: str
    agent_base_url: str
    notes: Optional[str] = None


class Capabilities(BaseModel):
    machine_id: str
    label: str
    tests: List[str] = Field(default_factory=list)
    llm_models: List[str] = Field(default_factory=list)
    whisper_models: List[str] = Field(default_factory=list)
    sdxl_profiles: List[str] = Field(default_factory=list)


# ----- Job Requests -----
class LLMRequest(BaseModel):
    test_type: Literal["llm_generate"] = "llm_generate"
    model: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.2
    num_ctx: int = 4096
    repeat: int = 1
    stream: bool = True


class WhisperRequest(BaseModel):
    test_type: Literal["whisper_transcribe"] = "whisper_transcribe"
    asset_id: str
    model: str = "large-v3"
    language: Optional[str] = None
    stream: bool = True


class SDXLRequest(BaseModel):
    test_type: Literal["sdxl_generate"] = "sdxl_generate"
    profile: str = "sdxl_1024_30steps"
    positive_prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 30
    seed: int = 12345
    n_images: int = 1
    stream: bool = True


# ----- Job & Events -----
class JobStartResponse(BaseModel):
    job_id: str


class Event(BaseModel):
    job_id: str
    type: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class LLMResult(BaseModel):
    ttft_ms: Optional[float] = None
    gen_tokens: Optional[int] = None
    gen_tokens_per_s: Optional[float] = None
    prompt_tokens: Optional[int] = None
    total_ms: Optional[float] = None
    model: str
    engine: str = "ollama"


class WhisperResult(BaseModel):
    audio_seconds: Optional[float] = None
    wall_seconds: Optional[float] = None
    x_realtime: Optional[float] = None
    model: str
    engine: str = "faster-whisper"


class SDXLResult(BaseModel):
    wall_seconds: Optional[float] = None
    seconds_per_image: Optional[float] = None
    images: int = 1
    steps: int = 30
    width: int = 1024
    height: int = 1024
    engine: str = "comfyui"
EOF

# ---------- agent ----------
cat > agent/requirements.txt <<'EOF'
fastapi==0.110.0
uvicorn[standard]==0.27.1
requests==2.31.0
pydantic==2.6.4
python-multipart==0.0.9
websockets==12.0
pyyaml==6.0.1
psutil==5.9.8
pillow==10.2.0

# Optional (enable when you implement Whisper on agents):
# faster-whisper==1.0.3
EOF

cat > agent/config/agent.yaml <<'EOF'
machine_id: "local-dev"
label: "Local Dev Agent"
bind_host: "0.0.0.0"
bind_port: 9001

# Local engine endpoints (running on the same machine as the agent)
ollama_base_url: "http://127.0.0.1:11434"
comfyui_base_url: "http://127.0.0.1:8188"

# Models to advertise for UI dropdowns (you can override per-machine)
llm_models:
  - "llama3.1:8b-instruct-q8_0"
  - "llama3.1:70b-instruct-q4_K_M"

whisper_models:
  - "large-v3"

sdxl_profiles:
  - "sdxl_1024_30steps"
EOF

cat > agent/agent_app.py <<'EOF'
from __future__ import annotations

import time
import uuid
from typing import Dict

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from shared.schemas import Capabilities, Event, JobStartResponse, LLMRequest

app = FastAPI(title="bench-race agent")

# In-memory event bus (simple stub). We'll replace with a real job runner.
SUBSCRIBERS: Dict[str, WebSocket] = {}


def load_config():
    with open("config/agent.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_config()


@app.get("/health")
def health():
    return {"ok": True, "machine_id": CFG["machine_id"], "label": CFG["label"]}


@app.get("/capabilities")
def capabilities():
    cap = Capabilities(
        machine_id=CFG["machine_id"],
        label=CFG["label"],
        tests=["llm_generate"],  # add whisper_transcribe, sdxl_generate later
        llm_models=CFG.get("llm_models", []),
        whisper_models=CFG.get("whisper_models", []),
        sdxl_profiles=CFG.get("sdxl_profiles", []),
    )
    return JSONResponse(cap.model_dump())


@app.post("/jobs", response_model=JobStartResponse)
async def start_job(req: LLMRequest):
    # Stub: immediately emits fake streaming tokens for UI wiring.
    job_id = str(uuid.uuid4())

    # fire-and-forget using background task would be ideal; for stub we keep simple.
    # The central UI will connect to /ws and receive events.
    # We'll stream tokens when a WS subscriber appears.

    return JobStartResponse(job_id=job_id)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    client_id = str(uuid.uuid4())
    SUBSCRIBERS[client_id] = ws
    try:
        # Basic echo loop; central will likely just listen.
        while True:
            _ = await ws.receive_text()
    except WebSocketDisconnect:
        SUBSCRIBERS.pop(client_id, None)


# Helper for later: send an event to all connected subscribers
async def broadcast(event: Event):
    dead = []
    for cid, sock in SUBSCRIBERS.items():
        try:
            await sock.send_json(event.model_dump())
        except Exception:
            dead.append(cid)
    for cid in dead:
        SUBSCRIBERS.pop(cid, None)
EOF

cat > agent/backends/ollama_backend.py <<'EOF'
"""
Ollama backend wrapper (stub).
Later this will:
- call Ollama /api/generate with stream=true
- compute TTFT/tok/s locally
- emit streaming events back to central
"""
EOF

cat > agent/backends/whisper_backend.py <<'EOF'
"""
Whisper backend wrapper (stub).
Later this will:
- accept staged asset_id or local path
- run faster-whisper
- stream segments/progress
- compute x realtime and total duration
"""
EOF

cat > agent/backends/comfyui_backend.py <<'EOF'
"""
ComfyUI backend wrapper (stub).
Later this will:
- submit workflow
- stream progress
- fetch images and emit to central
- compute seconds/image
"""
EOF

# ---------- central ----------
cat > central/requirements.txt <<'EOF'
flask==3.0.2
flask-socketio==5.3.6
eventlet==0.35.2
requests==2.31.0
pydantic==2.6.4
pyyaml==6.0.1
EOF

cat > central/config/machines.yaml <<'EOF'
machines:
  - machine_id: "rtx5090"
    label: "RTX 5090 (32GB)"
    agent_base_url: "http://10.0.0.10:9001"
  - machine_id: "m4mbp"
    label: "Apple M4 (128GB)"
    agent_base_url: "http://10.0.0.11:9001"
  - machine_id: "rtxpro6000"
    label: "RTX Pro 6000 (96GB)"
    agent_base_url: "http://10.0.0.12:9001"
  - machine_id: "gb10"
    label: "Nvidia GB10 (128GB)"
    agent_base_url: "http://10.0.0.13:9001"
EOF

cat > central/app.py <<'EOF'
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
EOF

cat > central/templates/index.html <<'EOF'
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>bench-race</title>
    <link rel="stylesheet" href="/static/css/app.css" />
    <script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
  </head>
  <body>
    <div class="container">
      <h1>bench-race</h1>

      <div class="controls card">
        <div class="row">
          <label>Model</label>
          <select id="model">
            <option value="llama3.1:8b-instruct-q8_0">llama3.1:8b-instruct-q8_0</option>
            <option value="llama3.1:70b-instruct-q4_K_M">llama3.1:70b-instruct-q4_K_M</option>
          </select>

          <label>Max tokens</label>
          <input id="max_tokens" type="number" value="256" />

          <label>Context</label>
          <input id="num_ctx" type="number" value="4096" />

          <label>Temp</label>
          <input id="temperature" type="number" step="0.1" value="0.2" />

          <label>Repeat</label>
          <input id="repeat" type="number" value="1" />
        </div>

        <div class="row">
          <label>Prompt</label>
          <textarea id="prompt" rows="4">Summarize the key tradeoffs between unified memory and discrete VRAM for local LLM inference.</textarea>
        </div>

        <button id="run">Run (LLM)</button>
        <div id="status"></div>
      </div>

      <div class="grid">
        {% for m in machines %}
        <div class="pane card">
          <div class="pane-title">{{ m.label }}</div>
          <pre class="output" id="out-{{ m.machine_id }}"></pre>
          <div class="metrics" id="metrics-{{ m.machine_id }}">
            <span class="muted">Waiting…</span>
          </div>
        </div>
        {% endfor %}
      </div>
    </div>

    <script src="/static/js/app.js"></script>
  </body>
</html>
EOF

cat > central/static/css/app.css <<'EOF'
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #f6f7f9; }
.container { max-width: 1400px; margin: 0 auto; padding: 18px; }
h1 { margin: 0 0 12px 0; }
.card { background: white; border-radius: 12px; padding: 12px; box-shadow: 0 1px 6px rgba(0,0,0,.08); }
.controls { margin-bottom: 12px; }
.row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }
label { font-size: 12px; color: #555; }
select, input, textarea { padding: 8px; border-radius: 10px; border: 1px solid #ddd; }
textarea { width: 100%; }
button { padding: 10px 14px; border: 0; border-radius: 10px; background: #111827; color: white; cursor: pointer; }
.grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
.pane-title { font-weight: 700; margin-bottom: 8px; }
.output { height: 280px; background: #0b1020; color: #d7e1ff; padding: 10px; border-radius: 10px; overflow: auto; }
.metrics { margin-top: 8px; font-size: 13px; color: #111; }
.muted { color: #666; }
EOF

cat > central/static/js/app.js <<'EOF'
const socket = io();

socket.on("status", (msg) => {
  document.getElementById("status").innerText = msg.ok ? "Connected" : "Not connected";
});

socket.on("llm_jobs_started", (results) => {
  console.log("Jobs started", results);
  // Stub: show job IDs for now
  results.forEach((r) => {
    const out = document.getElementById(`out-${r.machine_id}`);
    const metrics = document.getElementById(`metrics-${r.machine_id}`);
    if (!out || !metrics) return;

    out.textContent = "";
    if (r.error) {
      out.textContent = `Error starting job: ${r.error}`;
      metrics.innerHTML = `<span class="muted">Failed to start</span>`;
    } else {
      out.textContent = `Job started: ${JSON.stringify(r.job)}`;
      metrics.innerHTML = `<span class="muted">Streaming not wired yet</span>`;
    }
  });
});

document.getElementById("run").addEventListener("click", () => {
  const payload = {
    model: document.getElementById("model").value,
    prompt: document.getElementById("prompt").value,
    max_tokens: parseInt(document.getElementById("max_tokens").value, 10),
    num_ctx: parseInt(document.getElementById("num_ctx").value, 10),
    temperature: parseFloat(document.getElementById("temperature").value),
    repeat: parseInt(document.getElementById("repeat").value, 10),
  };

  socket.emit("llm_run", payload);
});
EOF

# ---------- scripts for venv ----------
cat > scripts/setup_venv_central.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../central"
PYTHON=${PYTHON:-python3.10}

$PYTHON -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
echo "Central venv ready. Run: python app.py"
EOF

cat > scripts/setup_venv_agent.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../agent"
PYTHON=${PYTHON:-python3.10}

$PYTHON -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
echo "Agent venv ready. Run: uvicorn agent_app:app --host 0.0.0.0 --port 9001"
EOF

chmod +x scripts/setup_venv_central.sh scripts/setup_venv_agent.sh scripts/scaffold.sh 2>/dev/null || true

# Optional: initialize git
if [ ! -d ".git" ]; then
  git init >/dev/null 2>&1 || true
fi

echo "Done."
echo "Next:"
echo "  1) Run: ./scripts/setup_venv_agent.sh   (then start agent)"
echo "  2) Run: ./scripts/setup_venv_central.sh (then start central UI)"
echo "  3) Commit & push to GitHub."


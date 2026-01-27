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

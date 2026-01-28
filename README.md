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

## Runbook / Operational Notes

### Ollama Backend

**Startup Scripts:**
- `scripts/start_agent.sh` now avoids double-starting Ollama by:
  1. Checking if port 11434 is already listening (via `lsof`)
  2. Checking if an `ollama serve` process already exists (via `pgrep`)
  3. Only starting a new instance if neither condition is met

**API Options:**
- Ollama uses `num_predict` (not `max_tokens`) to cap token generation in `/api/generate`.

**Streaming Configuration:**
The agent buffers small Ollama token fragments before broadcasting over WebSocket to reduce "jittery" streaming. Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_STREAM_FLUSH_CHARS` | `64` | Flush buffer when it reaches this many characters |
| `BENCH_STREAM_FLUSH_SECONDS` | `0.10` | Flush buffer after this many seconds since last flush |

Buffer is also flushed immediately when it contains a newline or when the stream completes.

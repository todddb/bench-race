# bench-race

**Local LLM benchmarking and multi-machine comparison toolkit.**

Run the same prompt across local or networked machines (agents) and collect streaming output + performance metrics to compare models, hardware, and settings.

---

## Table of Contents

- [Why bench-race?](#why-bench-race)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Scripts & Process Management](#scripts--process-management)
- [Running & Testing](#running--testing)
- [Multi-Machine Runs](#multi-machine-runs)
- [Results & Persistence](#results--persistence)
- [Troubleshooting](#troubleshooting)
- [Development Workflow](#development-workflow)
- [Contributing & License](#contributing--license)

---

## Why bench-race?

bench-race helps you benchmark local LLM inference across hardware and model variants:

- **Stream tokens in real-time** from each machine via WebSocket
- **Record metrics**: TTFT (time to first token), total generation time, tokens/sec, token counts
- **Compare machines** side-by-side with a web UI
- **Repeat runs** to get consistent, reproducible stats
- **Support multiple backends**: Ollama (LLM), with stubs for Whisper and ComfyUI/SDXL

Use cases:
- Compare Apple Silicon machines (M1 vs M2 vs M4)
- Test different quantization levels (Q4 vs Q8)
- Benchmark different model sizes on the same hardware
- Validate inference performance across a fleet of machines

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **macOS or Linux** | Tested on macOS (Apple Silicon) and Linux |
| **Python 3.9+** | Required for both agent and central |
| **pip** | Comes with Python |
| **virtualenv** or `python -m venv` | For isolated environments |
| **Ollama** (optional) | Default LLM backend; agent falls back to mock if unavailable |
| **curl** | For API testing |
| **websocat** (optional) | For WebSocket debugging (`brew install websocat` on macOS) |
| **gh** (optional) | GitHub CLI for repo operations |

### Installing Ollama

macOS (Homebrew):
```bash
brew install ollama
```

Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Pull a model for testing:
```bash
ollama pull llama3.1:8b-instruct-q8_0
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/todddb/bench-race.git
cd bench-race

# Create virtual environment for the agent
python3 -m venv agent/.venv
source agent/.venv/bin/activate
pip install -r agent/requirements.txt
deactivate

# Create virtual environment for central
python3 -m venv central/.venv
source central/.venv/bin/activate
pip install -r central/requirements.txt
deactivate
```

Alternatively, use the setup scripts:
```bash
./scripts/setup_venv_agent.sh
./scripts/setup_venv_central.sh
```

> **Tip**: You can use a single venv if preferred:
> ```bash
> python3 -m venv .venv
> source .venv/bin/activate
> pip install -r agent/requirements.txt -r central/requirements.txt
> ```

---

## Configuration

### Agent Configuration

Each agent reads `agent/config/agent.yaml`:

```yaml
# agent/config/agent.yaml
machine_id: "macbook"
label: "MacBook (M4, 128GB)"

bind_host: "0.0.0.0"
bind_port: 9001

ollama_base_url: "http://127.0.0.1:11434"

llm_models:
  - "llama3.1:8b-instruct-q8_0"
  - "llama3.1:70b-instruct-q4_K_M"

whisper_models:
  - "large-v3"

sdxl_profiles:
  - "sdxl_1024_30steps"
```

| Key | Description |
|-----|-------------|
| `machine_id` | Unique identifier for this machine |
| `label` | Human-readable name shown in UI |
| `bind_host` | Host to bind agent server (use `0.0.0.0` for network access) |
| `bind_port` | Port for agent API (default: 9001) |
| `ollama_base_url` | URL of local Ollama instance |
| `llm_models` | List of available models (shown in UI dropdown) |

### Central Configuration

Central reads `central/config/machines.yaml` to know which agents to connect to:

```yaml
# central/config/machines.yaml
machines:
  - machine_id: "macbook"
    label: "MacBook (M4, 128GB)"
    agent_base_url: "http://127.0.0.1:9001"
    notes: "Primary development machine"

  - machine_id: "macmini"
    label: "Mac Mini (M2, 24GB)"
    agent_base_url: "http://192.168.1.100:9001"
    notes: "Secondary test machine"
```

| Key | Description |
|-----|-------------|
| `machine_id` | Must match the agent's `machine_id` |
| `label` | Display name in the UI |
| `agent_base_url` | HTTP URL to reach the agent |
| `notes` | Optional description |

---

## Scripts & Process Management

Management scripts are located in `scripts/`:

### Agent Management

```bash
# Start agent (also ensures Ollama is running)
./scripts/agents start

# Start agent in daemon mode (background)
./scripts/agents start --daemon

# Stop agent
./scripts/agents stop

# Check status
./scripts/agents status

# Restart agent
./scripts/agents restart

# Tail logs
./scripts/agents logs
```

### Central Management

```bash
# Start central server
./scripts/central start

# Start in daemon mode
./scripts/central start --daemon

# Stop central
./scripts/central stop

# Check status
./scripts/central status

# Restart
./scripts/central restart

# Tail logs
./scripts/central logs
```

### What the Scripts Do

**`scripts/agents start`**:
1. Checks if Ollama is already running (port 11434)
2. Starts `ollama serve` if needed
3. Activates `agent/.venv`
4. Runs: `uvicorn agent.agent_app:app --host 0.0.0.0 --port 9001`
5. Writes PID to `run/agent.pid`, logs to `logs/agent.log`

**`scripts/central start`**:
1. Activates `central/.venv`
2. Runs: `python central/app.py`
3. Writes PID to `run/central.pid`, logs to `logs/central.log`

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_HOST` | `0.0.0.0` | Agent bind host |
| `AGENT_PORT` | `9001` | Agent bind port |
| `OLLAMA_HOST` | `127.0.0.1` | Ollama host |
| `OLLAMA_PORT` | `11434` | Ollama port |
| `CENTRAL_HOST` | `0.0.0.0` | Central bind host |
| `CENTRAL_PORT` | `8080` | Central bind port |

---

## Running & Testing

### 1. Start Ollama (if using local models)

```bash
# Option A: Let the agent script start Ollama automatically
./scripts/agents start

# Option B: Start Ollama manually first
ollama serve
```

### 2. Start the Agent

```bash
# From repo root
./scripts/agents start --daemon

# Or manually:
source agent/.venv/bin/activate
uvicorn agent.agent_app:app --host 0.0.0.0 --port 9001
```

### 3. Start Central

```bash
# From repo root
./scripts/central start --daemon

# Or manually:
source central/.venv/bin/activate
python central/app.py
```

### 4. Health Check

```bash
curl http://127.0.0.1:9001/health
```

Expected response:
```json
{"ok": true, "machine_id": "macbook", "label": "MacBook (M4, 128GB)"}
```

### 5. Get Capabilities

```bash
curl http://127.0.0.1:9001/capabilities
```

Expected response:
```json
{
  "machine_id": "macbook",
  "label": "MacBook (M4, 128GB)",
  "tests": ["llm_generate"],
  "llm_models": ["llama3.1:8b-instruct-q8_0", "llama3.1:70b-instruct-q4_K_M"],
  "whisper_models": ["large-v3"],
  "sdxl_profiles": ["sdxl_1024_30steps"]
}
```

### 6. Start a Job (curl example)

```bash
curl -X POST http://127.0.0.1:9001/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "test_type": "llm_generate",
    "model": "llama3.1:8b-instruct-q8_0",
    "prompt": "Explain the difference between unified memory and discrete VRAM in two sentences.",
    "max_tokens": 128,
    "temperature": 0.2,
    "num_ctx": 2048,
    "repeat": 1,
    "stream": true
  }'
```

Response:
```json
{"job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"}
```

### 7. Stream Events via WebSocket

```bash
websocat ws://127.0.0.1:9001/ws
```

Example output (NDJSON events):
```json
{"job_id":"a1b2c3d4...","type":"llm_token","payload":{"text":"Unified","timestamp_s":1706000000.123}}
{"job_id":"a1b2c3d4...","type":"llm_token","payload":{"text":" memory","timestamp_s":1706000000.234}}
{"job_id":"a1b2c3d4...","type":"llm_token","payload":{"text":" allows","timestamp_s":1706000000.345}}
...
{"job_id":"a1b2c3d4...","type":"job_done","payload":{"ttft_ms":123.4,"gen_tokens":45,"gen_tokens_per_s":32.1,"total_ms":1400.5,"model":"llama3.1:8b-instruct-q8_0","engine":"ollama"}}
```

### 8. Browser UI

Open: **http://127.0.0.1:8080**

The UI provides:
- Model selector dropdown
- Prompt textarea
- Controls for max tokens, context size, temperature, repeat count
- "Run (LLM)" button to start jobs
- Real-time token streaming output per machine
- Metrics display (TTFT, tokens/sec, total time)

---

## Multi-Machine Runs

### Setup

1. **Configure each agent**: Edit `agent/config/agent.yaml` on each machine with a unique `machine_id`

2. **Add machines to central**: Edit `central/config/machines.yaml`:
   ```yaml
   machines:
     - machine_id: "macbook-pro"
       label: "MacBook Pro (M4 Max, 128GB)"
       agent_base_url: "http://192.168.1.10:9001"

     - machine_id: "mac-mini"
       label: "Mac Mini (M2 Pro, 32GB)"
       agent_base_url: "http://192.168.1.20:9001"

     - machine_id: "linux-server"
       label: "Linux Server (RTX 4090)"
       agent_base_url: "http://192.168.1.30:9001"
   ```

3. **Start agents on each machine**:
   ```bash
   # On each machine
   ./scripts/agents start --daemon
   ```

4. **Verify connectivity**:
   ```bash
   # From the central machine
   curl http://192.168.1.10:9001/health
   curl http://192.168.1.20:9001/health
   curl http://192.168.1.30:9001/health
   ```

5. **Start central and open UI**: The UI will show all machines side-by-side

### Running Comparisons

1. Open **http://127.0.0.1:8080** on the central machine
2. Select a model from the dropdown
3. Enter your prompt
4. Click "Run (LLM)"
5. Watch results stream in real-time for all machines
6. Compare metrics in the summary

---

## Results & Persistence

Results are saved under `results/runs/<run_id>/`:

```
results/
└── runs/
    └── 2024-01-15_143022_abc123/
        ├── metadata.json    # Run settings, machine list, timestamps
        ├── events.ndjson    # Raw streamed events (one JSON per line)
        └── summary.json     # Aggregated metrics per machine
```

### Metadata Example

```json
{
  "run_id": "2024-01-15_143022_abc123",
  "started_at": "2024-01-15T14:30:22Z",
  "prompt": "Explain quantum computing in simple terms.",
  "model": "llama3.1:8b-instruct-q8_0",
  "machines": ["macbook", "macmini"],
  "settings": {
    "max_tokens": 256,
    "temperature": 0.2,
    "num_ctx": 4096
  }
}
```

### Summary Example

```json
{
  "macbook": {
    "ttft_ms": 89.2,
    "gen_tokens": 156,
    "gen_tokens_per_s": 45.3,
    "total_ms": 3442.1
  },
  "macmini": {
    "ttft_ms": 112.5,
    "gen_tokens": 156,
    "gen_tokens_per_s": 38.7,
    "total_ms": 4031.8
  }
}
```

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: config/agent.yaml` | Ensure `agent/config/agent.yaml` exists. Copy from the example in this README. |
| `ModuleNotFoundError: No module named 'backends'` | Run from repo root, not from `agent/` directory. The app adjusts `sys.path` accordingly. |
| `ollama not found` | Install Ollama or the agent will use mock streaming. See [Prerequisites](#prerequisites). |
| Agent not responding on network | Check firewall allows port 9001. Use `bind_host: "0.0.0.0"` in agent config. |
| Central can't reach agents | Verify `agent_base_url` in `machines.yaml`. Test with `curl <url>/health`. |
| WebSocket connection drops | Check network stability. Central auto-reconnects with exponential backoff. |
| Python package errors | Activate correct venv: `source agent/.venv/bin/activate` |
| Port already in use | Stop existing process: `./scripts/agents stop` or `lsof -ti:9001 | xargs kill` |

### Ollama Issues

```bash
# Check if Ollama is running
curl http://127.0.0.1:11434/api/tags

# Check Ollama logs
cat logs/ollama.log

# Restart Ollama
pkill -f "ollama serve"
ollama serve
```

### Viewing Logs

```bash
# Agent logs
./scripts/agents logs
# or: tail -f logs/agent.log

# Central logs
./scripts/central logs
# or: tail -f logs/central.log

# Ollama logs
tail -f logs/ollama.log
```

### Token Streaming Configuration

If streaming appears jittery, adjust buffering via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_STREAM_FLUSH_CHARS` | `64` | Flush buffer at this character count |
| `BENCH_STREAM_FLUSH_SECONDS` | `0.10` | Flush buffer after this interval |

```bash
BENCH_STREAM_FLUSH_CHARS=32 BENCH_STREAM_FLUSH_SECONDS=0.05 ./scripts/agents start
```

---

## Development Workflow

### Project Structure

```
bench-race/
├── agent/
│   ├── agent_app.py          # FastAPI agent server
│   ├── backends/             # Agent-side backend implementations
│   │   ├── ollama_backend.py
│   │   ├── whisper_backend.py  (stub)
│   │   └── comfyui_backend.py  (stub)
│   ├── config/
│   │   └── agent.yaml
│   └── requirements.txt
├── central/
│   ├── app.py                # Flask + Socket.IO server
│   ├── config/
│   │   └── machines.yaml
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   │   ├── css/app.css
│   │   └── js/app.js
│   └── requirements.txt
├── backends/                 # Shared backend library
│   └── ollama_backend.py
├── shared/
│   └── schemas.py            # Pydantic models
├── scripts/                  # Management scripts
├── results/                  # Benchmark results
├── logs/                     # Log files
└── run/                      # PID files
```

### Adding a New Backend

1. Create `agent/backends/mybackend_backend.py`:
   ```python
   import httpx

   async def check_mybackend_available(base_url: str) -> bool:
       try:
           async with httpx.AsyncClient() as client:
               resp = await client.get(f"{base_url}/health", timeout=5.0)
               return resp.status_code == 200
       except Exception:
           return False

   async def stream_mybackend_generate(params: dict):
       # Yield Event objects for each token/result
       yield {"type": "mybackend_token", "payload": {"text": "..."}}
       yield {"type": "job_done", "payload": {...}}
   ```

2. Register in `agent/agent_app.py`

3. Add request/result schemas to `shared/schemas.py`

### Running Tests

```bash
# Activate venv
source agent/.venv/bin/activate

# Run any tests (if present)
pytest tests/

# Type checking (if configured)
mypy agent/ central/
```

### Code Style

The project uses:
- **Black** for formatting (line length: 100)
- **Ruff** for linting (line length: 100)

```bash
# Format
black agent/ central/ --line-length 100

# Lint
ruff check agent/ central/
```

### Typical Development Loop

1. Create a feature branch
2. Make changes
3. Stop services: `./scripts/agents stop && ./scripts/central stop`
4. Start services: `./scripts/agents start && ./scripts/central start`
5. Test via UI at http://127.0.0.1:8080
6. Check logs: `./scripts/agents logs` / `./scripts/central logs`
7. Commit and push

### Using Codex/Claude for Tasks

The repo supports AI-assisted development:

```bash
# Example: Ask Claude to implement a feature
claude "Add a new endpoint to export results as CSV"

# Example: Ask Codex to fix an issue
codex "Fix the WebSocket reconnection logic in central/app.py"
```

---

## Contributing & License

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

For bugs or feature requests, open an issue on GitHub.

---

**License**: MIT

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

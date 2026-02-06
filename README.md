# bench-race

**Local LLM benchmarking and multi-machine comparison toolkit.**

Run the same prompt across local or networked machines (agents) and collect streaming output + performance metrics to compare models, hardware, and settings.

For runtime sampling and model-fit scoring details, see `docs/metrics_and_fit.md`.

---

## Table of Contents

- [Why bench-race?](#why-bench-race)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Scripts & Process Management](#scripts--process-management)
- [Running & Testing](#running--testing)
- [Multi-Machine Runs](#multi-machine-runs)
- [Compute Benchmarks](#compute-benchmarks)
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
- Compare raw CPU throughput with prime counting benchmarks

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

bench-race provides automated installer scripts for both agent and central components.

### Quick Start

```bash
# Clone the repository
git clone https://github.com/todddb/bench-race.git
cd bench-race

# Install agent (with Ollama and ComfyUI)
./scripts/install_agent.sh --central-url http://<central-host>:8080

# Install central server
./scripts/install_central.sh
```

The installers are **idempotent** - you can safely re-run them to update your installation.

### Agent Installation

The agent installer (`scripts/install_agent.sh`) sets up everything needed on agent machines:
- Platform detection (macOS/Linux, architecture, GPU support)
- Ollama installation and configuration
- ComfyUI with GPU-aware PyTorch selection
- Python environment and dependencies
- Agent configuration with automatic hardware detection

**Options:**
```bash
./scripts/install_agent.sh [OPTIONS]

Options:
  --agent-id ID         Agent identifier (default: hostname)
  --label "Label"       Human-readable label for UI
  --central-url URL     Central server URL (default: http://127.0.0.1:8080)
  --platform PLATFORM   Override platform detection (macos|linux|linux-gb10)
  --yes                 Non-interactive mode (use defaults)
  --no-service          Skip systemd/launchctl service installation
  --update              Update existing installation
  --skip-ollama         Skip Ollama installation
  --skip-comfyui        Skip ComfyUI installation
```

**Examples:**
```bash
# Interactive installation (prompts for configuration)
./scripts/install_agent.sh

# Non-interactive with all options
./scripts/install_agent.sh \
  --agent-id "gpu-server-1" \
  --label "GPU Server 1 (RTX 4090, 64GB)" \
  --central-url http://192.168.1.100:8080 \
  --yes

# Update existing installation
./scripts/install_agent.sh --update

# Install without Ollama or ComfyUI (minimal agent only)
./scripts/install_agent.sh --skip-ollama --skip-comfyui
```

### Central Installation

The central installer (`scripts/install_central.sh`) sets up the central server:
- Python environment and dependencies
- Configuration files (machines.yaml, model_policy.yaml)
- Web UI server

**Options:**
```bash
./scripts/install_central.sh [OPTIONS]

Options:
  --yes                 Non-interactive mode
  --no-service          Skip systemd/launchctl service installation
  --update              Update existing installation
  --platform PLATFORM   Override platform detection (macos|linux)
```

**Example:**
```bash
# Install central server
./scripts/install_central.sh --yes
```

### Manual Installation

If you prefer manual setup:

```bash
# Agent setup
python3 -m venv agent/.venv
source agent/.venv/bin/activate
pip install -r agent/requirements.txt
deactivate

# Central setup
python3 -m venv central/.venv
source central/.venv/bin/activate
pip install -r central/requirements.txt
deactivate
```

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
```

| Key | Description |
|-----|-------------|
| `machine_id` | Unique identifier for this machine |
| `label` | Human-readable name shown in UI |
| `bind_host` | Host to bind agent server (use `0.0.0.0` for network access) |
| `bind_port` | Port for agent API (default: 9001) |
| `ollama_base_url` | URL of local Ollama instance |

Central model policy (`central/config/model_policy.yaml`) is the single source of truth for required models; agent.yaml only defines networking and local endpoints.

### Central Configuration

Central reads `central/config/machines.yaml` to know which agents to connect to (see
`docs/machines_yaml.md` for optional hardware overrides):

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

## Service Control

bench-race provides a unified CLI (`bin/control`) for managing all services, plus compatibility wrapper scripts.

### Unified Control CLI

The primary way to manage services:

```bash
# Agent control
bin/control agent start          # Start agent in daemon mode
bin/control agent start --fg     # Start agent in foreground (for debugging)
bin/control agent stop           # Stop agent
bin/control agent status         # Show agent status
bin/control agent status --json  # Show agent status as JSON

# Central control
bin/control central start        # Start central in daemon mode
bin/control central start --fg   # Start central in foreground
bin/control central stop         # Stop central
bin/control central status       # Show central status
```

**Exit codes:**
- `0` - Success (or running for status)
- `1` - Failure (or not running for status)
- `2` - Invalid arguments

### Legacy Scripts (Compatibility)

The original scripts remain functional as thin wrappers:

```bash
# Agent management (wraps bin/control)
./scripts/agents start [--daemon]
./scripts/agents stop
./scripts/agents status
./scripts/agents restart
./scripts/agents logs

# Central management
./scripts/central start [--daemon]
./scripts/central stop
./scripts/central status
./scripts/central restart
./scripts/central logs

# Individual wrapper scripts
./scripts/start_agent
./scripts/stop_agent
./scripts/status_agent
./scripts/start_central
./scripts/stop_central
./scripts/status_central
```

### Web UI Controls

The web UI at http://127.0.0.1:8080 includes service control cards for Agent and Central:
- Green/gray status indicator
- Start/Stop buttons
- PID and connection info

### Service Control API

Backend endpoints for programmatic control:

```bash
# Get status
curl http://127.0.0.1:8080/api/service/agent/status
curl http://127.0.0.1:8080/api/service/central/status

# Start/stop (local requests only, or with token)
curl -X POST http://127.0.0.1:8080/api/service/agent/start
curl -X POST http://127.0.0.1:8080/api/service/agent/stop
```

**Response format:**
```json
{
  "component": "agent",
  "running": true,
  "pid": 12345,
  "info": "listening on 127.0.0.1:9001"
}
```

**Security:** Service control endpoints require either:
- Local access (127.0.0.1, localhost)
- Valid token: `Authorization: Bearer <token>` (set via `SERVICE_CONTROL_TOKEN` env var)

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_HOST` | `0.0.0.0` | Agent bind host |
| `AGENT_PORT` | `9001` | Agent bind port |
| `OLLAMA_HOST` | `127.0.0.1` | Ollama host |
| `OLLAMA_PORT` | `11434` | Ollama port |
| `CENTRAL_HOST` | `0.0.0.0` | Central bind host |
| `CENTRAL_PORT` | `8080` | Central bind port |
| `SERVICE_CONTROL_TOKEN` | (none) | Optional auth token for remote API access |

### systemd Deployment

For production deployment, use the provided systemd units:

```bash
# Copy unit files
sudo cp deploy/bench-agent.service /etc/systemd/system/
sudo cp deploy/bench-central.service /etc/systemd/system/

# Create bench user (optional)
sudo useradd -r -s /bin/false bench

# Adjust paths in unit files (default: /opt/bench-race)
sudo vim /etc/systemd/system/bench-agent.service

# Enable and start services
sudo systemctl daemon-reload
sudo systemctl enable bench-agent bench-central
sudo systemctl start bench-agent bench-central

# Check status
sudo systemctl status bench-agent bench-central
journalctl -u bench-agent -f
```

### Docker Deployment

For containerized deployment:

```bash
# Build images
docker build -f deploy/Dockerfile.agent -t bench-race-agent .
docker build -f deploy/Dockerfile.central -t bench-race-central .

# Run with docker-compose
cd deploy
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

See `deploy/docker-compose.yml` and `deploy/docker-compose.override.yml` for configuration options.

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
  "llm_models": [],
  "whisper_models": [],
  "sdxl_profiles": [],
  "ollama_reachable": true,
  "ollama_models": ["llama3.1:8b-instruct-q8_0"]
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

## Compute Benchmarks

Compute mode counts primes ≤ N to measure raw CPU throughput per agent. The UI streams the first *K* primes for visual feedback, then switches to periodic progress updates to avoid flooding the output panel.

**Recommended demo presets (2–30s runtimes on typical hardware):**
- Segmented Sieve: `N = 50_000_000`
- Simple Sieve: `N = 20_000_000`
- Trial Division: `N = 2_000_000`

**Streaming behavior:**
- The first K primes (default 100) are emitted as `prime[1]=2`, `prime[2]=3`, ...
- After K, the agent emits progress lines at the configured interval (default 1s).
- A final summary reports the total primes, elapsed time, and primes/sec.

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

### Why Did I Get Mock?

If benchmark results show `engine: mock` instead of `engine: ollama`, the agent fell back to mock streaming. The UI displays a warning badge with the reason:

| Fallback Reason | Meaning | Solution |
|-----------------|---------|----------|
| `ollama_unreachable` | Agent cannot connect to Ollama API | Start Ollama: `ollama serve` |
| `missing_model` | Selected model not installed on Ollama | Run: `ollama pull <model>` or use `pull_models.sh` |
| `stream_error` | Error occurred during Ollama streaming | Check Ollama logs, restart Ollama |

**Preflight Checks**: The UI performs preflight validation before starting runs:
- Shows status pills on each machine card: ✓ Ready, ⚠ Missing model, ✗ Agent/Ollama unreachable
- Blocks machines that aren't ready (configurable with "Run only ready machines" toggle)
- Shows summary banner listing blocked machines and reasons

---

## Model Sync via central policy + pull_models.sh

The `pull_models.sh` script reads `central/config/model_policy.yaml` and pulls the required models on any machine.

### Usage

```bash
# Pull all required models from the central policy (default)
./scripts/pull_models.sh

# Skip large 70B models
./scripts/pull_models.sh --skip-70b

# Use a custom policy path
MODEL_POLICY=/path/to/model_policy.yaml ./scripts/pull_models.sh
./scripts/pull_models.sh --policy /path/to/model_policy.yaml

# Legacy mode: use hardcoded model list
./scripts/pull_models.sh --legacy
```

### How It Works

1. Reads `central/config/model_policy.yaml` (or path in `MODEL_POLICY`/`--policy`)
2. Extracts required model lists:
   - `required.llm` → Pulled via `ollama pull`
   - `required.whisper` → Logged (not yet implemented)
   - `required.sdxl_profiles` → Logged (not yet implemented)
3. Shows planned pulls, executes them, and prints summary

### Example model policy

```yaml
required:
  llm:
    - "llama3.1:8b-instruct-q8_0"
    - "llama3.1:70b-instruct-q4_K_M"
```

### Multi-Machine Sync

To sync models across multiple machines:

1. Update the required model lists in `central/config/model_policy.yaml`
2. SSH to each machine and run:
   ```bash
   cd /path/to/bench-race
   ./scripts/pull_models.sh
   ```

The script is idempotent - running it multiple times is safe and will only download missing models.

### Script Output

```
==============================================
          PLANNED MODEL PULLS
==============================================

LLM Models (via Ollama):
  - llama3.1:8b-instruct-q8_0
  - llama3.1:70b-instruct-q4_K_M

Whisper Models:
  (none configured)

SDXL Profiles:
  (none configured)

==============================================

[pull_models] Pulling 2 LLM model(s)...
[pull_models] Pulling LLM model: llama3.1:8b-instruct-q8_0
...

==============================================
               SUMMARY
==============================================

LLM Models:
  Successfully pulled: 2
  Failed: 0
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

# Run all unit tests
pytest tests/ -v

# Run service control CLI tests
pytest tests/test_control_cli.py -v

# Run integration smoke tests
bash tests/integration/test_smoke_service_control.sh

# Type checking (if configured)
mypy agent/ central/
```

#### Service Control Test Coverage

The test suite includes:

**Unit tests** (`tests/test_control_cli.py`):
- `test_start_when_not_running_starts_process` - Verifies start creates PID file
- `test_start_when_already_running_is_idempotent` - Safe to start twice
- `test_status_reports_running_and_pid` - Status shows correct info
- `test_stop_terminates_process_and_cleans_pid` - Stop cleans up properly

**Integration smoke test** (`tests/integration/test_smoke_service_control.sh`):
- Full start → status → stop cycle for agent and central
- Port listening verification
- API endpoint validation

#### Manual Testing Checklist

- [ ] `bin/control agent start` starts the agent, `status` shows running + PID
- [ ] `bin/control agent stop` stops the agent, `status` shows stopped
- [ ] `scripts/start_agent` / `scripts/stop_agent` work as wrappers
- [ ] UI shows correct status (green dot = running, gray = stopped)
- [ ] UI Start/Stop buttons function correctly
- [ ] API endpoints return proper JSON and HTTP status codes

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

## Agent Reset & Diagnostics

The system includes robust agent reset functionality with detailed diagnostics for troubleshooting Ollama and ComfyUI service issues.

### Reset Agent via UI

Click the **Reset** button on any agent card to restart both Ollama and ComfyUI services. If the reset fails, a diagnostics modal will display:
- Detailed notes about what went wrong
- Full stdout/stderr logs from service startup
- Log file paths for deeper investigation
- Time-to-ready metrics for each service
- Full JSON response for debugging

### Reset Agent via API

**From central server:**
```bash
# Reset an agent through the central proxy
curl -X POST http://localhost:8080/api/agents/macbook/reset
```

**Directly on agent:**
```bash
# Reset agent services directly
curl -X POST http://localhost:9001/api/reset
```

**Example success response:**
```json
{
  "ok": true,
  "duration_ms": 45320,
  "ollama": {
    "stopped": true,
    "start_command": ["brew", "services", "start", "ollama"],
    "start_stdout_tail": "...",
    "start_stderr_tail": "",
    "start_log_file": "/path/to/logs/ollama_start_2026-02-05T16-53-36.log",
    "healthy": true,
    "time_to_ready_ms": 3450
  },
  "comfyui": {
    "stopped": true,
    "pid": 12345,
    "start_log_file": "/path/to/logs/comfyui_start_2026-02-05T16-53-41.log",
    "healthy": true,
    "time_to_ready_ms": 18200
  },
  "notes": []
}
```

**Example failure response:**
```json
{
  "ok": false,
  "duration_ms": 122450,
  "ollama": {
    "stopped": true,
    "start_command": ["systemctl", "start", "ollama"],
    "start_stdout_tail": "...",
    "start_stderr_tail": "Error: service failed to start...",
    "start_log_file": "/path/to/logs/ollama_start_2026-02-05T17-02-15.log",
    "healthy": false,
    "time_to_ready_ms": null
  },
  "comfyui": {...},
  "notes": ["Ollama unhealthy after 120s"]
}
```

### Image Benchmark Notes

- The `checkpoint` query parameter uses the checkpoint filename (e.g. `sd_xl_base_1.0.safetensors`). Digests are accepted for backwards compatibility and return a clear error if unknown.
- Image completion is determined via ComfyUI `/history/{prompt_id}` polling; websocket updates are best-effort for progress only.
- If you see sudo errors during agent resets, re-run `scripts/install_agent.sh --install-sudoers` to install the passwordless sudo drop-in.

### Reset Configuration

Configure reset timeouts and logging via environment variables on the **agent**:

```bash
# Agent environment variables (agent/.env or systemd service config)
OLLAMA_START_TIMEOUT_S=120        # Max time to wait for Ollama startup (default: 120)
COMFYUI_START_TIMEOUT_S=60        # Max time to wait for ComfyUI startup (default: 60)
HEALTH_POLL_INTERVAL_S=1.5        # Interval between health checks (default: 1.5)
RESET_LOG_DIR=./logs              # Directory for reset log files (default: ./logs)
```

### Troubleshooting Reset Failures

1. **Check the diagnostics modal** in the UI for detailed error messages
2. **Review log files** at the paths shown in the diagnostics (on the agent machine)
3. **Inspect stdout/stderr tails** for immediate context
4. **Verify service configuration**:
   ```bash
   # macOS
   brew services list | grep ollama

   # Linux
   systemctl status ollama
   systemctl --user status ollama
   ```
5. **Check health endpoints manually**:
   ```bash
   # Ollama health (used by reset)
   curl http://127.0.0.1:11434/api/tags

   # ComfyUI health
   curl http://127.0.0.1:8188/system_stats
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

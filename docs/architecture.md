# Architecture Overview

This document describes the internal architecture of bench-race, including
how the components fit together, the request/event flow, and the key design
decisions.

---

## Table of Contents

- [System Components](#system-components)
- [Agent Architecture](#agent-architecture)
- [Central Architecture](#central-architecture)
- [Frontend Architecture](#frontend-architecture)
- [Communication Flow](#communication-flow)
- [LLM Inference Flow](#llm-inference-flow)
- [Compute Benchmark Flow](#compute-benchmark-flow)
- [Image Generation Flow](#image-generation-flow)
- [Results Persistence](#results-persistence)
- [Service Lifecycle Management](#service-lifecycle-management)
- [Error Handling](#error-handling)
- [Project Directory Structure](#project-directory-structure)

---

## System Components

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ index.html│  │compute.html│ │image.html│  │admin.html│   │
│  │ (Inference)│ │ (Compute) │ │ (Image)  │  │ (Admin)  │   │
│  └─────┬─────┘  └─────┬─────┘ └─────┬────┘  └─────┬────┘  │
│        └───────────────┼─────────────┼─────────────┘        │
│                        │ HTTP + Socket.IO                    │
└────────────────────────┼────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            │     Central Server      │
            │   Flask + Flask-SocketIO │
            │   Port 8080             │
            │                         │
            │  - Route handlers       │
            │  - Run lifecycle mgmt   │
            │  - Model fit scoring    │
            │  - Result persistence   │
            │  - Service control API  │
            └───────────┬─────────────┘
                        │  HTTP REST + WebSocket
         ┌──────────────┼──────────────┐
         │              │              │
   ┌─────┴──────┐ ┌────┴───────┐ ┌────┴───────┐
   │  Agent 1   │ │  Agent 2   │ │  Agent 3   │
   │  FastAPI   │ │  FastAPI   │ │  FastAPI   │
   │  Port 9001 │ │  Port 9001 │ │  Port 9001 │
   │            │ │            │ │            │
   │ Backends:  │ │ Backends:  │ │ Backends:  │
   │ - Ollama   │ │ - Ollama   │ │ - Ollama   │
   │ - ComfyUI  │ │ - ComfyUI  │ │ - ComfyUI  │
   │ - Compute  │ │ - Compute  │ │ - Compute  │
   └────────────┘ └────────────┘ └────────────┘
```

---

## Agent Architecture

The agent (`agent/agent_app.py`) is a FastAPI application running on each
benchmarked machine.

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| Main app | `agent/agent_app.py` | FastAPI routes, job runners, WebSocket hub |
| Ollama backend | `backends/ollama_backend.py` | Streaming LLM generation via Ollama API |
| ComfyUI backend | `agent/backends/comfyui_backend.py` | Image generation via ComfyUI |
| ComfyUI WebSocket | `agent/comfy_ws.py` | WebSocket communication with ComfyUI |
| Hardware discovery | `agent/hardware_discovery.py` | GPU/CPU detection at startup |
| Runtime sampler | `agent/runtime_sampler.py` | Periodic CPU/GPU/memory metrics |
| Startup checks | `agent/startup_checks.py` | CUDA probe for ComfyUI compatibility |
| Reset helpers | `agent/reset_helpers.py` | Ollama/ComfyUI service restart |
| Error classifier | `agent/errors.py` | ComfyUI error categorization |
| Logging | `agent/logging_utils.py` | Structured JSON logging |
| HTTP middleware | `agent/http_logging_asgi.py` | Request logging middleware |
| Schemas | `shared/schemas.py` | Pydantic models for API contracts |

### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check (always returns 200) |
| GET | `/capabilities` | Available models, hardware info, fit data |
| POST | `/jobs` | Start an LLM inference job |
| POST | `/api/compute` | Start a compute benchmark |
| POST | `/api/image/txt2img` | Start image generation |
| POST | `/api/reset` | Restart Ollama and ComfyUI |
| POST | `/api/comfy/sync_checkpoints` | Download ComfyUI checkpoints |
| POST | `/models/sync` | Sync required models |
| GET | `/api/agent/runtime_metrics` | Current hardware metrics |
| WebSocket | `/ws` | Real-time event streaming |

### Job Execution Model

All benchmark jobs follow the same pattern:

1. HTTP POST creates a `job_id` and launches an async task.
2. The async task executes the workload (Ollama, compute, or ComfyUI).
3. Progress events are broadcast to all connected WebSocket clients.
4. A final `*_done` event carries the complete metrics.
5. The HTTP response returns the `job_id` (for LLM jobs) or the full
   result (for compute jobs).

### WebSocket Hub

The agent maintains a set of connected WebSocket clients. Every event
(token, progress, completion) is broadcast to all connected clients:

```python
WS_CLIENTS: Dict[str, WebSocket] = {}

async def _broadcast_event(event: Event):
    for client_id, ws in WS_CLIENTS.items():
        await ws.send_json(event.dict())
```

Central connects as a WebSocket client to each agent and relays events
to the browser via Socket.IO.

---

## Central Architecture

The central server (`central/app.py`) is a Flask application with
Flask-SocketIO for real-time communication with browsers.

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| Main app | `central/app.py` | Flask routes, SocketIO events, run management |
| Model fit scoring | `central/fit_util.py` | Memory-first fit heuristic |
| Service controller | `central/services/controller.py` | Wrapper for bin/control CLI |
| HTML templates | `central/templates/` | Jinja2 templates for web UI |
| JavaScript | `central/static/js/` | Frontend logic |
| CSS | `central/static/css/app.css` | Styling |

### Web Routes

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Inference UI |
| GET | `/compute` | Compute benchmark UI |
| GET | `/image` | Image generation UI |
| GET | `/admin` | Admin panel |
| POST | `/api/jobs` | Start benchmark run across agents |
| GET | `/api/runs` | List past runs |
| GET | `/api/runs/<id>` | Get run details |
| POST | `/api/runs/<id>/download` | Export run as CSV |
| POST | `/api/agents/<id>/reset` | Proxy reset to agent |
| GET/POST | `/api/service/<component>/<action>` | Service control |

### Run Lifecycle

A "run" is a coordinated benchmark across multiple agents:

```
1. User clicks "Run" in UI
2. Central creates a run_id
3. For each machine in machines.yaml:
   a. POST to agent's /jobs endpoint
   b. Map agent's job_id to the run_id
4. Emit "run_start" via SocketIO to UI
5. Agents stream events → Central relays to UI
6. When all agents complete:
   a. Emit "run_end" via SocketIO
   b. Save results to disk
```

---

## Frontend Architecture

The frontend is vanilla JavaScript (no framework) with HTML templates
rendered by Flask's Jinja2 engine.

### Pages

| Page | Template | JavaScript | Purpose |
|------|----------|-----------|---------|
| Inference | `index.html` | `app.js` | LLM benchmarking |
| Compute | `compute.html` | (inline + app.js) | CPU prime benchmarks |
| Image | `image.html` | `image.js` | ComfyUI image generation |
| Admin | `admin.html` | `service-control.js` | Service management |

### UI Layout (Inference Page)

```
┌──────────────────────────────────────────────────────────┐
│  Header: "Benchmark Race" | Compute | Inference | Image  │
├──────────────────────────────────────────────────────────┤
│  Controls:                                               │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Model: [llama3.1:8b-instruct-q8_0 ▼]               │ │
│  │ Prompt: [Enter your prompt here...              ]   │ │
│  │ Max tokens: [256]  Context: [4096]  Temp: [0.2]     │ │
│  │ Repeats: [1]                                        │ │
│  │ [Run (LLM)]                                         │ │
│  └─────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────┤
│  Machine Cards (responsive grid):                        │
│  ┌────────────────┐  ┌────────────────┐                  │
│  │ MacBook Pro     │  │ Mac Mini       │                 │
│  │ FIT: GOOD       │  │ FIT: RISK      │                 │
│  │                 │  │                │                  │
│  │ [streaming      │  │ [streaming     │                 │
│  │  tokens...]     │  │  tokens...]    │                 │
│  │                 │  │                │                  │
│  │ TTFT: 89ms      │  │ TTFT: 112ms   │                 │
│  │ Rate: 45.3 t/s  │  │ Rate: 38.7 t/s│                 │
│  │ Total: 3.4s     │  │ Total: 4.0s   │                 │
│  └────────────────┘  └────────────────┘                  │
├──────────────────────────────────────────────────────────┤
│  Summary: Comparative metrics table                      │
└──────────────────────────────────────────────────────────┘
```

### Real-Time Event Handling

The frontend connects to central via Socket.IO and handles events:

```javascript
socket.on('message', (event) => {
    switch (event.type) {
        case 'llm_token':
            appendTokenToMachineCard(event.machine_id, event.payload.text);
            break;
        case 'job_done':
            showMetrics(event.machine_id, event.payload);
            break;
        case 'compute_line':
            appendLineToOutput(event.machine_id, event.payload.line);
            break;
        case 'compute_done':
            showComputeResults(event.machine_id, event.payload);
            break;
        case 'run_lifecycle':
            handleRunLifecycle(event.payload);
            break;
    }
});
```

---

## Communication Flow

### REST API Pattern (Request/Response)

```
Browser  →  POST /api/jobs  →  Central  →  POST /jobs  →  Agent
Browser  ←  {run_id}        ←  Central  ←  {job_id}    ←  Agent
```

### WebSocket Event Pattern (Streaming)

```
Agent  →  ws://central:8080  →  Central  →  SocketIO  →  Browser
  [llm_token events]           [relay]     [broadcast]   [display]
```

### Combined Flow for a Benchmark Run

```
1. Browser → Central: POST /api/jobs {model, prompt, ...}
2. Central → Agent A: POST /jobs {model, prompt, ...}
   Central → Agent B: POST /jobs {model, prompt, ...}
3. Agent A → Central (WS): {type: "job_start", job_id: "..."}
4. Agent A → Central (WS): {type: "llm_token", payload: {text: "The"}}
   Agent B → Central (WS): {type: "llm_token", payload: {text: "The"}}
5. Central → Browser (SocketIO): relay each token to appropriate machine card
6. Agent A → Central (WS): {type: "job_done", payload: {ttft_ms: 89, ...}}
   Agent B → Central (WS): {type: "job_done", payload: {ttft_ms: 112, ...}}
7. Central → Browser (SocketIO): {type: "run_lifecycle", payload: {type: "run_end"}}
8. Central: Save results to disk
```

---

## LLM Inference Flow

Detailed flow for a single LLM inference job on one agent:

```
POST /jobs
  │
  ├─ Validate request (model, prompt, parameters)
  ├─ Generate job_id (UUID)
  ├─ Launch asyncio.Task(_job_runner_llm)
  └─ Return {job_id: "..."}
       │
       └─ _job_runner_llm() runs in background:
            │
            ├─ Broadcast: {type: "job_start"}
            ├─ Check Ollama health
            │   ├─ Reachable? → Continue with Ollama
            │   └─ Unreachable? → Fall back to mock streaming
            │
            ├─ Call stream_ollama_generate():
            │   ├─ POST to Ollama /api/generate (streaming)
            │   ├─ Read NDJSON response lines
            │   ├─ Buffer tokens (64 chars or 0.1s)
            │   ├─ On flush → callback → Broadcast {type: "llm_token"}
            │   ├─ Track: t_first (TTFT), gen_tokens, elapsed
            │   └─ On "done" marker → break
            │
            ├─ Compute final metrics:
            │   ├─ ttft_ms = (t_first - t_start) * 1000
            │   ├─ gen_tokens_per_s = tokens / elapsed
            │   └─ total_ms = (t_end - t_start) * 1000
            │
            └─ Broadcast: {type: "job_done", payload: {metrics}}
```

### Token Buffering

Ollama streams individual tokens (often single words or punctuation). Rather
than sending a WebSocket message for each tiny fragment, the agent buffers
tokens and flushes when:

- Buffer reaches 64 characters (`BENCH_STREAM_FLUSH_CHARS`)
- Buffer contains a newline
- 0.1 seconds have elapsed since last flush (`BENCH_STREAM_FLUSH_SECONDS`)
- The stream completes

This reduces WebSocket overhead and produces smoother UI rendering.

---

## Compute Benchmark Flow

```
POST /api/compute
  │
  ├─ Parse request (algorithm, n, threads, progress_interval_s)
  ├─ Generate job_id
  ├─ Broadcast: {type: "compute_line", line: "Compute: Count primes <= N"}
  │
  ├─ Select algorithm:
  │   ├─ "segmented_sieve" → _run_segmented_sieve()
  │   ├─ "simple_sieve"    → _run_simple_sieve()
  │   └─ "trial_division"  → _run_trial_division()
  │
  ├─ During execution:
  │   └─ Broadcast progress every interval:
  │      {type: "compute_line", line: "Progress: 45% | primes: 1.4M | elapsed: 6.3s"}
  │
  ├─ On completion:
  │   ├─ Broadcast summary lines (algorithm, N, primes, rate, elapsed)
  │   └─ Broadcast: {type: "compute_done", payload: {ok, algorithm, n, primes_found, ...}}
  │
  └─ Return result as HTTP response
```

See `docs/prime-algorithms.md` for algorithm details.

---

## Image Generation Flow

```
POST /api/image/txt2img
  │
  ├─ Validate checkpoint availability
  ├─ Generate job_id
  ├─ Build ComfyUI workflow JSON
  │
  ├─ POST workflow to ComfyUI /prompt
  │   └─ Get prompt_id
  │
  ├─ Poll /history/{prompt_id} for completion:
  │   ├─ Check for actual image outputs (not just "completed" flag)
  │   ├─ Broadcast progress via WebSocket (best-effort)
  │   └─ Timeout after configured duration → broadcast job_timeout
  │
  ├─ On completion:
  │   ├─ Extract images from ComfyUI output
  │   ├─ Compute metrics (wall_seconds, seconds_per_image)
  │   └─ Broadcast: {type: "image_done", payload: {images, metrics}}
  │
  └─ Return result
```

---

## Results Persistence

Completed runs are saved to `results/runs/<run_id>/`:

```
results/runs/2024-01-15_143022_abc123/
  ├─ metadata.json      # Run configuration and settings
  ├─ events.ndjson      # All streamed events (one JSON per line)
  └─ summary.json       # Final metrics per machine
```

### metadata.json

```json
{
    "run_id": "2024-01-15_143022_abc123",
    "started_at": "2024-01-15T14:30:22Z",
    "prompt": "Explain quantum computing.",
    "model": "llama3.1:8b-instruct-q8_0",
    "machines": ["macbook", "macmini"],
    "settings": {"max_tokens": 256, "temperature": 0.2, "num_ctx": 4096}
}
```

### summary.json

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

## Service Lifecycle Management

### Unified Control CLI

All services are managed through `bin/control`:

```
bin/control <component> <action> [--foreground] [--json]
```

Internally, the control CLI:

1. **start**: Activates the Python venv, launches the process, writes a PID
   file to `run/pids/<component>.pid`.
2. **stop**: Reads the PID file, sends SIGTERM, waits for exit, removes PID
   file.
3. **status**: Reads the PID file, checks if the process is running, returns
   status as text or JSON.

### Service Dependencies

```
Central depends on:
  └─ Agents (HTTP connectivity to agent URLs)
      └─ Ollama (local to each agent, for LLM workloads)
      └─ ComfyUI (local to each agent, for image workloads)
```

Start order: Ollama/ComfyUI first, then agents, then central.

---

## Error Handling

### Ollama Fallback

If Ollama is unreachable or a model isn't installed, the agent falls back
to mock streaming. The UI displays an `engine: mock` badge so users know
real inference didn't occur.

Fallback reasons:
- `ollama_unreachable`: Cannot connect to Ollama API
- `missing_model`: Selected model not installed
- `stream_error`: Error during Ollama streaming

### ComfyUI Error Classification

The error classifier (`agent/errors.py`) categorizes ComfyUI failures:

| Category | Trigger | Action |
|----------|---------|--------|
| `cuda_unsupported_arch` | "no kernel image is available" | Suggest PyTorch reinstall; optional CPU fallback |
| `oom` | Out of memory errors | Suggest smaller model or resolution |
| `missing_checkpoint` | Checkpoint file not found | Suggest sync_checkpoints |
| `unknown` | Any other error | Display raw error message |

### Preflight Validation

Before starting a run, the UI performs preflight checks:
- Agent reachability (HTTP health check)
- Model availability (agent capabilities)
- Memory fit scoring (GOOD/RISK/FAIL badges)
- Missing model warnings

---

## Project Directory Structure

```
bench-race/
├── agent/                        # Agent service (runs on each benchmarked machine)
│   ├── agent_app.py              # FastAPI application (main)
│   ├── backends/                 # Backend integrations
│   │   ├── ollama_backend.py     # Agent-side Ollama wrapper
│   │   ├── comfyui_backend.py    # ComfyUI integration
│   │   └── whisper_backend.py    # Whisper integration (stub)
│   ├── config/
│   │   ├── agent.yaml            # Per-machine config (gitignored)
│   │   └── agent.example.yaml    # Config template
│   ├── tests/                    # Agent-specific tests
│   ├── hardware_discovery.py     # GPU/CPU auto-detection
│   ├── runtime_sampler.py        # Hardware metrics collection
│   ├── startup_checks.py         # CUDA compatibility probe
│   ├── reset_helpers.py          # Service restart utilities
│   ├── errors.py                 # Error classification
│   ├── comfy_ws.py               # ComfyUI WebSocket client
│   ├── http_client.py            # HTTP client utilities
│   ├── middleware.py              # ASGI middleware
│   ├── http_logging_asgi.py      # Request logging middleware
│   ├── logging_utils.py          # Structured JSON logging
│   └── requirements.txt          # Python dependencies
│
├── central/                      # Central coordination server
│   ├── app.py                    # Flask + SocketIO application (main)
│   ├── fit_util.py               # Model fit scoring algorithm
│   ├── services/
│   │   └── controller.py         # Programmatic service control
│   ├── config/
│   │   ├── machines.yaml         # Machine registry (gitignored)
│   │   ├── machines.example.yaml # Machine registry template
│   │   ├── model_policy.example.yaml
│   │   └── comfyui.example.yaml
│   ├── templates/                # HTML pages
│   │   ├── index.html            # Inference UI
│   │   ├── compute.html          # Compute benchmark UI
│   │   ├── image.html            # Image generation UI
│   │   └── admin.html            # Admin panel
│   ├── static/
│   │   ├── css/app.css           # Styling
│   │   └── js/
│   │       ├── app.js            # Main UI logic
│   │       ├── image.js          # Image UI logic
│   │       ├── service-control.js # Service control UI
│   │       └── image_sparkline_utils.js
│   └── requirements.txt          # Python dependencies
│
├── backends/                     # Shared backend library
│   └── ollama_backend.py         # Ollama API wrapper (shared)
│
├── shared/                       # Shared data models
│   └── schemas.py                # Pydantic models
│
├── bin/
│   └── control                   # Unified service control CLI
│
├── scripts/                      # Installation and management
│   ├── install_agent.sh          # Agent installer
│   ├── install_central.sh        # Central installer
│   ├── agents                    # Agent management wrapper
│   ├── central                   # Central management wrapper
│   ├── pull_models.sh            # Model sync script
│   ├── setup_venv_agent.sh       # Agent venv setup
│   ├── setup_venv_central.sh     # Central venv setup
│   ├── _common.sh                # Shared shell utilities
│   ├── _python_pick.sh           # Python version detection
│   └── _read_agent_config.py     # Config parser
│
├── deploy/                       # Deployment configurations
│   ├── docker-compose.yml        # Docker Compose
│   ├── docker-compose.override.yml # Dev overrides
│   ├── Dockerfile.agent          # Agent container
│   ├── Dockerfile.central        # Central container
│   ├── bench-agent.service       # systemd unit (agent)
│   └── bench-central.service     # systemd unit (central)
│
├── tests/                        # Test suite
│   ├── test_control_cli.py       # Service control tests
│   ├── test_fit_util.py          # Fit scoring tests
│   ├── test_*.py                 # Other test files
│   └── integration/              # Integration tests
│
├── docs/                         # Documentation
│   ├── architecture.md           # This file
│   ├── configuration-guide.md    # Configuration reference
│   ├── gpu-benchmarking-guide.md # GPU and hardware guide
│   ├── prime-algorithms.md       # Prime algorithm deep-dive
│   ├── machines_yaml.md          # machines.yaml field reference
│   └── metrics_and_fit.md        # Metrics and fit scoring
│
├── results/runs/                 # Benchmark results (gitignored)
├── logs/                         # Log files (gitignored)
├── run/pids/                     # PID files (gitignored)
│
├── README.md                     # Main documentation
├── CHANGELOG.md                  # Version history
├── pyproject.toml                # Black/Ruff config
├── pytest.ini                    # Pytest config
└── .github/workflows/            # CI/CD
    └── installer-checks.yml      # ShellCheck + installer validation
```

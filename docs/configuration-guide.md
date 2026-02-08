# Configuration Guide

This document is the complete reference for configuring bench-race, including
network setup, IP addresses, all configuration files, and environment variables.

---

## Table of Contents

- [Network Architecture](#network-architecture)
- [IP Address Setup](#ip-address-setup)
- [Configuration Files Overview](#configuration-files-overview)
- [Agent Configuration (agent.yaml)](#agent-configuration-agentyaml)
- [Central Machine Registry (machines.yaml)](#central-machine-registry-machinesyaml)
- [Model Policy (model_policy.yaml)](#model-policy-model_policyyaml)
- [ComfyUI Configuration (comfyui.yaml)](#comfyui-configuration-comfyuiyaml)
- [Environment Variables](#environment-variables)
- [Port Reference](#port-reference)
- [Example: Two-Machine Setup](#example-two-machine-setup)
- [Example: Four-Machine Lab Setup](#example-four-machine-lab-setup)
- [Firewall and Security](#firewall-and-security)
- [Suggested Screenshots](#suggested-screenshots)

---

## Network Architecture

bench-race uses a hub-and-spoke architecture:

```
                    ┌──────────────────┐
                    │     Central      │
                    │  (Flask + SocketIO)  │
                    │  :8080 (Web UI)  │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────┴─────┐ ┌─────┴─────┐ ┌─────┴─────┐
        │  Agent 1   │ │  Agent 2   │ │  Agent 3   │
        │ (FastAPI)  │ │ (FastAPI)  │ │ (FastAPI)  │
        │ :9001      │ │ :9001      │ │ :9001      │
        └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
              │              │              │
        ┌─────┴──────┐ ┌────┴───────┐ ┌────┴───────┐
        │  Ollama    │ │  Ollama    │ │  Ollama    │
        │  :11434    │ │  :11434    │ │  :11434    │
        └────────────┘ └────────────┘ └────────────┘
```

**Communication paths:**
- **Browser -> Central**: HTTP + WebSocket on port 8080
- **Central -> Agents**: HTTP REST calls to agent URLs in machines.yaml
- **Agents -> Central**: WebSocket connection for event streaming
- **Agents -> Ollama**: HTTP to local Ollama API on port 11434
- **Agents -> ComfyUI**: HTTP + WebSocket to local ComfyUI on port 8188

---

## IP Address Setup

### Single-Machine Development

For development on a single machine, all services use `127.0.0.1` (localhost):

```
Central:  http://127.0.0.1:8080
Agent:    http://127.0.0.1:9001
Ollama:   http://127.0.0.1:11434
ComfyUI:  http://127.0.0.1:8188
```

No special network configuration is needed.

### Multi-Machine Setup

For benchmarking across multiple machines, each agent must be reachable
from the central server over the network.

**Step 1: Determine each machine's IP address**

```bash
# macOS
ipconfig getifaddr en0    # Wi-Fi
ipconfig getifaddr en1    # Ethernet

# Linux
ip addr show | grep "inet " | grep -v 127.0.0.1
hostname -I
```

**Step 2: Configure agents to bind to all interfaces**

In each agent's `agent/config/agent.yaml`:

```yaml
bind_host: "0.0.0.0"    # Listen on all interfaces (required for network access)
bind_port: 9001          # Default agent port
```

`bind_host: "0.0.0.0"` means the agent accepts connections from any
network interface. If you use `"127.0.0.1"`, the agent will only accept
local connections and other machines won't be able to reach it.

**Step 3: Register agents in central's machines.yaml**

On the central machine, edit `central/config/machines.yaml` with each agent's
IP address:

```yaml
machines:
  - machine_id: "macbook-pro"
    label: "MacBook Pro (M4 Max, 128GB)"
    agent_base_url: "http://192.168.1.10:9001"   # <- Agent's LAN IP

  - machine_id: "mac-mini"
    label: "Mac Mini (M4 Pro, 24GB)"
    agent_base_url: "http://192.168.1.20:9001"   # <- Agent's LAN IP

  - machine_id: "gb10-server"
    label: "Dell ProMax (GB10, 128GB)"
    agent_base_url: "http://192.168.1.30:9001"   # <- Agent's LAN IP
```

**Step 4: Configure each agent's central URL**

In each agent's `agent/config/agent.yaml`:

```yaml
central_base_url: "http://192.168.1.10:8080"  # <- Central machine's LAN IP
```

**Step 5: Verify connectivity**

From the central machine:

```bash
# Test each agent
curl http://192.168.1.10:9001/health
curl http://192.168.1.20:9001/health
curl http://192.168.1.30:9001/health
```

Each should return:
```json
{"ok": true, "machine_id": "...", "label": "..."}
```

### Using Hostnames Instead of IPs

If your network has working DNS or mDNS (Bonjour), you can use hostnames:

```yaml
# machines.yaml with hostnames
machines:
  - machine_id: "macbook-pro"
    agent_base_url: "http://macbook-pro.local:9001"

  - machine_id: "gb10-server"
    agent_base_url: "http://gb10-server.local:9001"
```

Hostnames are more readable but can cause issues if DNS is unreliable.
IP addresses are more predictable for benchmarking.

### Using Non-Default Ports

If you run multiple agents on the same machine (e.g., for testing), use
different ports:

```yaml
# agent.yaml on the first agent
bind_port: 9001

# agent.yaml on the second agent
bind_port: 9002
```

Update machines.yaml to match:

```yaml
machines:
  - machine_id: "agent-1"
    agent_base_url: "http://127.0.0.1:9001"
  - machine_id: "agent-2"
    agent_base_url: "http://127.0.0.1:9002"
```

---

## Configuration Files Overview

| File | Location | Purpose | Gitignored? |
|------|----------|---------|-------------|
| `agent.yaml` | `agent/config/agent.yaml` | Per-machine agent settings | Yes |
| `agent.example.yaml` | `agent/config/agent.example.yaml` | Agent config template | No |
| `machines.yaml` | `central/config/machines.yaml` | Machine registry for central | Yes |
| `machines.example.yaml` | `central/config/machines.example.yaml` | Machines config template | No |
| `model_policy.yaml` | `central/config/model_policy.example.yaml` | Required model list | No |
| `comfyui.yaml` | `central/config/comfyui.example.yaml` | Image generation config | No |

The `.example.yaml` files are committed to the repository. The actual
`.yaml` files (without `.example`) are gitignored because they contain
environment-specific values (IP addresses, hardware details, etc.).

**To set up configuration:**
```bash
# Agent
cp agent/config/agent.example.yaml agent/config/agent.yaml
# Edit agent.yaml for this machine

# Central
cp central/config/machines.example.yaml central/config/machines.yaml
# Edit machines.yaml for your environment
```

The installer scripts (`install_agent.sh`, `install_central.sh`) generate
these files automatically during installation.

---

## Agent Configuration (agent.yaml)

Location: `agent/config/agent.yaml`

This file controls how the agent runs on a particular machine. Every agent
must have its own copy with a unique `machine_id`.

### Full Reference

```yaml
########################################
# Identity
########################################

# Must match a machine_id in central's machines.yaml
machine_id: "example-machine"

# Human-friendly name (displayed in logs and diagnostics)
label: "Example Machine Agent"

########################################
# Agent HTTP server
########################################

# Use "0.0.0.0" for network access, "127.0.0.1" for local-only
bind_host: "0.0.0.0"
bind_port: 9001

########################################
# Central connectivity
########################################

# Base URL where the central server is running
central_base_url: "http://192.168.1.10:8080"

########################################
# Ollama configuration
########################################

ollama:
  enabled: true
  base_url: "http://127.0.0.1:11434"
  health_endpoint: "/api/tags"
  # start_command: ["ollama", "serve"]  # Uncomment for agent-managed Ollama

########################################
# ComfyUI configuration
########################################

comfyui:
  enabled: true
  host: "127.0.0.1"
  port: 8188
  install_dir: "agent/third_party/comfyui"
  cache_path: "agent/model_cache/comfyui"
  checkpoints_dir: "agent/third_party/comfyui/models/checkpoints"
  debug: true
  allow_cpu_fallback: false
  cpu_fallback_on:
    - "cuda_unsupported_arch"

########################################
# Compute benchmark defaults
########################################

compute:
  default_algorithm: "segmented_sieve"
  progress_interval_s: 1.0

########################################
# Runtime metrics sampling
########################################

runtime_sampler:
  enabled: true
  interval_s: 1
  buffer_len: 120

########################################
# Timeouts
########################################

timeouts:
  ollama_start_timeout_s: 120
  comfyui_start_timeout_s: 60

########################################
# Logging
########################################

logging:
  level: "info"
  log_dir: "./logs"
```

### Field-by-Field Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `machine_id` | string | (required) | Unique identifier. Must match central's machines.yaml. |
| `label` | string | (required) | Human-readable name for logs and UI. |
| `bind_host` | string | `"0.0.0.0"` | Network interface to bind. `0.0.0.0` = all interfaces. |
| `bind_port` | int | `9001` | TCP port for the agent API. |
| `central_base_url` | string | `"http://127.0.0.1:8080"` | URL of the central server. |
| `ollama.enabled` | bool | `true` | Enable Ollama LLM backend. |
| `ollama.base_url` | string | `"http://127.0.0.1:11434"` | Local Ollama API URL. |
| `ollama.health_endpoint` | string | `"/api/tags"` | Endpoint used for health checks. |
| `ollama.start_command` | list | (none) | Command to start Ollama (if agent-managed). |
| `comfyui.enabled` | bool | `true` | Enable ComfyUI image generation backend. |
| `comfyui.host` | string | `"127.0.0.1"` | ComfyUI API host. |
| `comfyui.port` | int | `8188` | ComfyUI API port. |
| `comfyui.install_dir` | string | `"agent/third_party/comfyui"` | ComfyUI installation directory. |
| `comfyui.cache_path` | string | `"agent/model_cache/comfyui"` | Checkpoint download cache. |
| `comfyui.checkpoints_dir` | string | (derived from install_dir) | Directory ComfyUI scans for models. |
| `comfyui.debug` | bool | `false` | Verbose logging for prompt failures. |
| `comfyui.allow_cpu_fallback` | bool | `false` | Allow CPU fallback on GPU errors. |
| `comfyui.cpu_fallback_on` | list | `["cuda_unsupported_arch"]` | Error categories that trigger fallback. |
| `compute.default_algorithm` | string | `"segmented_sieve"` | Default prime algorithm. |
| `compute.progress_interval_s` | float | `1.0` | Progress update interval (seconds). |
| `runtime_sampler.enabled` | bool | `true` | Enable hardware metrics sampling. |
| `runtime_sampler.interval_s` | float | `1` | Sampling interval (seconds). |
| `runtime_sampler.buffer_len` | int | `120` | Number of samples to retain. |
| `timeouts.ollama_start_timeout_s` | int | `120` | Max wait for Ollama startup. |
| `timeouts.comfyui_start_timeout_s` | int | `60` | Max wait for ComfyUI startup. |
| `logging.level` | string | `"info"` | Log level: debug, info, warning, error. |
| `logging.log_dir` | string | `"./logs"` | Directory for log files. |

---

## Central Machine Registry (machines.yaml)

Location: `central/config/machines.yaml`

This file tells the central server which agents exist, how to reach them,
and their hardware specifications for fit scoring.

### Full Reference

```yaml
machines:
  # Apple Silicon example
  - machine_id: "macbook"
    label: "MacBook Pro (M4 Max, 16 Cores, 128GB)"
    logo: "apple"                                    # UI vendor logo
    agent_base_url: "http://192.168.1.10:9001"
    notes: "Primary development machine"

    cpu_cores: 16                                    # Logical cores
    cpu_physical_cores: 12                           # Physical cores
    total_system_ram_bytes: 137438953472             # 128 GiB

    gpu:
      name: "Apple M4 Max"
      type: "unified"                                # Shared with CPU
      vram_bytes: 137438953472                       # Same as system RAM

  # NVIDIA discrete GPU example
  - machine_id: "alienware"
    label: "Alienware Area 51 (RTX5090, 32GB)"
    logo: "nvidia"
    agent_base_url: "http://192.168.1.20:9001"

    cpu_cores: 32
    cpu_physical_cores: 16
    total_system_ram_bytes: 68719476736              # 64 GiB

    gpu:
      name: "NVIDIA RTX5090"
      type: "discrete"
      vram_bytes: 34359738368                        # 32 GiB
      cuda_compute: [12, 0]

  # NVIDIA GB10 example
  - machine_id: "gb10"
    label: "Dell ProMax (GB10, 128GB)"
    logo: "nvidia"
    agent_base_url: "http://192.168.1.30:9001"

    cpu_cores: 64
    cpu_physical_cores: 32
    total_system_ram_bytes: 137438953472             # 128 GiB

    gpu:
      name: "NVIDIA GB10"
      type: "discrete"
      vram_bytes: 68719476736                        # 64 GiB
      cuda_compute: [12, 1]
      driver_version: "545.101"
      pci_bus: "0000:01:00.0"
```

### Field-by-Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `machine_id` | string | Yes | Must match the agent's `machine_id` in agent.yaml. |
| `label` | string | Yes | Display name in the UI. |
| `logo` | string | No | Vendor logo: `"apple"` or `"nvidia"`. |
| `agent_base_url` | string | Yes | Full URL to reach the agent (http://IP:port). |
| `notes` | string | No | Freeform notes displayed in admin view. |
| `cpu_cores` | int | No | Logical CPU cores. |
| `cpu_physical_cores` | int | No | Physical CPU cores. |
| `total_system_ram_bytes` | int | No | Total system RAM in bytes. |
| `gpu.name` | string | No | GPU product name. |
| `gpu.type` | string | No | `"unified"` (Apple) or `"discrete"` (NVIDIA). |
| `gpu.vram_bytes` | int | No | GPU memory in bytes. For unified, same as system RAM. |
| `gpu.cuda_compute` | list[int] | No | CUDA compute capability [major, minor]. |
| `gpu.driver_version` | string | No | NVIDIA driver version string. |
| `gpu.pci_bus` | string | No | PCI bus address. |

### How Hardware Fields Are Used

- **Model fit scoring**: `gpu.vram_bytes` is the primary input for determining
  whether a model will fit in memory. See `docs/metrics_and_fit.md`.
- **UI display**: `cpu_cores`, `gpu.name`, and `total_system_ram_bytes`
  are shown in machine cards for context.
- **Preflight warnings**: If hardware fields are missing, fit scoring is
  degraded (scores show as "unknown" instead of GOOD/RISK/FAIL).
- **Override vs discovery**: If hardware fields are present in machines.yaml,
  they take precedence over what the agent detects. This is useful when agent
  detection is incorrect or incomplete.

---

## Model Policy (model_policy.yaml)

Location: `central/config/model_policy.example.yaml`

This file is the single source of truth for which models agents should have
installed. It's read by `scripts/pull_models.sh`.

### Reference

```yaml
required:
  llm:
    - "llama3.1:8b-instruct-q8_0"
    - "llama3.1:70b-instruct-q4_K_M"

optional:
  whisper:
    - "large-v3"

optional_profiles:
  sdxl:
    - "sdxl_1024_30steps"
```

### Sections

| Section | Purpose | Pulled by |
|---------|---------|-----------|
| `required.llm` | Models every agent must have for LLM inference | `ollama pull` via `pull_models.sh` |
| `optional.whisper` | Whisper models for audio transcription | Logged only (not yet implemented) |
| `optional_profiles.sdxl` | ComfyUI image generation profiles | Logged only (not yet implemented) |

### Syncing Models

```bash
# Pull all required LLM models on the current machine
./scripts/pull_models.sh

# Skip 70B+ models (for machines with limited VRAM)
./scripts/pull_models.sh --skip-70b

# Use a custom policy file
./scripts/pull_models.sh --policy /path/to/custom_policy.yaml
```

---

## ComfyUI Configuration (comfyui.yaml)

Location: `central/config/comfyui.example.yaml`

This file configures image generation checkpoint downloads and paths.

### Reference

```yaml
comfyui:
  base_url: ''

  # Cache paths for downloaded checkpoints
  central_cache_path: central/model_cache/comfyui
  agent_cache_path: agent/model_cache/comfyui

  # Path where ComfyUI looks for checkpoint files
  comfyui_models_path: agent/third_party/comfyui/models/checkpoints

  # Checkpoint download URLs
  # Supported formats:
  #   - Simple URL: https://example.com/checkpoint.safetensors
  #   - URL with label: my-checkpoint | https://example.com/checkpoint.safetensors
  #   - URL with hash: https://example.com/checkpoint.safetensors | sha256:abc123...
  #   - Full: my-checkpoint | https://example.com/checkpoint.safetensors | sha256:abc123...
  checkpoint_urls:
    - https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
```

---

## Environment Variables

Environment variables override configuration file values. Set them in your
shell, systemd service file, or `.env` file.

### Agent Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_HOST` | `0.0.0.0` | Agent bind host (overrides agent.yaml `bind_host`) |
| `AGENT_PORT` | `9001` | Agent bind port (overrides agent.yaml `bind_port`) |
| `OLLAMA_HOST` | `127.0.0.1` | Ollama API host |
| `OLLAMA_PORT` | `11434` | Ollama API port |
| `BENCH_STREAM_FLUSH_CHARS` | `64` | Token buffer size before flush to WebSocket |
| `BENCH_STREAM_FLUSH_SECONDS` | `0.10` | Max time (seconds) before buffer flush |
| `OLLAMA_START_TIMEOUT_S` | `120` | Max wait for Ollama startup on reset |
| `COMFYUI_START_TIMEOUT_S` | `60` | Max wait for ComfyUI startup on reset |
| `HEALTH_POLL_INTERVAL_S` | `1.5` | Interval between health checks during reset |
| `RESET_LOG_DIR` | `./logs` | Directory for reset log files |
| `COMFY_FORCE_CPU` | `0` | Force ComfyUI to use CPU (set to `1` to enable) |
| `LOG_LEVEL` | `INFO` | Log level: DEBUG, INFO, WARNING, ERROR |
| `LOG_JSON` | `1` | JSON log output (1) or pretty text (0) |
| `LOG_HTTP_BODY` | `0` | Include HTTP request bodies in logs |
| `LOG_HTTP_MAXLEN` | `2000` | Max body length for logging |

### Central Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CENTRAL_HOST` | `0.0.0.0` | Central bind host |
| `CENTRAL_PORT` | `8080` | Central bind port |
| `SERVICE_CONTROL_TOKEN` | (none) | Auth token for remote service control API |

---

## Port Reference

| Port | Service | Protocol | Notes |
|------|---------|----------|-------|
| **8080** | Central web UI + API | HTTP + WebSocket | Configurable via `CENTRAL_PORT` |
| **9001** | Agent API (default) | HTTP + WebSocket | Configurable via `bind_port` or `AGENT_PORT` |
| **11434** | Ollama API | HTTP | Ollama's default port |
| **8188** | ComfyUI API | HTTP + WebSocket | ComfyUI's default port |

---

## Example: Two-Machine Setup

**Machine A** (MacBook Pro, also runs central):
- IP: 192.168.1.10
- Roles: Central + Agent

**Machine B** (Mac Mini, agent only):
- IP: 192.168.1.20
- Role: Agent

### Machine A: agent.yaml

```yaml
machine_id: "macbook-pro"
label: "MacBook Pro (M4 Max, 128GB)"
bind_host: "0.0.0.0"
bind_port: 9001
central_base_url: "http://127.0.0.1:8080"  # Central is local

ollama:
  enabled: true
  base_url: "http://127.0.0.1:11434"
```

### Machine B: agent.yaml

```yaml
machine_id: "mac-mini"
label: "Mac Mini (M4 Pro, 24GB)"
bind_host: "0.0.0.0"
bind_port: 9001
central_base_url: "http://192.168.1.10:8080"  # Points to Machine A

ollama:
  enabled: true
  base_url: "http://127.0.0.1:11434"
```

### Machine A: machines.yaml

```yaml
machines:
  - machine_id: "macbook-pro"
    label: "MacBook Pro (M4 Max, 128GB)"
    logo: "apple"
    agent_base_url: "http://127.0.0.1:9001"       # Local agent
    cpu_cores: 16
    total_system_ram_bytes: 137438953472
    gpu:
      name: "Apple M4 Max"
      type: "unified"
      vram_bytes: 137438953472

  - machine_id: "mac-mini"
    label: "Mac Mini (M4 Pro, 24GB)"
    logo: "apple"
    agent_base_url: "http://192.168.1.20:9001"     # Remote agent
    cpu_cores: 12
    total_system_ram_bytes: 25769803776
    gpu:
      name: "Apple M4 Pro"
      type: "unified"
      vram_bytes: 25769803776
```

### Startup Sequence

```bash
# Machine B (start agent first)
cd /path/to/bench-race
bin/control agent start

# Machine A (start agent, then central)
cd /path/to/bench-race
bin/control agent start
bin/control central start

# Verify from Machine A
curl http://127.0.0.1:9001/health         # Local agent
curl http://192.168.1.20:9001/health       # Remote agent

# Open UI
open http://127.0.0.1:8080
```

---

## Example: Four-Machine Lab Setup

A lab setup with one central and four agents of mixed hardware:

```
┌─────────────────────────────────────────────────────────────┐
│                     Lab Network (192.168.1.0/24)            │
│                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │ Central        │  │ MacBook Pro   │  │ Mac Mini       │  │
│  │ 192.168.1.10   │  │ 192.168.1.10  │  │ 192.168.1.20  │  │
│  │ :8080 (UI)     │  │ :9001 (Agent) │  │ :9001 (Agent)  │  │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
│                                                             │
│  ┌───────────────┐  ┌───────────────┐                       │
│  │ Alienware      │  │ Dell GB10     │                      │
│  │ 192.168.1.30   │  │ 192.168.1.40  │                     │
│  │ :9001 (Agent)  │  │ :9001 (Agent) │                      │
│  └───────────────┘  └───────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### machines.yaml for this setup

```yaml
machines:
  - machine_id: "macbook-pro"
    label: "MacBook Pro (M4 Max, 128GB)"
    logo: "apple"
    agent_base_url: "http://192.168.1.10:9001"
    cpu_cores: 16
    cpu_physical_cores: 12
    total_system_ram_bytes: 137438953472
    gpu:
      name: "Apple M4 Max"
      type: "unified"
      vram_bytes: 137438953472

  - machine_id: "mac-mini"
    label: "Mac Mini (M4 Pro, 24GB)"
    logo: "apple"
    agent_base_url: "http://192.168.1.20:9001"
    cpu_cores: 12
    cpu_physical_cores: 10
    total_system_ram_bytes: 25769803776
    gpu:
      name: "Apple M4 Pro"
      type: "unified"
      vram_bytes: 25769803776

  - machine_id: "alienware"
    label: "Alienware Area 51 (RTX5090, 32GB)"
    logo: "nvidia"
    agent_base_url: "http://192.168.1.30:9001"
    cpu_cores: 32
    cpu_physical_cores: 16
    total_system_ram_bytes: 68719476736
    gpu:
      name: "NVIDIA RTX5090"
      type: "discrete"
      vram_bytes: 34359738368
      cuda_compute: [12, 0]

  - machine_id: "gb10"
    label: "Dell ProMax (GB10, 128GB)"
    logo: "nvidia"
    agent_base_url: "http://192.168.1.40:9001"
    cpu_cores: 64
    cpu_physical_cores: 32
    total_system_ram_bytes: 137438953472
    gpu:
      name: "NVIDIA GB10"
      type: "discrete"
      vram_bytes: 68719476736
      cuda_compute: [12, 1]
      driver_version: "545.101"
```

---

## Firewall and Security

### Required Ports

Ensure these ports are open between machines:

| Direction | Port | Purpose |
|-----------|------|---------|
| Central -> Agents | 9001/tcp | HTTP REST API calls |
| Agents -> Central | 8080/tcp | WebSocket event streaming |
| Browser -> Central | 8080/tcp | Web UI access |

### Security Considerations

- **No built-in encryption.** All communication is plain HTTP. For production
  deployments over untrusted networks, use a VPN or reverse proxy with TLS.
- **Service control API** supports optional token auth:
  ```bash
  export SERVICE_CONTROL_TOKEN="your-secret-token"
  ```
  Remote callers must include `Authorization: Bearer your-secret-token`.
  Local requests (from 127.0.0.1) are allowed without a token.
- **Agent bind address.** Use `bind_host: "127.0.0.1"` if you don't need
  network access to prevent unauthorized connections.

---

## Suggested Screenshots

1. **machines.yaml editor** -- showing a configured machines.yaml in an
   editor with IP addresses and hardware fields highlighted.

2. **Network connectivity test** -- terminal showing `curl` health checks
   succeeding for all agents.

3. **Web UI with multiple machines** -- showing the machine cards for all
   four machines in the lab setup, with vendor logos and hardware info.

4. **Agent status panel** -- showing the admin view with agent connection
   status (green = connected, gray = offline).

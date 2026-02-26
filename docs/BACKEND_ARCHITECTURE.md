# Backend Architecture (Authoritative)

`./scripts/agent` is the lifecycle entrypoint for local backend orchestration.

| Backend | Platform | Prereqs | Launcher | Default endpoint | Lifecycle commands |
|---|---|---|---|---|---|
| ollama | macOS + Linux | `ollama` in `PATH` | direct (`ollama serve`) | `127.0.0.1:11434` | `./scripts/agent start-ollama`, `./scripts/agent stop-ollama` |
| mlx | macOS Apple Silicon | `./scripts/install_macos_mlx.sh` | direct (`agent.backends.mlx.server`) | `127.0.0.1:8321` | `./scripts/agent start-mlx`, `./scripts/agent stop-mlx` |
| trtllm | Linux + NVIDIA Docker | `./scripts/install_trtllm.sh` | `agent/backends/trtllm_run.sh` | `127.0.0.1:8000` | `./scripts/agent start-trtllm`, `./scripts/agent stop-trtllm` |
| comfyui | macOS + Linux | installed by `./scripts/install_agent.sh` | direct (`main.py`) | `127.0.0.1:8188` | `./scripts/agent start-comfyui`, `./scripts/agent stop-comfyui` |
| wrapper service | macOS + Linux | Python env for `agent.backends.wrapper` | `python -m agent.backends.wrapper` | `127.0.0.1:9002` | independent of `scripts/agent`; see wrapper README |

## LLM exclusivity

Only one LLM backend is active at a time: `ollama`, `mlx`, or `trtllm`.
`./scripts/agent start-backend <backend>` enforces this by stopping the other LLMs first.

## Install flow

1. `./scripts/install_agent.sh` for shared agent dependencies.
2. Optional platform backends:
   - MLX: `./scripts/install_macos_mlx.sh`
   - TRT-LLM: `./scripts/install_trtllm.sh`
3. Use lifecycle manager:
   - `./scripts/agent status`
   - `./scripts/agent start-mlx`

## vLLM status

vLLM artifacts are archived under `archive/vllm/`. Runtime lifecycle commands were removed from `./scripts/agent`.

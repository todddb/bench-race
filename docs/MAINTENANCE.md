# Maintenance and Archive Notes

## Archived components

- `scripts/install_x64_vllm.sh` → `archive/vllm/scripts/install_x64_vllm.sh`
- `agent/backends/vllm/*` → `archive/vllm/agent_backends/vllm/*`
- `agent/backends/vllm_backend.py` → `archive/vllm/agent_backends/vllm_backend.py`

Stubs remain at original locations to keep old paths discoverable.

## Rationale

The lifecycle manager now supports `ollama`, `mlx`, `trtllm`, and `comfyui` only. vLLM was removed from central lifecycle paths to reduce backend drift and simplify support.

## Restore/rollback

1. Copy archived files back to original paths.
2. Reintroduce vLLM commands in `scripts/agent`.
3. Re-enable install logic in `scripts/install_agent.sh`.

Use `git log -- archive/vllm scripts/agent scripts/install_agent.sh` to find the exact archival and restoration commits.

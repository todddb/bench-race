# bench-race configuration

## Configuration Hierarchy

1. `config/*.yaml` and `config/registry/models.json` (canonical)
2. Environment variables (override)
3. Script defaults (fallback)

## Files

- `registry/models.json`: canonical model registry consumed by central.
- `machines.yaml`: canonical machine inventory consumed by central.
- `backends.yaml`: backend runtime defaults (ports, image tags, sizing defaults).
- `policy.yaml`: central model policy (`required`, `optional`, `optional_profiles`).
- `env/dev.yaml`, `env/prod.yaml`: optional environment-specific overlays.

## Environment variable overrides

These remain supported as runtime overrides:

- `HUGGINGFACE_TOKEN`
- `NGC_API_KEY`
- `TRTLLM_MODEL`
- `TRTLLM_PORT`
- `MLX_HOST`
- `MLX_PORT`

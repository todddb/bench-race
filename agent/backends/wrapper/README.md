# bench-race unified wrapper (`agent.backends.wrapper`)

This service provides a stable API for bench-race clients and routes requests to backend adapters:

- **MLX adapter** (`http://127.0.0.1:8321` by default)
- **TRT-LLM adapter** (`http://127.0.0.1:8000` by default)

## Run

```bash
python -m agent.backends.wrapper
# or
uvicorn agent.backends.wrapper.app:app --host 127.0.0.1 --port 9002
```

Optional env overrides:

- `WRAPPER_MLX_BASE_URL`
- `WRAPPER_TRT_BASE_URL`

## Endpoints

- `GET /v1/health`
- `GET /v1/models`
- `POST /v1/models/start`
- `POST /v1/models/switch`
- `POST /v1/models/stop`
- `POST /v1/infer`
- `POST /v1/infer/stream` (SSE)

## Backend lifecycle integration

Wrapper lifecycle reuses existing scripts:

- MLX start/stop: `./scripts/agent start-mlx` / `./scripts/agent stop-mlx`
- TRT start/stop: `TRTLLM_MODEL=<engine> ./agent/backends/trtllm_run.sh restart` / `./agent/backends/trtllm_run.sh stop`

TRT engine IDs are normalized by replacing `/` with `__`.

## `scripts/agent` integration snippet (manual merge)

```bash
start-wrapper)
    nohup uvicorn agent.backends.wrapper.app:app --host 127.0.0.1 --port 9002 >> "$LOG_DIR/wrapper.log" 2>&1 &
    echo $! > "$RUN_DIR/wrapper.pid"
    ;;
stop-wrapper)
    stop_process "$RUN_DIR/wrapper.pid" "Wrapper"
    ;;
```

You can also invoke wrapper hooks from `start-mlx` / `stop-mlx` flows:

```bash
curl -s -X POST http://127.0.0.1:9002/v1/models/start -H 'Content-Type: application/json' -d '{"model_id":"..."}'
curl -s -X POST http://127.0.0.1:9002/v1/models/stop -H 'Content-Type: application/json' -d '{"backend":"mlx"}'
```

## Smoke test

```bash
agent/backends/wrapper/tests/smoke_tests.sh
```

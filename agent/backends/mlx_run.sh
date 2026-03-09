#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MLX_HOST="${MLX_HOST:-127.0.0.1}"
MLX_PORT="${MLX_PORT:-8321}"
MLX_PIDFILE="${REPO_ROOT}/agent/run/mlx.pid"
PY="${REPO_ROOT}/agent/backends/mlx/.venv/bin/python"

wait_ready() {
  for i in {1..60}; do
    if curl -sf "http://127.0.0.1:${MLX_PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  echo "MLX failed to become ready" >&2
  return 3
}

start() {
  local model="${1:-}"

  if [[ ! -x "${PY}" ]]; then
    echo "mlx not installed. Run ./scripts/install_macos_mlx.sh" >&2
    return 4
  fi

  # Verify mlx_lm is installed
  if ! "${PY}" -c "import mlx_lm" >/dev/null 2>&1; then
    echo "ERROR: mlx_lm not installed in MLX venv. Run ./scripts/install_agent.sh" >&2
    return 5
  fi

  (
    cd "${REPO_ROOT}"
    nohup "${PY}" -m uvicorn agent.backends.mlx.server:app --host "${MLX_HOST}" --port "${MLX_PORT}" >>"${REPO_ROOT}/agent/log/mlx.log" 2>&1 &
    echo $! >"${MLX_PIDFILE}"
  )

  if [[ -n "${model}" ]]; then
    echo "INFO: mlx start received model argument '${model}', backend will use its default model selection" >&2
  fi

  wait_ready
}

stop() {
  if [[ -f "${MLX_PIDFILE}" ]]; then
    pid="$(cat "${MLX_PIDFILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]]; then
      kill "${pid}" 2>/dev/null || true
      sleep 1
      kill -9 "${pid}" 2>/dev/null || true
    fi
    rm -f "${MLX_PIDFILE}"
  fi
}

case "${1:-start}" in
  start) start "${2:-}" ;;
  stop) stop ;;
  *) echo "Usage: $0 {start|stop} [model]" >&2; exit 2 ;;
esac

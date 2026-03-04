#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODELS_DIR="${REPO_ROOT}/agent/models/trtllm"
CONTAINER_NAME="bench-race-trtllm"
TRTLLM_IMAGE="${TRTLLM_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6.post3}"
TRTLLM_PORT="${TRTLLM_PORT:-8000}"
READY_TIMEOUT_SECONDS="120"

mkdir -p "${MODELS_DIR}"

wait_ready() {
  local deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
  local url="http://127.0.0.1:${TRTLLM_PORT}/v1/models"
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 2 "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  echo "Timed out waiting for TRT-LLM health at ${url}" >&2
  return 1
}

start() {
  local model_folder="${1:-}"
  if [[ -z "${model_folder}" ]]; then
    echo "Usage: $0 start <model_folder>" >&2
    return 2
  fi

  if [[ ! -d "${MODELS_DIR}/${model_folder}" ]]; then
    echo "Model directory not found: ${MODELS_DIR}/${model_folder}" >&2
    return 2
  fi

  docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true

  docker run -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    -p "${TRTLLM_PORT}:8000" \
    -v "${MODELS_DIR}:/models" \
    "${TRTLLM_IMAGE}" \
    trtllm-serve "/models/${model_folder}" \
      --host 0.0.0.0 \
      --port 8000

  wait_ready
}

stop() {
  docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
status() { docker ps -a --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"; }
logs() { docker logs -f "${CONTAINER_NAME}"; }
restart() { stop || true; start; }

case "${1:-start}" in
  start) start "${2:-}" ;;
  stop) stop ;;
  restart) restart ;;
  status) status ;;
  logs) logs ;;
  *) echo "Usage: $0 {start|stop|restart|status|logs}" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODELS_DIR="${REPO_ROOT}/agent/models/trtllm"
HF_CACHE_DIR="${HOME}/.cache/huggingface"
CONTAINER_NAME="${TRTLLM_CONTAINER_NAME:-bench-race-trtllm}"
TRTLLM_IMAGE="${TRTLLM_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6.post3}"
TRTLLM_PORT="${TRTLLM_PORT:-8000}"
DETACH="${TRTLLM_DETACH:-true}"

mkdir -p "${MODELS_DIR}" "${HF_CACHE_DIR}"

start_container() {
  stop_container >/dev/null 2>&1 || true
  local docker_args=(
    run
    --name "${CONTAINER_NAME}"
    --gpus=all
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    --publish "${TRTLLM_PORT}:8000"
    --mount "type=bind,src=${MODELS_DIR},dst=/workspace/trtllm,rw"
    --mount "type=bind,src=${HF_CACHE_DIR},dst=/root/.cache/huggingface,rw"
  )

  if [[ -n "${HUGGINGFACE_TOKEN:-}" ]]; then
    docker_args+=(--env "HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}")
  fi

  if [[ "${DETACH}" == "true" ]]; then
    docker_args+=(--detach)
  fi

  docker "${docker_args[@]}" "${TRTLLM_IMAGE}"
}

stop_container() {
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    docker stop "${CONTAINER_NAME}" >/dev/null || true
    docker rm "${CONTAINER_NAME}" >/dev/null || true
  fi
}

status_container() {
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "running (${CONTAINER_NAME})"
  elif docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "stopped (${CONTAINER_NAME})"
  else
    echo "not-created (${CONTAINER_NAME})"
  fi
}

case "${1:-start}" in
  start)
    start_container
    ;;
  stop)
    stop_container
    ;;
  restart)
    stop_container
    start_container
    ;;
  status)
    status_container
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status}" >&2
    exit 1
    ;;
esac

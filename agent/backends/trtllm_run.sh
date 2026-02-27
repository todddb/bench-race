#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODELS_DIR="${REPO_ROOT}/agent/models/trtllm"
HF_CACHE_DIR="${HOME}/.cache/huggingface"
CONTAINER_NAME="${TRTLLM_CONTAINER_NAME:-bench-race-trtllm}"
TRTLLM_IMAGE="${TRTLLM_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6.post3}"
TRTLLM_PORT="${TRTLLM_PORT:-8000}"
TRTLLM_MODEL="${TRTLLM_MODEL:-distilgpt2}"
TRTLLM_ENGINE_QUANT="${TRTLLM_ENGINE_QUANT:-auto}"
TRTLLM_MAX_BATCH_SIZE="${TRTLLM_MAX_BATCH_SIZE:-2048}"
DETACH="${TRTLLM_DETACH:-true}"

mkdir -p "${MODELS_DIR}" "${HF_CACHE_DIR}"

resolve_model_path() {
  local model_name="${TRTLLM_MODEL}"
  local quant="${TRTLLM_ENGINE_QUANT}"
  if [[ "${model_name}" == *:* ]]; then
    quant="${model_name##*:}"
    model_name="${model_name%%:*}"
  fi
  local base="/workspace/trtllm/engines/${model_name}"
  if [[ -d "${base}/${quant}" ]]; then
    echo "${base}/${quant}"
    return 0
  fi
  if [[ -d "${base}/float16" ]]; then
    echo "${base}/float16"
    return 0
  fi
  if [[ -d "${base}/bfloat16" ]]; then
    echo "${base}/bfloat16"
    return 0
  fi
  echo "${base}"
}

start() {
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
      echo "Container ${CONTAINER_NAME} already running."
      return 0
    fi
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi

  local model_path
  model_path="$(resolve_model_path)"
  local docker_args=(
    run --name "${CONTAINER_NAME}"
    --gpus=all --ipc=host
    --ulimit memlock=-1 --ulimit stack=67108864
    -p "${TRTLLM_PORT}:8000"
    -v "${MODELS_DIR}:/workspace/trtllm:ro"
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface:ro"
  )

  [[ -n "${HUGGINGFACE_TOKEN:-}" ]] && docker_args+=( -e "HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}" )
  [[ "${DETACH}" == "true" ]] && docker_args+=( -d )

  docker "${docker_args[@]}" "${TRTLLM_IMAGE}" \
    trtllm-serve serve \
      --backend tensorrt \
      --max_batch_size "${TRTLLM_MAX_BATCH_SIZE}" \
      --host 0.0.0.0 --port 8000 \
      "${model_path}"
}

stop() {
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    echo "Stopped ${CONTAINER_NAME}."
  else
    echo "Container ${CONTAINER_NAME} not present."
  fi
}
status() { docker ps -a --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"; }
logs() { docker logs -f "${CONTAINER_NAME}"; }
restart() { stop || true; start; }

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  restart) restart ;;
  status) status ;;
  logs) logs ;;
  *) echo "Usage: $0 {start|stop|restart|status|logs}" >&2; exit 2 ;;
esac

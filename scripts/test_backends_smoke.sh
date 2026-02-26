#!/usr/bin/env bash
set -euo pipefail

VLLM_PORT="${VLLM_PORT:-8000}"
TRITON_PORT="${TRITON_PORT:-8020}"
MLX_PORT="${MLX_PORT:-8321}"
TRTLLM_PORT="${TRTLLM_PORT:-8000}"

check_health() {
  local name="$1"; local url="$2"
  if curl -fsS "$url" >/dev/null 2>&1; then
    echo "[smoke] $name health ok"
  else
    echo "[smoke] $name health skipped/fail (not running)"
  fi
}

check_health "vllm" "http://127.0.0.1:${VLLM_PORT}/health"
check_health "triton" "http://127.0.0.1:${TRITON_PORT}/health"
check_health "mlx" "http://127.0.0.1:${MLX_PORT}/health"

if [[ -x "agent/backends/trtllm_run.sh" ]]; then
  if command -v docker >/dev/null 2>&1; then
    if curl -fsS "http://127.0.0.1:${TRTLLM_PORT}/v1/models" >/dev/null 2>&1 || curl -fsS "http://127.0.0.1:${TRTLLM_PORT}/health" >/dev/null 2>&1; then
      echo "[smoke] trtllm health ok"
    else
      echo "[smoke] trtllm health skipped/fail (not running)"
    fi
  else
    echo "[smoke] trtllm health skipped (docker unavailable)"
  fi
fi

echo "[smoke] websocket checks require websocat; using optional check"
if command -v websocat >/dev/null 2>&1; then
  printf '{"model":"test","prompt":"hello","params":{"max_tokens":5}}\n' | timeout 5 websocat "ws://127.0.0.1:${VLLM_PORT}/stream" || true
fi

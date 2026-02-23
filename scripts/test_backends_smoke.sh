#!/usr/bin/env bash
set -euo pipefail

VLLM_PORT="${VLLM_PORT:-8000}"
TRITON_PORT="${TRITON_PORT:-8020}"
MLX_PORT="${MLX_PORT:-8321}"

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

echo "[smoke] websocket checks require websocat; using optional check"
if command -v websocat >/dev/null 2>&1; then
  printf '{"model":"test","prompt":"hello","params":{"max_tokens":5}}\n' | timeout 5 websocat "ws://127.0.0.1:${VLLM_PORT}/stream" || true
fi

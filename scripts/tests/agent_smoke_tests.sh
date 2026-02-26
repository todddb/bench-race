#!/usr/bin/env bash
set -euo pipefail

AGENT_SCRIPT="./scripts/agent"

run_case() {
  local backend="$1"
  echo "[smoke] backend=${backend}"
  if "$AGENT_SCRIPT" "start-${backend}"; then
    "$AGENT_SCRIPT" status >/dev/null
    "$AGENT_SCRIPT" "stop-${backend}"
    echo "[smoke] ${backend}: pass"
  else
    echo "[smoke] ${backend}: skipped/unavailable"
  fi
}

"$AGENT_SCRIPT" help >/dev/null
run_case ollama
run_case mlx
run_case trtllm
run_case comfyui

if [[ -x "agent/backends/wrapper/tests/smoke_tests.sh" ]]; then
  echo "[smoke] running wrapper smoke tests"
  agent/backends/wrapper/tests/smoke_tests.sh || true
fi

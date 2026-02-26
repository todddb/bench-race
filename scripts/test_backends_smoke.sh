#!/usr/bin/env bash
set -euo pipefail

AGENT_SCRIPT="./scripts/agent"

for cmd in help status; do
  "$AGENT_SCRIPT" "$cmd" >/dev/null
  echo "[smoke] agent $cmd ok"
done

for backend in ollama mlx trtllm comfyui; do
  if "$AGENT_SCRIPT" "start-${backend}" >/dev/null 2>&1; then
    "$AGENT_SCRIPT" status >/dev/null
    "$AGENT_SCRIPT" "stop-${backend}" >/dev/null 2>&1 || true
    echo "[smoke] ${backend} lifecycle ok"
  else
    echo "[smoke] ${backend} lifecycle skipped/unavailable"
  fi
done

if [[ -x "agent/backends/wrapper/tests/smoke_tests.sh" ]]; then
  echo "[smoke] wrapper tests"
  agent/backends/wrapper/tests/smoke_tests.sh || true
fi

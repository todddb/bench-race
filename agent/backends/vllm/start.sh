#!/usr/bin/env bash
set -euo pipefail

: "${VLLM_MODEL:=facebook/opt-125m}"
: "${VLLM_OPENAI_PORT:=8001}"
: "${VLLM_PORT:=8010}"

python3 -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port "${VLLM_OPENAI_PORT}" \
  --model "${VLLM_MODEL}" \
  --trust-remote-code &

exec uvicorn wrapper:app --host 0.0.0.0 --port "${VLLM_PORT}"

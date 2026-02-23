#!/usr/bin/env bash
# NOTE: This start.sh is part of the legacy wrapper-based vLLM backend.
# It is no longer used. The current backend uses the upstream vLLM image
# built by scripts/install_x64_vllm.sh which runs vLLM's OpenAI server directly.
# This file is retained for reference only.

python3 /app/monkeypatch_tokenizer.py || true
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

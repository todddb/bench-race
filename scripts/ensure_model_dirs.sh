#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
AGENT_MODELS_DIR="$ROOT/agent/models"
OLLAMA_DIR="$AGENT_MODELS_DIR/ollama"
VLLM_DIR="$AGENT_MODELS_DIR/vllm"

mkdir -p "$OLLAMA_DIR" "$VLLM_DIR"
chmod 0755 "$AGENT_MODELS_DIR" "$OLLAMA_DIR" "$VLLM_DIR"

echo "Ensured: $OLLAMA_DIR and $VLLM_DIR"

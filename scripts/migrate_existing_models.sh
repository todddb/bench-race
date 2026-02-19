#!/usr/bin/env bash
# Migrate legacy model layout into agent/models/ollama non-destructively.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODELS_DIR="$ROOT/agent/models"
OLLAMA_DIR="$MODELS_DIR/ollama"

mkdir -p "$OLLAMA_DIR"

# Migrate old layout: agent/models/<model> -> agent/models/ollama/<model>
shopt -s nullglob
for src in "$MODELS_DIR"/*; do
  base=$(basename "$src")
  [[ "$base" == "ollama" || "$base" == "vllm" || "$base" == ".gitkeep" ]] && continue
  dst="$OLLAMA_DIR/$base"
  if [[ -e "$dst" ]]; then
    continue
  fi
  if [[ -d "$src" || -L "$src" ]]; then
    ln -s "$src" "$dst"
    echo "legacy-linked: $dst -> $src"
  fi
done

# Optional: expose host ollama model store if available.
if [[ -d "$HOME/.ollama/models" ]]; then
  dst="$OLLAMA_DIR/_host_ollama_models"
  [[ -e "$dst" ]] || ln -s "$HOME/.ollama/models" "$dst"
fi

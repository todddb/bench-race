#!/usr/bin/env bash
# idempotent: scan agent/models/ollama and make sanitized symlinks under agent/models/vllm
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OLLAMA_DIR="$ROOT/agent/models/ollama"
VLLM_DIR="$ROOT/agent/models/vllm"
SANITIZER="$ROOT/scripts/model_sanitize.sh"

if [[ ! -x "$SANITIZER" ]]; then
  echo "error: sanitizer not found or not executable: $SANITIZER" >&2
  exit 2
fi

mkdir -p "$OLLAMA_DIR" "$VLLM_DIR"

shopt -s nullglob
for src in "$OLLAMA_DIR"/*; do
  [[ -d "$src" ]] || continue
  base=$(basename "$src")
  sanitized=$(echo "$base" | "$SANITIZER")
  if [[ -z "$sanitized" ]]; then
    echo "skipping $base (sanitized name empty)" >&2
    continue
  fi

  dst="$VLLM_DIR/$sanitized"

  if [[ -L "$dst" ]]; then
    target=$(readlink "$dst")
    if [[ "$target" == "$src" ]]; then
      continue
    fi
    echo "warning: $dst is symlink but points to $target — replacing" >&2
    rm "$dst"
  fi

  if [[ -e "$dst" ]]; then
    echo "warning: $dst exists and is not symlink — skipping" >&2
    continue
  fi

  ln -s "$src" "$dst"
  echo "linked: $dst -> $src"
done

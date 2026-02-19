#!/usr/bin/env bash
# agent/scripts/sync_models.sh
# Downloads models into agent/models/ollama and creates sanitized vllm symlinks
set -euo pipefail
IFS=$'\n\t'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SANITIZER="$REPO_ROOT/scripts/model_sanitize.sh"
OLLAMA_DIR="$REPO_ROOT/agent/models/ollama"
VLLM_DIR="$REPO_ROOT/agent/models/vllm"
SYNC_HELPER="$REPO_ROOT/scripts/sync_models.sh"
LOG_PREFIX="[sync_models]"

mkdir -p "$OLLAMA_DIR"
mkdir -p "$VLLM_DIR"

if ! [ -x "$SANITIZER" ]; then
  echo "$LOG_PREFIX sanitizer script not found or not executable: $SANITIZER" >&2
  exit 1
fi

# Helper: download_model <model-id> <safe-local-name>
download_model() {
  local model_id="$1"
  local local_name="$2"

  if [ -x "$SYNC_HELPER" ]; then
    "$SYNC_HELPER" "$model_id" --target-dir ollama --local-name "$local_name"
    return $?
  fi

  if command -v bench-sync-model >/dev/null 2>&1; then
    bench-sync-model "$model_id" --outdir "$OLLAMA_DIR/$local_name"
    return $?
  fi

  echo "$LOG_PREFIX no model downloader configured. Expected $SYNC_HELPER or bench-sync-model in PATH." >&2
  return 2
}

if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  if [ -t 0 ]; then
    echo "$LOG_PREFIX Usage: $0 <model-id> [<model-id> ...]" >&2
    exit 2
  fi
  mapfile -t MODELS < /dev/stdin
fi

echo "$LOG_PREFIX Starting model sync for ${#MODELS[@]} models"

had_error=0
for model in "${MODELS[@]}"; do
  [ -n "$model" ] || continue
  echo "$LOG_PREFIX Processing model: $model"

  safe_dir_name="$(echo "$model" | sed -E 's|/|__|g')"
  target_dir="$OLLAMA_DIR/$safe_dir_name"

  if [ -d "$target_dir" ] && [ "$(find "$target_dir" -mindepth 1 -maxdepth 1 | wc -l)" -gt 0 ]; then
    echo "$LOG_PREFIX model already present at $target_dir — skipping download"
  else
    echo "$LOG_PREFIX downloading $model -> $target_dir"
    if ! download_model "$model" "$safe_dir_name"; then
      echo "$LOG_PREFIX ERROR downloading model $model" >&2
      had_error=1
      continue
    fi
    echo "$LOG_PREFIX downloaded $model"
  fi

  sanitized="$("$SANITIZER" "$model")"
  if [ -z "$sanitized" ]; then
    echo "$LOG_PREFIX sanitizer returned empty name for '$model'" >&2
    had_error=1
    continue
  fi

  vllm_link="$VLLM_DIR/$sanitized"
  real_target="$(cd "$target_dir" && pwd)"

  if [ -e "$vllm_link" ] || [ -L "$vllm_link" ]; then
    if [ -L "$vllm_link" ] && [ "$(readlink "$vllm_link")" = "$real_target" ]; then
      echo "$LOG_PREFIX symlink exists and is correct: $vllm_link -> $real_target"
    else
      echo "$LOG_PREFIX updating symlink: $vllm_link -> $real_target"
      ln -sfn "$real_target" "$vllm_link"
    fi
  else
    echo "$LOG_PREFIX creating symlink: $vllm_link -> $real_target"
    ln -sfn "$real_target" "$vllm_link"
  fi

done

if [ "$had_error" -ne 0 ]; then
  echo "$LOG_PREFIX Sync completed with errors. Ollama dir: $OLLAMA_DIR, vLLM symlinks: $VLLM_DIR" >&2
  exit 1
fi

echo "$LOG_PREFIX Sync complete. Ollama dir: $OLLAMA_DIR, vLLM symlinks: $VLLM_DIR"

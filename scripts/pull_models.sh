#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[pull_models] $*"
}

if ! command -v ollama >/dev/null 2>&1; then
  log "Ollama CLI not found. Install Ollama first."
  exit 1
fi

skip_70b=false
models=()

for arg in "$@"; do
  case "$arg" in
    --skip-70b)
      skip_70b=true
      ;;
    *)
      models+=("$arg")
      ;;
  esac
done

if [ ${#models[@]} -eq 0 ]; then
  models=("llama3.1:8b-instruct-q8_0" "llama3.1:70b-instruct-q4_K_M")
fi

filtered_models=()
for model in "${models[@]}"; do
  if $skip_70b && [[ "$model" == *"70b"* ]]; then
    log "Skipping $model due to --skip-70b."
    continue
  fi
  filtered_models+=("$model")
done

if [ ${#filtered_models[@]} -eq 0 ]; then
  log "No models to pull after filtering."
  exit 0
fi

for model in "${filtered_models[@]}"; do
  log "Pulling model: $model"
  ollama pull "$model"
done

log "Installed models:"
ollama list

#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[install_ollama_macos] $*"
}

if command -v brew >/dev/null 2>&1; then
  log "Homebrew detected."
  if ! command -v ollama >/dev/null 2>&1; then
    log "Installing Ollama via Homebrew."
    brew install ollama
  else
    log "Ollama already installed."
  fi
else
  log "Homebrew not found."
  log "Install Ollama from the macOS DMG and launch the app: https://ollama.com/download/mac"
fi

if command -v ollama >/dev/null 2>&1; then
  log "Ollama version: $(ollama --version)"
  if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    log "Ollama API not responding. Attempting to start 'ollama serve' in background."
    nohup ollama serve >"${TMPDIR:-/tmp}/ollama-serve.log" 2>&1 &
    sleep 1
  fi

  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    log "Ollama API reachable."
  else
    log "Ollama API still unreachable. If you installed the DMG, launch the Ollama app."
  fi
else
  log "Ollama CLI not found. Install from the DMG and launch the app."
fi

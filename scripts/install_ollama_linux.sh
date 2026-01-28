#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[install_ollama_linux] $*"
}

# Ensure zstd is installed (required for Ollama model extraction)
if ! command -v zstd >/dev/null 2>&1; then
  log "zstd not found. Installing..."
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y zstd
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y zstd
  elif command -v yum >/dev/null 2>&1; then
    sudo yum install -y zstd
  elif command -v pacman >/dev/null 2>&1; then
    sudo pacman -Sy --noconfirm zstd
  else
    log "WARN: Could not determine package manager. Please install zstd manually."
  fi
else
  log "zstd already installed."
fi


if ! command -v ollama >/dev/null 2>&1; then
  log "Installing Ollama via official install script."
  curl -fsSL https://ollama.com/install.sh | sh
else
  log "Ollama already installed."
fi

if command -v ollama >/dev/null 2>&1; then
  log "Ollama version: $(ollama --version)"
  if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    if command -v systemctl >/dev/null 2>&1; then
      log "Attempting to start Ollama via systemctl."
      systemctl start ollama >/dev/null 2>&1 || log "systemctl start ollama failed; trying manual serve."
    fi
  fi

  if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    log "Ollama API not responding. Attempting to start 'ollama serve' in background."
    nohup ollama serve >"/tmp/ollama-serve.log" 2>&1 &
    sleep 1
  fi

  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    log "Ollama API reachable."
  else
    log "Ollama API still unreachable. Start 'ollama serve' manually or check service logs."
  fi
else
  log "Ollama CLI not found after install attempt."
fi

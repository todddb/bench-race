#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[install_comfyui_linux] $*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMFY_DIR="${COMFY_DIR:-$REPO_ROOT/agent/third_party/comfyui}"

if ! command -v git >/dev/null 2>&1; then
  log "git is required to install ComfyUI."
  exit 1
fi

if [ ! -d "$COMFY_DIR" ]; then
  log "Cloning ComfyUI into $COMFY_DIR"
  git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
else
  log "ComfyUI already exists at $COMFY_DIR"
fi

if ! command -v python3 >/dev/null 2>&1; then
  log "python3 is required."
  exit 1
fi

VENV_DIR="$COMFY_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
  log "Creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

log "Installing dependencies"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$COMFY_DIR/requirements.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  log "Detected NVIDIA GPU. Installing CUDA-enabled PyTorch."
  "$VENV_DIR/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
  log "No NVIDIA GPU detected. Installing CPU PyTorch."
  "$VENV_DIR/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

log "ComfyUI installed."
log "Next steps:"
log "1) Copy agent/config/agent.example.yaml to agent/config/agent.yaml and set comfyui.checkpoints_dir."
log "2) Download SDXL checkpoints into your checkpoints directory."
log "3) Start the agent with: bin/control agent start"

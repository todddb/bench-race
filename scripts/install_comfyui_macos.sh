#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[install_comfyui_macos] $*"
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
log "Installing PyTorch (macOS default build)"
if "$VENV_DIR/bin/python" - <<'PYCODE' >/dev/null 2>&1
import importlib
importlib.import_module("torch")
importlib.import_module("torchvision")
importlib.import_module("torchaudio")
PYCODE
then
  log "Torch stack already installed; skipping reinstall."
else
  "$VENV_DIR/bin/pip" install --upgrade --force-reinstall --no-cache-dir torch torchvision torchaudio
fi

log "Installing remaining dependencies (excluding torch/tv/ta)"
REQS="$COMFY_DIR/requirements.txt"
TEMP_REQS="$(mktemp)"
grep -v -iE '^(torch|torchvision|torchaudio)([ =<>~]|$)' "$REQS" > "$TEMP_REQS"
"$VENV_DIR/bin/pip" install -r "$TEMP_REQS"
rm -f "$TEMP_REQS"

log "Verifying torch install"
"$VENV_DIR/bin/python" - <<'PYCODE'
import torch
print("torch.__version__=", torch.__version__)
print("torch.backends.mps.is_built=", torch.backends.mps.is_built())
print("torch.backends.mps.is_available=", torch.backends.mps.is_available())
PYCODE

# Create required directories for checkpoint sync
CACHE_DIR="$REPO_ROOT/agent/model_cache/comfyui"
CHECKPOINTS_DIR="$COMFY_DIR/models/checkpoints"

log "Creating checkpoint directories"
mkdir -p "$CACHE_DIR"
mkdir -p "$CHECKPOINTS_DIR"

log "ComfyUI installed at $COMFY_DIR"
log "Checkpoint cache: $CACHE_DIR"
log "ComfyUI models: $CHECKPOINTS_DIR"
log ""
log "Next steps:"
log "1) Copy agent/config/agent.example.yaml to agent/config/agent.yaml"
log "   The default paths are already set to use repo-relative directories."
log "2) Copy central/config/comfyui.example.yaml to central/config/comfyui.yaml"
log "   (This will happen automatically on first central startup if missing)"
log "3) Start the agent: bin/control agent start"
log "4) Start the central: bin/control central start"
log "5) Use the central UI to sync checkpoints to agents (no manual downloads needed!)"

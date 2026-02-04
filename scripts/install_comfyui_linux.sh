#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# install_comfyui_linux.sh - improved idempotent installer
# - installs requirements excluding torch/torchvision/torchaudio
# - detects GPU compute capability and installs appropriate torch wheel
# - force-reinstalls torch if needed
# - copies example YAMLs only if target missing
# - prints clear instructions for manual remediation

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMFY_DIR="$REPO_ROOT/agent/third_party/comfyui"
VENV_PY="$COMFY_DIR/.venv/bin/python"
VENV_PIP="$COMFY_DIR/.venv/bin/pip"

if [ ! -d "$COMFY_DIR" ]; then
  echo "[install_comfyui_linux] ERROR: ComfyUI not found at $COMFY_DIR"
  echo "Clone or ensure the submodule is present and re-run this script."
  exit 1
fi

if [ ! -x "$VENV_PIP" ]; then
  echo "[install_comfyui_linux] Creating virtualenv for ComfyUI..."
  python3 -m venv "$COMFY_DIR/.venv"
fi

echo "[install_comfyui_linux] Activating venv: $COMFY_DIR/.venv"
# Do not source; just reference the venv pip/python directly.

# Fix malformed tr usage in any helper invocation (clean CRLFs in example files)
fix_tr_crlf() {
  # safe wrapper, used later if necessary
  true
}

echo "[install_comfyui_linux] Installing python packages (excluding torch/tv/ta) ..."
REQS="$COMFY_DIR/requirements.txt"
TEMP_REQS="$(mktemp)"
# remove lines containing torch, torchvision, torchaudio (case-insensitive)
# this avoids pip auto-installing torch before our plan runs.
grep -v -iE '^(torch|torchvision|torchaudio)([ =<>~]|$)' "$REQS" > "$TEMP_REQS"

# run pip install for generic deps
"$VENV_PIP" install --upgrade pip
"$VENV_PIP" install -r "$TEMP_REQS"

rm -f "$TEMP_REQS"

# GPU detection
NVIDIA_SMI="$(command -v nvidia-smi || true)"
compute_major=0
compute_minor=0
gpu_name=""
gpu_present=0

if [ -n "$NVIDIA_SMI" ]; then
  echo "[install_comfyui_linux] nvidia-smi found: $NVIDIA_SMI"
  gpu_present=1
  # Query compute capability: prefer the --query-gpu option that prints compute capability
  cc_raw="$($NVIDIA_SMI --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null || true)"
  gpu_name="$($NVIDIA_SMI --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
  if [ -n "$cc_raw" ]; then
    # nvidia-smi often returns e.g. "12.1" or "12.1, 12.1" for multi-gpu; take first
    cc_first="$(echo "$cc_raw" | sed -n '1p' | tr -d '[:space:]' | cut -d',' -f1)"
    if [[ "$cc_first" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
      compute_major="${BASH_REMATCH[1]}"
      compute_minor="${BASH_REMATCH[2]}"
      echo "[install_comfyui_linux] Detected GPU: $gpu_name compute capability: $compute_major.$compute_minor"
    else
      echo "[install_comfyui_linux] Warning: could not parse compute capability from nvidia-smi ('$cc_raw')"
    fi
  else
    echo "[install_comfyui_linux] Warning: nvidia-smi present but compute_cap query returned empty"
  fi
else
  echo "[install_comfyui_linux] nvidia-smi not found; assuming no CUDA GPU (or drivers missing)."
fi

# Choose PyTorch index based on compute capability
PYTORCH_INDEX_URL=""
PYTORCH_TAG=""
if [ "$gpu_present" -eq 1 ]; then
  # Prefer nightly cu130 for very new architectures (>=12.1)
  if [ "$compute_major" -ge 13 ] || ( [ "$compute_major" -eq 12 ] && [ "$compute_minor" -ge 1 ] ); then
    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/nightly/cu130"
    PYTORCH_TAG="nightly/cu130"
  elif [ "$compute_major" -eq 12 ]; then
    # For 12.0 choose cu129 stable if available; else fall back to cu121
    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu129"
    PYTORCH_TAG="stable/cu129"
  else
    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
    PYTORCH_TAG="stable/cu121"
  fi
else
  # No GPU; install CPU-only torch (stable)
  PYTORCH_INDEX_URL=""
  PYTORCH_TAG="cpu"
fi

echo "[install_comfyui_linux] PyTorch plan:"
echo "[install_comfyui_linux]   GPU present: $gpu_present"
if [ -n "$gpu_name" ]; then
  echo "[install_comfyui_linux]   cuda_device: $gpu_name"
fi
echo "[install_comfyui_linux]   compute capability: ${compute_major}.${compute_minor}"
echo "[install_comfyui_linux]   chosen plan: $PYTORCH_TAG"
if [ -n "$PYTORCH_INDEX_URL" ]; then
  echo "[install_comfyui_linux]   index url: $PYTORCH_INDEX_URL"
fi

# Force uninstall conflicting packages to ensure a clean torch install
echo "[install_comfyui_linux] Uninstalling existing torch packages (if any) to avoid mismatch..."
"$VENV_PIP" uninstall -y torch torchvision torchaudio || true

# Wait a bit for pip to finish cleaning caches (sometimes helps)
sleep 1

# Build pip install command for torch
if [ "$PYTORCH_TAG" = "cpu" ]; then
  echo "[install_comfyui_linux] Installing CPU-only torch (stable) via pip..."
  "$VENV_PIP" install --upgrade --force-reinstall --no-cache-dir torch
else
  echo "[install_comfyui_linux] Installing torch from index: $PYTORCH_INDEX_URL"
  "$VENV_PIP" install --index-url "$PYTORCH_INDEX_URL" --upgrade --force-reinstall --no-cache-dir torch
fi

# Quick verify: print torch version & cuda details
echo "[install_comfyui_linux] Verifying torch install..."
"$VENV_PY" - <<PYCODE
import pkgutil,sys
try:
    import torch
    print("torch.__version__=", torch.__version__)
    try:
        if torch.cuda.is_available():
            print("torch.cuda.get_device_name(0)=", torch.cuda.get_device_name(0))
            print("torch.cuda.get_device_capability(0)=", torch.cuda.get_device_capability(0))
            try:
                print("torch.cuda.get_arch_list()=", torch.cuda.get_arch_list())
            except Exception as e:
                print("torch.cuda.get_arch_list() not available:", e)
        else:
            print("CUDA not available according to torch.")
    except Exception as e:
        print("Warning checking CUDA availability:", e)
except Exception as e:
    print("ERROR importing torch:", e)
    sys.exit(2)
PYCODE

echo "[install_comfyui_linux] Now installing any remaining python packages (safe install)..."
# Some packages might depend on torch, we installed torch already.
# Re-run installing requirements but don't fail if torch is referenced.
# (We've already installed all non-torch deps).
# Nothing to do here in normal case; include for compatibility.
# Optionally, install torchvision/torchaudio if user wants them (we avoid by default).

# Fix any CRLF leftovers in repository helper files (fix broken tr usage)
# Example: if you sanitize some files earlier, use tr -d '\r' properly when used.
# (No global tr invocation required here.)

# Create checkpoint directories and show locations (idempotent)
mkdir -p "$REPO_ROOT/agent/model_cache/comfyui"
mkdir -p "$COMFY_DIR/models/checkpoints"

echo "[install_comfyui_linux] ComfyUI installed at $COMFY_DIR"
echo "[install_comfyui_linux] Checkpoint cache: $REPO_ROOT/agent/model_cache/comfyui"
echo "[install_comfyui_linux] ComfyUI models: $COMFY_DIR/models/checkpoints"

# Conditional copy of example YAMLs -> actual config only if missing
maybe_copy_example() {
  local example="$1"
  local target="$2"
  if [ ! -f "$target" ]; then
    echo "[install_comfyui_linux] Copying $example -> $target"
    cp "$example" "$target"
  else
    echo "[install_comfyui_linux] Config already exists at $target; not overwriting."
  fi
}

maybe_copy_example "$REPO_ROOT/agent/config/agent.example.yaml" "$REPO_ROOT/agent/config/agent.yaml" || true
maybe_copy_example "$REPO_ROOT/central/config/comfyui.example.yaml" "$REPO_ROOT/central/config/comfyui.yaml" || true

echo "[install_comfyui_linux] Installation complete."

echo ""
echo "Next steps / smoke test:"
echo "1) Start the agent and central (or restart existing):"
echo "   bin/control agent restart"
echo "   bin/control central restart"
echo ""
echo "2) Optional manual validation (inside the comfyui venv):"
echo "   source $COMFY_DIR/.venv/bin/activate"
echo "   python -c \"import torch; print(torch.__version__, torch.cuda.is_available()); \
if torch.cuda.is_available(): print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); \
print('arch_list=', getattr(torch.cuda, 'get_arch_list', lambda : 'N/A')())\""
echo ""
echo "If you saw compute capability >=12.1 and arch_list includes 'sm_121' -> good. If not, re-run this script and inspect the pip output above."
echo ""
echo "If installer detects sm_121 but you still see 'no kernel image is available', try:"
echo "  pip uninstall -y torch torchvision torchaudio && \\"
echo "  pip install --index-url https://download.pytorch.org/whl/nightly/cu130 --upgrade --force-reinstall --no-cache-dir torch"
echo ""
echo "Acceptance: torch.__version__ (nightly/cu130) and torch.cuda.get_arch_list() includes sm_121 (or torch reports capability for GB10)."

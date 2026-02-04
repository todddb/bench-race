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

"$VENV_PIP" install --upgrade pip

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

# Try to get compute capability via torch if already installed
torch_cc="$("$VENV_PY" - <<'PYCODE' 2>/dev/null || true
import importlib.util

if importlib.util.find_spec("torch") is None:
    raise SystemExit

import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"{cap[0]}.{cap[1]}")
PYCODE
)"

if [ -n "$torch_cc" ] && [[ "$torch_cc" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
  compute_major="${BASH_REMATCH[1]}"
  compute_minor="${BASH_REMATCH[2]}"
  echo "[install_comfyui_linux] Detected compute capability via torch: $compute_major.$compute_minor"
elif [ "$gpu_present" -eq 1 ] && [ "$compute_major" -eq 0 ] && [ "$compute_minor" -eq 0 ]; then
  echo "[install_comfyui_linux] Warning: unable to determine compute capability; defaulting to stable plan."
fi

# Choose PyTorch index based on compute capability
PYTORCH_INDEX_URL=""
PYTORCH_TAG=""
PYTORCH_EXPECTED_CUDA=""
if [ "$gpu_present" -eq 1 ]; then
  # Prefer nightly cu130 for very new architectures (>=12.1)
  if [ "$compute_major" -ge 13 ] || ( [ "$compute_major" -eq 12 ] && [ "$compute_minor" -ge 1 ] ); then
    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/nightly/cu130"
    PYTORCH_TAG="nightly/cu130"
    PYTORCH_EXPECTED_CUDA="13.0"
  elif [ "$compute_major" -eq 12 ]; then
    # For 12.0 choose cu129 stable if available; else fall back to cu121
    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu129"
    PYTORCH_TAG="stable/cu129"
    PYTORCH_EXPECTED_CUDA="12.9"
  else
    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
    PYTORCH_TAG="stable/cu121"
    PYTORCH_EXPECTED_CUDA="12.1"
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

# Detect current torch stack to decide whether to reinstall
HAS_TORCH=0
HAS_TORCHVISION=0
HAS_TORCHAUDIO=0
TORCH_CUDA_VERSION=""
TORCH_VERSION=""

torch_status="$("$VENV_PY" - <<'PYCODE' 2>/dev/null || true
import importlib.util

def has_module(name):
    return importlib.util.find_spec(name) is not None

has_torch = has_module("torch")
print(f"HAS_TORCH={1 if has_torch else 0}")
if has_torch:
    import torch
    print(f"TORCH_VERSION={torch.__version__}")
    print(f"TORCH_CUDA_VERSION={torch.version.cuda}")
else:
    print("TORCH_VERSION=")
    print("TORCH_CUDA_VERSION=")
print(f"HAS_TORCHVISION={1 if has_module('torchvision') else 0}")
print(f"HAS_TORCHAUDIO={1 if has_module('torchaudio') else 0}")
PYCODE
)"

while IFS='=' read -r key value; do
  case "$key" in
    HAS_TORCH) HAS_TORCH="$value" ;;
    HAS_TORCHVISION) HAS_TORCHVISION="$value" ;;
    HAS_TORCHAUDIO) HAS_TORCHAUDIO="$value" ;;
    TORCH_CUDA_VERSION) TORCH_CUDA_VERSION="$value" ;;
    TORCH_VERSION) TORCH_VERSION="$value" ;;
  esac
done <<< "$torch_status"

needs_torch_reinstall=0
if [ "$HAS_TORCH" -ne 1 ] || [ "$HAS_TORCHVISION" -ne 1 ] || [ "$HAS_TORCHAUDIO" -ne 1 ]; then
  needs_torch_reinstall=1
elif [ "$PYTORCH_TAG" = "cpu" ]; then
  if [ -n "$TORCH_CUDA_VERSION" ] && [ "$TORCH_CUDA_VERSION" != "None" ]; then
    needs_torch_reinstall=1
  fi
else
  if [ -z "$TORCH_CUDA_VERSION" ] || [ "$TORCH_CUDA_VERSION" = "None" ]; then
    needs_torch_reinstall=1
  elif [ -n "$PYTORCH_EXPECTED_CUDA" ] && [ "$TORCH_CUDA_VERSION" != "$PYTORCH_EXPECTED_CUDA" ]; then
    needs_torch_reinstall=1
  fi
fi

if [ "$needs_torch_reinstall" -eq 1 ]; then
  echo "[install_comfyui_linux] Uninstalling existing torch packages (if any) to avoid mismatch..."
  "$VENV_PIP" uninstall -y torch torchvision torchaudio || true

  # Wait a bit for pip to finish cleaning caches (sometimes helps)
  sleep 1

  # Build pip install command for torch stack
  if [ "$PYTORCH_TAG" = "cpu" ]; then
    echo "[install_comfyui_linux] Installing CPU-only torch stack (stable) via pip..."
    "$VENV_PIP" install --upgrade --force-reinstall --no-cache-dir torch torchvision torchaudio
  else
    echo "[install_comfyui_linux] Installing torch stack from index: $PYTORCH_INDEX_URL"
    if [ "$PYTORCH_TAG" = "nightly/cu130" ]; then
      "$VENV_PIP" install --pre --index-url "$PYTORCH_INDEX_URL" --upgrade --force-reinstall --no-cache-dir torch torchvision torchaudio
    else
      "$VENV_PIP" install --index-url "$PYTORCH_INDEX_URL" --upgrade --force-reinstall --no-cache-dir torch torchvision torchaudio
    fi
  fi
else
  echo "[install_comfyui_linux] Torch stack already matches plan; skipping reinstall."
fi

echo "[install_comfyui_linux] Installing python packages (excluding torch/tv/ta) ..."
REQS="$COMFY_DIR/requirements.txt"
TEMP_REQS="$(mktemp)"
grep -v -iE '^(torch|torchvision|torchaudio)([ =<>~]|$)' "$REQS" > "$TEMP_REQS"
"$VENV_PIP" install -r "$TEMP_REQS"
rm -f "$TEMP_REQS"

echo "[install_comfyui_linux] Verifying torch + torchvision CUDA smoke test..."
if ! "$VENV_PY" - <<'PYCODE'
import sys
import torch
import torchvision
from torchvision.models import resnet18

print("torch.__version__=", torch.__version__)
print("torch.version.cuda=", torch.version.cuda)
print("torchvision.__version__=", torchvision.__version__)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("torch.cuda.get_device_name(0)=", torch.cuda.get_device_name(0))
    print("torch.cuda.get_device_capability(0)=", torch.cuda.get_device_capability(0))
    x = torch.randn(4096, 4096, device=device)
    y = torch.randn(4096, 4096, device=device)
    _ = x @ y
    torch.cuda.synchronize()
    model = resnet18().to(device)
    model.eval()
    with torch.no_grad():
        _ = model(torch.randn(1, 3, 224, 224, device=device))
    torch.cuda.synchronize()
    print("CUDA smoke test passed.")
else:
    print("CUDA not available according to torch; skipping GPU smoke test.")
sys.exit(0)
PYCODE
then
  echo "[install_comfyui_linux] ERROR: CUDA smoke test failed."
  echo "[install_comfyui_linux] Suggested remediation (inside the comfyui venv):"
  if [ "$PYTORCH_TAG" = "nightly/cu130" ]; then
    echo "  pip uninstall -y torch torchvision torchaudio"
    echo "  pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu130 --upgrade --force-reinstall --no-cache-dir torch torchvision torchaudio"
  elif [ "$PYTORCH_TAG" = "stable/cu129" ]; then
    echo "  pip uninstall -y torch torchvision torchaudio"
    echo "  pip install --index-url https://download.pytorch.org/whl/cu129 --upgrade --force-reinstall --no-cache-dir torch torchvision torchaudio"
  else
    echo "  pip uninstall -y torch torchvision torchaudio"
    echo "  pip install --index-url https://download.pytorch.org/whl/cu121 --upgrade --force-reinstall --no-cache-dir torch torchvision torchaudio"
  fi
  exit 1
fi

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
echo "If compute capability is 12.x and the matmul + torchvision forward pass succeed, you're good."

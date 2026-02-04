#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[install_comfyui_linux] $*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMFY_DIR="${COMFY_DIR:-$REPO_ROOT/agent/third_party/comfyui}"
TORCH_STABLE_CUDA_CHANNEL="${TORCH_STABLE_CUDA_CHANNEL:-cu121}"
TORCH_CPU_INDEX_URL="https://download.pytorch.org/whl/cpu"
TORCH_NIGHTLY_CU130_INDEX_URL="https://download.pytorch.org/whl/nightly/cu130"

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

gpu_present() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

torch_installed() {
  "$VENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torch") else 1)
PY
}

detect_compute_capability() {
  local raw_caps=""
  local cap_list=()
  if command -v nvidia-smi >/dev/null 2>&1; then
    raw_caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr '\r' | sed '/^$/d' || true)"
  fi

  if [ -n "$raw_caps" ]; then
    mapfile -t cap_list <<<"$raw_caps"
  elif torch_installed; then
    raw_caps="$("$VENV_DIR/bin/python" - <<'PY' || true
import torch
if not torch.cuda.is_available():
    raise SystemExit(1)
caps = []
for idx in range(torch.cuda.device_count()):
    major, minor = torch.cuda.get_device_capability(idx)
    caps.append(f"{major}.{minor}")
print("\n".join(caps))
PY
)"
    if [ -n "$raw_caps" ]; then
      mapfile -t cap_list <<<"$raw_caps"
    fi
  fi

  if [ "${#cap_list[@]}" -eq 0 ]; then
    return 1
  fi

  local unique_caps
  unique_caps="$(printf '%s\n' "${cap_list[@]}" | sort -u)"
  if [ "$(printf '%s\n' "$unique_caps" | wc -l | tr -d ' ')" -gt 1 ]; then
    log "Multiple GPU compute capabilities detected: $(printf '%s' "$unique_caps" | tr '\n' ' '). Choosing the maximum."
  fi

  printf '%s\n' "$unique_caps" | sort -V | tail -n1
}

print_sm121_guidance() {
  local cap="$1"
  log "Detected NVIDIA GPU with compute capability ${cap}."
  log "Stable PyTorch CUDA wheels do not support sm_121 yet."
  log "Remediation options:"
  log "  1) Install nightly cu130:"
  log "     BENCH_TORCH_CHANNEL=nightly-cu130"
  log "     BENCH_TORCH_INDEX_URL=$TORCH_NIGHTLY_CU130_INDEX_URL"
  log "  2) Build from source with TORCH_CUDA_ARCH_LIST=\"12.1\""
}

select_torch_install_plan() {
  local compute_capability="$1"
  TORCH_PACKAGES="${BENCH_TORCH_PACKAGES:-torch}"

  if [ -n "${BENCH_TORCH_INDEX_URL:-}" ]; then
    TORCH_INDEX_URL="$BENCH_TORCH_INDEX_URL"
    TORCH_PLAN_REASON="user override BENCH_TORCH_INDEX_URL"
    return
  fi

  if [ -n "${BENCH_TORCH_CHANNEL:-}" ]; then
    case "$BENCH_TORCH_CHANNEL" in
      nightly-cu130)
        TORCH_INDEX_URL="$TORCH_NIGHTLY_CU130_INDEX_URL"
        TORCH_PLAN_REASON="user override nightly-cu130"
        ;;
      stable-*)
        local channel="${BENCH_TORCH_CHANNEL#stable-}"
        TORCH_INDEX_URL="https://download.pytorch.org/whl/${channel}"
        TORCH_PLAN_REASON="user override stable-${channel}"
        ;;
      cpu)
        TORCH_INDEX_URL="$TORCH_CPU_INDEX_URL"
        TORCH_PLAN_REASON="user override cpu"
        ;;
      *)
        log "Unknown BENCH_TORCH_CHANNEL: $BENCH_TORCH_CHANNEL"
        exit 1
        ;;
    esac
    return
  fi

  if gpu_present; then
    if [ -z "$compute_capability" ]; then
      log "Detected NVIDIA GPU, but unable to determine compute capability."
      log "Ensure nvidia-smi is functional, or install torch first to allow a probe."
      exit 1
    fi

    if [ "$compute_capability" = "12.1" ]; then
      TORCH_INDEX_URL="$TORCH_NIGHTLY_CU130_INDEX_URL"
      TORCH_PLAN_REASON="sm_121 detected -> nightly cu130"
    else
      TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_STABLE_CUDA_CHANNEL}"
      TORCH_PLAN_REASON="stable CUDA ${TORCH_STABLE_CUDA_CHANNEL}"
    fi
  else
    TORCH_INDEX_URL="$TORCH_CPU_INDEX_URL"
    TORCH_PLAN_REASON="no NVIDIA GPU detected"
  fi
}

verify_torch_install() {
  local compute_capability="$1"
  local output
  local status
  set +e
  output="$("$VENV_DIR/bin/python" - <<'PY' 2>&1
import torch
print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"cuda_device={torch.cuda.get_device_name(device)}")
    print(f"cuda_capability={torch.cuda.get_device_capability(device)}")
    print(f"cuda_arch_list={torch.cuda.get_arch_list()}")
    x = torch.randn(1, device=device)
    y = x * 2
    print(f"cuda_smoke_ok={y.item()}")
else:
    print("cuda_available=False")
PY
)"
  status=$?
  set -e
  printf '%s\n' "$output" | while IFS= read -r line; do log "$line"; done

  if [ "$status" -ne 0 ]; then
    if printf '%s' "$output" | grep -i "no kernel image" >/dev/null 2>&1; then
      print_sm121_guidance "${compute_capability:-unknown}"
      exit 1
    fi
    log "PyTorch smoke test failed."
    exit 1
  fi
}
if [ ! -d "$VENV_DIR" ]; then
  log "Creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

log "Installing dependencies"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$COMFY_DIR/requirements.txt"

COMPUTE_CAPABILITY=""
if gpu_present; then
  COMPUTE_CAPABILITY="$(detect_compute_capability || true)"
fi

select_torch_install_plan "$COMPUTE_CAPABILITY"

log "PyTorch install plan:"
log "  GPU present: $(gpu_present && echo yes || echo no)"
log "  compute capability: ${COMPUTE_CAPABILITY:-unknown}"
log "  chosen plan: ${TORCH_PLAN_REASON}"
log "  packages: ${TORCH_PACKAGES}"
log "  index url: ${TORCH_INDEX_URL}"

if gpu_present && [ "$COMPUTE_CAPABILITY" = "12.1" ] && [ "$TORCH_INDEX_URL" != "$TORCH_NIGHTLY_CU130_INDEX_URL" ]; then
  print_sm121_guidance "$COMPUTE_CAPABILITY"
  exit 1
fi

log "Installing PyTorch packages"
if ! "$VENV_DIR/bin/pip" install --index-url "$TORCH_INDEX_URL" $TORCH_PACKAGES; then
  log "PyTorch install failed."
  if gpu_present && [ "$COMPUTE_CAPABILITY" = "12.1" ]; then
    print_sm121_guidance "$COMPUTE_CAPABILITY"
  fi
  exit 1
fi
verify_torch_install "$COMPUTE_CAPABILITY"

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

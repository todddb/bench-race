#!/usr/bin/env bash
# install_vllm.sh
#
# Safe, idempotent installer for vLLM on Linux (Ubuntu/Debian, RHEL-family).
#
# Guarantees:
#   - NEVER uninstalls or overwrites existing Ollama, PyTorch, CUDA, or NVIDIA drivers
#   - NEVER auto-downgrades PyTorch
#   - Detects existing installations and reuses them when compatible
#   - Supports dry-run mode and rollback (--uninstall-vllm)
#   - Idempotent: safe to run multiple times
#   - Clearly logs every decision
#   - Fails safely
#
# Usage:
#   ./install_vllm.sh [OPTIONS]
#
# Options:
#   --yes                 Non-interactive mode (use defaults)
#   --dry-run             Show what would be done without doing it
#   --venv-path PATH      Custom venv path (default: /opt/bench-race/vllm-venv)
#   --model-dir PATH      Model storage directory (default: /mnt/models/vllm)
#   --vllm-port PORT      vLLM server port (default: 8000)
#   --force-managed-venv  Force creating a managed venv even if PyTorch is already installed
#   --skip-service        Skip systemd service installation
#   --uninstall-vllm      Remove vLLM service and venv (leaves Ollama and system PyTorch intact)
#   --gpu-required        Fail if no GPU detected

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============================================================================
# Color Output
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $*" >&2; }
log_success() { echo -e "${GREEN}[OK]${NC} $*" >&2; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_step()    { echo -e "${CYAN}[STEP]${NC} $*" >&2; }
log_decision(){ echo -e "${GREEN}[DECISION]${NC} $*" >&2; }

# ============================================================================
# Defaults
# ============================================================================
DRY_RUN=false
YES_MODE=false
SKIP_SERVICE=false
GPU_REQUIRED=false
FORCE_MANAGED_VENV=false
UNINSTALL=false
VENV_PATH="/opt/bench-race/vllm-venv"
MODEL_DIR="/mnt/models/vllm"
OLLAMA_MODEL_DIR="/mnt/models/ollama"
VLLM_PORT=8000

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --yes)             YES_MODE=true; shift ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --venv-path)       VENV_PATH="$2"; shift 2 ;;
        --model-dir)       MODEL_DIR="$2"; shift 2 ;;
        --vllm-port)       VLLM_PORT="$2"; shift 2 ;;
        --force-managed-venv) FORCE_MANAGED_VENV=true; shift ;;
        --skip-service)    SKIP_SERVICE=true; shift ;;
        --uninstall-vllm)  UNINSTALL=true; shift ;;
        --gpu-required)    GPU_REQUIRED=true; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --yes                Non-interactive mode"
            echo "  --dry-run            Show what would be done"
            echo "  --venv-path PATH     Custom venv path (default: /opt/bench-race/vllm-venv)"
            echo "  --model-dir PATH     Model storage directory (default: /mnt/models/vllm)"
            echo "  --vllm-port PORT     vLLM server port (default: 8000)"
            echo "  --force-managed-venv Force new venv even if PyTorch exists"
            echo "  --skip-service       Skip systemd service installation"
            echo "  --uninstall-vllm     Remove vLLM (keeps Ollama, system PyTorch)"
            echo "  --gpu-required       Fail if no GPU detected"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Dry-run wrapper
# ============================================================================
run() {
    if $DRY_RUN; then
        log_info "[DRY-RUN] Would execute: $*"
    else
        "$@"
    fi
}

confirm() {
    if $YES_MODE; then
        return 0
    fi
    printf "%s [y/N] " "$1" >&2
    read -r answer
    [[ "$answer" =~ ^[Yy] ]]
}

# ============================================================================
# Uninstall mode
# ============================================================================
if $UNINSTALL; then
    log_step "Uninstalling vLLM (preserving Ollama and system PyTorch)"

    # Stop and disable systemd service
    if systemctl is-active --quiet bench-race-vllm 2>/dev/null; then
        log_info "Stopping bench-race-vllm service"
        run sudo systemctl stop bench-race-vllm
    fi
    if systemctl is-enabled --quiet bench-race-vllm 2>/dev/null; then
        log_info "Disabling bench-race-vllm service"
        run sudo systemctl disable bench-race-vllm
    fi
    if [[ -f /etc/systemd/system/bench-race-vllm.service ]]; then
        log_info "Removing systemd unit file"
        run sudo rm -f /etc/systemd/system/bench-race-vllm.service
        run sudo systemctl daemon-reload
    fi

    # Remove venv
    if [[ -d "$VENV_PATH" ]]; then
        log_info "Removing vLLM venv at $VENV_PATH"
        run sudo rm -rf "$VENV_PATH"
    fi

    log_success "vLLM uninstalled."
    log_info "Ollama installation: UNTOUCHED"
    log_info "System PyTorch: UNTOUCHED"
    log_info "Model files in $MODEL_DIR: PRESERVED (remove manually if desired)"
    exit 0
fi

# ============================================================================
# Phase 1: Platform & Hardware Detection
# ============================================================================
log_step "Phase 1: Detecting platform and hardware"

# OS detection
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    OS_ID="${ID:-unknown}"
    OS_VERSION="${VERSION_ID:-unknown}"
    OS_PRETTY="${PRETTY_NAME:-$OS_ID $OS_VERSION}"
else
    OS_ID="unknown"
    OS_VERSION="unknown"
    OS_PRETTY="Unknown Linux"
fi
log_info "OS: $OS_PRETTY"

ARCH=$(uname -m)
log_info "Architecture: $ARCH"

# GPU detection
GPU_DETECTED=false
GPU_NAME=""
CUDA_VERSION=""
CUDA_MAJOR=""
IS_BLACKWELL=false

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs || true)
    if [[ -n "$GPU_NAME" ]]; then
        GPU_DETECTED=true
        log_info "GPU detected: $GPU_NAME"

        # Check for Blackwell/GB10
        if echo "$GPU_NAME" | grep -qiE "gb10|blackwell|b100|b200"; then
            IS_BLACKWELL=true
            log_info "Blackwell/GB10 GPU detected"
        fi
    fi

    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | xargs || true)
    NVCC_VERSION=""
    if command -v nvcc &>/dev/null; then
        NVCC_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' || true)
    fi

    if [[ -n "$NVCC_VERSION" ]]; then
        CUDA_VERSION="$NVCC_VERSION"
    elif [[ -f /usr/local/cuda/version.txt ]]; then
        CUDA_VERSION=$(cat /usr/local/cuda/version.txt | sed -n 's/CUDA Version \([0-9]*\.[0-9]*\).*/\1/p' || true)
    fi

    if [[ -n "$CUDA_VERSION" ]]; then
        CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
        log_info "CUDA version: $CUDA_VERSION"
    else
        log_warning "NVIDIA GPU detected but CUDA version could not be determined"
    fi
else
    log_info "No NVIDIA GPU detected"
fi

if $GPU_REQUIRED && ! $GPU_DETECTED; then
    log_error "GPU required but none detected. Aborting."
    exit 1
fi

# ============================================================================
# Phase 2: Check existing installations (NEVER modify them)
# ============================================================================
log_step "Phase 2: Checking existing installations"

# Check Ollama
if command -v ollama &>/dev/null; then
    OLLAMA_VER=$(ollama --version 2>/dev/null || echo "unknown")
    log_info "Ollama installation found: $OLLAMA_VER"
    log_decision "Ollama will NOT be modified"
else
    log_info "Ollama not found (this is OK for vLLM installation)"
fi

# Check existing PyTorch
EXISTING_TORCH=false
EXISTING_TORCH_VERSION=""
EXISTING_TORCH_CUDA=""
EXISTING_TORCH_PATH=""

check_torch() {
    local python_bin="$1"
    local result
    result=$("$python_bin" -c "
import torch
cuda_available = torch.cuda.is_available()
cuda_version = torch.version.cuda if cuda_available else 'none'
print(f'{torch.__version__}|{cuda_version}|{cuda_available}')
" 2>/dev/null) || return 1

    IFS='|' read -r ver cuda avail <<< "$result"
    EXISTING_TORCH_VERSION="$ver"
    EXISTING_TORCH_CUDA="$cuda"
    EXISTING_TORCH=true
    EXISTING_TORCH_PATH="$python_bin"
    return 0
}

# Check system Python first
for py in python3 python; do
    if command -v "$py" &>/dev/null; then
        if check_torch "$py"; then
            log_info "Existing PyTorch found: $EXISTING_TORCH_VERSION (CUDA: $EXISTING_TORCH_CUDA) via $EXISTING_TORCH_PATH"
            log_decision "System PyTorch will NOT be modified or downgraded"
            break
        fi
    fi
done

# Check if existing torch supports current CUDA
TORCH_CUDA_COMPATIBLE=false
if $EXISTING_TORCH && $GPU_DETECTED; then
    if [[ "$EXISTING_TORCH_CUDA" != "none" && -n "$EXISTING_TORCH_CUDA" ]]; then
        EXISTING_CUDA_MAJOR=$(echo "$EXISTING_TORCH_CUDA" | cut -d. -f1)
        if [[ "$EXISTING_CUDA_MAJOR" == "$CUDA_MAJOR" ]]; then
            TORCH_CUDA_COMPATIBLE=true
            log_info "Existing PyTorch CUDA support ($EXISTING_TORCH_CUDA) is compatible with system CUDA ($CUDA_VERSION)"
        else
            log_warning "Existing PyTorch CUDA ($EXISTING_TORCH_CUDA) may not match system CUDA ($CUDA_VERSION)"
        fi
    fi
fi

# ============================================================================
# Phase 3: Determine installation mode
# ============================================================================
log_step "Phase 3: Determining installation mode"

# Mode 1: Existing PyTorch mode (reuse compatible torch)
# Mode 2: Managed venv mode (create isolated venv with correct torch)
INSTALL_MODE=""

if $EXISTING_TORCH && $TORCH_CUDA_COMPATIBLE && ! $FORCE_MANAGED_VENV; then
    INSTALL_MODE="existing_pytorch"
    log_decision "Mode 1: Using existing compatible PyTorch ($EXISTING_TORCH_VERSION)"
    log_decision "Will install vLLM into isolated venv that inherits system torch"
elif $EXISTING_TORCH && $IS_BLACKWELL && ! $FORCE_MANAGED_VENV; then
    # GB10/Blackwell special handling: ask before doing anything
    log_warning "Blackwell/GB10 GPU detected with existing PyTorch ($EXISTING_TORCH_VERSION)"
    if ! $TORCH_CUDA_COMPATIBLE; then
        log_warning "Existing PyTorch may not support Blackwell GPU"
        log_info "A nightly PyTorch build may be required for full Blackwell support"
        if ! $YES_MODE; then
            if confirm "Install nightly PyTorch in isolated venv? (system PyTorch will NOT be modified)"; then
                INSTALL_MODE="managed_venv"
                log_decision "Mode 2: Creating isolated managed venv with nightly PyTorch"
            else
                INSTALL_MODE="existing_pytorch"
                log_decision "Mode 1: Proceeding with existing PyTorch (may have limited GPU support)"
            fi
        else
            INSTALL_MODE="managed_venv"
            log_decision "Mode 2: Creating managed venv with nightly PyTorch (--yes mode, GB10)"
        fi
    else
        INSTALL_MODE="existing_pytorch"
        log_decision "Mode 1: Using existing compatible PyTorch ($EXISTING_TORCH_VERSION) on GB10"
    fi
else
    INSTALL_MODE="managed_venv"
    log_decision "Mode 2: Creating managed venv with PyTorch + vLLM"
fi

# ============================================================================
# Phase 4: Select correct PyTorch wheel (for managed venv mode)
# ============================================================================
TORCH_INSTALL_CMD=""
TORCH_REASON=""

if [[ "$INSTALL_MODE" == "managed_venv" ]]; then
    log_step "Phase 4: Selecting PyTorch wheel"

    if $IS_BLACKWELL; then
        # Blackwell needs nightly with CUDA 12.8+
        TORCH_INSTALL_CMD="pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128"
        TORCH_REASON="Blackwell/GB10 GPU requires nightly PyTorch with CUDA 12.8+ support"
    elif $GPU_DETECTED && [[ -n "$CUDA_MAJOR" ]]; then
        case "$CUDA_MAJOR" in
            12)
                TORCH_INSTALL_CMD="pip install torch --index-url https://download.pytorch.org/whl/cu121"
                TORCH_REASON="CUDA $CUDA_VERSION detected; using stable PyTorch with CUDA 12.1 support"
                ;;
            11)
                TORCH_INSTALL_CMD="pip install torch --index-url https://download.pytorch.org/whl/cu118"
                TORCH_REASON="CUDA $CUDA_VERSION detected; using stable PyTorch with CUDA 11.8 support"
                ;;
            *)
                TORCH_INSTALL_CMD="pip install torch"
                TORCH_REASON="CUDA $CUDA_VERSION detected; using default PyTorch (may need manual CUDA config)"
                ;;
        esac
    else
        TORCH_INSTALL_CMD="pip install torch --index-url https://download.pytorch.org/whl/cpu"
        TORCH_REASON="No GPU detected; using CPU-only PyTorch"
    fi

    log_decision "PyTorch selection: $TORCH_REASON"
    log_info "Install command: $TORCH_INSTALL_CMD"
fi

# ============================================================================
# Phase 5: Create directories (never destroy existing ones)
# ============================================================================
log_step "Phase 5: Creating directories"

create_dir_safe() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        log_info "Directory exists: $dir (preserving)"
    else
        log_info "Creating directory: $dir"
        run sudo mkdir -p "$dir"
    fi
    # Ensure permissions without changing ownership of existing files
    run sudo chmod 755 "$dir"
}

create_dir_safe "$MODEL_DIR"
create_dir_safe "$OLLAMA_MODEL_DIR"
create_dir_safe "$(dirname "$VENV_PATH")"

# ============================================================================
# Phase 6: Create/update venv and install vLLM
# ============================================================================
log_step "Phase 6: Installing vLLM"

if [[ "$INSTALL_MODE" == "existing_pytorch" ]]; then
    # Mode 1: Install vLLM into isolated venv, inherit system site-packages
    if [[ -d "$VENV_PATH" ]]; then
        log_info "vLLM venv already exists at $VENV_PATH"
        # Check if vLLM is already installed
        if "$VENV_PATH/bin/python" -c "import vllm" 2>/dev/null; then
            VLLM_VER=$("$VENV_PATH/bin/python" -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
            log_info "vLLM already installed: $VLLM_VER"
            log_decision "Skipping vLLM reinstall (already present)"
        else
            log_info "venv exists but vLLM not found; installing"
            run "$VENV_PATH/bin/pip" install --upgrade pip
            run "$VENV_PATH/bin/pip" install vllm
        fi
    else
        log_info "Creating venv at $VENV_PATH (with system site-packages for PyTorch)"
        run sudo python3 -m venv --system-site-packages "$VENV_PATH"
        run sudo "$VENV_PATH/bin/pip" install --upgrade pip
        run sudo "$VENV_PATH/bin/pip" install vllm
    fi

elif [[ "$INSTALL_MODE" == "managed_venv" ]]; then
    # Mode 2: Create fully managed venv
    if [[ -d "$VENV_PATH" ]]; then
        log_info "Managed venv already exists at $VENV_PATH"
        if "$VENV_PATH/bin/python" -c "import vllm" 2>/dev/null; then
            VLLM_VER=$("$VENV_PATH/bin/python" -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
            log_info "vLLM already installed: $VLLM_VER"
            log_decision "Skipping vLLM reinstall (already present)"
        else
            log_info "venv exists but vLLM not found; installing torch + vLLM"
            run "$VENV_PATH/bin/pip" install --upgrade pip
            # shellcheck disable=SC2086
            run "$VENV_PATH/bin/$TORCH_INSTALL_CMD"
            run "$VENV_PATH/bin/pip" install vllm
        fi
    else
        log_info "Creating managed venv at $VENV_PATH"
        run sudo python3 -m venv "$VENV_PATH"
        run sudo "$VENV_PATH/bin/pip" install --upgrade pip
        log_info "Installing PyTorch: $TORCH_INSTALL_CMD"
        # shellcheck disable=SC2086
        run sudo "$VENV_PATH/bin/$TORCH_INSTALL_CMD"
        log_info "Installing vLLM"
        run sudo "$VENV_PATH/bin/pip" install vllm
    fi
fi

# Install uvicorn for serving
if [[ -d "$VENV_PATH" ]] && ! $DRY_RUN; then
    if ! "$VENV_PATH/bin/python" -c "import uvicorn" 2>/dev/null; then
        log_info "Installing uvicorn"
        run sudo "$VENV_PATH/bin/pip" install uvicorn
    fi
fi

# ============================================================================
# Phase 7: Systemd service (idempotent)
# ============================================================================
if ! $SKIP_SERVICE; then
    log_step "Phase 7: Configuring systemd service"

    SERVICE_FILE="/etc/systemd/system/bench-race-vllm.service"
    SERVICE_TEMPLATE="${REPO_ROOT}/deploy/bench-race-vllm.service"

    if [[ -f "$SERVICE_TEMPLATE" ]]; then
        # Generate service file from template
        GENERATED_SERVICE=$(sed \
            -e "s|__VENV_PATH__|${VENV_PATH}|g" \
            -e "s|__MODEL_DIR__|${MODEL_DIR}|g" \
            -e "s|__VLLM_PORT__|${VLLM_PORT}|g" \
            "$SERVICE_TEMPLATE")

        if [[ -f "$SERVICE_FILE" ]]; then
            EXISTING_SERVICE=$(cat "$SERVICE_FILE")
            if [[ "$EXISTING_SERVICE" == "$GENERATED_SERVICE" ]]; then
                log_info "Systemd service unchanged, leaving as-is"
            else
                log_info "Systemd service configuration changed, updating"
                run bash -c "echo '$GENERATED_SERVICE' | sudo tee '$SERVICE_FILE' > /dev/null"
                run sudo systemctl daemon-reload
                if systemctl is-active --quiet bench-race-vllm 2>/dev/null; then
                    log_info "Restarting service due to config change"
                    run sudo systemctl restart bench-race-vllm
                fi
            fi
        else
            log_info "Installing systemd service"
            run bash -c "echo '$GENERATED_SERVICE' | sudo tee '$SERVICE_FILE' > /dev/null"
            run sudo systemctl daemon-reload
            run sudo systemctl enable bench-race-vllm
        fi
    else
        log_warning "Service template not found at $SERVICE_TEMPLATE; skipping service setup"
    fi
else
    log_info "Skipping systemd service installation (--skip-service)"
fi

# ============================================================================
# Phase 8: Update agent config
# ============================================================================
log_step "Phase 8: Updating agent configuration"

AGENT_CONFIG="${REPO_ROOT}/agent/config/agent.yaml"
if [[ -f "$AGENT_CONFIG" ]]; then
    if grep -q "vllm:" "$AGENT_CONFIG" 2>/dev/null; then
        log_info "vLLM config already present in agent.yaml"
    else
        log_info "Adding vLLM config to agent.yaml"
        if ! $DRY_RUN; then
            cat >> "$AGENT_CONFIG" <<EOF

# vLLM backend configuration (added by install_vllm.sh)
vllm:
  enabled: true
  base_url: "http://127.0.0.1:${VLLM_PORT}"
EOF
        fi
    fi
else
    log_warning "Agent config not found at $AGENT_CONFIG; run install_agent.sh first"
fi

# ============================================================================
# Phase 9: Smoke tests
# ============================================================================
log_step "Phase 9: Running smoke tests"

SMOKE_PASS=true

if ! $DRY_RUN && [[ -d "$VENV_PATH" ]]; then
    # Test 1: torch import
    if "$VENV_PATH/bin/python" -c "import torch; print(f'torch {torch.__version__} OK')" 2>/dev/null; then
        log_success "Smoke test: torch import passed"
    else
        log_warning "Smoke test: torch import failed (vLLM may still work for CPU)"
    fi

    # Test 2: GPU availability
    if $GPU_DETECTED; then
        if "$VENV_PATH/bin/python" -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(f'CUDA OK: {torch.cuda.get_device_name(0)}')" 2>/dev/null; then
            log_success "Smoke test: GPU available in venv"
        else
            log_warning "Smoke test: GPU not available in vLLM venv"
            if $GPU_REQUIRED; then
                SMOKE_PASS=false
            fi
        fi
    fi

    # Test 3: vLLM import
    if "$VENV_PATH/bin/python" -c "import vllm; print(f'vllm {vllm.__version__} OK')" 2>/dev/null; then
        log_success "Smoke test: vLLM import passed"
    else
        log_error "Smoke test: vLLM import failed"
        SMOKE_PASS=false
    fi

    # Test 4: Check if vLLM server is running
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
        log_success "Smoke test: vLLM server healthcheck passed"
    else
        log_info "Smoke test: vLLM server not running (expected if no model loaded yet)"
    fi
else
    log_info "Skipping smoke tests (dry-run or venv not found)"
fi

if ! $SMOKE_PASS; then
    log_error "Smoke tests failed. vLLM installation may not be functional."
    exit 1
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
log_success "============================================"
log_success "vLLM installation complete"
log_success "============================================"
echo ""
log_info "Installation mode: $INSTALL_MODE"
log_info "vLLM venv:         $VENV_PATH"
log_info "Model directory:   $MODEL_DIR"
log_info "vLLM port:         $VLLM_PORT"
echo ""
log_info "Preserved (not modified):"
log_info "  - Ollama installation"
log_info "  - System PyTorch"
log_info "  - CUDA/NVIDIA drivers"
log_info "  - Existing model files"
echo ""
log_info "Next steps:"
log_info "  1. Start a model:  $VENV_PATH/bin/vllm serve <model-name> --port $VLLM_PORT"
log_info "  2. Or enable the systemd service: sudo systemctl start bench-race-vllm"
log_info "  3. Restart the bench-race agent to detect vLLM"
echo ""
log_info "To remove vLLM: $0 --uninstall-vllm"

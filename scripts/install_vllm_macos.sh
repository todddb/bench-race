#!/usr/bin/env bash
# install_vllm_macos.sh
#
# Safe, idempotent installer for vLLM on macOS (Intel and Apple Silicon).
#
# Guarantees:
#   - NEVER modifies existing Ollama, PyTorch, or Homebrew packages
#   - Idempotent: safe to run multiple times
#   - Supports dry-run mode and rollback (--uninstall-vllm)
#   - Clearly logs every decision
#
# Note: vLLM on macOS has limited GPU support. Apple Silicon uses MPS backend.
#       For production GPU inference on macOS, Ollama is typically preferred.
#
# Usage:
#   ./install_vllm_macos.sh [OPTIONS]
#
# Options:
#   --yes                 Non-interactive mode
#   --dry-run             Show what would be done
#   --venv-path PATH      Custom venv path (default: ~/bench-race/vllm-venv)
#   --model-dir PATH      Model storage directory (default: ~/bench-race/models/vllm)
#   --vllm-port PORT      vLLM server port (default: 8000)
#   --skip-service        Skip launchd service installation
#   --uninstall-vllm      Remove vLLM (keeps Ollama, system PyTorch)

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
UNINSTALL=false
VENV_PATH="$HOME/bench-race/vllm-venv"
MODEL_DIR="$HOME/bench-race/models/vllm"
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
        --skip-service)    SKIP_SERVICE=true; shift ;;
        --uninstall-vllm)  UNINSTALL=true; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --yes                Non-interactive mode"
            echo "  --dry-run            Show what would be done"
            echo "  --venv-path PATH     Custom venv path (default: ~/bench-race/vllm-venv)"
            echo "  --model-dir PATH     Model storage directory (default: ~/bench-race/models/vllm)"
            echo "  --vllm-port PORT     vLLM server port (default: 8000)"
            echo "  --skip-service       Skip launchd service installation"
            echo "  --uninstall-vllm     Remove vLLM (keeps Ollama, system PyTorch)"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Helpers
# ============================================================================
run() {
    if $DRY_RUN; then
        log_info "[DRY-RUN] Would execute: $*"
    else
        "$@"
    fi
}

confirm() {
    if $YES_MODE; then return 0; fi
    printf "%s [y/N] " "$1" >&2
    read -r answer
    [[ "$answer" =~ ^[Yy] ]]
}

PLIST_LABEL="com.bench-race.vllm"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_LABEL}.plist"

# ============================================================================
# Uninstall mode
# ============================================================================
if $UNINSTALL; then
    log_step "Uninstalling vLLM (preserving Ollama and system PyTorch)"

    # Unload launchd agent
    if launchctl list | grep -q "$PLIST_LABEL" 2>/dev/null; then
        log_info "Unloading launchd agent"
        run launchctl unload "$PLIST_PATH" 2>/dev/null || true
    fi
    if [[ -f "$PLIST_PATH" ]]; then
        log_info "Removing launchd plist"
        run rm -f "$PLIST_PATH"
    fi

    # Remove venv
    if [[ -d "$VENV_PATH" ]]; then
        log_info "Removing vLLM venv at $VENV_PATH"
        run rm -rf "$VENV_PATH"
    fi

    log_success "vLLM uninstalled."
    log_info "Ollama installation: UNTOUCHED"
    log_info "System PyTorch: UNTOUCHED"
    log_info "Model files in $MODEL_DIR: PRESERVED (remove manually if desired)"
    exit 0
fi

# ============================================================================
# Platform detection
# ============================================================================
log_step "Phase 1: Detecting platform"

if [[ "$(uname)" != "Darwin" ]]; then
    log_error "This script is for macOS only. Use install_vllm.sh for Linux."
    exit 1
fi

ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    log_info "Platform: macOS Apple Silicon ($ARCH)"
    HAS_MPS=true
else
    log_info "Platform: macOS Intel ($ARCH)"
    HAS_MPS=false
fi

MACOS_VER=$(sw_vers -productVersion)
log_info "macOS version: $MACOS_VER"

# Check Ollama
if command -v ollama &>/dev/null; then
    log_info "Ollama found (will NOT be modified)"
fi

# Check existing PyTorch
EXISTING_TORCH=false
for py in python3 python; do
    if command -v "$py" &>/dev/null; then
        if "$py" -c "import torch; print(f'torch {torch.__version__}')" 2>/dev/null; then
            EXISTING_TORCH=true
            log_info "Existing PyTorch found (will NOT be modified)"
            break
        fi
    fi
done

# ============================================================================
# Create directories
# ============================================================================
log_step "Phase 2: Creating directories"

mkdir -p "$MODEL_DIR" 2>/dev/null || true
mkdir -p "$(dirname "$VENV_PATH")" 2>/dev/null || true
log_info "Model directory: $MODEL_DIR"

# ============================================================================
# Create venv and install vLLM
# ============================================================================
log_step "Phase 3: Installing vLLM"

if [[ -d "$VENV_PATH" ]]; then
    log_info "venv exists at $VENV_PATH"
    if "$VENV_PATH/bin/python" -c "import vllm" 2>/dev/null; then
        VLLM_VER=$("$VENV_PATH/bin/python" -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
        log_info "vLLM already installed: $VLLM_VER"
        log_decision "Skipping reinstall"
    else
        log_info "venv exists but vLLM not found; installing"
        run "$VENV_PATH/bin/pip" install --upgrade pip
        run "$VENV_PATH/bin/pip" install torch torchvision
        run "$VENV_PATH/bin/pip" install vllm
    fi
else
    log_info "Creating venv at $VENV_PATH"
    run python3 -m venv "$VENV_PATH"
    run "$VENV_PATH/bin/pip" install --upgrade pip
    log_info "Installing PyTorch"
    run "$VENV_PATH/bin/pip" install torch torchvision
    log_info "Installing vLLM"
    run "$VENV_PATH/bin/pip" install vllm
fi

# ============================================================================
# Launchd service (idempotent)
# ============================================================================
if ! $SKIP_SERVICE; then
    log_step "Phase 4: Configuring launchd service"

    LAUNCHD_TEMPLATE="${REPO_ROOT}/deploy/com.bench-race.vllm.plist"

    if [[ -f "$LAUNCHD_TEMPLATE" ]]; then
        GENERATED_PLIST=$(sed \
            -e "s|__VENV_PATH__|${VENV_PATH}|g" \
            -e "s|__MODEL_DIR__|${MODEL_DIR}|g" \
            -e "s|__VLLM_PORT__|${VLLM_PORT}|g" \
            "$LAUNCHD_TEMPLATE")

        mkdir -p "$HOME/Library/LaunchAgents"

        if [[ -f "$PLIST_PATH" ]]; then
            EXISTING_PLIST=$(cat "$PLIST_PATH")
            if [[ "$EXISTING_PLIST" == "$GENERATED_PLIST" ]]; then
                log_info "Launchd plist unchanged"
            else
                log_info "Updating launchd plist"
                launchctl unload "$PLIST_PATH" 2>/dev/null || true
                run bash -c "echo '$GENERATED_PLIST' > '$PLIST_PATH'"
                run launchctl load "$PLIST_PATH"
            fi
        else
            log_info "Installing launchd plist"
            run bash -c "echo '$GENERATED_PLIST' > '$PLIST_PATH'"
            run launchctl load "$PLIST_PATH"
        fi
    else
        log_warning "Launchd template not found at $LAUNCHD_TEMPLATE; skipping service setup"
    fi
else
    log_info "Skipping launchd service installation"
fi

# ============================================================================
# Update agent config
# ============================================================================
log_step "Phase 5: Updating agent configuration"

AGENT_CONFIG="${REPO_ROOT}/agent/config/agent.yaml"
if [[ -f "$AGENT_CONFIG" ]]; then
    if grep -q "vllm:" "$AGENT_CONFIG" 2>/dev/null; then
        log_info "vLLM config already present in agent.yaml"
    else
        log_info "Adding vLLM config to agent.yaml"
        if ! $DRY_RUN; then
            cat >> "$AGENT_CONFIG" <<EOF

# vLLM backend configuration (added by install_vllm_macos.sh)
vllm:
  enabled: true
  base_url: "http://127.0.0.1:${VLLM_PORT}"
EOF
        fi
    fi
else
    log_warning "Agent config not found at $AGENT_CONFIG"
fi

# ============================================================================
# Smoke tests
# ============================================================================
log_step "Phase 6: Running smoke tests"

SMOKE_PASS=true
if ! $DRY_RUN && [[ -d "$VENV_PATH" ]]; then
    if "$VENV_PATH/bin/python" -c "import torch; print(f'torch {torch.__version__} OK')" 2>/dev/null; then
        log_success "Smoke test: torch import passed"
    else
        log_warning "Smoke test: torch import failed"
    fi

    if $HAS_MPS; then
        if "$VENV_PATH/bin/python" -c "import torch; assert torch.backends.mps.is_available(), 'no MPS'; print('MPS OK')" 2>/dev/null; then
            log_success "Smoke test: MPS (Apple GPU) available"
        else
            log_warning "Smoke test: MPS not available"
        fi
    fi

    if "$VENV_PATH/bin/python" -c "import vllm; print(f'vllm {vllm.__version__} OK')" 2>/dev/null; then
        log_success "Smoke test: vLLM import passed"
    else
        log_error "Smoke test: vLLM import failed"
        SMOKE_PASS=false
    fi
fi

if ! $SMOKE_PASS; then
    log_error "Smoke tests failed."
    exit 1
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
log_success "============================================"
log_success "vLLM installation complete (macOS)"
log_success "============================================"
echo ""
log_info "vLLM venv:       $VENV_PATH"
log_info "Model directory: $MODEL_DIR"
log_info "vLLM port:       $VLLM_PORT"
echo ""
log_info "Preserved (not modified):"
log_info "  - Ollama installation"
log_info "  - System PyTorch"
log_info "  - Homebrew packages"
echo ""
log_info "To remove vLLM: $0 --uninstall-vllm"

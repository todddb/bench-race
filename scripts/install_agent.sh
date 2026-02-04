#!/usr/bin/env bash
# install_agent.sh
#
# Consolidated installer for bench-race agent
# - Detects platform (macOS/Linux, architecture, GB10 variant)
# - Installs Ollama, ComfyUI, and Agent Python environment
# - Generates minimal agent.yaml configuration
# - Runs smoke tests
# - Idempotent: safe to re-run for updates
#
# Usage:
#   ./install_agent.sh [OPTIONS]
#
# Options:
#   --agent-id ID         Agent identifier (default: hostname)
#   --label "Label"       Human-readable label for UI
#   --central-url URL     Central server URL (default: http://127.0.0.1:8080)
#   --platform PLATFORM   Override platform detection (macos|linux|linux-gb10)
#   --yes                 Non-interactive mode (use defaults)
#   --no-service          Skip systemd/launchctl service installation
#   --update              Update existing installation
#   --dry-run             Show what would be done without doing it
#   --skip-ollama         Skip Ollama installation
#   --skip-comfyui        Skip ComfyUI installation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AGENT_DIR="${REPO_ROOT}/agent"
CONFIG_DIR="${AGENT_DIR}/config"
CONFIG_FILE="${CONFIG_DIR}/agent.yaml"
COMFY_DIR="${AGENT_DIR}/third_party/comfyui"

# ============================================================================
# Color Output
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $*" >&2
}

# Prompt helper - outputs to stderr
prompt() {
    printf "%s" "$*" >&2
}

# ============================================================================
# Global Variables
# ============================================================================

PLATFORM=""
OS_TYPE=""
ARCH=""
IS_GB10=false
DRY_RUN=false
YES_MODE=false
NO_SERVICE=false
UPDATE_MODE=false
SKIP_OLLAMA=false
SKIP_COMFYUI=false

AGENT_ID=""
LABEL=""
CENTRAL_URL=""

# ============================================================================
# Helper Functions
# ============================================================================

sanitize_id() {
    local input="$1"
    echo "$input" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]-' '-' | sed 's/^-//;s/-$//'
}

# Sanitize values by removing ASCII control characters and ANSI escape sequences
sanitize() {
    # Remove ASCII control chars (0x00-0x1F), DEL (0x7F), and CR
    printf '%s' "$1" | LC_ALL=C tr -d '\000-\037\177'
}

# Read a value from YAML file
# Usage: read_yaml_value "key_name" "file_path"
read_yaml_value() {
    local key="$1"
    local file="$2"

    if [[ ! -f "$file" ]]; then
        return 1
    fi

    # Parse key: value lines, handling quoted and unquoted values
    awk -v k="$key" '
        $0 ~ "^[[:space:]]*"k":[[:space:]]*" {
            sub("^[[:space:]]*"k":[[:space:]]*", "", $0);
            gsub(/[[:space:]]+$/, "", $0);
            # Strip surrounding double quotes
            if ($0 ~ /^".*"$/) {
                sub(/^"/, "", $0);
                sub(/"$/, "", $0);
            }
            # Strip surrounding single quotes
            if ($0 ~ /^\047.*\047$/) {
                sub(/^\047/, "", $0);
                sub(/\047$/, "", $0);
            }
            print $0;
            exit
        }
    ' "$file"
}

# Validate agent_id format
validate_agent_id() {
    local id="$1"

    if [[ -z "$id" ]]; then
        log_error "agent_id cannot be empty"
        return 1
    fi

    # Check if it contains only allowed characters
    if [[ ! "$id" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        log_warning "agent_id contains unusual characters: $id"
        log_warning "Recommended format: letters, numbers, underscores, hyphens only"
    fi

    return 0
}

run_with_timeout() {
    local timeout_seconds="$1"
    shift
    if [[ "$OS_TYPE" == "macos" ]]; then
        # macOS doesn't have timeout by default, use perl fallback
        perl -e "alarm ${timeout_seconds}; exec @ARGV" "$@" 2>/dev/null || true
    else
        timeout "${timeout_seconds}" "$@" 2>/dev/null || true
    fi
}

run_command() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] Would execute: $*"
        return 0
    fi
    "$@"
}

# ============================================================================
# Platform Detection
# ============================================================================

detect_platform() {
    log_step "Detecting platform..."

    local uname_s
    uname_s=$(uname -s)

    case "$uname_s" in
        Darwin)
            OS_TYPE="macos"
            ;;
        Linux)
            OS_TYPE="linux"
            ;;
        *)
            log_error "Unsupported operating system: $uname_s"
            exit 1
            ;;
    esac

    ARCH=$(uname -m)

    # Detect GB10 variant on Linux (example detection logic)
    if [[ "$OS_TYPE" == "linux" ]]; then
        local cpu_model
        cpu_model=$(run_with_timeout 1 lscpu | grep "Model name" | sed 's/Model name:[[:space:]]*//' | head -1 || echo "")

        # Check for GB10 indicators in CPU model or other system info
        if [[ "$cpu_model" =~ GB10 ]] || [[ "$cpu_model" =~ "Grace" ]]; then
            IS_GB10=true
            PLATFORM="linux-gb10"
        else
            PLATFORM="linux"
        fi
    else
        PLATFORM="macos"
    fi

    log_info "Platform: ${PLATFORM}"
    log_info "OS: ${OS_TYPE}"
    log_info "Architecture: ${ARCH}"
    if [[ "$IS_GB10" == true ]]; then
        log_info "GB10 variant detected"
    fi
}

detect_hostname() {
    hostname -s 2>/dev/null || hostname 2>/dev/null || echo "unknown-host"
}

detect_cpu_cores() {
    if [[ "$OS_TYPE" == "macos" ]]; then
        sysctl -n hw.ncpu 2>/dev/null || echo "unknown"
    else
        nproc 2>/dev/null || echo "unknown"
    fi
}

detect_cpu_name() {
    if [[ "$OS_TYPE" == "macos" ]]; then
        local cpu_name
        cpu_name=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "")
        # Simplify Apple Silicon naming
        if [[ "$cpu_name" == *"Apple"* ]]; then
            cpu_name=$(echo "$cpu_name" | grep -o "Apple M[0-9][^,]*" | sed 's/Apple //')
        fi
        echo "${cpu_name:-Unknown CPU}"
    else
        local cpu_name
        cpu_name=$(run_with_timeout 1 lscpu | grep "Model name" | sed 's/Model name:[[:space:]]*//' | head -1)
        if [[ -z "$cpu_name" ]]; then
            cpu_name=$(run_with_timeout 1 grep "model name" /proc/cpuinfo | head -1 | sed 's/.*: //')
        fi
        echo "${cpu_name:-Unknown CPU}"
    fi
}

detect_ram_gb() {
    if [[ "$OS_TYPE" == "macos" ]]; then
        local ram_bytes
        ram_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
        if [[ -n "$ram_bytes" && "$ram_bytes" != "0" ]]; then
            echo $((ram_bytes / 1024 / 1024 / 1024))
        else
            echo "unknown"
        fi
    else
        local ram_bytes
        ram_bytes=$(run_with_timeout 1 free -b | awk '/^Mem:/ {print $2}')
        if [[ -n "$ram_bytes" && "$ram_bytes" != "0" ]]; then
            echo $((ram_bytes / 1024 / 1024 / 1024))
        else
            echo "unknown"
        fi
    fi
}

detect_gpu_info() {
    if [[ "$OS_TYPE" == "macos" ]]; then
        # For Apple Silicon, use unified memory
        local cpu_brand
        cpu_brand=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "")
        if [[ "$cpu_brand" == *"Apple"* ]]; then
            local gpu_name
            gpu_name=$(detect_cpu_name)
            local ram_gb
            ram_gb=$(detect_ram_gb)
            echo "${gpu_name} (Unified ${ram_gb}GB)"
        else
            # Intel Mac
            local gpu_data
            gpu_data=$(run_with_timeout 5 system_profiler SPDisplaysDataType 2>/dev/null | grep "Chipset Model:" | head -1 | sed 's/.*: //' || echo "Unknown GPU")
            echo "$gpu_data"
        fi
    else
        # Linux GPU detection
        if command -v nvidia-smi &>/dev/null; then
            local gpu_data
            gpu_data=$(run_with_timeout 2 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
            if [[ -n "$gpu_data" ]]; then
                local gpu_name=$(echo "$gpu_data" | cut -d',' -f1 | xargs)
                local vram_mib=$(echo "$gpu_data" | cut -d',' -f2 | sed 's/[^0-9]//g')
                if [[ -n "$vram_mib" ]]; then
                    local gpu_vram_gb=$((vram_mib / 1024))
                    echo "${gpu_name} (${gpu_vram_gb}GB VRAM)"
                else
                    echo "${gpu_name}"
                fi
                return
            fi
        fi

        # Fallback to lspci
        if command -v lspci &>/dev/null; then
            local gpu_name
            gpu_name=$(run_with_timeout 1 lspci | grep -i "VGA\|3D\|Display" | head -1 | sed 's/.*: //' || echo "")
            if [[ -n "$gpu_name" ]]; then
                echo "$gpu_name"
                return
            fi
        fi

        echo "CPU-only"
    fi
}

# ============================================================================
# Prerequisites Check
# ============================================================================

ensure_prereqs() {
    log_step "Checking prerequisites..."

    local missing_tools=()

    # Check for required tools
    if ! command -v curl &>/dev/null; then
        missing_tools+=("curl")
    fi

    if ! command -v git &>/dev/null; then
        missing_tools+=("git")
    fi

    if ! command -v python3 &>/dev/null; then
        missing_tools+=("python3")
    fi

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install them and re-run this script."

        if [[ "$OS_TYPE" == "linux" ]]; then
            log_info "On Debian/Ubuntu: sudo apt-get install ${missing_tools[*]}"
            log_info "On RHEL/CentOS: sudo dnf install ${missing_tools[*]}"
        elif [[ "$OS_TYPE" == "macos" ]]; then
            log_info "On macOS: brew install ${missing_tools[*]}"
        fi

        exit 1
    fi

    # Check for Python venv module
    if ! python3 -m venv --help &>/dev/null; then
        log_error "Python venv module not available"
        if [[ "$OS_TYPE" == "linux" ]]; then
            log_info "Install with: sudo apt-get install python3-venv"
        fi
        exit 1
    fi

    log_success "All prerequisites satisfied"
}

# ============================================================================
# Ollama Installation
# ============================================================================

install_or_update_ollama() {
    if [[ "$SKIP_OLLAMA" == true ]]; then
        log_info "Skipping Ollama installation (--skip-ollama)"
        return 0
    fi

    log_step "Installing/updating Ollama..."

    if command -v ollama &>/dev/null; then
        log_info "Ollama already installed: $(ollama --version 2>/dev/null || echo 'version unknown')"
        if [[ "$UPDATE_MODE" != true ]]; then
            log_info "Use --update to upgrade Ollama"
            return 0
        fi
    fi

    if [[ "$OS_TYPE" == "macos" ]]; then
        if command -v brew &>/dev/null; then
            log_info "Installing Ollama via Homebrew..."
            run_command brew install ollama
        else
            log_warning "Homebrew not found."
            log_info "Install Ollama from the macOS DMG: https://ollama.com/download/mac"
            if [[ "$YES_MODE" != true ]]; then
                read -rp "Press Enter to continue after installing Ollama..."
            fi
        fi
    else
        # Linux installation
        # Check for zstd (required by Ollama)
        if ! command -v zstd &>/dev/null; then
            log_info "Installing zstd (required by Ollama)..."
            if command -v apt-get &>/dev/null; then
                run_command sudo apt-get update && run_command sudo apt-get install -y zstd
            elif command -v dnf &>/dev/null; then
                run_command sudo dnf install -y zstd
            elif command -v yum &>/dev/null; then
                run_command sudo yum install -y zstd
            elif command -v pacman &>/dev/null; then
                run_command sudo pacman -Sy --noconfirm zstd
            else
                log_warning "Could not determine package manager. Please install zstd manually."
            fi
        fi

        log_info "Installing Ollama via official install script..."
        if [[ "$DRY_RUN" == true ]]; then
            echo "[DRY-RUN] Would execute: curl -fsSL https://ollama.com/install.sh | sh"
        else
            curl -fsSL https://ollama.com/install.sh | sh
        fi
    fi

    # Verify installation
    if command -v ollama &>/dev/null; then
        log_success "Ollama installed: $(ollama --version 2>/dev/null || echo 'version unknown')"

        # Try to start Ollama
        if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
            log_info "Starting Ollama service..."

            if [[ "$OS_TYPE" == "linux" ]] && command -v systemctl &>/dev/null; then
                run_command systemctl start ollama 2>/dev/null || true
                sleep 1
            fi

            # If still not running, start manually
            if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
                log_info "Starting 'ollama serve' in background..."
                if [[ "$DRY_RUN" != true ]]; then
                    nohup ollama serve >"/tmp/ollama-serve.log" 2>&1 &
                    sleep 2
                fi
            fi
        fi

        # Final check
        if curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
            log_success "Ollama API is reachable"
        else
            log_warning "Ollama API not responding. You may need to start 'ollama serve' manually."
        fi
    else
        log_error "Ollama installation failed"
        return 1
    fi
}

# ============================================================================
# ComfyUI Installation
# ============================================================================

install_or_update_comfyui() {
    if [[ "$SKIP_COMFYUI" == true ]]; then
        log_info "Skipping ComfyUI installation (--skip-comfyui)"
        return 0
    fi

    log_step "Installing/updating ComfyUI..."

    # Clone ComfyUI if not present
    if [[ ! -d "$COMFY_DIR" ]]; then
        log_info "Cloning ComfyUI into $COMFY_DIR..."
        run_command git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
    else
        log_info "ComfyUI already exists at $COMFY_DIR"
        if [[ "$UPDATE_MODE" == true ]]; then
            log_info "Updating ComfyUI..."
            if [[ "$DRY_RUN" != true ]]; then
                (cd "$COMFY_DIR" && git pull)
            fi
        fi
    fi

    # Create venv if not present
    if [[ ! -d "$COMFY_DIR/.venv" ]]; then
        log_info "Creating Python venv for ComfyUI..."
        run_command python3 -m venv "$COMFY_DIR/.venv"
    fi

    local VENV_PY="$COMFY_DIR/.venv/bin/python"
    local VENV_PIP="$COMFY_DIR/.venv/bin/pip"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Would install ComfyUI dependencies"
        return 0
    fi

    # Upgrade pip
    "$VENV_PIP" install --quiet --upgrade pip

    if [[ "$OS_TYPE" == "macos" ]]; then
        # macOS installation (simpler - use default PyTorch)
        log_info "Installing PyTorch (macOS)..."

        # Check if torch is already installed
        if "$VENV_PY" -c "import torch, torchvision, torchaudio" 2>/dev/null; then
            log_info "Torch stack already installed; skipping reinstall."
        else
            "$VENV_PIP" install --upgrade --force-reinstall --no-cache-dir torch torchvision torchaudio
        fi

        # Install other dependencies (excluding torch packages)
        log_info "Installing remaining dependencies..."
        local TEMP_REQS
        TEMP_REQS=$(mktemp)
        grep -v -iE '^(torch|torchvision|torchaudio)([ =<>~]|$)' "$COMFY_DIR/requirements.txt" > "$TEMP_REQS"
        "$VENV_PIP" install -r "$TEMP_REQS"
        rm -f "$TEMP_REQS"

    else
        # Linux installation (GPU-aware)
        install_comfyui_linux
    fi

    # Create checkpoint directories
    run_command mkdir -p "$REPO_ROOT/agent/model_cache/comfyui"
    run_command mkdir -p "$COMFY_DIR/models/checkpoints"

    log_success "ComfyUI installed at $COMFY_DIR"
}

install_comfyui_linux() {
    log_info "Installing ComfyUI for Linux with GPU detection..."

    local VENV_PY="$COMFY_DIR/.venv/bin/python"
    local VENV_PIP="$COMFY_DIR/.venv/bin/pip"

    # GPU detection
    local gpu_present=0
    local compute_major=0
    local compute_minor=0

    if command -v nvidia-smi &>/dev/null; then
        gpu_present=1
        log_info "NVIDIA GPU detected"

        # Get compute capability
        local cc_raw
        cc_raw=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "")
        if [[ -n "$cc_raw" ]]; then
            cc_raw=$(echo "$cc_raw" | tr -d '[:space:]' | cut -d',' -f1)
            if [[ "$cc_raw" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
                compute_major="${BASH_REMATCH[1]}"
                compute_minor="${BASH_REMATCH[2]}"
                log_info "Compute capability: $compute_major.$compute_minor"
            fi
        fi
    else
        log_info "No NVIDIA GPU detected; installing CPU-only PyTorch"
    fi

    # Choose PyTorch index based on compute capability
    local PYTORCH_INDEX_URL=""
    local PYTORCH_TAG=""

    if [[ "$gpu_present" -eq 1 ]]; then
        if [[ "$compute_major" -ge 13 ]] || [[ "$compute_major" -eq 12 && "$compute_minor" -ge 1 ]]; then
            PYTORCH_INDEX_URL="https://download.pytorch.org/whl/nightly/cu130"
            PYTORCH_TAG="nightly/cu130"
        elif [[ "$compute_major" -eq 12 ]]; then
            PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu129"
            PYTORCH_TAG="stable/cu129"
        else
            PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
            PYTORCH_TAG="stable/cu121"
        fi
        log_info "Using PyTorch: $PYTORCH_TAG"
    else
        PYTORCH_TAG="cpu"
        log_info "Using PyTorch: CPU-only"
    fi

    # Check if we need to reinstall torch
    local needs_torch_reinstall=false
    if ! "$VENV_PY" -c "import torch, torchvision, torchaudio" 2>/dev/null; then
        needs_torch_reinstall=true
    fi

    if [[ "$needs_torch_reinstall" == true ]] || [[ "$UPDATE_MODE" == true ]]; then
        log_info "Installing PyTorch stack..."

        # Uninstall existing torch packages
        "$VENV_PIP" uninstall -y torch torchvision torchaudio 2>/dev/null || true
        sleep 1

        # Install torch stack
        if [[ "$PYTORCH_TAG" == "cpu" ]]; then
            "$VENV_PIP" install --upgrade --force-reinstall --no-cache-dir torch torchvision torchaudio
        elif [[ "$PYTORCH_TAG" == "nightly/cu130" ]]; then
            "$VENV_PIP" install --pre --index-url "$PYTORCH_INDEX_URL" --upgrade --force-reinstall --no-cache-dir torch torchvision torchaudio
        else
            "$VENV_PIP" install --index-url "$PYTORCH_INDEX_URL" --upgrade --force-reinstall --no-cache-dir torch torchvision torchaudio
        fi
    else
        log_info "Torch stack already installed; skipping reinstall."
    fi

    # Install other dependencies
    log_info "Installing remaining dependencies..."
    local TEMP_REQS
    TEMP_REQS=$(mktemp)
    grep -v -iE '^(torch|torchvision|torchaudio)([ =<>~]|$)' "$COMFY_DIR/requirements.txt" > "$TEMP_REQS"
    "$VENV_PIP" install -r "$TEMP_REQS"
    rm -f "$TEMP_REQS"

    # Verify installation
    log_info "Verifying PyTorch installation..."
    "$VENV_PY" - <<'PYCODE' || log_warning "PyTorch verification failed"
import torch
print(f"torch.__version__: {torch.__version__}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    print(f"torch.cuda.get_device_capability(0): {torch.cuda.get_device_capability(0)}")
PYCODE
}

# ============================================================================
# Agent Python Environment
# ============================================================================

install_or_update_agent_venv() {
    log_step "Setting up Agent Python environment..."

    if [[ ! -d "$AGENT_DIR/.venv" ]]; then
        log_info "Creating Python venv for agent..."
        run_command python3 -m venv "$AGENT_DIR/.venv"
    else
        log_info "Agent venv already exists"
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Would install agent dependencies"
        return 0
    fi

    # shellcheck disable=SC1091
    source "$AGENT_DIR/.venv/bin/activate"

    log_info "Installing/updating Python dependencies..."
    pip install --quiet --upgrade pip
    pip install --quiet -r "$AGENT_DIR/requirements.txt" || {
        log_error "Failed to install Python dependencies"
        return 1
    }

    deactivate

    log_success "Agent Python environment ready"
}

# ============================================================================
# Configuration Generation
# ============================================================================

prompt_for_values() {
    # Read existing config values if file exists
    local existing_agent_id=""
    local existing_label=""

    if [[ -f "$CONFIG_FILE" ]]; then
        existing_agent_id=$(read_yaml_value "machine_id" "$CONFIG_FILE" || echo "")
        existing_label=$(read_yaml_value "label" "$CONFIG_FILE" || echo "")
        if [[ -n "$existing_agent_id" ]]; then
            log_info "Found existing config with agent_id: ${existing_agent_id}"
        fi
    fi

    # Skip prompts in YES_MODE
    if [[ "$YES_MODE" == true ]]; then
        # Prefer existing values, fall back to command-line args or detected defaults
        if [[ -z "$AGENT_ID" ]]; then
            if [[ -n "$existing_agent_id" ]]; then
                AGENT_ID="$existing_agent_id"
            else
                AGENT_ID=$(sanitize_id "$(detect_hostname)")
            fi
        fi

        if [[ -z "$LABEL" ]]; then
            if [[ -n "$existing_label" ]]; then
                LABEL="$existing_label"
            else
                local cpu_name
                cpu_name=$(detect_cpu_name)
                local ram_gb
                ram_gb=$(detect_ram_gb)
                LABEL="$(detect_hostname) (${cpu_name}, ${ram_gb}GB)"
            fi
        fi

        return 0
    fi

    # Interactive mode: prompt with defaults
    if [[ -z "$AGENT_ID" ]]; then
        # Determine default: existing > detected
        local detected_id
        detected_id=$(sanitize_id "$(detect_hostname)")
        local default_agent_id="${existing_agent_id:-$detected_id}"

        echo "" >&2
        log_info "Enter a unique machine ID for this agent"
        log_info "This ID must match the machine_id in central/config/machines.yaml"
        prompt "Machine ID [${default_agent_id}]: "
        read -r AGENT_ID_INPUT
        AGENT_ID="${AGENT_ID_INPUT:-$default_agent_id}"
        AGENT_ID=$(sanitize_id "$AGENT_ID")
    fi

    if [[ -z "$LABEL" ]]; then
        # Determine default: existing > detected
        local detected_label=""
        if [[ -z "$existing_label" ]]; then
            local cpu_name
            cpu_name=$(detect_cpu_name)
            local ram_gb
            ram_gb=$(detect_ram_gb)
            detected_label="$(detect_hostname) (${cpu_name}, ${ram_gb}GB)"
        fi
        local default_label="${existing_label:-$detected_label}"

        echo "" >&2
        log_info "Enter a human-friendly label for this machine"
        log_info "This label will be displayed in the UI"
        prompt "Label [${default_label}]: "
        read -r LABEL_INPUT
        LABEL="${LABEL_INPUT:-$default_label}"
    fi

    if [[ -z "$CENTRAL_URL" ]]; then
        local default_url="http://127.0.0.1:8080"
        echo "" >&2
        log_info "Enter the URL of the central server"
        prompt "Central URL [${default_url}]: "
        read -r CENTRAL_URL_INPUT
        CENTRAL_URL="${CENTRAL_URL_INPUT:-${default_url}}"
    fi
}

write_agent_config() {
    log_step "Generating agent configuration..."

    # Set defaults if values not set
    if [[ -z "$AGENT_ID" ]]; then
        AGENT_ID=$(sanitize_id "$(detect_hostname)")
    fi

    if [[ -z "$LABEL" ]]; then
        local cpu_name
        cpu_name=$(detect_cpu_name)
        local ram_gb
        ram_gb=$(detect_ram_gb)
        LABEL="$(detect_hostname) (${cpu_name}, ${ram_gb}GB)"
    fi

    # Sanitize values to remove any ANSI codes or control characters
    AGENT_ID=$(sanitize "$AGENT_ID")
    LABEL=$(sanitize "$LABEL")

    # Validate agent_id
    if ! validate_agent_id "$AGENT_ID"; then
        log_error "Invalid agent_id: ${AGENT_ID}"
        exit 1
    fi

    run_command mkdir -p "$CONFIG_DIR"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Would write config to $CONFIG_FILE"
        log_info "[DRY-RUN]   agent_id: ${AGENT_ID}"
        log_info "[DRY-RUN]   label: ${LABEL}"
        return 0
    fi

    log_info "Writing minimal agent.yaml..."
    log_info "  agent_id: ${AGENT_ID}"
    log_info "  label: ${LABEL}"

    # Write to temp file first (atomic write)
    local tmp_file
    tmp_file=$(mktemp)

    # Write only the minimal required fields
    # Note: The key is "machine_id" (not "agent_id") to match what the agent app expects
    {
        printf "machine_id: %s\n" "$AGENT_ID"
        if [[ -n "$LABEL" ]]; then
            printf "label: %s\n" "$LABEL"
        fi
    } > "$tmp_file"

    # Move temp file to final location
    mv "$tmp_file" "$CONFIG_FILE"

    log_success "Config written to $CONFIG_FILE"

    # Validate the written config
    if ! validate_yaml_config "$CONFIG_FILE"; then
        log_error "Generated config failed validation"
        exit 1
    fi
}

# Validate that the YAML config is clean (no control chars)
validate_yaml_config() {
    local config_file="$1"

    if [[ ! -f "$config_file" ]]; then
        log_error "Config file not found: $config_file"
        return 1
    fi

    log_info "Validating generated YAML..."

    # Check for control characters in the file
    if command -v python3 &>/dev/null; then
        python3 - "$config_file" <<'PYCODE'
import sys
import pathlib

config_path = pathlib.Path(sys.argv[1])
content = config_path.read_bytes()

# Check for control characters (excluding tab, newline, carriage return)
bad_chars = [c for c in content if c < 0x20 and c not in (0x09, 0x0a, 0x0d)]

if bad_chars:
    print(f"ERROR: Control characters found in config: {bad_chars[:10]}", file=sys.stderr)
    sys.exit(1)

# Try to parse as YAML (if PyYAML available)
try:
    import yaml
    config_text = config_path.read_text(encoding="utf-8")
    data = yaml.safe_load(config_text)

    # Check that machine_id exists and is clean
    if 'machine_id' not in data:
        print("ERROR: machine_id not found in config", file=sys.stderr)
        sys.exit(1)

    machine_id = str(data['machine_id'])
    if any(ord(c) < 0x20 for c in machine_id):
        print(f"ERROR: machine_id contains control characters: {repr(machine_id)}", file=sys.stderr)
        sys.exit(1)

    print("✓ YAML validation passed", file=sys.stderr)

except ImportError:
    # PyYAML not available, basic checks only
    print("✓ Basic validation passed (PyYAML not available for full check)", file=sys.stderr)
except Exception as e:
    print(f"ERROR: YAML parsing failed: {e}", file=sys.stderr)
    sys.exit(1)
PYCODE
        return $?
    else
        log_warning "Python3 not available, skipping YAML validation"
        return 0
    fi
}

# ============================================================================
# Service Installation
# ============================================================================

install_services() {
    if [[ "$NO_SERVICE" == true ]]; then
        log_info "Skipping service installation (--no-service)"
        return 0
    fi

    log_step "Installing services..."

    if [[ "$OS_TYPE" == "linux" ]] && command -v systemctl &>/dev/null; then
        log_info "Installing systemd services..."
        # TODO: Implement systemd service installation
        log_warning "Systemd service installation not yet implemented"
    elif [[ "$OS_TYPE" == "macos" ]]; then
        log_info "macOS service installation via launchctl..."
        log_warning "launchctl service installation not yet implemented"
        log_info "You can start the agent manually: bin/control agent start"
    fi
}

# ============================================================================
# Smoke Tests
# ============================================================================

start_agent_background() {
    log_info "Starting agent in background for testing..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Would start agent"
        return 0
    fi

    # shellcheck disable=SC1091
    source "$AGENT_DIR/.venv/bin/activate"

    cd "$AGENT_DIR"
    nohup uvicorn agent_app:app --host 0.0.0.0 --port 9001 > /tmp/agent-install-test.log 2>&1 &
    local agent_pid=$!
    echo "$agent_pid" > /tmp/agent-install-test.pid

    log_info "Agent PID: ${agent_pid}"
    sleep 3 # Give it time to start
}

stop_agent_background() {
    if [[ "$DRY_RUN" == true ]]; then
        return 0
    fi

    if [[ -f /tmp/agent-install-test.pid ]]; then
        local pid
        pid=$(cat /tmp/agent-install-test.pid)
        if kill -0 "$pid" 2>/dev/null; then
            log_info "Stopping test agent (PID ${pid})..."
            kill "$pid" 2>/dev/null || true
            sleep 1
        fi
        rm -f /tmp/agent-install-test.pid
    fi
}

test_health_endpoint() {
    log_info "Testing /health endpoint..."

    local response
    response=$(curl -s --max-time 5 http://localhost:9001/health 2>/dev/null || echo "FAILED")

    if [[ "$response" == *"ok"* ]]; then
        log_success "/health endpoint is responding"
        return 0
    else
        log_error "/health endpoint failed: ${response}"
        return 1
    fi
}

test_capabilities_endpoint() {
    log_info "Testing /capabilities endpoint..."

    local response
    response=$(curl -s --max-time 5 http://localhost:9001/capabilities 2>/dev/null || echo "FAILED")

    if [[ "$response" == *"machine_id"* ]]; then
        log_success "/capabilities endpoint is responding"
        return 0
    else
        log_error "/capabilities endpoint failed"
        return 1
    fi
}

test_ollama_reachable() {
    if [[ "$SKIP_OLLAMA" == true ]]; then
        return 0
    fi

    log_info "Testing Ollama connectivity..."

    local response
    response=$(curl -s --max-time 2 http://localhost:11434/api/tags 2>/dev/null || echo "FAILED")

    if [[ "$response" == *"models"* ]]; then
        log_success "Ollama is reachable"
        return 0
    else
        log_warning "Ollama not reachable (optional)"
        return 1
    fi
}

test_comfyui_reachable() {
    if [[ "$SKIP_COMFYUI" == true ]]; then
        return 0
    fi

    log_info "Testing ComfyUI connectivity..."

    local response
    response=$(curl -s --max-time 2 http://localhost:8188/ 2>/dev/null || echo "FAILED")

    if [[ "$response" != "FAILED" ]]; then
        log_success "ComfyUI is reachable"
        return 0
    else
        log_warning "ComfyUI not reachable (optional)"
        return 1
    fi
}

run_smoke_tests() {
    log_step "Running smoke tests..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Would run smoke tests"
        return 0
    fi

    local tests_passed=0
    local tests_failed=0

    start_agent_background

    if test_health_endpoint; then
        ((tests_passed++))
    else
        ((tests_failed++))
    fi

    if test_capabilities_endpoint; then
        ((tests_passed++))
    else
        ((tests_failed++))
    fi

    # Optional tests
    test_ollama_reachable || true
    test_comfyui_reachable || true

    stop_agent_background

    echo ""
    if [[ $tests_failed -eq 0 ]]; then
        log_success "All required smoke tests passed! (${tests_passed}/${tests_passed})"
        return 0
    else
        log_error "Some smoke tests failed (${tests_passed}/$((tests_passed + tests_failed)) passed)"
        log_info "Check logs at /tmp/agent-install-test.log"
        return 1
    fi
}

# ============================================================================
# Main Installation Flow
# ============================================================================

print_usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Consolidated installer for bench-race agent

Options:
  --agent-id ID         Agent identifier (default: hostname)
  --label "Label"       Human-readable label for UI
  --central-url URL     Central server URL (default: http://127.0.0.1:8080)
  --platform PLATFORM   Override platform detection (macos|linux|linux-gb10)
  --yes                 Non-interactive mode (use defaults)
  --no-service          Skip systemd/launchctl service installation
  --update              Update existing installation
  --dry-run             Show what would be done without doing it
  --skip-ollama         Skip Ollama installation
  --skip-comfyui        Skip ComfyUI installation
  --help                Show this help message

EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --agent-id)
                AGENT_ID=$(sanitize_id "$2")
                shift 2
                ;;
            --label)
                LABEL="$2"
                shift 2
                ;;
            --central-url)
                CENTRAL_URL="$2"
                shift 2
                ;;
            --platform)
                PLATFORM="$2"
                shift 2
                ;;
            --yes)
                YES_MODE=true
                shift
                ;;
            --no-service)
                NO_SERVICE=true
                shift
                ;;
            --update)
                UPDATE_MODE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-ollama)
                SKIP_OLLAMA=true
                shift
                ;;
            --skip-comfyui)
                SKIP_COMFYUI=true
                shift
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                log_error "Unknown argument: $1"
                print_usage
                exit 1
                ;;
        esac
    done
}

main() {
    parse_args "$@"

    echo ""
    log_info "==================================================="
    log_info "bench-race Agent Installer"
    log_info "==================================================="
    echo ""

    # Detect platform (unless overridden)
    if [[ -z "$PLATFORM" ]]; then
        detect_platform
    else
        log_info "Using platform override: $PLATFORM"
        case "$PLATFORM" in
            macos)
                OS_TYPE="macos"
                ;;
            linux)
                OS_TYPE="linux"
                IS_GB10=false
                ;;
            linux-gb10)
                OS_TYPE="linux"
                IS_GB10=true
                ;;
            *)
                log_error "Invalid platform: $PLATFORM"
                exit 1
                ;;
        esac
        ARCH=$(uname -m)
    fi

    # Display system info
    log_info "Detecting system specifications..."
    local hostname
    hostname=$(detect_hostname)
    local cpu_cores
    cpu_cores=$(detect_cpu_cores)
    local cpu_name
    cpu_name=$(detect_cpu_name)
    local ram_gb
    ram_gb=$(detect_ram_gb)
    local gpu_info
    gpu_info=$(detect_gpu_info)

    echo ""
    log_info "System information:"
    log_info "  Hostname: ${hostname}"
    log_info "  Platform: ${PLATFORM}"
    log_info "  CPU: ${cpu_name} (${cpu_cores} cores)"
    log_info "  RAM: ${ram_gb}GB"
    log_info "  GPU: ${gpu_info}"
    echo ""

    # Check prerequisites
    ensure_prereqs

    # Install components
    install_or_update_ollama || log_warning "Ollama installation had issues"
    install_or_update_comfyui || log_warning "ComfyUI installation had issues"
    install_or_update_agent_venv || {
        log_error "Failed to setup Agent Python environment"
        exit 1
    }

    # Get user input and write config
    prompt_for_values
    write_agent_config

    # Install services
    install_services

    # Run smoke tests
    echo ""
    run_smoke_tests || {
        log_warning "Smoke tests had issues, but installation may still be usable"
    }

    # Print next steps
    echo ""
    log_success "==================================================="
    log_success "Agent installation complete!"
    log_success "==================================================="
    echo ""
    log_info "Configuration:"
    log_info "  Agent ID: ${AGENT_ID}"
    log_info "  Label: ${LABEL}"
    log_info "  Central URL: ${CENTRAL_URL}"
    echo ""
    log_info "Next steps:"
    log_info "  1. Start the agent: bin/control agent start"
    log_info "  2. Check status: bin/control agent status"
    log_info "  3. Add this machine to central/config/machines.yaml:"
    echo ""
    echo "    - machine_id: \"${AGENT_ID}\""
    echo "      label: \"${LABEL}\""
    echo "      agent_base_url: \"http://<this-machine-ip>:9001\""
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        echo ""
        log_info "This was a DRY RUN - no changes were made"
        echo ""
    fi
}

# Run main function
main "$@"

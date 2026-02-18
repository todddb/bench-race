#!/usr/bin/env bash
# install_central.sh
#
# Consolidated installer for bench-race central server
# - Detects platform (macOS/Linux)
# - Sets up Python virtual environment
# - Ensures minimal config files exist
# - Runs smoke tests
# - Idempotent: safe to re-run for updates
#
# Usage:
#   ./install_central.sh [OPTIONS]
#
# Options:
#   --yes                 Non-interactive mode (use defaults)
#   --no-service          Skip systemd/launchctl service installation
#   --update              Update existing installation
#   --dry-run             Show what would be done without doing it
#   --platform PLATFORM   Override platform detection (macos|linux)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CENTRAL_DIR="${REPO_ROOT}/central"
CONFIG_DIR="${CENTRAL_DIR}/config"

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
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $*"
}

# ============================================================================
# Global Variables
# ============================================================================

PLATFORM=""
OS_TYPE=""
ARCH=""
DRY_RUN=false
YES_MODE=false
NO_SERVICE=false
UPDATE_MODE=false

# ============================================================================
# Helper Functions
# ============================================================================

# ---------------------------------------------------------------------------
# Prefer Homebrew python3.12 on macOS if available
# This sets PYTHON_BIN to a sensible python executable (defaults to python3)
PYTHON_BIN=python3
if [[ "$(uname -s)" == "Darwin" && -x /opt/homebrew/bin/python3.12 ]]; then
    PYTHON_BIN=/opt/homebrew/bin/python3.12
fi
# ---------------------------------------------------------------------------



run_command() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] Would execute: $*"
        return 0
    fi
    "$@"
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
            PLATFORM="macos"
            ;;
        Linux)
            OS_TYPE="linux"
            PLATFORM="linux"
            ;;
        *)
            log_error "Unsupported operating system: $uname_s"
            exit 1
            ;;
    esac

    ARCH=$(uname -m)

    log_info "Platform: ${PLATFORM}"
    log_info "OS: ${OS_TYPE}"
    log_info "Architecture: ${ARCH}"
}

# ============================================================================
# Prerequisites Check
# ============================================================================

ensure_prereqs() {
    log_step "Checking prerequisites..."

    local missing_tools=()

    # Check for required tools
    if ! command -v "$PYTHON_BIN" &>/dev/null; then
        missing_tools+=("python3")
    fi

    if ! command -v curl &>/dev/null; then
        missing_tools+=("curl")
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
    if ! $PYTHON_BIN -m venv --help &>/dev/null; then
        log_error "Python venv module not available"
        if [[ "$OS_TYPE" == "linux" ]]; then
            log_info "Install with: sudo apt-get install python3-venv"
        fi
        exit 1
    fi

    # Check Python version (reject 3.14+ per the existing setup_venv_central.sh logic)
    local python_version
    python_version=$($PYTHON_BIN -c \'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")\')
    local major minor
    IFS='.' read -r major minor <<< "$python_version"

    if [[ "$major" -eq 3 && "$minor" -ge 14 ]]; then
        log_error "Python 3.14+ is not yet supported. Please use Python 3.9-3.13."
        exit 1
    fi

    log_success "All prerequisites satisfied (Python ${python_version})"
}

# ============================================================================
# Central Python Environment
# ============================================================================

install_or_update_central_venv() {
    log_step "Setting up Central Python environment..."

    if [[ ! -d "$CENTRAL_DIR/.venv" ]]; then
        log_info "Creating Python venv for central..."
        run_command $PYTHON_BIN -m venv "$CENTRAL_DIR/.venv"
    else
        log_info "Central venv already exists"
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Would install central dependencies"
        return 0
    fi

    # shellcheck disable=SC1091
    source "$CENTRAL_DIR/.venv/bin/activate"

    log_info "Installing/updating Python dependencies..."
    pip install --quiet --upgrade pip setuptools wheel
    pip install --quiet -r "$CENTRAL_DIR/requirements.txt" || {
        log_error "Failed to install Python dependencies"
        return 1
    }

    deactivate

    log_success "Central Python environment ready"
}

# ============================================================================
# Configuration Generation
# ============================================================================

write_central_config_if_missing() {
    log_step "Checking central configuration..."

    run_command mkdir -p "$CONFIG_DIR"

    # Check for machines.yaml
    if [[ ! -f "$CONFIG_DIR/machines.yaml" ]]; then
        if [[ -f "$CONFIG_DIR/machines.example.yaml" ]]; then
            log_info "Creating machines.yaml from example..."
            if [[ "$DRY_RUN" != true ]]; then
                cp "$CONFIG_DIR/machines.example.yaml" "$CONFIG_DIR/machines.yaml"
            fi
        else
            log_info "Creating minimal machines.yaml..."
            if [[ "$DRY_RUN" != true ]]; then
                cat > "$CONFIG_DIR/machines.yaml" <<'EOF'
# central/config/machines.yaml
#
# Auto-generated by install_central.sh
# Add your agent machines here

machines: []

# Example:
# machines:
#   - machine_id: "macbook"
#     label: "MacBook (M4, 128GB)"
#     agent_base_url: "http://127.0.0.1:9001"
#     notes: "Primary development machine"
EOF
            fi
        fi
        log_success "Created machines.yaml"
    else
        log_info "machines.yaml already exists"
    fi

    # Check for other config files and copy from examples if they exist
    local config_files=("model_policy.yaml" "comfyui.yaml" "central.yaml")
    for config_file in "${config_files[@]}"; do
        if [[ ! -f "$CONFIG_DIR/$config_file" ]] && [[ -f "$CONFIG_DIR/${config_file%.yaml}.example.yaml" ]]; then
            log_info "Creating $config_file from example..."
            if [[ "$DRY_RUN" != true ]]; then
                cp "$CONFIG_DIR/${config_file%.yaml}.example.yaml" "$CONFIG_DIR/$config_file"
            fi
        fi
    done

    log_success "Central configuration ready"
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
        log_info "You can start central manually: bin/control central start"
    fi
}

# ============================================================================
# Smoke Tests
# ============================================================================

start_central_background() {
    log_info "Starting central in background for testing..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Would start central"
        return 0
    fi

    # shellcheck disable=SC1091
    source "$CENTRAL_DIR/.venv/bin/activate"

    cd "$CENTRAL_DIR"
    nohup python app.py > /tmp/central-install-test.log 2>&1 &
    local central_pid=$!
    echo "$central_pid" > /tmp/central-install-test.pid

    log_info "Central PID: ${central_pid}"
    sleep 4 # Give it time to start
}

stop_central_background() {
    if [[ "$DRY_RUN" == true ]]; then
        return 0
    fi

    if [[ -f /tmp/central-install-test.pid ]]; then
        local pid
        pid=$(cat /tmp/central-install-test.pid)
        if kill -0 "$pid" 2>/dev/null; then
            log_info "Stopping test central (PID ${pid})..."
            kill "$pid" 2>/dev/null || true
            sleep 1
        fi
        rm -f /tmp/central-install-test.pid
    fi
}

test_central_health() {
    log_info "Testing central server health..."

    local response
    # Try the root endpoint (UI) or /health if it exists
    response=$(curl -s --max-time 5 http://localhost:8080/ 2>/dev/null || echo "FAILED")

    if [[ "$response" != "FAILED" ]] && [[ -n "$response" ]]; then
        log_success "Central server is responding"
        return 0
    else
        # Try alternate port if default fails
        response=$(curl -s --max-time 5 http://localhost:5000/ 2>/dev/null || echo "FAILED")
        if [[ "$response" != "FAILED" ]] && [[ -n "$response" ]]; then
            log_success "Central server is responding on port 5000"
            return 0
        fi

        log_error "Central server not responding"
        log_info "Check logs at /tmp/central-install-test.log"
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

    start_central_background

    if test_central_health; then
        ((tests_passed++))
    else
        ((tests_failed++))
    fi

    stop_central_background

    echo ""
    if [[ $tests_failed -eq 0 ]]; then
        log_success "All smoke tests passed! (${tests_passed}/${tests_passed})"
        return 0
    else
        log_error "Some smoke tests failed (${tests_passed}/$((tests_passed + tests_failed)) passed)"
        log_info "Check logs at /tmp/central-install-test.log"
        return 1
    fi
}

# ============================================================================
# Main Installation Flow
# ============================================================================

print_usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Consolidated installer for bench-race central server

Options:
  --yes                 Non-interactive mode (use defaults)
  --no-service          Skip systemd/launchctl service installation
  --update              Update existing installation
  --dry-run             Show what would be done without doing it
  --platform PLATFORM   Override platform detection (macos|linux)
  --help                Show this help message

EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
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
            --platform)
                PLATFORM="$2"
                shift 2
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
    log_info "bench-race Central Server Installer"
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
                ;;
            *)
                log_error "Invalid platform: $PLATFORM"
                exit 1
                ;;
        esac
        ARCH=$(uname -m)
    fi

    # Check prerequisites
    ensure_prereqs

    # Install Python environment
    install_or_update_central_venv || {
        log_error "Failed to setup Central Python environment"
        exit 1
    }

    # Write config if missing
    write_central_config_if_missing

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
    log_success "Central server installation complete!"
    log_success "==================================================="
    echo ""
    log_info "Next steps:"
    log_info "  1. Configure agents in central/config/machines.yaml"
    log_info "  2. Start the central server: bin/control central start"
    log_info "  3. Check status: bin/control central status"
    log_info "  4. Open the UI: http://localhost:8080"
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        echo ""
        log_info "This was a DRY RUN - no changes were made"
        echo ""
    fi
}

# Run main function
main "$@"

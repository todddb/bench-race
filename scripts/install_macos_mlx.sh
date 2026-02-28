#!/usr/bin/env bash
# scripts/install_macos_mlx.sh — Install MLX backend for bench-race on macOS
#
# Creates the Python venv at agent/backends/mlx/.venv, installs dependencies,
# and optionally drops a launchctl plist for auto-start on macOS.
#
# Usage:
#   ./scripts/install_macos_mlx.sh [OPTIONS]
#
# Options:
#   --service       Install a launchctl user agent plist (macOS only)
#   --force-venv    Recreate the venv even if it already exists
#   --dry-run       Show what would be done without making changes
#   --help          Show this help message

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MLX_DIR="${REPO_ROOT}/agent/backends/mlx"
VENV_DIR="${MLX_DIR}/.venv"
REQUIREMENTS="${MLX_DIR}/requirements.txt"
BACKENDS_CONFIG="${REPO_ROOT}/config/backends.yaml"

read_backend_cfg() {
    local key="$1"
    local fallback="$2"
    if [[ -f "$BACKENDS_CONFIG" ]] && command -v yq >/dev/null 2>&1; then
        local value
        value="$(yq -r "${key} // \"${fallback}\"" "$BACKENDS_CONFIG" 2>/dev/null || true)"
        if [[ -n "$value" && "$value" != "null" ]]; then
            echo "$value"
            return
        fi
    fi
    echo "$fallback"
}

if [[ -f "$BACKENDS_CONFIG" ]] && ! command -v yq >/dev/null 2>&1; then
    echo "[WARN] config/backends.yaml present but yq not installed; using built-in fallbacks." >&2
fi

# Defaults
DEFAULT_MLX_HOST="$(read_backend_cfg '.mlx.host' "127.0.0.1")"
DEFAULT_MLX_PORT="$(read_backend_cfg '.mlx.port' "8321")"
MLX_HOST="${MLX_HOST:-${DEFAULT_MLX_HOST}}"
MLX_PORT="${MLX_PORT:-${DEFAULT_MLX_PORT}}"
INSTALL_SERVICE=false
FORCE_VENV=false
DRY_RUN=false

# ============================================================================
# Colours
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()      { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_err()     { echo -e "${RED}[ERROR]${NC} $*"; }

# ============================================================================
# Argument parsing
# ============================================================================

print_usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Install the MLX backend for bench-race on macOS (Apple Silicon).

Options:
  --service       Install a launchctl user agent plist for auto-start
  --force-venv    Recreate the venv even if it already exists
  --dry-run       Show what would be done without making changes
  --help          Show this help message

Environment:
  MLX_HOST        Bind address  (default: 127.0.0.1)
  MLX_PORT        Bind port     (default: 8321)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --service)      INSTALL_SERVICE=true; shift ;;
        --force-venv)   FORCE_VENV=true; shift ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --help|-h)      print_usage; exit 0 ;;
        *)
            log_err "Unknown argument: $1"
            print_usage
            exit 1
            ;;
    esac
done

# ============================================================================
# Pre-flight checks
# ============================================================================

log_info "=== bench-race MLX Backend Installer ==="

# Must be macOS
if [[ "$(uname -s)" != "Darwin" ]]; then
    log_err "This installer is for macOS only (detected: $(uname -s))."
    exit 1
fi

# Must be Apple Silicon (arm64)
if [[ "$(uname -m)" != "arm64" ]]; then
    log_warn "Apple Silicon (arm64) expected; detected $(uname -m)."
    log_warn "MLX requires Apple Silicon. Proceeding anyway…"
fi

# Python 3 available?
if ! command -v python3 &>/dev/null; then
    log_err "python3 not found. Install via Homebrew: brew install python@3.12"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
log_info "Detected Python ${PYTHON_VERSION}"

if ! python3 -c "import venv" &>/dev/null; then
    log_err "Python venv module not available. Reinstall Python or run: xcode-select --install"
    exit 1
fi

# Requirements file exists?
if [[ ! -f "$REQUIREMENTS" ]]; then
    log_err "Requirements file not found at ${REQUIREMENTS}"
    log_err "Make sure the repository is intact."
    exit 1
fi

# ============================================================================
# Create / recreate the venv
# ============================================================================

if [[ "$FORCE_VENV" == true && -d "$VENV_DIR" ]]; then
    log_info "Removing existing venv (--force-venv)..."
    [[ "$DRY_RUN" != true ]] && rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
    log_info "Creating Python venv at ${VENV_DIR}..."
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] python3 -m venv ${VENV_DIR}"
    else
        # Prefer Homebrew Python 3.12 if available
        if [[ -x /opt/homebrew/bin/python3.12 ]]; then
            /opt/homebrew/bin/python3.12 -m venv "$VENV_DIR"
        else
            python3 -m venv "$VENV_DIR"
        fi
    fi
else
    log_info "Venv already exists at ${VENV_DIR}"
fi

VENV_PY="${VENV_DIR}/bin/python"

# ============================================================================
# Install dependencies
# ============================================================================

log_info "Upgrading pip..."
if [[ "$DRY_RUN" == true ]]; then
    log_info "[DRY-RUN] ${VENV_PY} -m pip install --upgrade pip"
else
    "$VENV_PY" -m pip install --quiet --upgrade pip
fi

log_info "Installing MLX backend dependencies..."
if [[ "$DRY_RUN" == true ]]; then
    log_info "[DRY-RUN] ${VENV_PY} -m pip install -r ${REQUIREMENTS}"
else
    "$VENV_PY" -m pip install --quiet -r "$REQUIREMENTS"
fi

# ============================================================================
# Verify MLX import
# ============================================================================

log_info "Verifying MLX installation..."
if [[ "$DRY_RUN" != true ]]; then
    "$VENV_PY" - <<'PYEOF' || {
import sys
try:
    import mlx.core as mx
    print(f"  mlx version : {mx.__version__ if hasattr(mx, '__version__') else 'OK'}")
except ImportError as e:
    print(f"ERROR: Cannot import mlx.core: {e}", file=sys.stderr)
    sys.exit(1)

try:
    import mlx_lm
    print("  mlx_lm      : OK")
except ImportError as e:
    print(f"ERROR: Cannot import mlx_lm: {e}", file=sys.stderr)
    sys.exit(1)

try:
    import fastapi, uvicorn
    print("  fastapi      : OK")
    print("  uvicorn      : OK")
except ImportError as e:
    print(f"ERROR: Cannot import fastapi/uvicorn: {e}", file=sys.stderr)
    sys.exit(1)

print("MLX backend dependencies verified.")
PYEOF
        log_err "MLX dependency verification failed. Check the output above."
        exit 1
    }
    log_ok "All MLX dependencies verified"
else
    log_info "[DRY-RUN] Would verify MLX imports"
fi

# ============================================================================
# Launchctl service (optional)
# ============================================================================

if [[ "$INSTALL_SERVICE" == true ]]; then
    PLIST_LABEL="com.bench-race.mlx"
    PLIST_DIR="$HOME/Library/LaunchAgents"
    PLIST_FILE="${PLIST_DIR}/${PLIST_LABEL}.plist"
    LOG_DIR="${REPO_ROOT}/agent/log"

    mkdir -p "$PLIST_DIR" "$LOG_DIR"

    log_info "Installing launchctl plist at ${PLIST_FILE}..."

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Would write ${PLIST_FILE}"
    else
        cat > "$PLIST_FILE" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>${VENV_DIR}/bin/python</string>
        <string>-m</string>
        <string>uvicorn</string>
        <string>agent.backends.mlx.server:app</string>
        <string>--host</string>
        <string>${MLX_HOST}</string>
        <string>--port</string>
        <string>${MLX_PORT}</string>
    </array>

    <key>WorkingDirectory</key>
    <string>${REPO_ROOT}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>MLX_HOST</key>
        <string>${MLX_HOST}</string>
        <key>MLX_PORT</key>
        <string>${MLX_PORT}</string>
    </dict>

    <key>RunAtLoad</key>
    <false/>

    <key>KeepAlive</key>
    <false/>

    <key>StandardOutPath</key>
    <string>${LOG_DIR}/mlx.log</string>

    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/mlx.log</string>
</dict>
</plist>
PLIST

        log_ok "Plist written: ${PLIST_FILE}"
        log_info "Load with:   launchctl load ${PLIST_FILE}"
        log_info "Unload with: launchctl unload ${PLIST_FILE}"
    fi
fi

# ============================================================================
# Done
# ============================================================================

echo ""
log_ok "=== MLX Backend installation complete ==="
echo ""
log_info "Quick start:"
log_info "  Start server : ./scripts/agent start-mlx"
log_info "  Health check : curl http://${MLX_HOST}:${MLX_PORT}/health"
log_info "  Stop server  : ./scripts/agent stop-mlx"
echo ""
log_info "To load a model:"
log_info "  curl -X POST http://${MLX_HOST}:${MLX_PORT}/start \\"
log_info "       -H 'Content-Type: application/json' \\"
log_info "       -d '{\"model_id\": \"mlx-community/Meta-Llama-3-8B-Instruct-4bit\"}'"
echo ""

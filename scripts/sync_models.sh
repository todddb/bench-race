#!/usr/bin/env bash
# sync_models.sh — Download HuggingFace models to agent/models/
#
# Uses huggingface_hub.snapshot_download to fetch canonical HF safetensors
# into agent/models/<name> as the single source of truth for weights.
#
# Usage:
#   ./scripts/sync_models.sh <hf-id> [--local-name <name>] [--ollama-create]
#
# Examples:
#   ./scripts/sync_models.sh meta-llama/Llama-2-70b-chat --local-name llama-70b
#   ./scripts/sync_models.sh mistralai/Mistral-7B-Instruct-v0.3
#   ./scripts/sync_models.sh meta-llama/Llama-2-7b-chat --ollama-create
#
# Environment:
#   HF_TOKEN       HuggingFace API token (required for gated/private models)
#   AGENT_DIR      Override agent directory (default: <repo>/agent)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AGENT_DIR="${AGENT_DIR:-${REPO_ROOT}/agent}"
MODELS_DIR="${AGENT_DIR}/models"

# ============================================================================
# Colors
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()      { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_err()     { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ============================================================================
# Parse args
# ============================================================================

HF_ID=""
LOCAL_NAME=""
OLLAMA_CREATE=false

usage() {
    cat <<EOF
Usage: $(basename "$0") <hf-repo-id> [OPTIONS]

Download a HuggingFace model to agent/models/<name>.

Arguments:
  <hf-repo-id>           HuggingFace model identifier (e.g. meta-llama/Llama-2-70b-chat)

Options:
  --local-name <name>    Local directory name (default: derived from hf-repo-id)
  --ollama-create        Create an Ollama model from the downloaded safetensors
  --help, -h             Show this help

Environment:
  HF_TOKEN               HuggingFace token for gated/private models
  AGENT_DIR              Override agent directory (default: <repo>/agent)

Examples:
  $(basename "$0") meta-llama/Llama-2-70b-chat --local-name llama-70b
  $(basename "$0") mistralai/Mistral-7B-Instruct-v0.3
  HF_TOKEN=hf_... $(basename "$0") meta-llama/Llama-2-70b-chat --ollama-create
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local-name)
            LOCAL_NAME="$2"
            shift 2
            ;;
        --ollama-create)
            OLLAMA_CREATE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            log_err "Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            if [[ -z "$HF_ID" ]]; then
                HF_ID="$1"
            else
                log_err "Unexpected argument: $1"
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$HF_ID" ]]; then
    log_err "Missing required argument: <hf-repo-id>"
    usage
    exit 1
fi

# Derive local name from HF ID if not specified
if [[ -z "$LOCAL_NAME" ]]; then
    # meta-llama/Llama-2-70b-chat -> Llama-2-70b-chat
    LOCAL_NAME="${HF_ID##*/}"
    # Lowercase and replace spaces/special chars
    LOCAL_NAME=$(echo "$LOCAL_NAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
fi

MODEL_DIR="${MODELS_DIR}/${LOCAL_NAME}"

# ============================================================================
# Find Python with huggingface_hub
# ============================================================================

find_python() {
    # Prefer vLLM venv (has huggingface_hub), then agent venv, then system
    local candidates=(
        "${AGENT_DIR}/third_party/vllm/.venv/bin/python"
        "${AGENT_DIR}/.venv/bin/python"
        "python3"
    )

    for py in "${candidates[@]}"; do
        if command -v "$py" &>/dev/null || [[ -x "$py" ]]; then
            if "$py" -c "import huggingface_hub" 2>/dev/null; then
                echo "$py"
                return 0
            fi
        fi
    done

    return 1
}

PYTHON=""
PYTHON=$(find_python) || {
    log_err "Could not find Python with huggingface_hub installed"
    log_err "Install it: pip install huggingface_hub"
    log_err "Or run: ./scripts/install_agent.sh (installs it in vLLM venv)"
    exit 1
}

log_info "Using Python: $PYTHON"

# ============================================================================
# Download model
# ============================================================================

mkdir -p "$MODELS_DIR"

log_info "Downloading model: $HF_ID"
log_info "  Local path: $MODEL_DIR"

if [[ -n "${HF_TOKEN:-}" ]]; then
    log_info "  HF_TOKEN: set"
else
    log_warn "  HF_TOKEN: not set (may fail for gated/private models)"
fi

echo ""

"$PYTHON" - "$HF_ID" "$MODEL_DIR" <<'PYEOF'
import os
import sys

repo_id = sys.argv[1]
local_dir = sys.argv[2]

token = os.environ.get("HF_TOKEN") or None

from huggingface_hub import snapshot_download

print(f"Downloading {repo_id} to {local_dir}...")
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    token=token,
    # Download safetensors files preferentially
    ignore_patterns=["*.bin", "*.pt", "*.ckpt", "original/**", "*.pth"],
)
print(f"Download complete: {local_dir}")
PYEOF

download_status=$?
if [[ $download_status -ne 0 ]]; then
    log_err "Model download failed (exit code $download_status)"
    if [[ -z "${HF_TOKEN:-}" ]]; then
        log_err "This may be a gated model. Set HF_TOKEN and try again."
    fi
    exit 1
fi

# ============================================================================
# Validate downloaded model
# ============================================================================

log_info "Validating downloaded model..."

"$PYTHON" - "$MODEL_DIR" <<'PYEOF'
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])

# Check for safetensors files
safetensors_files = list(model_dir.glob("*.safetensors"))
config_file = model_dir / "config.json"
tokenizer_file = model_dir / "tokenizer.json"
tokenizer_model = model_dir / "tokenizer.model"

print(f"  Safetensors files: {len(safetensors_files)}")
print(f"  config.json: {'found' if config_file.exists() else 'missing'}")
print(f"  tokenizer: {'found' if (tokenizer_file.exists() or tokenizer_model.exists()) else 'missing'}")

if not safetensors_files:
    # Check for GGUF files (Ollama format)
    gguf_files = list(model_dir.glob("*.gguf"))
    if gguf_files:
        print(f"  GGUF files: {len(gguf_files)} (Ollama-compatible format)")
    else:
        print("WARNING: No safetensors or GGUF files found", file=sys.stderr)
        # Don't fail - some models use other formats
PYEOF

log_ok "Model synced to: $MODEL_DIR"

# ============================================================================
# Optional: Create Ollama model
# ============================================================================

if [[ "$OLLAMA_CREATE" == true ]]; then
    log_info "Creating Ollama model from downloaded files..."

    if ! command -v ollama &>/dev/null; then
        log_err "ollama not found in PATH; cannot create Ollama model"
        exit 1
    fi

    # Check for GGUF file
    GGUF_FILE=$(find "$MODEL_DIR" -name "*.gguf" -print -quit 2>/dev/null || true)
    if [[ -z "$GGUF_FILE" ]]; then
        log_warn "No GGUF file found in $MODEL_DIR"
        log_warn "Ollama requires GGUF format. You may need to convert safetensors to GGUF first."
        log_warn "Skipping Ollama model creation."
    else
        MODELFILE="${MODEL_DIR}/Modelfile"
        log_info "Creating Modelfile at $MODELFILE..."

        cat > "$MODELFILE" <<MFEOF
FROM ${GGUF_FILE}

TEMPLATE """{{ .Prompt }}"""
MFEOF

        log_info "Running: ollama create $LOCAL_NAME -f $MODELFILE"
        if ollama create "$LOCAL_NAME" -f "$MODELFILE"; then
            log_ok "Ollama model '$LOCAL_NAME' created successfully"
        else
            log_err "Failed to create Ollama model"
            exit 1
        fi
    fi
fi

echo ""
log_ok "Sync complete: $HF_ID -> $MODEL_DIR"
echo ""

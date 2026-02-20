#!/usr/bin/env bash
# ollama_sync_and_convert.sh — Pull a model via Ollama and convert for vLLM.
#
# Outputs structured JSON to stdout for parsing by the agent.
#
# Usage:
#   ./agent/scripts/ollama_sync_and_convert.sh <model_display_name> [--hf-repo <hf_id>] [--skip-convert]
#
# Environment:
#   OLLAMA_HOST       Ollama API base (default: http://127.0.0.1:11434)
#   AGENT_DIR         Override agent directory (default: <repo>/agent)
#   HF_TOKEN          HuggingFace token for gated models
#   AUTO_CONVERT      Set to "false" to skip conversion (default: true)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
AGENT_DIR="${AGENT_DIR:-${REPO_ROOT}/agent}"
MODELS_DIR="${AGENT_DIR}/models"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
SANITIZER="${SCRIPT_DIR}/sanitize_name.sh"
AUTO_CONVERT="${AUTO_CONVERT:-true}"

# Colors (stderr only)
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()  { echo -e "${BLUE}[INFO]${NC} $*" >&2; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $*" >&2; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
log_err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# JSON output helper — write final result to stdout
emit_result() {
    local status="$1" error_msg="${2:-}"
    cat <<JSONEOF
{
  "id": "${MODEL_ID}",
  "display_name": "${DISPLAY_NAME}",
  "sanitized_name": "${SANITIZED}",
  "ollama_path": "${OLLAMA_PATH}",
  "vllm_path": "${VLLM_PATH}",
  "status": "${status}",
  "error_message": "${error_msg}"
}
JSONEOF
}

# ---- Parse args ----
DISPLAY_NAME=""
HF_REPO=""
SKIP_CONVERT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf-repo)  HF_REPO="$2"; shift 2 ;;
        --skip-convert) SKIP_CONVERT=true; shift ;;
        --id)       MODEL_ID="$2"; shift 2 ;;
        -*)         log_err "Unknown option: $1"; exit 1 ;;
        *)          if [[ -z "$DISPLAY_NAME" ]]; then DISPLAY_NAME="$1"; else log_err "Unexpected arg: $1"; exit 1; fi; shift ;;
    esac
done

if [[ -z "$DISPLAY_NAME" ]]; then
    log_err "Usage: $0 <model_display_name> [--hf-repo <hf_id>] [--skip-convert] [--id <model_id>]"
    exit 1
fi

MODEL_ID="${MODEL_ID:-$DISPLAY_NAME}"

# ---- Sanitize ----
if [[ -x "$SANITIZER" ]]; then
    SANITIZED=$("$SANITIZER" "$DISPLAY_NAME")
else
    SANITIZED=$(echo "$DISPLAY_NAME" | sed -E 's/[^A-Za-z0-9._-]/_/g; s/_+/_/g; s/^_+//; s/_+$//' | cut -c1-90)
fi

OLLAMA_DIR="${MODELS_DIR}/ollama"
VLLM_DIR="${MODELS_DIR}/vllm"
CONVERTED_DIR="${MODELS_DIR}/converted/${SANITIZED}"
OLLAMA_PATH="${OLLAMA_DIR}/${SANITIZED}"
VLLM_PATH="${VLLM_DIR}/${SANITIZED}"

mkdir -p "$OLLAMA_DIR" "$VLLM_DIR" "${MODELS_DIR}/converted"

log_info "Model: ${DISPLAY_NAME}"
log_info "Sanitized: ${SANITIZED}"
log_info "Ollama host: ${OLLAMA_HOST}"

# ---- Step 1: Pull via Ollama ----
log_info "Pulling model via Ollama: ${DISPLAY_NAME}"

PULL_RESPONSE=$(curl -sf "${OLLAMA_HOST}/api/pull" \
    -d "{\"name\": \"${DISPLAY_NAME}\", \"stream\": false}" \
    2>&1) || {
    log_err "Ollama pull failed for ${DISPLAY_NAME}"
    emit_result "failed" "Ollama pull failed: ${PULL_RESPONSE}"
    exit 1
}

log_ok "Ollama pull complete for ${DISPLAY_NAME}"

# ---- Step 2: Locate Ollama model files ----
# Try to find the model blob directory via ollama show
OLLAMA_MODEL_DIR=""

# Check common Ollama storage locations
for candidate_base in \
    "${HOME}/.ollama/models" \
    "/usr/share/ollama/.ollama/models" \
    "/var/lib/ollama/models"; do
    if [[ -d "$candidate_base" ]]; then
        OLLAMA_MODEL_DIR="$candidate_base"
        break
    fi
done

if [[ -z "$OLLAMA_MODEL_DIR" ]]; then
    log_warn "Could not locate Ollama model storage directory"
fi

# ---- Step 3: HuggingFace download (if hf_repo provided) ----
HF_SNAPSHOT_DIR=""
if [[ -n "$HF_REPO" ]]; then
    log_info "Downloading from HuggingFace: ${HF_REPO}"

    # Find Python with huggingface_hub
    PYTHON=""
    for py in "${AGENT_DIR}/third_party/vllm/.venv/bin/python" "${AGENT_DIR}/.venv/bin/python" "python3"; do
        if command -v "$py" &>/dev/null || [[ -x "$py" ]]; then
            if "$py" -c "import huggingface_hub" 2>/dev/null; then
                PYTHON="$py"
                break
            fi
        fi
    done

    if [[ -z "$PYTHON" ]]; then
        log_warn "No Python with huggingface_hub found; skipping HF download"
    else
        HF_SNAPSHOT_DIR=$("$PYTHON" - "$HF_REPO" "$CONVERTED_DIR" <<'PYEOF'
import os, sys
repo_id, local_dir = sys.argv[1], sys.argv[2]
token = os.environ.get("HF_TOKEN") or None
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    token=token,
    ignore_patterns=["*.bin", "*.pt", "*.ckpt", "original/**", "*.pth"],
)
print(path)
PYEOF
        ) || {
            log_warn "HF download failed; will attempt conversion from Ollama GGUF instead"
            HF_SNAPSHOT_DIR=""
        }
    fi
fi

# ---- Step 4: Convert GGUF to safetensors (if no HF snapshot and convert not skipped) ----
if [[ -n "$HF_SNAPSHOT_DIR" && -d "$HF_SNAPSHOT_DIR" ]]; then
    log_ok "Using HF snapshot at: ${HF_SNAPSHOT_DIR}"
    # If HF snapshot has config.json and safetensors, it's ready for vLLM
    if [[ -f "${HF_SNAPSHOT_DIR}/config.json" ]]; then
        CONVERTED_DIR="$HF_SNAPSHOT_DIR"
    fi
elif [[ "$SKIP_CONVERT" == "true" || "$AUTO_CONVERT" == "false" ]]; then
    log_info "Skipping conversion (AUTO_CONVERT=${AUTO_CONVERT}, SKIP_CONVERT=${SKIP_CONVERT})"
    emit_result "success" ""
    exit 0
else
    log_info "No HF snapshot available; conversion from GGUF not implemented in this version"
    log_info "Set --hf-repo to provide HuggingFace repo ID for direct safetensors download"
    # Still mark the Ollama pull as success — just no vLLM conversion
    VLLM_PATH=""
fi

# ---- Step 5: Create vLLM symlink ----
if [[ -n "$CONVERTED_DIR" && -d "$CONVERTED_DIR" ]]; then
    # Mark directory as agent-managed
    touch "${CONVERTED_DIR}/.benchrace_converted_by_agent=true"

    # Create or update symlink
    if [[ -L "$VLLM_PATH" ]]; then
        rm "$VLLM_PATH"
    fi
    ln -sf "$CONVERTED_DIR" "$VLLM_PATH"
    log_ok "vLLM symlink: ${VLLM_PATH} -> ${CONVERTED_DIR}"
fi

# ---- Step 6: Emit result ----
log_ok "Sync and convert complete for ${DISPLAY_NAME}"
emit_result "success" ""

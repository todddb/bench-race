#!/usr/bin/env bash
#
# scripts/install_trtllm.sh
#
# One-stop installer for TensorRT-LLM backend (containerized)
# - Pulls NVIDIA TRT-LLM container (configurable)
# - Downloads HF model snapshot inside the container
# - Converts HF checkpoint -> TRT-LLM checkpoint -> builds TensorRT engines inside container
# - Copies tokenizer files into engine directory so trtllm-serve can find them
# - Persists artifacts under agent/models/trtllm/
# - Installs agent/backends/trtllm_run.sh launcher (start/stop/status/logs/restart)
# - Starts TRT-LLM server (serving from engines) and runs smoke test
#
# Usage:
#   ./scripts/install_trtllm.sh [--yes] [--image IMAGE] [--port PORT] [--model MODEL_ID]
#                               [--precision float16|bfloat16|float32] [--system-service]
#                               [--skip-docker-login] [--no-detach] [--keep-artifacts]
#                               [--max-batch-size N] [--max-input-len N] [--max-seq-len N]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AGENT_DIR="${REPO_ROOT}/agent"
MODELS_ROOT="${AGENT_DIR}/models/trtllm"
BACKENDS_DIR="${AGENT_DIR}/backends"
LAUNCHER="${BACKENDS_DIR}/trtllm_run.sh"
SECRETS_FILE="${HOME}/.bench-race-secrets"

# Defaults
DEFAULT_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6.post3"
DEFAULT_PORT="8000"
DEFAULT_CONTAINER_NAME="bench-race-trtllm"
DEFAULT_MODEL="distilgpt2"
DEFAULT_PRECISION="float16"        # must match convert_checkpoint.py choices: auto, float16, bfloat16, float32
DEFAULT_MAX_BATCH_SIZE="2048"
DEFAULT_MAX_INPUT_LEN="512"
DEFAULT_MAX_SEQ_LEN="1024"

# CLI flags (defaults)
YES_MODE=false
IMAGE="${DEFAULT_IMAGE}"
PORT="${DEFAULT_PORT}"
CONTAINER_NAME="${DEFAULT_CONTAINER_NAME}"
MODEL_ID="${DEFAULT_MODEL}"
PRECISION="${DEFAULT_PRECISION}"
MAX_BATCH_SIZE="${DEFAULT_MAX_BATCH_SIZE}"
MAX_INPUT_LEN="${DEFAULT_MAX_INPUT_LEN}"
MAX_SEQ_LEN="${DEFAULT_MAX_SEQ_LEN}"
SYSTEM_SERVICE=false
SKIP_DOCKER_LOGIN=false
NO_DETACH=false
KEEP_ARTIFACTS=false

# Colors for logs
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERR]${NC} $*"; }
log_step()    { echo -e "${CYAN}[STEP]${NC} $*"; }

print_usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --yes                         Assume yes for prompts
  --image IMAGE                 TRT-LLM container image (default: ${DEFAULT_IMAGE})
  --port PORT                   Host port to map container 8000 to (default: ${DEFAULT_PORT})
  --container-name NAME         Docker container name (default: ${DEFAULT_CONTAINER_NAME})
  --model MODEL_ID              HuggingFace model id (default: ${DEFAULT_MODEL})
  --precision float16|bfloat16|float32
                                Precision for checkpoint conversion (default: ${DEFAULT_PRECISION})
  --max-batch-size N            Max batch size for engine build (default: ${DEFAULT_MAX_BATCH_SIZE})
  --max-input-len N             Max input length for engine build (default: ${DEFAULT_MAX_INPUT_LEN})
  --max-seq-len N               Max sequence length for engine build (default: ${DEFAULT_MAX_SEQ_LEN})
  --system-service              Install user-level systemd unit
  --skip-docker-login           Skip docker login to nvcr.io (NGC)
  --no-detach                   Start server in foreground during install
  --keep-artifacts              Keep intermediate checkpoint dirs after engine build
  -h, --help                    Show this help
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes) YES_MODE=true; shift ;;
    --image) IMAGE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --container-name) CONTAINER_NAME="$2"; shift 2 ;;
    --model) MODEL_ID="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --max-batch-size) MAX_BATCH_SIZE="$2"; shift 2 ;;
    --max-input-len) MAX_INPUT_LEN="$2"; shift 2 ;;
    --max-seq-len) MAX_SEQ_LEN="$2"; shift 2 ;;
    --system-service) SYSTEM_SERVICE=true; shift ;;
    --skip-docker-login) SKIP_DOCKER_LOGIN=true; shift ;;
    --no-detach) NO_DETACH=true; shift ;;
    --keep-artifacts) KEEP_ARTIFACTS=true; shift ;;
    -h|--help) print_usage; exit 0 ;;
    *) log_error "Unknown arg: $1"; print_usage; exit 2 ;;
  esac
done

# Validate precision
case "${PRECISION}" in
  float16|bfloat16|float32|auto) ;;
  fp16)  log_warn "--precision fp16 is not valid for convert_checkpoint.py; using 'float16' instead."; PRECISION="float16" ;;
  bf16)  log_warn "--precision bf16 is not valid for convert_checkpoint.py; using 'bfloat16' instead."; PRECISION="bfloat16" ;;
  fp32)  log_warn "--precision fp32 is not valid for convert_checkpoint.py; using 'float32' instead."; PRECISION="float32" ;;
  *)     log_error "Invalid precision '${PRECISION}'. Choose from: auto, float16, bfloat16, float32"; exit 2 ;;
esac

# Platform guard
if [[ "$(uname -s)" != "Linux" ]]; then
  log_error "This installer targets Linux hosts with NVIDIA GPUs. Exiting."
  exit 3
fi

# Flatten model name for directory paths (meta-llama/Meta-Llama-3-8B -> meta-llama--Meta-Llama-3-8B)
MODEL_FLAT="${MODEL_ID//\//__}"

# Load secrets (HUGGINGFACE_TOKEN, NGC_API_KEY) if present
if [[ -r "${SECRETS_FILE}" ]]; then
  log_info "Loading secrets from ${SECRETS_FILE}"
  # shellcheck disable=SC1090
  source "${SECRETS_FILE}"
else
  log_warn "Secrets file ${SECRETS_FILE} not found; ensure HUGGINGFACE_TOKEN/NGC_API_KEY are in environment if needed."
fi

# Basic checks
command -v docker >/dev/null 2>&1 || { log_error "docker not installed/in PATH"; exit 4; }
log_info "Docker CLI available: $(docker --version 2>/dev/null || echo 'unknown')"
if command -v nvidia-smi >/dev/null 2>&1; then
  log_info "Detected local GPUs:"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
else
  log_warn "nvidia-smi not present on host (container may still access GPUs via nvidia-docker toolkit)."
fi

# Create dirs
mkdir -p "${MODELS_ROOT}/engines/${MODEL_FLAT}"
mkdir -p "${BACKENDS_DIR}"
mkdir -p "${HOME}/.cache/huggingface"

# -------------------------------------------------------------------
# Helper: docker login to nvcr if key present
# -------------------------------------------------------------------
docker_login_ngc() {
  if [[ "${SKIP_DOCKER_LOGIN}" == "true" ]]; then
    log_info "--skip-docker-login set; skipping nvcr.io login"
    return 0
  fi
  if [[ -n "${NGC_API_KEY:-}" ]]; then
    log_info "Attempting docker login to nvcr.io"
    if printf '%s' "${NGC_API_KEY}" | docker login nvcr.io --username '$oauthtoken' --password-stdin >/dev/null 2>&1; then
      log_success "Logged into nvcr.io"
      return 0
    else
      log_warn "docker login to nvcr.io failed."
      return 2
    fi
  else
    log_warn "NGC_API_KEY not set; relying on local image cache."
    return 1
  fi
}

# Pull image (non-fatal)
pull_image() {
  log_step "Pulling image: ${IMAGE}"
  if docker pull "${IMAGE}"; then
    log_success "Image pulled: ${IMAGE}"
  else
    log_warn "docker pull failed; if image exists locally this may be OK."
  fi
}

# -------------------------------------------------------------------
# Install launcher script (idempotent)
# -------------------------------------------------------------------
install_launcher() {
  log_step "Installing launcher: ${LAUNCHER}"

  # We write the launcher with the build-time values baked in as defaults,
  # but every value can be overridden via environment variables at runtime.
  cat > "${LAUNCHER}" <<LAUNCHER_EOF
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="\$(cd "\${SCRIPT_DIR}/../.." && pwd)"
MODELS_DIR="\${REPO_ROOT}/agent/models/trtllm"
HF_CACHE_DIR="\${HOME}/.cache/huggingface"

# Overridable via environment
CONTAINER_NAME="\${TRTLLM_CONTAINER_NAME:-${CONTAINER_NAME}}"
TRTLLM_IMAGE="\${TRTLLM_IMAGE:-${IMAGE}}"
TRTLLM_PORT="\${TRTLLM_PORT:-${PORT}}"
TRTLLM_MODEL="\${TRTLLM_MODEL:-${MODEL_FLAT}}"
TRTLLM_MAX_BATCH_SIZE="\${TRTLLM_MAX_BATCH_SIZE:-${MAX_BATCH_SIZE}}"
DETACH="\${TRTLLM_DETACH:-true}"

mkdir -p "\${MODELS_DIR}" "\${HF_CACHE_DIR}"

start() {
  # Remove stopped container if exists
  if docker ps -a --format '{{.Names}}' | grep -qx "\${CONTAINER_NAME}"; then
    if docker ps --format '{{.Names}}' | grep -qx "\${CONTAINER_NAME}"; then
      echo "Container \${CONTAINER_NAME} already running."
      return 0
    else
      echo "Removing stopped container \${CONTAINER_NAME}..."
      docker rm -f "\${CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi
  fi

  local docker_args=(
    run --name "\${CONTAINER_NAME}"
    --gpus=all --ipc=host
    --ulimit memlock=-1 --ulimit stack=67108864
    -p "\${TRTLLM_PORT}:8000"
    -v "\${MODELS_DIR}:/workspace/trtllm:ro"
    -v "\${HF_CACHE_DIR}:/root/.cache/huggingface:ro"
  )

  [[ -n "\${HUGGINGFACE_TOKEN:-}" ]] && docker_args+=( -e "HUGGINGFACE_TOKEN=\${HUGGINGFACE_TOKEN}" )
  [[ "\${DETACH}" == "true" ]] && docker_args+=( -d )

  docker "\${docker_args[@]}" "\${TRTLLM_IMAGE}" \
    trtllm-serve serve \
      --backend tensorrt \
      --max_batch_size "\${TRTLLM_MAX_BATCH_SIZE}" \
      --host 0.0.0.0 --port 8000 \
      "/workspace/trtllm/engines/\${TRTLLM_MODEL}"
}

stop() {
  if docker ps -a --format '{{.Names}}' | grep -qx "\${CONTAINER_NAME}"; then
    docker rm -f "\${CONTAINER_NAME}" >/dev/null 2>&1 || true
    echo "Stopped \${CONTAINER_NAME}."
  else
    echo "Container \${CONTAINER_NAME} not present."
  fi
}

status() {
  docker ps -a --filter "name=\${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

logs() {
  docker logs -f "\${CONTAINER_NAME}"
}

restart() {
  stop || true
  start
}

case "\${1:-start}" in
  start) start ;;
  stop) stop ;;
  restart) restart ;;
  status) status ;;
  logs) logs ;;
  *) echo "Usage: \$0 {start|stop|restart|status|logs}" >&2; exit 2 ;;
esac
LAUNCHER_EOF

  chmod +x "${LAUNCHER}"
  log_success "Launcher installed at ${LAUNCHER}"
}

# -------------------------------------------------------------------
# Create systemd user unit (optional)
# -------------------------------------------------------------------
install_systemd() {
  if [[ "${SYSTEM_SERVICE}" != "true" ]]; then return 0; fi
  if ! command -v systemctl >/dev/null 2>&1; then
    log_warn "systemctl not available; skipping systemd unit"
    return 0
  fi
  local unit_dir="${HOME}/.config/systemd/user"
  mkdir -p "${unit_dir}"
  local unit_file="${unit_dir}/bench-race-trtllm.service"
  cat > "${unit_file}" <<UNIT
[Unit]
Description=bench-race TensorRT-LLM backend
After=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${REPO_ROOT}
Environment=TRTLLM_PORT=${PORT}
Environment=TRTLLM_IMAGE=${IMAGE}
Environment=TRTLLM_MODEL=${MODEL_FLAT}
Environment=TRTLLM_MAX_BATCH_SIZE=${MAX_BATCH_SIZE}
ExecStart=${LAUNCHER} start
ExecStop=${LAUNCHER} stop

[Install]
WantedBy=default.target
UNIT

  systemctl --user daemon-reload || true
  systemctl --user enable bench-race-trtllm.service >/dev/null 2>&1 || true
  log_success "Installed systemd user unit at ${unit_file}"
}

# -------------------------------------------------------------------
# Detect model family — determines which convert script to use
# Returns: gpt, llama, falcon, mpt, baichuan, chatglm, qwen, phi,
#          gemma, mixtral, dbrx, recurrentgemma, mamba, etc.
# -------------------------------------------------------------------
detect_model_family() {
  local model_lower
  model_lower="$(echo "${MODEL_ID}" | tr '[:upper:]' '[:lower:]')"
  case "${model_lower}" in
    *llama*|*meta-llama*|*codellama*|*vicuna*|*alpaca*)  echo "llama" ;;
    *mistral*|*mixtral*)     echo "llama" ;;   # Mistral/Mixtral use llama converter
    *gpt2*|*distilgpt*)      echo "gpt" ;;
    *starcoder*)             echo "gpt" ;;
    *falcon*)                echo "falcon" ;;
    *mpt*)                   echo "mpt" ;;
    *phi-3*|*phi-2*|*phi-1*) echo "phi" ;;
    *gemma*)                 echo "gemma" ;;
    *qwen*)                  echo "qwen" ;;
    *baichuan*)              echo "baichuan" ;;
    *chatglm*|*glm*)         echo "chatglm" ;;
    *bloom*)                 echo "bloom" ;;
    *opt-*)                  echo "opt" ;;
    *mamba*)                 echo "mamba" ;;
    *)                       echo "auto" ;;
  esac
}

# For GPT-family models, detect the specific variant
detect_gpt_variant() {
  local model_lower
  model_lower="$(echo "${MODEL_ID}" | tr '[:upper:]' '[:lower:]')"
  case "${model_lower}" in
    *gpt2*|*distilgpt*)  echo "gpt2" ;;
    *starcoder2*)        echo "starcoder2" ;;
    *starcoder*)         echo "starcoder" ;;
    *santacoder*)        echo "santacoder" ;;
    *persimmon*)         echo "persimmon" ;;
    *nemotron*)          echo "nemotron" ;;
    *)                   echo "" ;;
  esac
}

# -------------------------------------------------------------------
# Build model: download, convert, build engines — all inside container
# -------------------------------------------------------------------
prepare_and_build_model() {
  local engine_out="${MODELS_ROOT}/engines/${MODEL_FLAT}"

  # Skip if engines already exist
  if ls "${engine_out}"/*.engine 2>/dev/null | head -1 >/dev/null 2>&1; then
    log_info "Engine files already exist at ${engine_out}; skipping build."
    return 0
  fi

  local model_family
  model_family="$(detect_model_family)"
  local gpt_variant=""
  if [[ "${model_family}" == "gpt" ]]; then
    gpt_variant="$(detect_gpt_variant)"
  fi

  log_step "Building TRT-LLM engines inside container"
  log_info "  Model:          ${MODEL_ID}"
  log_info "  Model family:   ${model_family}"
  log_info "  Precision:      ${PRECISION}"
  [[ -n "${gpt_variant}" ]] && log_info "  GPT variant:    ${gpt_variant}"
  log_info "  Max batch size: ${MAX_BATCH_SIZE}"
  log_info "  Max input len:  ${MAX_INPUT_LEN}"
  log_info "  Max seq len:    ${MAX_SEQ_LEN}"

  # Build the inner script.
  # Everything runs inside the container under /tmp/build.
  # Only the final engine + tokenizer files get written to /output (which is volume-mounted).
  local INNER
  read -r -d '' INNER <<'INNER_SCRIPT' || true
set -euo pipefail

MODEL_ID="${MODEL_ENV}"
MODEL_FAMILY="${MODEL_FAMILY_ENV}"
PRECISION="${PRECISION_ENV}"
GPT_VARIANT="${GPT_VARIANT_ENV}"
MAX_BATCH="${MAX_BATCH_ENV}"
MAX_INPUT="${MAX_INPUT_ENV}"
MAX_SEQ="${MAX_SEQ_ENV}"

WORK="/tmp/build"
HF_DIR="${WORK}/hf_model"
CKPT_DIR="${WORK}/checkpoint"
ENGINE_DIR="/output"   # volume-mounted to host

mkdir -p "${HF_DIR}" "${CKPT_DIR}"

echo "============================================"
echo "[build] Step 1/4: Downloading model ${MODEL_ID}"
echo "============================================"
python3 - <<PY
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="${MODEL_ID}",
    local_dir="${HF_DIR}",
    cache_dir="/root/.cache/huggingface",
    token=os.getenv("HUGGINGFACE_TOKEN"),
    ignore_patterns=["*.h5", "*.msgpack", "*.tflite", "*.ot"]
)
print("[build] Download complete")
PY

echo "============================================"
echo "[build] Step 2/4: Converting checkpoint (family=${MODEL_FAMILY})"
echo "============================================"

# Find the right convert script based on model family.
# TRT-LLM organizes converters by model architecture under /app/tensorrt_llm/examples/
# The path structure varies by TRT-LLM version so we search multiple locations.
find_convert_script() {
  local family="$1"
  local candidates=()

  case "${family}" in
    llama)
      candidates=(
        /app/tensorrt_llm/examples/models/core/llama/convert_checkpoint.py
        /app/tensorrt_llm/examples/llama/convert_checkpoint.py
      )
      ;;
    gpt)
      candidates=(
        /app/tensorrt_llm/examples/models/core/gpt/convert_checkpoint.py
        /app/tensorrt_llm/examples/gpt/convert_checkpoint.py
      )
      ;;
    falcon)
      candidates=(
        /app/tensorrt_llm/examples/models/core/falcon/convert_checkpoint.py
        /app/tensorrt_llm/examples/falcon/convert_checkpoint.py
      )
      ;;
    phi)
      candidates=(
        /app/tensorrt_llm/examples/models/core/phi/convert_checkpoint.py
        /app/tensorrt_llm/examples/phi/convert_checkpoint.py
      )
      ;;
    gemma)
      candidates=(
        /app/tensorrt_llm/examples/models/core/gemma/convert_checkpoint.py
        /app/tensorrt_llm/examples/gemma/convert_checkpoint.py
      )
      ;;
    qwen)
      candidates=(
        /app/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py
        /app/tensorrt_llm/examples/qwen/convert_checkpoint.py
      )
      ;;
    bloom)
      candidates=(
        /app/tensorrt_llm/examples/models/core/bloom/convert_checkpoint.py
        /app/tensorrt_llm/examples/bloom/convert_checkpoint.py
      )
      ;;
    opt)
      candidates=(
        /app/tensorrt_llm/examples/models/core/opt/convert_checkpoint.py
        /app/tensorrt_llm/examples/opt/convert_checkpoint.py
      )
      ;;
    mpt)
      candidates=(
        /app/tensorrt_llm/examples/models/core/mpt/convert_checkpoint.py
        /app/tensorrt_llm/examples/mpt/convert_checkpoint.py
      )
      ;;
    auto|*)
      # Fallback: try to find any convert script and hope for the best
      candidates=(
        /app/tensorrt_llm/examples/models/core/gpt/convert_checkpoint.py
        /app/tensorrt_llm/examples/gpt/convert_checkpoint.py
      )
      echo "[build] WARNING: Unknown model family '${family}', falling back to GPT converter"
      ;;
  esac

  for p in "${candidates[@]}"; do
    if [ -f "$p" ]; then
      echo "$p"
      return 0
    fi
  done

  # Last resort: search the filesystem
  echo "[build] Searching for any convert_checkpoint.py in /app/tensorrt_llm/examples/..." >&2
  find /app/tensorrt_llm/examples/ -name "convert_checkpoint.py" -print >&2 || true
  return 1
}

CONVERT="$(find_convert_script "${MODEL_FAMILY}")" || {
  echo "[build] ERROR: Could not find convert_checkpoint.py for family '${MODEL_FAMILY}'"
  echo "[build] Available convert scripts in container:"
  find /app/tensorrt_llm/examples/ -name "convert_checkpoint.py" 2>/dev/null || true
  exit 2
}
echo "[build] Using convert script: ${CONVERT}"

CONVERT_ARGS=(
  --model_dir "${HF_DIR}"
  --output_dir "${CKPT_DIR}"
  --dtype "${PRECISION}"
  --tp_size 1
)

# Only add --gpt_variant for GPT-family models
if [ "${MODEL_FAMILY}" = "gpt" ] && [ -n "${GPT_VARIANT}" ]; then
  CONVERT_ARGS+=(--gpt_variant "${GPT_VARIANT}")
fi

python3 "${CONVERT}" "${CONVERT_ARGS[@]}"
echo "[build] Checkpoint conversion complete"

echo "============================================"
echo "[build] Step 3/4: Building TensorRT engines"
echo "============================================"
trtllm-build \
  --checkpoint_dir "${CKPT_DIR}" \
  --gemm_plugin float16 \
  --output_dir "${ENGINE_DIR}" \
  --max_batch_size "${MAX_BATCH}" \
  --max_input_len "${MAX_INPUT}" \
  --max_seq_len "${MAX_SEQ}" \
  --workers 1

echo "[build] Engine build complete"

echo "============================================"
echo "[build] Step 4/4: Copying tokenizer files"
echo "============================================"
# Copy tokenizer + generation config into engine dir so trtllm-serve finds them
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt \
         special_tokens_map.json generation_config.json tokenizer.model; do
  if [ -f "${HF_DIR}/${f}" ]; then
    cp "${HF_DIR}/${f}" "${ENGINE_DIR}/"
    echo "  copied ${f}"
  fi
done

echo "============================================"
echo "[build] ALL DONE — engine files in ${ENGINE_DIR}"
echo "============================================"
ls -lh "${ENGINE_DIR}/"
INNER_SCRIPT

  # Docker args: mount only the engine output dir (rw) and HF cache
  local docker_args=(
    --rm --gpus=all --ipc=host
    -v "${engine_out}:/output:rw"
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface:rw"
    -e "MODEL_ENV=${MODEL_ID}"
    -e "MODEL_FAMILY_ENV=${model_family}"
    -e "PRECISION_ENV=${PRECISION}"
    -e "GPT_VARIANT_ENV=${gpt_variant}"
    -e "MAX_BATCH_ENV=${MAX_BATCH_SIZE}"
    -e "MAX_INPUT_ENV=${MAX_INPUT_LEN}"
    -e "MAX_SEQ_ENV=${MAX_SEQ_LEN}"
  )
  [[ -n "${HUGGINGFACE_TOKEN:-}" ]] && docker_args+=( -e "HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}" )

  log_info "Running build container (this can take a while)..."
  docker run "${docker_args[@]}" "${IMAGE}" /bin/bash -c "${INNER}"

  # Verify engines
  if ls "${engine_out}"/*.engine 2>/dev/null | head -1 >/dev/null 2>&1; then
    log_success "Engine files present at ${engine_out}"
    ls -lh "${engine_out}/"
  else
    log_error "No .engine files found at ${engine_out} after build!"
    log_error "Check the build output above for errors."
    exit 5
  fi
}

# -------------------------------------------------------------------
# Smoke test
# -------------------------------------------------------------------
run_smoke() {
  log_step "Waiting for TRT-LLM server on http://127.0.0.1:${PORT}"
  for i in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      log_success "Server responded to /v1/models"
      return 0
    fi
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      log_success "Server responded to /health"
      return 0
    fi
    log_info "Waiting for server... (${i}/30)"
    sleep 2
  done
  log_warn "Server did not respond after 60s."
  return 1
}

# ===================================================================
# MAIN
# ===================================================================
log_info "Starting TRT-LLM installer"
log_info "Repo root:        ${REPO_ROOT}"
log_info "Model:            ${MODEL_ID} (flat: ${MODEL_FLAT})"
log_info "Precision:        ${PRECISION}"
log_info "Max batch size:   ${MAX_BATCH_SIZE}"
log_info "Max input len:    ${MAX_INPUT_LEN}"
log_info "Max seq len:      ${MAX_SEQ_LEN}"

# Login + pull
docker_login_ngc || log_warn "NGC login did not succeed or was skipped"
pull_image

# Install launcher
install_launcher

# Build model (idempotent — skips if engines already present)
prepare_and_build_model

# Export for launcher
export TRTLLM_IMAGE="${IMAGE}"
export TRTLLM_PORT="${PORT}"
export TRTLLM_CONTAINER_NAME="${CONTAINER_NAME}"
export TRTLLM_MODEL="${MODEL_FLAT}"
export TRTLLM_MAX_BATCH_SIZE="${MAX_BATCH_SIZE}"

# Start server
if [[ "${NO_DETACH}" == "true" ]]; then
  log_info "Starting TRT-LLM server in foreground (Ctrl-C to stop)"
  TRTLLM_DETACH=false "${LAUNCHER}" start
  exit 0
else
  log_step "Starting TRT-LLM server (detached)"
  TRTLLM_DETACH=true "${LAUNCHER}" start >/dev/null 2>&1 || {
    log_warn "Start failed; retrying after cleanup..."
    "${LAUNCHER}" stop || true
    sleep 1
    TRTLLM_DETACH=true "${LAUNCHER}" start >/dev/null 2>&1 || {
      log_error "Failed to start container. Check: docker logs ${CONTAINER_NAME}"
      exit 6
    }
  }
fi

# Systemd (optional)
install_systemd

# Smoke test
if ! run_smoke; then
  log_warn "Smoke test failed. Check logs: ${LAUNCHER} logs"
  exit 6
fi

log_success "TRT-LLM server is up! Test with:"
echo ""
echo "  curl -s http://localhost:${PORT}/v1/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"${MODEL_ID}\",\"prompt\":\"Hello world\",\"max_tokens\":32}' | python3 -m json.tool"
echo ""
log_info "Manage with: ${LAUNCHER} {start|stop|restart|status|logs}"
exit 0

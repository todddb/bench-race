#!/usr/bin/env bash
#
# scripts/install_trtllm.sh
#
# One-stop installer for TensorRT-LLM backend (containerized)
# - Pulls NVIDIA TRT-LLM container (configurable)
# - Downloads HF model snapshot inside the container (no host venv)
# - Converts HF checkpoint -> TRT-LLM checkpoint -> builds TensorRT engines inside container
# - Persists artifacts under agent/models/trtllm/
# - Installs agent/backends/trtllm_run.sh launcher (start/stop/status/logs/restart)
# - Starts TRT-LLM server (serving from engines) and runs smoke test
#
# Usage:
#   ./scripts/install_trtllm.sh [--yes] [--image IMAGE] [--port PORT] [--model MODEL_ID]
#                               [--precision fp16|fp32|fp8] [--system-service]
#                               [--skip-docker-login] [--no-detach] [--keep-artifacts]
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
DEFAULT_MODEL="distilgpt2"   # small test model (fast to download/build)
DEFAULT_PRECISION="fp16"     # build precision: fp16 recommended for speed/size

# CLI flags (defaults)
YES_MODE=false
IMAGE="${DEFAULT_IMAGE}"
PORT="${DEFAULT_PORT}"
CONTAINER_NAME="${DEFAULT_CONTAINER_NAME}"
MODEL_ID="${DEFAULT_MODEL}"
PRECISION="${DEFAULT_PRECISION}"
SYSTEM_SERVICE=false
SKIP_DOCKER_LOGIN=false
NO_DETACH=false
KEEP_ARTIFACTS=false   # if true, don't remove intermediate checkpoint dir after build

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
  --yes                     Assume yes for prompts
  --image IMAGE             TRT-LLM container image (default: ${DEFAULT_IMAGE})
  --port PORT               Host port to map container 8000 to (default: ${DEFAULT_PORT})
  --container-name NAME     Docker container name (default: ${DEFAULT_CONTAINER_NAME})
  --model MODEL_ID          HuggingFace model id to download/build (default: ${DEFAULT_MODEL})
  --precision fp16|fp32|fp8 Precision for engine build (default: ${DEFAULT_PRECISION})
  --system-service          Install user-level systemd unit to manage backend
  --skip-docker-login       Skip docker login to nvcr.io (NGC)
  --no-detach               Start server in foreground during install
  --keep-artifacts          Keep intermediate checkpoint dirs after engine build
  -h, --help                Show this help
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
    --system-service) SYSTEM_SERVICE=true; shift ;;
    --skip-docker-login) SKIP_DOCKER_LOGIN=true; shift ;;
    --no-detach) NO_DETACH=true; shift ;;
    --keep-artifacts) KEEP_ARTIFACTS=true; shift ;;
    -h|--help) print_usage; exit 0 ;;
    *) log_error "Unknown arg: $1"; print_usage; exit 2 ;;
  esac
done

# Platform guard
if [[ "$(uname -s)" != "Linux" ]]; then
  log_error "This installer targets Linux hosts with NVIDIA GPUs (nvidia-docker). Exiting."
  exit 3
fi

# Load secrets (HUGGINGFACE_TOKEN, NGC_API_KEY) if present
if [[ -r "${SECRETS_FILE}" ]]; then
  log_info "Loading secrets from ${SECRETS_FILE} (not echoed)"
  # shellcheck disable=SC1090
  source "${SECRETS_FILE}"
else
  log_warn "Secrets file ${SECRETS_FILE} not readable; ensure HUGGINGFACE_TOKEN/NGC_API_KEY are available in environment if needed."
fi

# Basic checks
command -v docker >/dev/null 2>&1 || { log_error "docker not installed/in PATH"; exit 4; }
log_info "Docker CLI available: $(docker --version 2>/dev/null || echo 'unknown')"
if command -v nvidia-smi >/dev/null 2>&1; then
  log_info "Detected local GPUs:"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
else
  log_warn "nvidia-smi not present; host GPU detection limited (but docker container may still access GPUs via toolkit)."
fi

# Create dirs
mkdir -p "${MODELS_ROOT}"
mkdir -p "${BACKENDS_DIR}"
mkdir -p "${HOME}/.cache/huggingface"

# Helper: docker login to nvcr if key present
docker_login_ngc() {
  if [[ "${SKIP_DOCKER_LOGIN}" == "true" ]]; then
    log_info "--skip-docker-login set; skipping nvcr.io login"
    return 0
  fi
  if [[ -n "${NGC_API_KEY:-}" ]]; then
    log_info "Attempting docker login to nvcr.io with NGC_API_KEY (username literal '\$oauthtoken')"
    if printf '%s' "${NGC_API_KEY}" | docker login nvcr.io --username '$oauthtoken' --password-stdin >/dev/null 2>&1; then
      log_success "Logged into nvcr.io"
      return 0
    else
      log_warn "docker login to nvcr.io failed. If image is cached locally that may be ok; otherwise re-run with a valid NGC_API_KEY."
      return 2
    fi
  else
    log_warn "NGC_API_KEY not set; relying on local cache or public access."
    return 1
  fi
}

# Pull image (non-fatal)
pull_image() {
  log_step "Pulling image: ${IMAGE}"
  if docker pull "${IMAGE}"; then
    log_success "Image: ${IMAGE}"
    return 0
  else
    log_warn "docker pull failed for ${IMAGE}. If image exists locally this may be OK; otherwise container run will likely fail."
    return 2
  fi
}

# Install launcher script (idempotent)
install_launcher() {
  local launcher="${LAUNCHER}"
  log_step "Installing launcher: ${launcher}"
  cat > "${launcher}" <<'LAUNCHER_EOF'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODELS_DIR="${REPO_ROOT}/agent/models/trtllm"
HF_CACHE_DIR="${HOME}/.cache/huggingface"
CONTAINER_NAME="${TRTLLM_CONTAINER_NAME:-bench-race-trtllm}"
TRTLLM_IMAGE="${TRTLLM_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6.post3}"
TRTLLM_PORT="${TRTLLM_PORT:-8000}"
DETACH="${TRTLLM_DETACH:-true}"

mkdir -p "${MODELS_DIR}" "${HF_CACHE_DIR}"

start() {
  # remove stopped container if exists (avoid name clashes)
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
      echo "Container ${CONTAINER_NAME} already running."
      return 0
    else
      echo "Removing existing stopped container ${CONTAINER_NAME}..."
      docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi
  fi

  local docker_args=(run --name "${CONTAINER_NAME}" --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p "${TRTLLM_PORT}:8000" -v "${MODELS_DIR}:/workspace/trtllm:rw" -v "${HF_CACHE_DIR}:/root/.cache/huggingface:rw")

  if [[ -n "${HUGGINGFACE_TOKEN:-}" ]]; then
    docker_args+=( -e "HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}" )
  fi
  if [[ "${DETACH}" == "true" ]]; then
    docker_args+=( -d )
  fi

  docker "${docker_args[@]}" "${TRTLLM_IMAGE}" /bin/bash -c "trtllm-serve serve --host 0.0.0.0 --port 8000"
}

stop() {
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    echo "Stopped ${CONTAINER_NAME}."
  else
    echo "Container ${CONTAINER_NAME} not present."
  fi
}

status() {
  docker ps -a --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

logs() {
  docker logs -f "${CONTAINER_NAME}"
}

restart() {
  stop || true
  start
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  restart) restart ;;
  status) status ;;
  logs) logs ;;
  *) echo "Usage: $0 {start|stop|restart|status|logs}" >&2; exit 2 ;;
esac
LAUNCHER_EOF

  chmod +x "${launcher}"
  log_success "Launcher installed at ${launcher}"
}

# Create systemd user unit (optional)
install_systemd() {
  if [[ "${SYSTEM_SERVICE}" != "true" ]]; then return 0; fi
  if ! command -v systemctl >/dev/null 2>&1; then
    log_warn "systemctl not available; skipping systemd unit installation"
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
ExecStart=${LAUNCHER} start
ExecStop=${LAUNCHER} stop

[Install]
WantedBy=default.target
UNIT

  systemctl --user daemon-reload || true
  systemctl --user enable bench-race-trtllm.service >/dev/null 2>&1 || true
  log_success "Installed systemd user unit at ${unit_file}"
}

# Run model download + convert + build inside container (idempotent)
# Layout under MODELS_ROOT:
#   models/<MODEL_ID>    <- HF snapshot (tokenizer/config)
#   checkpoints/<MODEL_ID>/checkpoint  <- converted checkpoint (trt-llm format)
#   engines/<MODEL_ID>   <- output engines (trt plans etc.)
prepare_and_build_model() {
  local model="${MODEL_ID}"
  local precision="${PRECISION}"
  local model_dir="${MODELS_ROOT}/models/${model}"
  local checkpoint_dir="${MODELS_ROOT}/checkpoints/${model}"
  local engine_out="${MODELS_ROOT}/engines/${model}"
  local tmp_checkpoint="${checkpoint_dir}/checkpoint"   # trtllm examples expect --output_dir to be /checkpoint

  mkdir -p "${model_dir}" "${checkpoint_dir}" "${engine_out}"

  # If engines already exist, skip build
  if [[ -n "$(ls -A "${engine_out}" 2>/dev/null || true)" ]]; then
    log_info "Engines already exist for model ${model} at ${engine_out}; skipping build."
    return 0
  fi

  log_step "Starting containerized flow to download model and build TensorRT engines (model=${model}, precision=${precision})"
  # Build command that will run inside the container
  # 1) try to use huggingface_hub.snapshot_download to fetch model into /workspace/trtllm/models/<model>
  # 2) locate convert_checkpoint.py inside container
  # 3) run conversion -> checkpoint_dir
  # 4) run trtllm-build to produce engines into /workspace/trtllm/engines/<model>
  #
  # NOTE: Pass HUGGINGFACE_TOKEN via env if available.
  docker_args=(--rm --gpus=all -v "${model_dir}:/workspace/trtllm/models/${model}:rw" -v "${checkpoint_dir}:/workspace/trtllm/checkpoints/${model}:rw" -v "${engine_out}:/workspace/trtllm/engines/${model}:rw" -v "${HOME}/.cache/huggingface:/root/.cache/huggingface:rw" --ipc=host)

  # Compose inner script (here-doc)
  read -r -d '' INNER <<'INNER_EOF' || true
set -euo pipefail
MODEL_ID="'${MODEL_PLACEHOLDER}'"
MODEL_DIR="/workspace/trtllm/models/${MODEL_PLACEHOLDER}"
CHECKPOINT_DIR="/workspace/trtllm/checkpoints/${MODEL_PLACEHOLDER}"
ENGINE_OUT="/workspace/trtllm/engines/${MODEL_PLACEHOLDER}"

echo "[container] python snapshot_download for ${MODEL_PLACEHOLDER} (if not present)"
python - <<PY
from huggingface_hub import snapshot_download
import os,sys
repo_id = os.getenv("MODEL_ENV", "${MODEL_PLACEHOLDER}")
out_dir = "${MODEL_DIR}"
token = os.getenv("HUGGINGFACE_TOKEN")
if not os.path.exists(out_dir) or not os.listdir(out_dir):
    print("Downloading model snapshot to", out_dir)
    snapshot_download(repo_id=repo_id, local_dir=out_dir, cache_dir='/root/.cache/huggingface', token=token, ignore_patterns=['*.h5','*.msgpack','*.tflite'])
else:
    print("Model dir exists and is non-empty; skipping download")
PY

# find convert script (GPT examples vary across image versions)
CONVERT=""
for p in /app/tensorrt_llm/examples/models/core/gpt/convert_checkpoint.py /app/tensorrt_llm/examples/gpt/convert_checkpoint.py /app/tensorrt_llm/examples/models/core/gpt/convert_checkpoint.py; do
  if [ -f "$p" ]; then CONVERT="$p"; break; fi
done
if [ -z "$CONVERT" ]; then
  echo "[container] convert_checkpoint.py not found; attempting to use generic convert script"
fi
echo "[container] convert script: $CONVERT"

# Create checkpoint dir and run conversion (idempotent)
mkdir -p "${CHECKPOINT_DIR}"
python "${CONVERT}" --model_dir "${MODEL_DIR}" --output_dir "${CHECKPOINT_DIR}" --dtype ${PRECISION} --tp_size 1 || {
  echo "[container] convert step may have failed (continuing if checkpoint exists)"
}

echo "[container] Running trtllm-build (may take time)"
trtllm-build --checkpoint_dir "${CHECKPOINT_DIR}" --gemm_plugin float16 --output_dir "${ENGINE_OUT}" --max_batch_size 1 --max_input_len 128 --max_seq_len 512 --workers 1 || {
  echo "[container] trtllm-build reported non-zero exit; check logs. If engines exist at ${ENGINE_OUT} they may still be usable."
}

echo "[container] build finished"
INNER_EOF

  # Replace placeholder MODEL_PLACEHOLDER with actual model name; and set precision placeholder
  INNER="${INNER//MODEL_PLACEHOLDER/${model}}"
  INNER="${INNER//PRECISION/${precision}}"

  # Run the container with environment to pass HUGGINGFACE_TOKEN and MODEL
  log_info "Running conversion & build in container (this can take time; logs will stream here)"
  if [[ -n "${HUGGINGFACE_TOKEN:-}" ]]; then
    docker run "${docker_args[@]}" -e HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN}" -e MODEL_ENV="${model}" "${IMAGE}" /bin/bash -lc "${INNER}"
  else
    docker run "${docker_args[@]}" -e MODEL_ENV="${model}" "${IMAGE}" /bin/bash -lc "${INNER}"
  fi

  # If KEEP_ARTIFACTS is false we could remove large intermediate files; leave them by default
  if [[ "${KEEP_ARTIFACTS}" == "false" ]]; then
    log_info "Keeping engine artifacts at ${engine_out} and checkpoint at ${checkpoint_dir} (KEEP_ARTIFACTS=false means we DO NOT delete them)."
  else
    log_info "KEEP_ARTIFACTS=true; preserving all artifacts."
  fi

  # Verify engines output
  if [[ -n "$(ls -A "${engine_out}" 2>/dev/null || true)" ]]; then
    log_success "Engine files present at ${engine_out}"
  else
    log_warn "No engine files found at ${engine_out} after build; check container logs for errors."
  fi
}

# Robust smoke test (tries v1/models and /health)
run_smoke() {
  log_step "Waiting for TRT-LLM server on http://127.0.0.1:${PORT} (may take while engines load)"
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
  log_warn "Server did not respond after retries."
  return 1
}

# MAIN flow
log_info "Starting TRT-LLM installer (combined: container + model build + launcher)"
log_info "Repo root: ${REPO_ROOT}"
log_info "Model: ${MODEL_ID}  Precision: ${PRECISION}"

# login/pull
docker_login_ngc || log_warn "NGC login did not succeed or skipped"

if ! pull_image; then
  log_warn "Image pull failed or warned; ensure image is present locally or fix NGC auth if necessary."
fi

# create launcher and layout
install_launcher
mkdir -p "${MODELS_ROOT}/models" "${MODELS_ROOT}/checkpoints" "${MODELS_ROOT}/engines"

# Build model if necessary (idempotent)
prepare_and_build_model

# Export envs for launcher / run
export TRTLLM_IMAGE="${IMAGE}"
export TRTLLM_PORT="${PORT}"
export TRTLLM_CONTAINER_NAME="${CONTAINER_NAME}"
# HUGGINGFACE_TOKEN already exported if present

# Start container via launcher
if [[ "${NO_DETACH}" == "true" ]]; then
  log_info "Starting TRT-LLM server in foreground (no detach). Ctrl-C to stop."
  TRTLLM_DETACH=false TRTLLM_IMAGE="${IMAGE}" TRTLLM_PORT="${PORT}" TRTLLM_CONTAINER_NAME="${CONTAINER_NAME}" "${LAUNCHER}" start
  # foreground mode -> skip smoke test (user watches logs)
  exit 0
else
  log_step "Starting TRT-LLM server (detached)"
  TRTLLM_DETACH=true TRTLLM_IMAGE="${IMAGE}" TRTLLM_PORT="${PORT}" TRTLLM_CONTAINER_NAME="${CONTAINER_NAME}" "${LAUNCHER}" start >/dev/null 2>&1 || {
    log_warn "Initial start failed; attempting retry after cleanup"
    TRTLLM_DETACH=true "${LAUNCHER}" stop || true
    sleep 1
    TRTLLM_DETACH=true TRTLLM_IMAGE="${IMAGE}" TRTLLM_PORT="${PORT}" TRTLLM_CONTAINER_NAME="${CONTAINER_NAME}" "${LAUNCHER}" start >/dev/null 2>&1 || {
      log_error "Failed to start container. Check docker logs for container ${CONTAINER_NAME} and ensure image ${IMAGE} exists."
      exit 6
    }
  }
fi

# optionally systemd
if [[ "${SYSTEM_SERVICE}" == "true" ]]; then
  install_systemd
fi

# smoke test
if ! run_smoke; then
  log_warn "Smoke test failed. Engines may still be building or server may need more time. Check logs with: ${LAUNCHER} logs"
  log_info "You can inspect GPU activity via: nvidia-smi"
  exit 6
fi

log_success "TRT-LLM server is up and responding. Test a chat request with:
curl -sS -X POST \"http://localhost:${PORT}/v1/chat/completions\" -H \"Content-Type: application/json\" -d '{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":16}' | jq .
"

log_info "Installer finished. You can safely delete the old install_trtllm_env_and_model.sh file if you want (artifacts are under ${MODELS_ROOT})."
exit 0

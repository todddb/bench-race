#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AGENT_DIR="${REPO_ROOT}/agent"
SECRETS_FILE="${HOME}/.bench-race-secrets"
DEFAULT_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6.post3"
DEFAULT_PORT="8000"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step() { echo -e "${CYAN}[STEP]${NC} $*"; }

YES_MODE=false
SYSTEM_SERVICE=false
SKIP_DOCKER_LOGIN=false
NO_DETACH=false
IMAGE="${DEFAULT_IMAGE}"
PORT="${DEFAULT_PORT}"

print_usage() {
    cat <<USAGE
Usage: $0 [OPTIONS]

Install/update TensorRT-LLM backend for bench-race.

Options:
  --yes                 Non-interactive mode
  --system-service      Install/update a minimal systemd user service (Linux only)
  --image IMAGE         TensorRT-LLM image (default: ${DEFAULT_IMAGE})
  --port PORT           Host port mapped to container :8000 (default: ${DEFAULT_PORT})
  --skip-docker-login   Skip docker login to nvcr.io
  --no-detach           Start backend in foreground during this install run
  --help                Show this help
USAGE
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --yes)
                YES_MODE=true
                shift
                ;;
            --system-service)
                SYSTEM_SERVICE=true
                shift
                ;;
            --image)
                IMAGE="$2"
                shift 2
                ;;
            --port)
                PORT="$2"
                shift 2
                ;;
            --skip-docker-login)
                SKIP_DOCKER_LOGIN=true
                shift
                ;;
            --no-detach)
                NO_DETACH=true
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

ensure_linux_supported() {
    if [[ "$(uname -s)" != "Linux" ]]; then
        log_error "TensorRT-LLM installer currently supports Linux with NVIDIA GPUs only."
        exit 2
    fi
}

source_secrets_if_present() {
    if [[ -r "${SECRETS_FILE}" ]]; then
        log_info "Loading secrets from ${SECRETS_FILE}"
        # shellcheck disable=SC1090
        source "${SECRETS_FILE}"
    else
        log_warning "Secrets file not found/readable at ${SECRETS_FILE}; continuing without it"
    fi
}

print_system_info() {
    log_info "==================================================="
    log_info "TensorRT-LLM Installer"
    log_info "==================================================="
    log_info "Repo root: ${REPO_ROOT}"
    log_info "Agent dir: ${AGENT_DIR}"
    log_info "Image: ${IMAGE}"
    log_info "Port: ${PORT}"

    if command -v nvidia-smi >/dev/null 2>&1; then
        log_info "Detected GPU(s):"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
    else
        log_warning "nvidia-smi not found; GPU capability checks will be limited"
    fi

    if command -v docker >/dev/null 2>&1; then
        log_info "Docker: $(docker --version 2>/dev/null || echo unavailable)"
    else
        log_error "Docker is required but not installed/in PATH"
        exit 3
    fi
}

check_docker_gpu_access() {
    log_step "Checking Docker GPU access..."
    if docker run --rm --gpus=all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
        log_success "Docker GPU access verified"
    else
        log_warning "Could not verify Docker GPU access with --gpus=all. Continuing, but runtime may fail."
    fi
}

docker_login_if_possible() {
    if [[ "${SKIP_DOCKER_LOGIN}" == "true" ]]; then
        log_info "Skipping docker login to nvcr.io (--skip-docker-login)"
        return
    fi

    if [[ -n "${NGC_API_KEY:-}" ]]; then
        log_step "Logging in to nvcr.io using NGC_API_KEY"
        if ! printf '%s' "${NGC_API_KEY}" | docker login nvcr.io --username '$oauthtoken' --password-stdin >/dev/null; then
            log_error "Failed to login to nvcr.io with NGC_API_KEY."
            log_error "Troubleshooting: verify ~/.bench-race-secrets has a valid NGC_API_KEY and retry."
            exit 4
        fi
        log_success "Docker login to nvcr.io succeeded"
    else
        log_warning "NGC_API_KEY not set; assuming existing docker auth for nvcr.io"
    fi
}

pull_image() {
    log_step "Pulling TensorRT-LLM image: ${IMAGE}"
    if ! docker pull "${IMAGE}"; then
        log_error "Failed to pull ${IMAGE}"
        log_error "Hints: ensure docker login to nvcr.io, check proxy/firewall, and verify image tag access permissions."
        exit 5
    fi
    log_success "Image pull complete"
}

create_layout() {
    log_step "Ensuring backend/model directories"
    mkdir -p "${AGENT_DIR}/models/trtllm" "${AGENT_DIR}/backends"
}

install_launcher_script() {
    local launcher="${AGENT_DIR}/backends/trtllm_run.sh"
    log_step "Installing launcher script at ${launcher}"
    cat > "${launcher}" <<'LAUNCHER'
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

start_container() {
  stop_container >/dev/null 2>&1 || true
  local docker_args=(
    run
    --name "${CONTAINER_NAME}"
    --gpus=all
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    --publish "${TRTLLM_PORT}:8000"
    --mount "type=bind,src=${MODELS_DIR},dst=/workspace/trtllm,rw"
    --mount "type=bind,src=${HF_CACHE_DIR},dst=/root/.cache/huggingface,rw"
  )

  if [[ -n "${HUGGINGFACE_TOKEN:-}" ]]; then
    docker_args+=(--env "HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}")
  fi

  if [[ "${DETACH}" == "true" ]]; then
    docker_args+=(--detach)
  fi

  docker "${docker_args[@]}" "${TRTLLM_IMAGE}"
}

stop_container() {
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    docker stop "${CONTAINER_NAME}" >/dev/null || true
    docker rm "${CONTAINER_NAME}" >/dev/null || true
  fi
}

status_container() {
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "running (${CONTAINER_NAME})"
  elif docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "stopped (${CONTAINER_NAME})"
  else
    echo "not-created (${CONTAINER_NAME})"
  fi
}

case "${1:-start}" in
  start)
    start_container
    ;;
  stop)
    stop_container
    ;;
  restart)
    stop_container
    start_container
    ;;
  status)
    status_container
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status}" >&2
    exit 1
    ;;
esac
LAUNCHER
    chmod +x "${launcher}"
}

install_system_service() {
    if [[ "${SYSTEM_SERVICE}" != "true" ]]; then
        return
    fi
    if ! command -v systemctl >/dev/null 2>&1; then
        log_warning "systemctl unavailable; skipping --system-service setup"
        return
    fi

    local service_dir="${HOME}/.config/systemd/user"
    local service_file="${service_dir}/bench-race-trtllm.service"
    mkdir -p "${service_dir}"

    cat > "${service_file}" <<SERVICE
[Unit]
Description=bench-race TensorRT-LLM backend
After=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${REPO_ROOT}
Environment=TRTLLM_PORT=${PORT}
Environment=TRTLLM_IMAGE=${IMAGE}
ExecStart=${AGENT_DIR}/backends/trtllm_run.sh start
ExecStop=${AGENT_DIR}/backends/trtllm_run.sh stop

[Install]
WantedBy=default.target
SERVICE

    systemctl --user daemon-reload
    systemctl --user enable bench-race-trtllm.service >/dev/null 2>&1 || true
    log_success "Installed systemd user service: ${service_file}"
}

run_launcher_and_smoke_test() {
    local launcher="${AGENT_DIR}/backends/trtllm_run.sh"
    local health_ok=false

    log_step "Starting TensorRT-LLM backend"
    if [[ "${NO_DETACH}" == "true" ]]; then
        TRTLLM_DETACH=false TRTLLM_IMAGE="${IMAGE}" TRTLLM_PORT="${PORT}" "${launcher}" start
    else
        TRTLLM_DETACH=true TRTLLM_IMAGE="${IMAGE}" TRTLLM_PORT="${PORT}" "${launcher}" start >/dev/null
    fi

    if [[ "${NO_DETACH}" == "true" ]]; then
        log_info "--no-detach was used; skipping smoke test because command runs foreground"
        return
    fi

    log_step "Running smoke test on http://127.0.0.1:${PORT}"
    for _ in {1..30}; do
        if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
            health_ok=true
            break
        fi
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            health_ok=true
            break
        fi
        sleep 2
    done

    if [[ "${health_ok}" != "true" ]]; then
        log_error "TensorRT-LLM smoke test failed on port ${PORT}"
        log_error "Check container logs: docker logs bench-race-trtllm"
        exit 6
    fi

    log_success "TensorRT-LLM backend responded on port ${PORT}"
}

print_next_steps() {
    echo ""
    log_success "TensorRT-LLM installation complete"
    log_info "Next steps:"
    log_info "  1. Start backend: ${AGENT_DIR}/backends/trtllm_run.sh start"
    log_info "  2. Stop backend:  ${AGENT_DIR}/backends/trtllm_run.sh stop"
    log_info "  3. Status:        ${AGENT_DIR}/backends/trtllm_run.sh status"
    log_info "  4. Place/prepare engines under: ${AGENT_DIR}/models/trtllm"
    log_info "  5. Verify endpoint: curl -sS http://127.0.0.1:${PORT}/v1/models"
}

main() {
    parse_args "$@"
    ensure_linux_supported
    source_secrets_if_present
    print_system_info
    check_docker_gpu_access
    docker_login_if_possible
    pull_image
    create_layout
    install_launcher_script
    install_system_service
    run_launcher_and_smoke_test
    print_next_steps
}

main "$@"

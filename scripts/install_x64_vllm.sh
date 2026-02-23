#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VLLM_BUILD_DIR="${VLLM_BUILD_DIR:-/tmp/vllm-build}"
CUDA_ARCH="${CUDA_ARCH:-9.0 12.0}"
FRESH=false

# ============================================================================
# Parse flags
# ============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fresh)
            FRESH=true
            shift
            ;;
        --arch)
            CUDA_ARCH="$2"
            shift 2
            ;;
        --build-dir)
            VLLM_BUILD_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown flag: $1" >&2
            echo "Usage: $0 [--fresh] [--arch \"x.x x.x\"] [--build-dir /path]" >&2
            exit 1
            ;;
    esac
done

# ============================================================================
# Check Docker is available and daemon is running
# ============================================================================
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker Engine is required but not installed." >&2
  if command -v apt-get >/dev/null 2>&1; then
    echo "Install hint (Debian/Ubuntu):" >&2
    echo "  sudo apt-get update && sudo apt-get install -y docker.io && sudo systemctl enable --now docker" >&2
  elif command -v dnf >/dev/null 2>&1; then
    echo "Install hint (RHEL/Fedora):" >&2
    echo "  sudo dnf install -y docker && sudo systemctl enable --now docker" >&2
  elif command -v pacman >/dev/null 2>&1; then
    echo "Install hint (Arch):" >&2
    echo "  sudo pacman -S docker && sudo systemctl enable --now docker" >&2
  else
    echo "Visit: https://docs.docker.com/engine/install/" >&2
  fi
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not reachable. Start Docker and retry." >&2
  exit 1
fi

# ============================================================================
# Clone or reuse vLLM source
# ============================================================================
echo "[install_x64_vllm] CUDA arch list: ${CUDA_ARCH}"
echo "[install_x64_vllm] Build dir: ${VLLM_BUILD_DIR}"

if [[ "$FRESH" == true ]] || [[ ! -d "$VLLM_BUILD_DIR" ]]; then
    echo "[install_x64_vllm] Cloning vLLM source into ${VLLM_BUILD_DIR}..."
    rm -rf "$VLLM_BUILD_DIR"
    git clone https://github.com/vllm-project/vllm.git "$VLLM_BUILD_DIR"
    echo "[install_x64_vllm] Clone complete."
else
    echo "[install_x64_vllm] Reusing existing source at ${VLLM_BUILD_DIR} (pass --fresh to re-clone)."
fi

# ============================================================================
# Locate Dockerfile (handle both old root location and new docker/ subdirectory)
# ============================================================================
if [[ -f "$VLLM_BUILD_DIR/docker/Dockerfile" ]]; then
    DOCKERFILE="$VLLM_BUILD_DIR/docker/Dockerfile"
elif [[ -f "$VLLM_BUILD_DIR/Dockerfile" ]]; then
    DOCKERFILE="$VLLM_BUILD_DIR/Dockerfile"
else
    echo "ERROR: Cannot find vLLM Dockerfile in $VLLM_BUILD_DIR" >&2
    exit 1
fi
echo "[install_x64_vllm] Using Dockerfile: ${DOCKERFILE}"

# ============================================================================
# Build
# ============================================================================
echo "[install_x64_vllm] Building bench-race/vllm:blackwell ..."
echo "[install_x64_vllm] (This compiles CUDA kernels from source — expect 30-60 min)"

docker build \
    --no-cache \
    --build-arg CUDA_VERSION=12.8.1 \
    --build-arg torch_cuda_arch_list="${CUDA_ARCH}" \
    --tag bench-race/vllm:blackwell \
    --tag bench-race/vllm:latest \
    -f "$DOCKERFILE" \
    "$VLLM_BUILD_DIR"

# ============================================================================
# Print usage
# ============================================================================
echo ""
echo "[install_x64_vllm] Build complete."
echo ""
echo "Image tags:"
echo "  bench-race/vllm:blackwell"
echo "  bench-race/vllm:latest"
echo ""
echo "Usage:"
echo "  # Start with a HuggingFace model:"
echo "  ./scripts/agent start-vllm Qwen/Qwen2.5-72B-Instruct"
echo "  ./scripts/agent start-vllm Qwen/Qwen2.5-72B-Instruct-AWQ"
echo ""
echo "  # Start with a local model directory:"
echo "  ./scripts/agent start-vllm agent/models/my-model"
echo ""
echo "  # Or use docker-compose (if docker-compose.vllm.yml is present):"
echo "  docker compose -f docker-compose.vllm.yml up"
echo ""
echo "  # Stop and release GPU:"
echo "  ./scripts/agent stop-vllm"
echo ""
echo "  # Health check:"
echo "  curl http://127.0.0.1:\${VLLM_PORT:-8000}/v1/models"

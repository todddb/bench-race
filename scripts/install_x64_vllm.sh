#!/usr/bin/env bash
set -euo pipefail

# Robust vLLM build script for bench-race
#
# Usage:
#   scripts/install_x64_vllm.sh [TAG] [IMAGE_NAME] [options]
#
# Positional arguments:
#   TAG         vLLM git tag/branch to clone (default: v0.8.5)
#   IMAGE_NAME  image tag to produce (default: bench-race/vllm:blackwell)
#
# Options:
#   --build-dir DIR      Build directory (default: /tmp/vllm-build)
#   --fresh              Force clean local build dir + partial images/cache first
#   --no-cache           Pass --no-cache to docker build
#   --retries N          Number of build attempts before hard fail (default: 2)
#   --python X.Y         Override PYTHON_VERSION build arg (default auto-detected)
#   --arch "A B"         CUDA arch list passed as torch_cuda_arch_list (default: "9.0 12.0")
#   --build-arg K=V      Additional docker build-arg (repeatable)
#   -h, --help           Show help

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_TAG="v0.8.5"
DEFAULT_IMAGE="bench-race/vllm:blackwell"
DEFAULT_BUILD_DIR="/tmp/vllm-build"
DEFAULT_RETRIES=2
DEFAULT_ARCH="9.0 12.0"

TAG="${TAG:-}"
IMAGE_NAME="${IMAGE_NAME:-}"
BUILD_DIR="${VLLM_BUILD_DIR:-$DEFAULT_BUILD_DIR}"
NO_CACHE=false
FRESH=false
RETRIES="$DEFAULT_RETRIES"
OVERRIDE_PYTHON="${PYTHON_VERSION:-}"
CUDA_ARCH="${CUDA_ARCH:-$DEFAULT_ARCH}"

EXTRA_BUILD_ARGS=()
POSITIONALS=()

usage() {
  sed -n '1,35p' "$0"
}

timestamp() { date --utc +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(timestamp)] [install_x64_vllm] $*"; }
log_err() { echo "[$(timestamp)] [install_x64_vllm] ERROR: $*" >&2; }

validate_int() {
  local value="$1"
  [[ "$value" =~ ^[0-9]+$ ]]
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      BUILD_DIR="${2:-}"
      shift 2
      ;;
    --no-cache)
      NO_CACHE=true
      shift
      ;;
    --fresh)
      FRESH=true
      shift
      ;;
    --retries)
      RETRIES="${2:-}"
      shift 2
      ;;
    --python)
      OVERRIDE_PYTHON="${2:-}"
      shift 2
      ;;
    --arch)
      CUDA_ARCH="${2:-}"
      shift 2
      ;;
    --build-arg)
      EXTRA_BUILD_ARGS+=("--build-arg" "${2:-}")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        POSITIONALS+=("$1")
        shift
      done
      ;;
    -*)
      log_err "Unknown flag: $1"
      usage
      exit 2
      ;;
    *)
      POSITIONALS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONALS[@]} -gt 2 ]]; then
  log_err "Too many positional arguments: ${POSITIONALS[*]}"
  usage
  exit 2
fi

if [[ -z "$TAG" ]]; then
  TAG="${POSITIONALS[0]:-$DEFAULT_TAG}"
fi
if [[ -z "$IMAGE_NAME" ]]; then
  IMAGE_NAME="${POSITIONALS[1]:-$DEFAULT_IMAGE}"
fi

if [[ -z "$BUILD_DIR" ]]; then
  log_err "--build-dir requires a non-empty directory"
  exit 2
fi
if ! validate_int "$RETRIES" || [[ "$RETRIES" -lt 1 ]]; then
  log_err "--retries must be an integer >= 1 (got: $RETRIES)"
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  log_err "Docker Engine is required but not installed."
  exit 2
fi
if ! docker info >/dev/null 2>&1; then
  log_err "Docker daemon is not reachable. Start Docker and retry."
  exit 2
fi

cleanup_partial() {
  log "Cleaning partial build state"
  docker image rm -f "$IMAGE_NAME" bench-race/vllm:latest >/dev/null 2>&1 || true
  docker builder prune -a -f >/dev/null 2>&1 || true
  rm -rf "$BUILD_DIR" || true
}

clone_source() {
  log "Cloning vLLM tag=${TAG} into ${BUILD_DIR}"
  rm -rf "$BUILD_DIR"
  git clone --depth 1 --branch "$TAG" https://github.com/vllm-project/vllm.git "$BUILD_DIR"
}

detect_dockerfile() {
  if [[ -f "$BUILD_DIR/docker/Dockerfile" ]]; then
    echo "$BUILD_DIR/docker/Dockerfile"
  elif [[ -f "$BUILD_DIR/Dockerfile" ]]; then
    echo "$BUILD_DIR/Dockerfile"
  else
    return 1
  fi
}

detect_first_from() {
  local dockerfile_path="$1"
  awk 'toupper($1)=="FROM" {print; exit}' "$dockerfile_path"
}

set_python_for_base() {
  if [[ -n "$OVERRIDE_PYTHON" ]]; then
    PYTHON_VERSION="$OVERRIDE_PYTHON"
    log "Using override PYTHON_VERSION=${PYTHON_VERSION}"
    return
  fi

  local dockerfile_path from_line
  dockerfile_path="$(detect_dockerfile)" || {
    PYTHON_VERSION="3.10"
    log "Dockerfile not found for base detection; defaulting PYTHON_VERSION=${PYTHON_VERSION}"
    return
  }

  from_line="$(detect_first_from "$dockerfile_path")"
  if echo "$from_line" | grep -qiE 'ubuntu:20\.04|focal'; then
    PYTHON_VERSION="3.10"
    log "Detected focal/ubuntu:20.04 base -> PYTHON_VERSION=${PYTHON_VERSION}"
  else
    PYTHON_VERSION="3.12"
    log "Detected non-focal base -> PYTHON_VERSION=${PYTHON_VERSION}"
  fi
}

build_image() {
  local dockerfile_path
  dockerfile_path="$(detect_dockerfile)" || {
    log_err "Cannot find Dockerfile under ${BUILD_DIR}"
    return 1
  }

  local no_cache_args=()
  if [[ "$NO_CACHE" == true ]]; then
    no_cache_args=(--no-cache)
  fi

  local build_args=(
    --build-arg "CUDA_VERSION=12.8.1"
    --build-arg "torch_cuda_arch_list=${CUDA_ARCH}"
    --build-arg "PYTHON_VERSION=${PYTHON_VERSION}"
  )

  log "Building image ${IMAGE_NAME} with Dockerfile ${dockerfile_path}"
  docker build \
    "${no_cache_args[@]}" \
    "${build_args[@]}" \
    "${EXTRA_BUILD_ARGS[@]}" \
    --tag "$IMAGE_NAME" \
    --tag "bench-race/vllm:latest" \
    -f "$dockerfile_path" \
    "$BUILD_DIR"
}

archive_logs_on_failure() {
  local attempt="$1"
  local log_dir="$REPO_ROOT/logs"
  local log_file="$log_dir/install_x64_vllm_failed_attempt_${attempt}_$(date +%Y%m%d_%H%M%S).txt"
  mkdir -p "$log_dir"
  {
    echo "timestamp=$(timestamp)"
    echo "tag=$TAG"
    echo "image=$IMAGE_NAME"
    echo "build_dir=$BUILD_DIR"
    echo "attempt=$attempt"
    echo "docker_images="
    docker image ls | head -n 30 || true
  } > "$log_file" || true
  log "Failure details archived to ${log_file}"
}

log "TAG=${TAG} IMAGE=${IMAGE_NAME} BUILD_DIR=${BUILD_DIR} NO_CACHE=${NO_CACHE} FRESH=${FRESH} RETRIES=${RETRIES}"

if [[ "$FRESH" == true ]]; then
  cleanup_partial
fi

clone_source
set_python_for_base

attempt=1
while [[ "$attempt" -le "$RETRIES" ]]; do
  log "Build attempt ${attempt}/${RETRIES}"

  if build_image; then
    log "Build complete: ${IMAGE_NAME}"
    log "Also tagged: bench-race/vllm:latest"
    exit 0
  fi

  log_err "Build failed on attempt ${attempt}/${RETRIES}"
  archive_logs_on_failure "$attempt"

  if [[ "$attempt" -ge "$RETRIES" ]]; then
    log_err "Reached max retries (${RETRIES}); performing final cleanup"
    cleanup_partial
    exit 1
  fi

  log "Cleaning partial state before retry"
  cleanup_partial
  clone_source
  set_python_for_base
  sleep 2
  attempt=$((attempt + 1))
done

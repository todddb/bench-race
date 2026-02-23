#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TRITON_DOCKER_DIR="$REPO_ROOT/agent/backends/triton"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA drivers/tools missing. Install GPU driver + nvidia-container-toolkit." >&2
  echo "Ubuntu hint:" >&2
  echo "  sudo apt-get install -y nvidia-driver-550 nvidia-container-toolkit" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker Engine required. Install Docker first." >&2
  exit 1
fi

echo "[install_gb10_inference] Checking GPU visibility..."
nvidia-smi >/dev/null

echo "[install_gb10_inference] Building bench-race/triton image..."
docker build -t bench-race/triton "$TRITON_DOCKER_DIR"

mkdir -p "$REPO_ROOT/agent/models/triton"
cat <<'TXT'
[install_gb10_inference] Expected Triton model repo layout:
  agent/models/triton/
    <model_name>/
      config.pbtxt
      1/
        model.plan (or backend-specific artifacts)
TXT

echo "[install_gb10_inference] Done. Start with: ./scripts/agent start-nvidia"

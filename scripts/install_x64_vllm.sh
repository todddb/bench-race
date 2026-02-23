#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VLLM_DOCKER_DIR="$REPO_ROOT/agent/backends/vllm"

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

echo "[install_x64_vllm] Building bench-race/vllm image..."
docker build -t bench-race/vllm "$VLLM_DOCKER_DIR"

echo "[install_x64_vllm] Done. Start with: ./scripts/agent start-vllm"

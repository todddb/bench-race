#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[INFO] install_vllm_macos.sh is deprecated; forwarding to install_agent.sh" >&2
exec "${SCRIPT_DIR}/install_agent.sh" "$@"

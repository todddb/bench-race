#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root no matter where script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_DIR="${REPO_ROOT}/run"
LOG_DIR="${REPO_ROOT}/logs"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# PID helpers
pid_is_running() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1
}

read_pid() {
  local pidfile="$1"
  [[ -f "${pidfile}" ]] && cat "${pidfile}" || true
}

write_pid() {
  local pidfile="$1"
  local pid="$2"
  echo "${pid}" > "${pidfile}"
}

remove_pid() {
  local pidfile="$1"
  rm -f "${pidfile}"
}

# Choose python for a component venv (agent/central)
activate_venv() {
  local component="$1"  # "agent" or "central"
  local venv="${REPO_ROOT}/${component}/.venv"
  if [[ ! -d "${venv}" ]]; then
    echo "ERROR: venv not found: ${venv}" >&2
    echo "Run: ./scripts/setup_venv_${component}.sh" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${venv}/bin/activate"
}

# Parse args (very small)
is_daemon=false
for arg in "$@"; do
  if [[ "${arg}" == "--daemon" ]]; then
    is_daemon=true
  fi
done


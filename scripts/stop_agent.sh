#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=scripts/_common.sh
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

PIDFILE="${RUN_DIR}/agent.pid"
OLLAMA_PIDFILE="${RUN_DIR}/ollama.pid"

stop_pidfile() {
  local name="$1"
  local pidfile="$2"
  local pid
  pid="$(read_pid "${pidfile}")"

  if [[ -z "${pid}" ]]; then
    log "${name}: not running (no pidfile)."
    return 0
  fi

  if pid_is_running "${pid}"; then
    log "Stopping ${name} (pid ${pid})..."
    kill "${pid}" || true
    # wait up to 5s
    for i in {1..25}; do
      if ! pid_is_running "${pid}"; then
        break
      fi
      sleep 0.2
    done
    if pid_is_running "${pid}"; then
      log "Force killing ${name} (pid ${pid})..."
      kill -9 "${pid}" || true
    fi
  else
    log "${name}: pidfile present but pid not running (${pid})."
  fi

  remove_pid "${pidfile}"
  log "${name}: stopped."
}

stop_pidfile "agent" "${PIDFILE}"
# Optional: also stop ollama started by script
stop_pidfile "ollama" "${OLLAMA_PIDFILE}"

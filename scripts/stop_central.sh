#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=scripts/_common.sh
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

PIDFILE="${RUN_DIR}/central.pid"

pid="$(read_pid "${PIDFILE}")"
if [[ -z "${pid}" ]]; then
  log "Central: not running (no pidfile)."
  exit 0
fi

if pid_is_running "${pid}"; then
  log "Stopping central (pid ${pid})..."
  kill "${pid}" || true
  for i in {1..25}; do
    if ! pid_is_running "${pid}"; then
      break
    fi
    sleep 0.2
  done
  if pid_is_running "${pid}"; then
    log "Force killing central (pid ${pid})..."
    kill -9 "${pid}" || true
  fi
else
  log "Central: pidfile present but pid not running (${pid})."
fi

remove_pid "${PIDFILE}"
log "Central: stopped."

#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=scripts/_common.sh
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

PIDFILE="${RUN_DIR}/central.pid"

pid="$(read_pid "${PIDFILE}")"
if [[ -n "${pid}" ]] && pid_is_running "${pid}"; then
  log "Central: RUNNING (pid ${pid})"
else
  log "Central: STOPPED"
fi

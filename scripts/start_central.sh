#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=scripts/_common.sh
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

PIDFILE="${RUN_DIR}/central.pid"
LOGFILE="${LOG_DIR}/central.log"

start_central() {
  local pid
  pid="$(read_pid "${PIDFILE}")"
  if [[ -n "${pid}" ]] && pid_is_running "${pid}"; then
    log "Central already running (pid ${pid})."
    exit 0
  fi

  activate_venv "central"

  log "Starting central (Flask) ..."
  if "${is_daemon}"; then
    nohup python "${REPO_ROOT}/central/app.py" >> "${LOGFILE}" 2>&1 &
    write_pid "${PIDFILE}" "$!"
    log "Central started (pid $(read_pid "${PIDFILE}")). Logs: ${LOGFILE}"
  else
    log "Logs: ${LOGFILE}"
    python "${REPO_ROOT}/central/app.py" 2>&1 | tee -a "${LOGFILE}"
  fi
}

start_central

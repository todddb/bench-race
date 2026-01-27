#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=scripts/_common.sh
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

PIDFILE="${RUN_DIR}/agent.pid"
OLLAMA_PIDFILE="${RUN_DIR}/ollama.pid"

pid="$(read_pid "${PIDFILE}")"
if [[ -n "${pid}" ]] && pid_is_running "${pid}"; then
  log "Agent: RUNNING (pid ${pid})"
else
  log "Agent: STOPPED"
fi

opid="$(read_pid "${OLLAMA_PIDFILE}")"
if [[ -n "${opid}" ]] && pid_is_running "${opid}"; then
  log "Ollama: RUNNING (pid ${opid})"
else
  log "Ollama: unknown/stopped (script pidfile missing or not running)"
fi

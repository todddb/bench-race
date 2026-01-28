#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=scripts/_common.sh
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

PIDFILE="${RUN_DIR}/agent.pid"
LOGFILE="${LOG_DIR}/agent.log"
OLLAMA_PIDFILE="${RUN_DIR}/ollama.pid"
OLLAMA_LOGFILE="${LOG_DIR}/ollama.log"

AGENT_HOST="${AGENT_HOST:-0.0.0.0}"
AGENT_PORT="${AGENT_PORT:-9001}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
OLLAMA_URL="http://${OLLAMA_HOST}:${OLLAMA_PORT}"

ensure_ollama() {
  # If ollama isn't installed, just warn (agent will fallback to mock)
  if ! command -v ollama >/dev/null 2>&1; then
    log "WARN: 'ollama' not found in PATH. Agent will fallback to mock backend."
    return 0
  fi

  # Step 1: Check if port is already listening (most reliable check)
  # Use lsof to detect any process listening on the Ollama port
  if lsof -nP -iTCP:"${OLLAMA_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    log "Ollama already listening on port ${OLLAMA_PORT}"
    # Wait for API to be ready (it might be starting up)
    for i in {1..30}; do
      if curl -fsS "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
        log "Ollama is reachable at ${OLLAMA_URL}"
        return 0
      fi
      sleep 0.2
    done
    log "WARN: Port ${OLLAMA_PORT} is listening but API not responding. Continuing anyway."
    return 0
  fi

  # Step 2: Check if an 'ollama serve' process exists (may be starting up)
  local existing_pid
  existing_pid="$(pgrep -f 'ollama serve' 2>/dev/null | head -1 || true)"
  if [[ -n "${existing_pid}" ]]; then
    log "Found existing ollama serve process (pid ${existing_pid}), waiting for API..."
    # Don't start another instance, just wait for this one to come up
    for i in {1..30}; do
      if curl -fsS "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
        log "Ollama is now reachable at ${OLLAMA_URL}"
        return 0
      fi
      sleep 0.2
    done
    log "WARN: Existing ollama process found but API not responding at ${OLLAMA_URL}."
    return 0
  fi

  # Step 3: No listener and no process - safe to start Ollama
  log "Starting Ollama (ollama serve) ..."
  OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE}" nohup ollama serve >> "${OLLAMA_LOGFILE}" 2>&1 &
  write_pid "${OLLAMA_PIDFILE}" "$!"
  log "Ollama started (pid $!)"

  # Wait for API to come up
  for i in {1..30}; do
    if curl -fsS "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
      log "Ollama is now reachable at ${OLLAMA_URL}"
      return 0
    fi
    sleep 0.2
  done

  log "WARN: Ollama still not reachable at ${OLLAMA_URL}. Agent may fallback to mock."
  return 0
}

start_agent() {
  local pid
  pid="$(read_pid "${PIDFILE}")"
  if [[ -n "${pid}" ]] && pid_is_running "${pid}"; then
    log "Agent already running (pid ${pid})."
    exit 0
  fi

  activate_venv "agent"

  log "Starting agent on ${AGENT_HOST}:${AGENT_PORT} ..."
  if "${is_daemon}"; then
    nohup uvicorn agent.agent_app:app --host "${AGENT_HOST}" --port "${AGENT_PORT}" >> "${LOGFILE}" 2>&1 &
    write_pid "${PIDFILE}" "$!"
    log "Agent started (pid $(read_pid "${PIDFILE}")). Logs: ${LOGFILE}"
  else
    log "Logs: ${LOGFILE}"
    uvicorn agent.agent_app:app --host "${AGENT_HOST}" --port "${AGENT_PORT}" 2>&1 | tee -a "${LOGFILE}"
  fi
}

ensure_ollama
start_agent

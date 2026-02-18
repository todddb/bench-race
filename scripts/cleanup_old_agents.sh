#!/usr/bin/env bash
set -euo pipefail

# Usage:
#  ./scripts/cleanup_old_agents.sh        # dry-run, prints what would be removed
#  ./scripts/cleanup_old_agents.sh --yes # performs the removals

YES=false
for arg in "$@"; do
  case "$arg" in
    -y|--yes) YES=true ;;
    *) ;;
  esac
done

# Resolve repo root (assumes script sits in ./scripts)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
USER_HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
echo "Repo root: ${REPO_ROOT}"
echo "User home: ${USER_HOME}"

# Candidate paths (adjust / add if your install used different paths)
declare -a STOP_CMD_CANDIDATES=(
  "sudo systemctl stop bench-race-vllm || true"
  "sudo systemctl disable bench-race-vllm || true"
  "systemctl --user stop bench-race-vllm || true"
  "systemctl --user disable bench-race-vllm || true"
)

declare -a SYSTEMD_SYSTEM_UNIT=(
  "/etc/systemd/system/bench-race-vllm.service"
)
declare -a SYSTEMD_USER_UNIT=(
  "${USER_HOME}/.config/systemd/user/bench-race-vllm.service"
)

declare -a LAUNCHD_PLISTS=(
  "/Library/LaunchDaemons/com.bench-race.vllm.plist"
  "${USER_HOME}/Library/LaunchAgents/com.bench-race.vllm.plist"
)

# venvs and model dirs
declare -a VENVS=(
  "${REPO_ROOT}/agent/.venv"
  "${USER_HOME}/bench-race/vllm-venv"
  "${USER_HOME}/bench-race"        # top-level repo-local bench-race (models under it)
  "/opt/bench-race"               # system install path (if used)
)

# comfyui and ollama
declare -a COMFYUI_DIRS=(
  "${REPO_ROOT}/agent/third_party/comfyui"
  "${USER_HOME}/comfyui"
  "${USER_HOME}/.local/share/comfyui"
)
declare -a OLLAMA_CANDIDATES=(
  "/usr/local/bin/ollama"
  "/usr/bin/ollama"
  "${USER_HOME}/bin/ollama"
)

# pid/log files
declare -a PIDFILES=(
  "${REPO_ROOT}/run/agent.pid"
  "${REPO_ROOT}/run/ollama.pid"
  "${REPO_ROOT}/run/bench-race-vllm.pid"
  "${USER_HOME}/bench-race/run/agent.pid"
  "${USER_HOME}/bench-race/run/bench-race-vllm.pid"
)
declare -a LOGFILES=(
  "${REPO_ROOT}/run/agent.log"
  "${REPO_ROOT}/run/ollama.log"
  "${REPO_ROOT}/run/bench-race-vllm.log"
  "${USER_HOME}/bench-race/logs/agent.log"
)

# helper functions
dryrun_rm(){
  local p="$1"
  if [[ -e "$p" ]]; then
    if [[ "${YES}" == "true" ]]; then
      echo "rm -rf ${p}"
      rm -rf -- "${p}"
      echo "  removed: ${p}"
    else
      echo "[DRY-RUN] would remove: ${p}"
    fi
  else
    echo "not found: ${p}"
  fi
}

dryrun_cmd(){
  local cmd="$1"
  if [[ "${YES}" == "true" ]]; then
    echo "RUN: ${cmd}"
    bash -c "${cmd}" || true
  else
    echo "[DRY-RUN] would run: ${cmd}"
  fi
}

echo
echo ">>> STOP any running services/processes (system & user)"
for cmd in "${STOP_CMD_CANDIDATES[@]}"; do
  dryrun_cmd "${cmd}"
done

echo
echo ">>> User/systemd unit files to remove"
for f in "${SYSTEMD_USER_UNIT[@]}"; do
  dryrun_rm "${f}"
done
for f in "${SYSTEMD_SYSTEM_UNIT[@]}"; do
  dryrun_rm "${f}"
done

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo
  echo ">>> macOS LaunchAgents/Daemons to unload/remove"
  for f in "${LAUNCHD_PLISTS[@]}"; do
    if [[ -f "$f" ]]; then
      if [[ "${YES}" == "true" ]]; then
        echo "Unloading plist: ${f}"
        # try user unload then system unload (best-effort)
        if [[ "$f" == "${USER_HOME}"* ]]; then
          launchctl unload "${f}" 2>/dev/null || true
        else
          sudo launchctl bootout system "${f}" 2>/dev/null || true
        fi
      else
        echo "[DRY-RUN] would unload plist: ${f}"
      fi
    else
      echo "not found: ${f}"
    fi
    dryrun_rm "${f}"
  done
fi

echo
echo ">>> Virtualenvs / model dirs / repo-local installs to remove"
for v in "${VENVS[@]}"; do
  dryrun_rm "${v}"
done

echo
echo ">>> ComfyUI candidate dirs to remove"
for d in "${COMFYUI_DIRS[@]}"; do
  dryrun_rm "${d}"
done

echo
echo ">>> PID / LOG files to remove"
for p in "${PIDFILES[@]}" "${LOGFILES[@]}"; do
  dryrun_rm "${p}"
done

echo
echo ">>> Ollama binary candidates (will NOT remove global package managers like apt/homebrew)."
echo "If one of these files exists and you want it removed, run the script with --yes."
for o in "${OLLAMA_CANDIDATES[@]}"; do
  if [[ -f "${o}" ]]; then
    if [[ "${YES}" == "true" ]]; then
      echo "Removing ollama at ${o}"
      sudo rm -f -- "${o}" || rm -f -- "${o}" || true
    else
      echo "[DRY-RUN] would remove ollama candidate: ${o}"
    fi
  else
    echo "ollama candidate not found: ${o}"
  fi
done

echo
echo ">>> Extra audit: list possible bench-race system locations"
echo "Checking /etc, /opt and user locations for bench-race files:"
if [[ "${YES}" == "true" ]]; then
  ls -ld /etc/systemd/system/bench-race* 2>/dev/null || true
  ls -ld /opt/bench-race* 2>/dev/null || true
  ls -ld "${USER_HOME}/bench-race"* 2>/dev/null || true
else
  echo "[DRY-RUN] would check: /etc/systemd/system/bench-race* , /opt/bench-race* , ${USER_HOME}/bench-race*"
fi

echo
if [[ "${YES}" == "true" ]]; then
  echo "Cleanup complete. If you removed system units, run: sudo systemctl daemon-reload"
  echo "If you removed user units, run: systemctl --user daemon-reload (or logout/login)"
else
  echo "DRY-RUN finished. Re-run with --yes to perform removals."
fi

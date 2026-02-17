#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
log_step(){ echo -e "${CYAN}[STEP]${NC} $*"; }
log_info(){ echo -e "${BLUE}[INFO]${NC} $*"; }
log_warn(){ echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error(){ echo -e "${RED}[ERROR]${NC} $*"; }
log_ok(){ echo -e "${GREEN}[OK]${NC} $*"; }

DRY_RUN=false
YES_MODE=false
INSTALL_VLLM=true
FORCE_RECREATE_VENV=false
SKIP_SERVICE=false
UNINSTALL_VLLM=false
SYSTEM_INSTALL=false
PLATFORM_OVERRIDE=""
AGENT_ID=""
LABEL=""
CENTRAL_URL="http://127.0.0.1:8080"
VLLM_PORT="8000"
VLLM_INSTALL_VERSION="${VLLM_INSTALL_VERSION:-}"

INVOKER_USER="${SUDO_USER:-${USER:-$(id -un)}}"
INVOKER_UID="$(id -u "${INVOKER_USER}" 2>/dev/null || id -u)"
INVOKER_HOME="$(getent passwd "${INVOKER_USER}" 2>/dev/null | cut -d: -f6 || true)"
if [[ -z "${INVOKER_HOME}" ]]; then
  INVOKER_HOME="$(eval echo "~${INVOKER_USER}")"
fi

MODEL_DIR=""
VENV_PATH=""

run(){
  if [[ "${DRY_RUN}" == true ]]; then
    echo "[DRY-RUN] $*"
    return 0
  fi
  "$@"
}

run_as_invoker(){
  if [[ "${DRY_RUN}" == true ]]; then
    echo "[DRY-RUN] (as ${INVOKER_USER}) $*"
    return 0
  fi
  if [[ "$(id -u)" -eq 0 && "${INVOKER_USER}" != "root" ]]; then
    sudo -u "${INVOKER_USER}" "$@"
  else
    "$@"
  fi
}

usage(){
cat <<USAGE
Usage: ./scripts/install_agent.sh [OPTIONS]

Core options:
  -y, --yes                   Non-interactive mode
  --dry-run                   Print actions only
  --no-vllm                   Skip vLLM install/update
  --force-recreate-venv       Recreate vLLM venv even if existing
  --skip-service              Do not install/enable vLLM service
  --uninstall-vllm            Remove managed vLLM venv + managed service
  --venv-path PATH            vLLM venv path
  --model-dir PATH            vLLM model dir
  --system                    System-level service defaults (/opt venv)
  --platform PLATFORM         Override detected platform

Compatibility options:
  --agent-id ID               agent/config machine id
  --label LABEL               agent label
  --central-url URL           central URL in agent config
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) YES_MODE=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --no-vllm) INSTALL_VLLM=false; shift ;;
    --force-recreate-venv) FORCE_RECREATE_VENV=true; shift ;;
    --skip-service) SKIP_SERVICE=true; shift ;;
    --uninstall-vllm) UNINSTALL_VLLM=true; shift ;;
    --venv-path) VENV_PATH="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --system) SYSTEM_INSTALL=true; shift ;;
    --platform) PLATFORM_OVERRIDE="$2"; shift 2 ;;
    --agent-id) AGENT_ID="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --central-url) CENTRAL_URL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) log_error "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${VENV_PATH}" ]]; then
  if [[ "${SYSTEM_INSTALL}" == true ]]; then
    VENV_PATH="/opt/bench-race/vllm-venv"
  else
    VENV_PATH="${INVOKER_HOME}/bench-race/vllm-venv"
  fi
fi
if [[ -z "${MODEL_DIR}" ]]; then
  if [[ "${SYSTEM_INSTALL}" == true ]]; then
    MODEL_DIR="/mnt/models/vllm"
  else
    MODEL_DIR="${INVOKER_HOME}/bench-race/models/vllm"
  fi
fi

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
PLATFORM=""
if [[ -n "${PLATFORM_OVERRIDE}" ]]; then
  PLATFORM="${PLATFORM_OVERRIDE}"
else
  case "${OS}" in
    linux) PLATFORM="linux" ;;
    darwin) PLATFORM="macos" ;;
    *) PLATFORM="unknown" ;;
  esac
fi

GPU_NAME=""
CUDA_VERSION=""
IS_BLACKWELL=false
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
  CUDA_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
  if echo "${GPU_NAME}" | tr '[:upper:]' '[:lower:]' | grep -Eq 'gb10|blackwell|b100|b200'; then
    IS_BLACKWELL=true
  fi
fi

create_or_update_agent_venv(){
  log_step "Ensuring agent venv is installed"
  local py
  py="$("${SCRIPT_DIR}/_python_pick.sh")"
  if [[ ! -d "${REPO_ROOT}/agent/.venv" ]]; then
    run "$py" -m venv "${REPO_ROOT}/agent/.venv"
  fi
  run "${REPO_ROOT}/agent/.venv/bin/python" -m pip install -U pip setuptools wheel
  run "${REPO_ROOT}/agent/.venv/bin/python" -m pip install -r "${REPO_ROOT}/agent/requirements.txt"
}

write_agent_config_if_missing(){
  local cfg="${REPO_ROOT}/agent/config/agent.yaml"
  mkdir -p "$(dirname "${cfg}")"
  if [[ ! -f "${cfg}" ]]; then
    local machine_id
    machine_id="${AGENT_ID:-$(hostname | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]-' '-') }"
    if [[ "${DRY_RUN}" == true ]]; then
      echo "[DRY-RUN] Would create ${cfg}"
    else
      cat > "${cfg}" <<CFG
machine_id: "${machine_id}"
label: "${LABEL:-$(hostname)}"
bind_host: "0.0.0.0"
bind_port: 9001
central_url: "${CENTRAL_URL}"
ollama_base_url: "http://127.0.0.1:11434"
CFG
    fi
    log_ok "Created ${cfg}"
  fi
}

ensure_vllm_config_block(){
  local cfg="${REPO_ROOT}/agent/config/agent.yaml"
  mkdir -p "$(dirname "${cfg}")"
  touch "${cfg}"
  if grep -q '^vllm:' "${cfg}"; then
    log_info "agent.yaml already has vllm block; leaving unchanged"
    return 0
  fi
  if [[ "${DRY_RUN}" == true ]]; then
    echo "[DRY-RUN] Would append vllm block to ${cfg}"
    return 0
  fi
  cat >> "${cfg}" <<CFG

vllm:
  enabled: true
  base_url: "http://127.0.0.1:${VLLM_PORT}"
  venv_path: "${VENV_PATH}"
  model_dir: "${MODEL_DIR}"
CFG
  log_ok "Added vllm block to agent/config/agent.yaml"
}

path_owner_uid(){
  local p="$1"
  if stat -c '%u' "$p" >/dev/null 2>&1; then stat -c '%u' "$p"; else stat -f '%u' "$p"; fi
}

ensure_vllm_venv(){
  local recreate=false
  if [[ -d "${VENV_PATH}" ]]; then
    local owner
    owner="$(path_owner_uid "${VENV_PATH}")"
    if [[ "${owner}" != "${INVOKER_UID}" ]]; then
      log_warn "Existing venv owner uid=${owner}, invoker uid=${INVOKER_UID}"
      if [[ "${FORCE_RECREATE_VENV}" == true ]]; then
        recreate=true
      elif [[ "${SYSTEM_INSTALL}" == false ]]; then
        local fallback="${INVOKER_HOME}/bench-race/vllm-venv"
        log_warn "Switching to user-owned venv path: ${fallback}"
        VENV_PATH="${fallback}"
      elif [[ "${YES_MODE}" == true ]]; then
        recreate=true
      else
        read -r -p "Recreate venv at ${VENV_PATH} as ${INVOKER_USER}? [y/N] " ans
        [[ "${ans}" =~ ^[Yy]$ ]] && recreate=true
      fi
    elif [[ "${FORCE_RECREATE_VENV}" == true ]]; then
      recreate=true
    fi
  fi

  run mkdir -p "$(dirname "${VENV_PATH}")" "${MODEL_DIR}"

  if [[ "${recreate}" == true && -d "${VENV_PATH}" ]]; then
    log_step "Recreating venv at ${VENV_PATH} (models preserved under ${MODEL_DIR})"
    if [[ "$(id -u)" -eq 0 ]]; then
      run rm -rf "${VENV_PATH}"
    else
      run sudo rm -rf "${VENV_PATH}"
    fi
  fi

  if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
    log_step "Creating vLLM venv as ${INVOKER_USER}: ${VENV_PATH}"
    if [[ "$(id -u)" -eq 0 && "${SYSTEM_INSTALL}" == true ]]; then
      run mkdir -p "${VENV_PATH}"
      run chown -R "${INVOKER_USER}":"$(id -gn "${INVOKER_USER}")" "${VENV_PATH}" "$(dirname "${MODEL_DIR}")" || true
    fi
    local py
    py="$("${SCRIPT_DIR}/_python_pick.sh")"
    run_as_invoker "$py" -m venv "${VENV_PATH}"
  fi
}

# --- Begin vllm setuptools compatibility helpers ---
# $VENV_PATH must be set to the venv path in the installer context
PIP_BIN="${VENV_PATH}/bin/pip"
PY_BIN="${VENV_PATH}/bin/python"

# Try to install vllm, detect setuptools-related pip errors, and retry with pinned setuptools if needed.
# Usage: try_install_vllm [<vllm_spec>]
# <vllm_spec> is optional, e.g. "vllm==0.14.1" or "vllm" (latest)
try_install_vllm() {
  local vllm_spec="${1:-vllm}"
  local tmpfile

  PIP_BIN="${VENV_PATH}/bin/pip"
  PY_BIN="${VENV_PATH}/bin/python"
  tmpfile="$(mktemp -t vllm_pip_log.XXXXXX)" || tmpfile="/tmp/vllm_pip_log.$$"

  if [[ "${DRY_RUN}" == true ]]; then
    log_info "[DRY-RUN] Would attempt vLLM install via try_install_vllm (${vllm_spec})"
    rm -f "$tmpfile"
    return 0
  fi

  echo "[INFO] Attempting to install ${vllm_spec} into ${VENV_PATH}"
  # First try: normal install (isolation on)
  if run_as_invoker "$PIP_BIN" install --upgrade pip setuptools >/dev/null 2>&1; then
    :
  fi
  if run_as_invoker "$PIP_BIN" install "${vllm_spec}" >"$tmpfile" 2>&1; then
    echo "[INFO] Installed ${vllm_spec} successfully."
    rm -f "$tmpfile"
    return 0
  fi

  # Install failed: inspect log for setuptools requirement/restriction
  local logcat
  logcat="$(cat "$tmpfile")"
  echo "[WARN] vllm install failed; inspecting pip output..."

  # Common pip wording examples we want to detect:
  #   vllm 0.14.1 requires setuptools==77.0.3
  #   vllm requires setuptools<81.0.0,>=77.0.3
  #   Could not find a version that satisfies the requirement torch==2.6.0 (skip)
  # We'll attempt to extract a setuptools exact version or lower-than constraint.
  local setuptools_pin=""
  # exact equals e.g. "requires setuptools==77.0.3"
  setuptools_pin="$(printf "%s\n" "$logcat" | grep -oE "setuptools==[0-9]+\.[0-9]+\.[0-9]+" | head -n1 || true)"
  if [ -z "$setuptools_pin" ]; then
    # range e.g. "requires setuptools<81.0.0,>=77.0.3" -> pick lower-bound if present
    setuptools_pin="$(printf "%s\n" "$logcat" | grep -oE "setuptools[^[:space:]]*" | sed -n '1p' || true)"
    # fallback: look for 'setuptools<81' style
    if [ -n "$setuptools_pin" ]; then
      # Try to extract a lower bound, e.g. ">=77.0.3"
      local lb
      lb="$(printf "%s\n" "$setuptools_pin" | grep -oE ">=([0-9]+(\.[0-9]+){1,2})" | sed -e 's/>=//' | head -n1 || true)"
      if [ -n "$lb" ]; then
        setuptools_pin="setuptools==${lb}"
      else
        setuptools_pin=""
      fi
    fi
  fi

  if [ -n "$setuptools_pin" ]; then
    echo "[INFO] Pip output indicates vllm needs specific setuptools: ${setuptools_pin}. Will pin and retry."
    # Attempt pin, then reinstall vllm with no-build-isolation fallback
    if run_as_invoker "$PIP_BIN" install --force-reinstall "${setuptools_pin}"; then
      echo "[INFO] Pinned ${setuptools_pin} in venv"
      if run_as_invoker "$PIP_BIN" install --force-reinstall --no-build-isolation "${vllm_spec}" >"$tmpfile" 2>&1; then
        echo "[INFO] Installed ${vllm_spec} successfully after pinning setuptools."
        rm -f "$tmpfile"
        return 0
      else
        echo "[ERROR] Retry install after pinning setuptools still failed. See ${tmpfile} for details."
        cat "$tmpfile" >&2
        return 2
      fi
    else
      echo "[ERROR] Failed to pin setuptools to ${setuptools_pin}. See ${tmpfile} for details."
      cat "$tmpfile" >&2
      return 3
    fi
  fi

  # Generic fallback: try again with --no-build-isolation (may work if dependency resolution was build-time only)
  echo "[INFO] No setuptools pin detected. Retrying with --no-build-isolation..."
  if run_as_invoker "$PIP_BIN" install --force-reinstall --no-build-isolation "${vllm_spec}" >"$tmpfile" 2>&1; then
    echo "[INFO] Installed ${vllm_spec} successfully with --no-build-isolation."
    rm -f "$tmpfile"
    return 0
  fi

  # Give up and print logs for debugging
  echo "[ERROR] vllm install ultimately failed. See $tmpfile for pip output."
  cat "$tmpfile" >&2
  return 4
}
# --- End vllm setuptools compatibility helpers ---

install_vllm_packages(){
  local pip="${VENV_PATH}/bin/pip"
  local py="${VENV_PATH}/bin/python"
  run_as_invoker "$pip" install -U pip setuptools wheel

  if "$py" -c "import torch" >/dev/null 2>&1; then
    log_info "torch already present in venv; not forcing downgrade/replace"
  else
    if [[ "${PLATFORM}" == macos* ]]; then
      log_step "Installing torch for macOS (${ARCH})"
      run_as_invoker "$pip" install torch
      log_warn "macOS vLLM support may be limited; CPU/MPS path selected"
    elif [[ -n "${GPU_NAME}" ]]; then
      if [[ "${IS_BLACKWELL}" == true ]]; then
        log_warn "Detected GB10/Blackwell GPU (${GPU_NAME})"
        if [[ "${YES_MODE}" == true ]]; then
          run_as_invoker "$pip" install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
        else
          read -r -p "Install nightly torch (cu128) for Blackwell? [Y/n] " ans
          if [[ -z "${ans}" || "${ans}" =~ ^[Yy]$ ]]; then
            run_as_invoker "$pip" install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
          else
            run_as_invoker "$pip" install torch
          fi
        fi
      else
        log_step "Installing CUDA torch (stable cu121)"
        run_as_invoker "$pip" install torch --index-url https://download.pytorch.org/whl/cu121
      fi
    else
      log_step "Installing CPU torch"
      run_as_invoker "$pip" install torch --index-url https://download.pytorch.org/whl/cpu
    fi
  fi

  if ! "$py" -c "import vllm" >/dev/null 2>&1; then
    log_step "Installing vLLM"
    local vllm_install_spec
    if [[ -n "${VLLM_INSTALL_VERSION}" ]]; then
      vllm_install_spec="vllm==${VLLM_INSTALL_VERSION}"
    else
      vllm_install_spec="vllm"
    fi

    if ! try_install_vllm "${vllm_install_spec}"; then
      echo "[ERROR] vLLM install failed. Aborting installer."
      exit 1
    fi
    run_as_invoker "$pip" install uvicorn
  else
    log_info "vLLM already installed in ${VENV_PATH}"
  fi

  # Post-install verification
  echo "[STEP] Verifying installed vLLM and dependencies..."
  if [[ "${DRY_RUN}" == true ]]; then
    log_info "[DRY-RUN] Skipping pip check/import verification"
    return 0
  fi
  run_as_invoker "$VENV_PATH/bin/pip" check || {
    echo "[ERROR] pip check failed - dependency conflicts detected. Printing 'pip check' output above."
    run_as_invoker "$VENV_PATH/bin/python" -c "import pkgutil; print('installed pkg list sample:', [m.name for m in pkgutil.iter_modules()][:20])" || true
    exit 1
  }

  # Quick import test
  if ! run_as_invoker "$VENV_PATH/bin/python" -c "import vllm, torch; print('vllm', getattr(vllm,'__version__','n/a'), 'torch', getattr(torch,'__version__','n/a'))"; then
    echo "[ERROR] Python import check for vllm/torch failed in venv at ${VENV_PATH}"
    exit 1
  fi
}

install_vllm_service(){
  [[ "${SKIP_SERVICE}" == true ]] && { log_info "Skipping service setup (--skip-service)"; return 0; }
  local cmd="${VENV_PATH}/bin/python -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port ${VLLM_PORT} --download-dir ${MODEL_DIR}"

  if [[ "${PLATFORM}" == linux* ]]; then
    if command -v systemctl >/dev/null 2>&1; then
      if [[ "${SYSTEM_INSTALL}" == true ]]; then
        local unit="/etc/systemd/system/bench-race-vllm.service"
        run bash -lc "cat > '${unit}' <<UNIT
[Unit]
Description=bench-race vLLM service
After=network.target

[Service]
Type=simple
User=${INVOKER_USER}
Group=$(id -gn "${INVOKER_USER}")
Environment=HF_HOME=${MODEL_DIR}
ExecStart=${cmd}
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT"
        run systemctl daemon-reload
        run systemctl enable --now bench-race-vllm
      else
        local user_unit="${INVOKER_HOME}/.config/systemd/user/bench-race-vllm.service"
        run_as_invoker mkdir -p "${INVOKER_HOME}/.config/systemd/user"
        run_as_invoker bash -lc "cat > '${user_unit}' <<UNIT
[Unit]
Description=bench-race vLLM (user)
After=default.target

[Service]
Type=simple
Environment=HF_HOME=${MODEL_DIR}
ExecStart=${cmd}
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
UNIT"
        if run_as_invoker systemctl --user daemon-reload 2>/dev/null; then
          run_as_invoker systemctl --user enable --now bench-race-vllm
        else
          log_warn "systemctl --user unavailable (WSL/minimal env). Start vLLM manually via scripts/agents start"
        fi
      fi
    else
      log_warn "systemctl not found; skipping service install"
    fi
  elif [[ "${PLATFORM}" == macos* ]]; then
    if [[ "${SYSTEM_INSTALL}" == true ]]; then
      local plist="/Library/LaunchDaemons/com.bench-race.vllm.plist"
      run bash -lc "cat > '${plist}' <<PLIST
<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\"><dict>
<key>Label</key><string>com.bench-race.vllm</string>
<key>ProgramArguments</key><array><string>${VENV_PATH}/bin/python</string><string>-m</string><string>vllm.entrypoints.openai.api_server</string><string>--host</string><string>127.0.0.1</string><string>--port</string><string>${VLLM_PORT}</string><string>--download-dir</string><string>${MODEL_DIR}</string></array>
<key>EnvironmentVariables</key><dict><key>HF_HOME</key><string>${MODEL_DIR}</string></dict>
<key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
</dict></plist>
PLIST"
      run launchctl unload "${plist}" >/dev/null 2>&1 || true
      run launchctl load -w "${plist}"
    else
      local plist="${INVOKER_HOME}/Library/LaunchAgents/com.bench-race.vllm.plist"
      run_as_invoker mkdir -p "${INVOKER_HOME}/Library/LaunchAgents"
      run_as_invoker bash -lc "cat > '${plist}' <<PLIST
<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\"><dict>
<key>Label</key><string>com.bench-race.vllm</string>
<key>ProgramArguments</key><array><string>${VENV_PATH}/bin/python</string><string>-m</string><string>vllm.entrypoints.openai.api_server</string><string>--host</string><string>127.0.0.1</string><string>--port</string><string>${VLLM_PORT}</string><string>--download-dir</string><string>${MODEL_DIR}</string></array>
<key>EnvironmentVariables</key><dict><key>HF_HOME</key><string>${MODEL_DIR}</string></dict>
<key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
</dict></plist>
PLIST"
      run_as_invoker launchctl unload "${plist}" >/dev/null 2>&1 || true
      run_as_invoker launchctl load -w "${plist}"
    fi
  fi
}

uninstall_vllm(){
  log_step "Uninstalling managed vLLM artifacts"
  if command -v systemctl >/dev/null 2>&1; then
    run systemctl stop bench-race-vllm >/dev/null 2>&1 || true
    run systemctl disable bench-race-vllm >/dev/null 2>&1 || true
    run_as_invoker systemctl --user stop bench-race-vllm >/dev/null 2>&1 || true
    run_as_invoker systemctl --user disable bench-race-vllm >/dev/null 2>&1 || true
    run rm -f /etc/systemd/system/bench-race-vllm.service
    run_as_invoker rm -f "${INVOKER_HOME}/.config/systemd/user/bench-race-vllm.service"
    run systemctl daemon-reload >/dev/null 2>&1 || true
    run_as_invoker systemctl --user daemon-reload >/dev/null 2>&1 || true
  fi
  run rm -f /Library/LaunchDaemons/com.bench-race.vllm.plist >/dev/null 2>&1 || true
  run_as_invoker rm -f "${INVOKER_HOME}/Library/LaunchAgents/com.bench-race.vllm.plist" >/dev/null 2>&1 || true
  if [[ -d "${VENV_PATH}" ]]; then
    run rm -rf "${VENV_PATH}"
  fi
  log_ok "vLLM uninstall complete (models preserved in ${MODEL_DIR})"
}

smoke_checks(){
  [[ "${INSTALL_VLLM}" == false ]] && return 0
  local py="${VENV_PATH}/bin/python"
  log_step "Running smoke checks"
  run "$py" -c "import torch, vllm; print(torch.__version__, vllm.__version__)"
  run "$py" -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
}

main(){
  log_info "Platform=${PLATFORM} arch=${ARCH} gpu='${GPU_NAME:-none}'"
  create_or_update_agent_venv
  write_agent_config_if_missing

  if [[ "${UNINSTALL_VLLM}" == true ]]; then
    uninstall_vllm
    exit 0
  fi

  if [[ "${INSTALL_VLLM}" == true ]]; then
    ensure_vllm_venv
    install_vllm_packages
    ensure_vllm_config_block
    install_vllm_service
  else
    log_info "Skipping vLLM install (--no-vllm)"
  fi

  smoke_checks
  log_ok "Install complete"
}

main "$@"

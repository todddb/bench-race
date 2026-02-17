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
  --vllm-version VERSION      Pin vLLM version (e.g. 0.14.1)
  --system                    System-level service defaults (/opt venv)
  --platform PLATFORM         Override detected platform

Compatibility options:
  --agent-id ID               agent/config machine id
  --label LABEL               agent label
  --central-url URL           central URL in agent config

Environment variables:
  VLLM_INSTALL_VERSION        Same as --vllm-version (CLI flag takes priority)
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
    --vllm-version) VLLM_INSTALL_VERSION="$2"; shift 2 ;;
    --system) SYSTEM_INSTALL=true; shift ;;
    --platform) PLATFORM_OVERRIDE="$2"; shift 2 ;;
    --agent-id) AGENT_ID="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --central-url) CENTRAL_URL="$2"; shift 2 ;;
    --skip-ollama|--skip-comfyui) shift ;;  # accepted for compat; no-op in this script
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
      log_warn "Existing venv at ${VENV_PATH} owned by uid=${owner}, invoker uid=${INVOKER_UID}"
      if [[ "${FORCE_RECREATE_VENV}" == true || "${YES_MODE}" == true ]]; then
        recreate=true
      else
        read -r -p "venv owned by another user — recreate as ${INVOKER_USER}? [y/N] " ans
        if [[ "${ans}" =~ ^[Yy]$ ]]; then
          recreate=true
        elif [[ "${SYSTEM_INSTALL}" == false ]]; then
          local fallback="${INVOKER_HOME}/bench-race/vllm-venv"
          log_warn "Switching to user-owned venv path: ${fallback}"
          VENV_PATH="${fallback}"
        fi
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

# --- Begin vllm compatibility and deterministic install helpers ---
# Requires: VENV_PATH set, INVOKER_USER/INVOKER_UID set, MODEL_DIR set.
# Optional: VLLM_INSTALL_VERSION set to '0.14.1' etc.

# Lookup compatibility mapping for a given vLLM version.
# Returns JSON object for version if available (jq required).
# Falls back gracefully if jq is missing or file not found.
_vllm_compat_lookup() {
  local version="${1:-}"
  local compat_file="${REPO_ROOT}/scripts/vllm_compat.json"
  if [[ -f "$compat_file" && -n "$version" ]]; then
    if command -v jq >/dev/null 2>&1; then
      jq -r --arg v "$version" '.mappings[$v] // empty' "$compat_file" || true
      return 0
    else
      log_warn "jq not found; skipping vllm_compat.json lookup (install jq for deterministic pinning)"
    fi
  fi
  return 1
}

# Deterministic vLLM install routine.
# 1) Upgrades pip/wheel
# 2) If compat mapping exists, pre-pins setuptools & torch
# 3) Tries normal pip install
# 4) Parses pip output for setuptools pins and retries
# 5) Falls back to --no-build-isolation
# 6) Fails with log output
_try_install_vllm_deterministic() {
  local v_spec="${1:-vllm}"
  local pip_bin="${VENV_PATH}/bin/pip"
  local py_bin="${VENV_PATH}/bin/python"
  local logf
  logf="$(mktemp -t vllm_install.XXXXXX)"

  if [[ "${DRY_RUN}" == true ]]; then
    log_info "[DRY-RUN] Would attempt deterministic vLLM install (${v_spec})"
    rm -f "$logf"
    return 0
  fi

  # 1) Ensure pip/setuptools/wheel up-to-date as first step
  run_as_invoker "$pip_bin" install --upgrade pip wheel >/dev/null 2>&1 || true

  # 2) If mapping exists for user-specified VLLM_INSTALL_VERSION, pre-pin setuptools & torch
  if [[ -n "${VLLM_INSTALL_VERSION:-}" ]]; then
    local compat_json
    compat_json="$(_vllm_compat_lookup "${VLLM_INSTALL_VERSION}" 2>/dev/null || true)"
    if [[ -n "$compat_json" ]]; then
      log_info "Using compatibility mapping for vLLM ${VLLM_INSTALL_VERSION}"
      # extract pin values if present
      local s_pin t_pin t_tag
      s_pin="$(printf "%s\n" "$compat_json" | jq -r '.setuptools // empty' 2>/dev/null || true)"
      t_pin="$(printf "%s\n" "$compat_json" | jq -r '.torch // empty' 2>/dev/null || true)"
      t_tag="$(printf "%s\n" "$compat_json" | jq -r '.torch_cuda_tag // empty' 2>/dev/null || true)"
      if [[ -n "$s_pin" ]]; then
        log_info "Pinning setuptools==${s_pin} from compatibility mapping"
        run_as_invoker "$pip_bin" install --force-reinstall "setuptools==${s_pin}"
      fi
      if [[ -n "$t_pin" ]]; then
        # install torch first (choose index-url if cuda tag present)
        if [[ "${t_tag}" =~ cu[0-9]+ ]]; then
          local idx_url="https://download.pytorch.org/whl/${t_tag}"
          # For Blackwell/GB10, prefer nightly index for cu128
          if [[ "${IS_BLACKWELL}" == true && "${t_tag}" == "cu128" ]]; then
            idx_url="https://download.pytorch.org/whl/nightly/cu128"
            log_info "GB10/Blackwell detected: using nightly PyTorch index for cu128"
            run_as_invoker "$pip_bin" install --force-reinstall --pre "torch==${t_pin}" --index-url "$idx_url" \
              || run_as_invoker "$pip_bin" install --force-reinstall --pre torch --index-url "$idx_url"
          else
            log_info "Pinning torch==${t_pin}+${t_tag} from compatibility mapping"
            run_as_invoker "$pip_bin" install --force-reinstall "torch==${t_pin}+${t_tag}" --index-url "$idx_url" \
              || run_as_invoker "$pip_bin" install --force-reinstall "torch==${t_pin}" --index-url "$idx_url"
          fi
          # try matching torchvision/torchaudio if necessary (best-effort)
          run_as_invoker "$pip_bin" install --force-reinstall "torchvision" --index-url "$idx_url" 2>/dev/null || true
        else
          log_info "Pinning torch==${t_pin} from compatibility mapping"
          run_as_invoker "$pip_bin" install --force-reinstall "torch==${t_pin}"
        fi
      fi
    fi
  fi

  # 3) Normal attempt to install vllm
  log_info "Attempting to install ${v_spec} into ${VENV_PATH}"
  if run_as_invoker "$pip_bin" install "${v_spec}" >"$logf" 2>&1; then
    log_ok "Installed ${v_spec} successfully"
    rm -f "$logf"
    return 0
  fi

  # 4) Parse for setuptools/tensor pins and retry
  log_warn "vLLM install failed; inspecting pip output for constraints..."
  local logcat
  logcat="$(sed -n '1,200p' "$logf" || true)"
  # detect exact setuptools pin
  local s_exact
  s_exact="$(printf "%s\n" "$logcat" | grep -oE 'setuptools==[0-9]+(\.[0-9]+){1,2}' | head -n1 || true)"
  if [[ -n "$s_exact" ]]; then
    log_info "Pip output requires ${s_exact}; pinning and retrying"
    run_as_invoker "$pip_bin" install --force-reinstall "$s_exact"
    # try again with no-build-isolation
    if run_as_invoker "$pip_bin" install --force-reinstall --no-build-isolation "${v_spec}" >>"$logf" 2>&1; then
      log_ok "Installed ${v_spec} after pinning setuptools"
      rm -f "$logf"
      return 0
    fi
  fi

  # 5) Generic fallback: try --no-build-isolation
  log_info "Retrying with --no-build-isolation..."
  if run_as_invoker "$pip_bin" install --force-reinstall --no-build-isolation "${v_spec}" >>"$logf" 2>&1; then
    log_ok "Installed ${v_spec} with --no-build-isolation"
    rm -f "$logf"
    return 0
  fi

  # 6) If still failing, show log and return non-zero
  log_error "vLLM install ultimately failed. Pip output:"
  cat "$logf" >&2
  rm -f "$logf"
  return 2
}
# --- End vllm compatibility and deterministic install helpers ---

install_vllm_packages(){
  local pip="${VENV_PATH}/bin/pip"
  local py="${VENV_PATH}/bin/python"

  local vllm_install_spec
  if [[ -n "${VLLM_INSTALL_VERSION}" ]]; then
    vllm_install_spec="vllm==${VLLM_INSTALL_VERSION}"
  else
    vllm_install_spec="vllm"
  fi

  # Check whether the compatibility mapping covers this version.
  # If so, _try_install_vllm_deterministic handles torch+setuptools+vllm.
  local has_compat=false
  if [[ -n "${VLLM_INSTALL_VERSION:-}" ]]; then
    local compat_json
    compat_json="$(_vllm_compat_lookup "${VLLM_INSTALL_VERSION}" 2>/dev/null || true)"
    [[ -n "$compat_json" ]] && has_compat=true
  fi

  if [[ "${has_compat}" == true ]]; then
    log_step "Installing vLLM ${VLLM_INSTALL_VERSION} via compatibility mapping"
    if ! _try_install_vllm_deterministic "${vllm_install_spec}"; then
      log_error "vLLM install failed via compatibility mapping. Aborting."
      exit 1
    fi
    run_as_invoker "$pip" install uvicorn 2>/dev/null || true
  else
    # No compat mapping: install torch first based on platform/GPU, then vllm.
    run_as_invoker "$pip" install -U pip setuptools wheel

    if [[ "${DRY_RUN}" != true ]] && "$py" -c "import torch" >/dev/null 2>&1; then
      log_info "torch already present in venv; not forcing downgrade/replace"
    else
      if [[ "${PLATFORM}" == macos* ]]; then
        log_step "Installing torch for macOS (${ARCH})"
        run_as_invoker "$pip" install torch
        log_warn "macOS vLLM support may be limited; CPU/MPS path selected"
      elif [[ -n "${GPU_NAME}" ]]; then
        if [[ "${IS_BLACKWELL}" == true ]]; then
          log_warn "Detected GB10/Blackwell GPU (${GPU_NAME})"
          log_info "Blackwell GPUs require nightly PyTorch (cu128) for full compatibility"
          if [[ "${YES_MODE}" == true ]]; then
            run_as_invoker "$pip" install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
          else
            read -r -p "Install nightly torch (cu128) for Blackwell? [Y/n] " ans
            if [[ -z "${ans}" || "${ans}" =~ ^[Yy]$ ]]; then
              run_as_invoker "$pip" install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
            else
              log_warn "Using default torch — GPU support may be limited on Blackwell"
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

    if [[ "${DRY_RUN}" != true ]] && "$py" -c "import vllm" >/dev/null 2>&1; then
      log_info "vLLM already installed in ${VENV_PATH}"
    else
      log_step "Installing vLLM"
      if ! _try_install_vllm_deterministic "${vllm_install_spec}"; then
        log_error "vLLM install failed. Aborting installer."
        exit 1
      fi
    fi
    run_as_invoker "$pip" install uvicorn 2>/dev/null || true
  fi

  # Post-install verification
  log_step "Verifying installed vLLM and dependencies..."
  if [[ "${DRY_RUN}" == true ]]; then
    log_info "[DRY-RUN] Skipping pip check/import verification"
    return 0
  fi
  run_as_invoker "${VENV_PATH}/bin/pip" check || {
    log_error "pip check failed — dependency conflicts detected"
    run_as_invoker "${VENV_PATH}/bin/python" -c "import pkgutil; print('installed pkg list sample:', [m.name for m in pkgutil.iter_modules()][:20])" || true
    exit 1
  }

  # Quick import test
  if ! run_as_invoker "${VENV_PATH}/bin/python" -c "import vllm, torch; print('vllm', getattr(vllm,'__version__','n/a'), 'torch', getattr(torch,'__version__','n/a'))"; then
    log_error "Python import check for vllm/torch failed in venv at ${VENV_PATH}"
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
  log_step "Running smoke checks"

  if [[ "${DRY_RUN}" == true ]]; then
    log_info "[DRY-RUN] Would run smoke checks on ${VENV_PATH}"
    return 0
  fi

  # Use the full smoke-test script if available
  local smoke_script="${SCRIPT_DIR}/smoke_test_vllm.sh"
  if [[ -x "${smoke_script}" ]]; then
    if ! "${smoke_script}" --venv-path "${VENV_PATH}" --skip-server --skip-generate; then
      log_error "Smoke tests failed — see output above"
      exit 1
    fi
  else
    # Inline fallback
    local py="${VENV_PATH}/bin/python"
    run "$py" -c "import torch, vllm; print(torch.__version__, vllm.__version__)"
    run "$py" -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
  fi
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

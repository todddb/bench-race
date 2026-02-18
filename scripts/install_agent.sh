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
LOCAL_ONLY=false
PRE_FLIGHT=false
UNINSTALL_ALL=false
PLATFORM_OVERRIDE=""
AGENT_ID=""
LABEL=""
CENTRAL_URL="http://127.0.0.1:8080"
VLLM_PORT="8000"
VLLM_INSTALL_VERSION="${VLLM_INSTALL_VERSION:-}"
USE_CONDA=false
FORCE_NIGHTLY="${FORCE_NIGHTLY:-false}"

# Resolved compatibility values (populated by resolve_vllm_compat())
RESOLVED_VLLM_VER=""
RESOLVED_TORCH_VER=""
RESOLVED_TORCHVISION_VER=""
RESOLVED_TORCHAUDIO_VER=""
RESOLVED_SETUPTOOLS_VER=""
RESOLVED_TORCH_INDEX_URL=""
RESOLVED_TORCH_TAG=""
RESOLVED_NIGHTLY="false"
RESOLVED_CONDA_PREFERRED="false"

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


# run_noabort: run command but do NOT exit script on non-zero.
# Returns exit code in RC variable (global), and prints the command for dry-run mode.
run_noabort() {
  RC=0
  if [[ "${DRY_RUN}" == true ]]; then
    echo "[DRY-RUN] (no-abort) $*"
    RC=0
    return 0
  fi
  set +e
  "$@"
  RC=$?
  set -e
  return ${RC}
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
  --platform PLATFORM         Override detected arch-platform (e.g. linux-x86_64)
  --use-conda                 Use conda/mamba env instead of venv for vLLM
                              (preferred on aarch64+CUDA where pip wheels may be scarce)

Local-only / cleanup options:
  --local-only                Force all paths to repo-local/user-local locations;
                              refuse system-wide writes and sudo for service installs
  --preflight                 Print what will be changed/created and exit (no writes)
  --uninstall-all             Remove all managed artifacts (venvs, models, services,
                              pid/log files). Requires --confirm or -y to proceed
  --confirm                   Non-interactive confirmation for --uninstall-all

Compatibility options:
  --agent-id ID               agent/config machine id
  --label LABEL               agent label
  --central-url URL           central URL in agent config

Environment variables:
  VLLM_INSTALL_VERSION        Same as --vllm-version (CLI flag takes priority)
  BENCH_RACE_CUDA_VERSION     Override CUDA version detection (e.g. 12.8; useful for testing)
  BENCH_RACE_ARCH_PLATFORM    Override arch-platform detection (e.g. linux-aarch64)
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
    --use-conda) USE_CONDA=true; shift ;;
    --agent-id) AGENT_ID="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --central-url) CENTRAL_URL="$2"; shift 2 ;;
    --local-only) LOCAL_ONLY=true; shift ;;
    --preflight|--preflight-only) PRE_FLIGHT=true; shift ;;
    --uninstall-all) UNINSTALL_ALL=true; shift ;;
    --confirm) YES_MODE=true; shift ;;
    --skip-ollama|--skip-comfyui) shift ;;  # accepted for compat; no-op in this script
    -h|--help) usage; exit 0 ;;
    *) log_error "Unknown option: $1"; usage; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# --local-only / --system conflict check
# ---------------------------------------------------------------------------
if [[ "${LOCAL_ONLY}" == true && "${SYSTEM_INSTALL}" == true ]]; then
  log_error "--local-only and --system are mutually exclusive."
  log_error "Use --local-only for user-local/repo-local installs (no sudo, no /etc, no /opt)."
  log_error "Use --system for system-wide installs (requires root/sudo)."
  exit 1
fi

# ---------------------------------------------------------------------------
# --local-only: force user-local/repo-local paths
# ---------------------------------------------------------------------------
if [[ "${LOCAL_ONLY}" == true ]]; then
  SYSTEM_INSTALL=false
  # Set defaults to user-local unless explicitly overridden via --venv-path / --model-dir
  if [[ -z "${VENV_PATH}" ]]; then
    VENV_PATH="${INVOKER_HOME}/bench-race/vllm-venv"
  fi
  if [[ -z "${MODEL_DIR}" ]]; then
    MODEL_DIR="${INVOKER_HOME}/bench-race/models/vllm"
  fi
else
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
fi

# ---------------------------------------------------------------------------
# Guard: refuse accidental root installs without --system
# ---------------------------------------------------------------------------
if [[ "$(id -u)" -eq 0 && "${SYSTEM_INSTALL}" == false && "${UNINSTALL_ALL}" == false && "${UNINSTALL_VLLM}" == false && "${PRE_FLIGHT}" == false ]]; then
  log_error "Running as root without --system is not allowed."
  log_error ""
  log_error "  For a LOCAL install (no system-wide writes):"
  log_error "    Re-run as a normal user:  ./scripts/install_agent.sh --local-only"
  log_error ""
  log_error "  For a SYSTEM-WIDE install:"
  log_error "    sudo ./scripts/install_agent.sh --system"
  exit 1
fi

# ---------------------------------------------------------------------------
# Guard: --local-only must not run as root (except for preflight / uninstall)
# ---------------------------------------------------------------------------
if [[ "${LOCAL_ONLY}" == true && "$(id -u)" -eq 0 && "${PRE_FLIGHT}" == false && "${UNINSTALL_ALL}" == false ]]; then
  log_error "--local-only cannot run as root. Re-run as a normal user."
  exit 1
fi

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
UNAME_M="$(uname -m)"
ARCH="${UNAME_M}"
PLATFORM=""
if [[ -n "${PLATFORM_OVERRIDE}" ]]; then
  PLATFORM="${PLATFORM_OVERRIDE}"
else
  if [[ "${OS}" == "darwin" ]]; then
    PLATFORM="macos-${UNAME_M}"
  elif [[ "${UNAME_M}" == "aarch64" || "${UNAME_M}" == "arm64" ]]; then
    PLATFORM="linux-aarch64"
  else
    PLATFORM="linux-x86_64"
  fi
fi

# Canonical architecture-specific platform string (linux-x86_64, linux-aarch64, macos-arm64, …)
# Used by the compat resolver.  Can be overridden via --platform or BENCH_RACE_ARCH_PLATFORM.
ARCH_PLATFORM="${BENCH_RACE_ARCH_PLATFORM:-}"
if [[ -z "${ARCH_PLATFORM}" ]]; then
  if [[ -n "${PLATFORM_OVERRIDE}" ]]; then
    ARCH_PLATFORM="${PLATFORM_OVERRIDE}"
  else
    case "${OS}-${ARCH}" in
      linux-x86_64)  ARCH_PLATFORM="linux-x86_64" ;;
      linux-aarch64|linux-arm64) ARCH_PLATFORM="linux-aarch64" ;;
      darwin-arm64|darwin-aarch64)  ARCH_PLATFORM="macos-arm64" ;;
      darwin-x86_64) ARCH_PLATFORM="macos-x86_64" ;;
      *)             ARCH_PLATFORM="${OS}-${ARCH}" ;;
    esac
  fi
fi

GPU_NAME=""
CUDA_VERSION=""
CUDA_MAJOR=""
IS_BLACKWELL=false
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
  # Parse CUDA toolkit version from nvidia-smi header output (e.g. "CUDA Version: 12.8")
  # Using the header is more reliable than --query-gpu=driver_version for CUDA toolkit ver.
  CUDA_VERSION="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | awk '{print $NF}' | head -n1 || true)"
fi
# Allow env var override for testability / headless installs
if [[ -n "${BENCH_RACE_CUDA_VERSION:-}" ]]; then
  CUDA_VERSION="${BENCH_RACE_CUDA_VERSION}"
fi
if [[ -n "${CUDA_VERSION}" ]]; then
  CUDA_MAJOR="${CUDA_VERSION%%.*}"
fi
if echo "${GPU_NAME}" | tr '[:upper:]' '[:lower:]' | grep -Eq 'gb10|blackwell|b100|b200'; then
  IS_BLACKWELL=true
fi

# ---------------------------------------------------------------------------
# Compute service path for preflight / uninstall
# ---------------------------------------------------------------------------
_compute_service_paths() {
  # Sets: SERVICE_TARGET, SERVICE_PATH, SERVICE_DESCRIPTION
  if [[ "${OS}" == "linux" ]]; then
    if [[ "${SYSTEM_INSTALL}" == true ]]; then
      SERVICE_TARGET="system (systemd)"
      SERVICE_PATH="/etc/systemd/system/bench-race-vllm.service"
    else
      SERVICE_TARGET="user (systemd user)"
      SERVICE_PATH="${INVOKER_HOME}/.config/systemd/user/bench-race-vllm.service"
    fi
  elif [[ "${OS}" == "darwin" ]]; then
    if [[ "${SYSTEM_INSTALL}" == true ]]; then
      SERVICE_TARGET="system (LaunchDaemon)"
      SERVICE_PATH="/Library/LaunchDaemons/com.bench-race.vllm.plist"
    else
      SERVICE_TARGET="user (LaunchAgent)"
      SERVICE_PATH="${INVOKER_HOME}/Library/LaunchAgents/com.bench-race.vllm.plist"
    fi
  else
    SERVICE_TARGET="unknown"
    SERVICE_PATH="(platform not recognized)"
  fi
}
_compute_service_paths

# ---------------------------------------------------------------------------
# --preflight: print planned paths and exit
# ---------------------------------------------------------------------------
if [[ "${PRE_FLIGHT}" == true ]]; then
  local_label="default"
  [[ "${LOCAL_ONLY}" == true ]] && local_label="local-only"
  [[ "${SYSTEM_INSTALL}" == true ]] && local_label="system"
  cat <<EOF
[PREFLIGHT] ─────────────────────────────────────────────────
[PREFLIGHT] Mode:             ${local_label}
[PREFLIGHT] VENV_PATH:        ${VENV_PATH}
[PREFLIGHT] MODEL_DIR:        ${MODEL_DIR}
[PREFLIGHT] Agent venv:       ${REPO_ROOT}/agent/.venv
[PREFLIGHT] Agent config:     ${REPO_ROOT}/agent/config/agent.yaml
[PREFLIGHT] COMFYUI_DIR:      ${REPO_ROOT}/agent/third_party/comfyui
[PREFLIGHT] SERVICE_TARGET:   ${SERVICE_TARGET}
[PREFLIGHT] SERVICE_PATH:     ${SERVICE_PATH}
[PREFLIGHT] Platform:         ${ARCH_PLATFORM}
[PREFLIGHT] CUDA:             ${CUDA_VERSION:-none}
[PREFLIGHT] GPU:              ${GPU_NAME:-none}
[PREFLIGHT] Run dir:          ${REPO_ROOT}/run
[PREFLIGHT] Log dir:          ${REPO_ROOT}/logs
[PREFLIGHT] LOCAL_ONLY:       ${LOCAL_ONLY}
[PREFLIGHT] SYSTEM_INSTALL:   ${SYSTEM_INSTALL}
[PREFLIGHT] INSTALL_VLLM:     ${INSTALL_VLLM}
[PREFLIGHT] USE_CONDA:        ${USE_CONDA}
[PREFLIGHT] SKIP_SERVICE:     ${SKIP_SERVICE}
[PREFLIGHT] ─────────────────────────────────────────────────
EOF
  if [[ "${LOCAL_ONLY}" == true ]]; then
    echo "[PREFLIGHT] Will NOT write to /etc, /opt, or /Library (local-only mode)."
  elif [[ "${SYSTEM_INSTALL}" == true ]]; then
    echo "[PREFLIGHT] Will write system-level service files (--system mode)."
  else
    echo "[PREFLIGHT] Will write user-level service files only."
  fi
  exit 0
fi

# ---------------------------------------------------------------------------
# uninstall_all – comprehensive cleanup
# ---------------------------------------------------------------------------
uninstall_all() {
  log_step "Uninstalling ALL managed bench-race agent artifacts"

  if [[ "${YES_MODE}" != true ]]; then
    log_warn "This will remove venvs, model directories, services, and pid/log files."
    log_warn "Paths that will be removed:"
    log_info "  vLLM venv:      ${VENV_PATH}"
    log_info "  Agent venv:     ${REPO_ROOT}/agent/.venv"
    log_info "  Model dir:      ${MODEL_DIR}"
    log_info "  Service:        ${SERVICE_PATH}"
    log_info "  Run dir:        ${REPO_ROOT}/run"
    log_info "  Log dir:        ${REPO_ROOT}/logs"
    log_info "  Local marker:   ${REPO_ROOT}/.bench-race-local-only"
    read -r -p "Proceed? [y/N] " ans
    if [[ ! "${ans}" =~ ^[Yy]$ ]]; then
      log_info "Aborted."
      exit 0
    fi
  fi

  local removed=()

  # Stop and remove services
  if [[ "${OS}" == "linux" ]] && command -v systemctl >/dev/null 2>&1; then
    # User-level service
    local user_unit="${INVOKER_HOME}/.config/systemd/user/bench-race-vllm.service"
    if [[ -f "${user_unit}" ]]; then
      run_as_invoker systemctl --user stop bench-race-vllm >/dev/null 2>&1 || true
      run_as_invoker systemctl --user disable bench-race-vllm >/dev/null 2>&1 || true
      run_as_invoker rm -f "${user_unit}"
      run_as_invoker systemctl --user daemon-reload >/dev/null 2>&1 || true
      removed+=("${user_unit}")
    fi
    # System-level service (only if --system or running as root)
    if [[ "${SYSTEM_INSTALL}" == true || "$(id -u)" -eq 0 ]]; then
      local sys_unit="/etc/systemd/system/bench-race-vllm.service"
      if [[ -f "${sys_unit}" ]]; then
        run systemctl stop bench-race-vllm >/dev/null 2>&1 || true
        run systemctl disable bench-race-vllm >/dev/null 2>&1 || true
        run rm -f "${sys_unit}"
        run systemctl daemon-reload >/dev/null 2>&1 || true
        removed+=("${sys_unit}")
      fi
    fi
  fi
  if [[ "${OS}" == "darwin" ]]; then
    local user_plist="${INVOKER_HOME}/Library/LaunchAgents/com.bench-race.vllm.plist"
    if [[ -f "${user_plist}" ]]; then
      run_as_invoker launchctl unload "${user_plist}" >/dev/null 2>&1 || true
      run_as_invoker rm -f "${user_plist}"
      removed+=("${user_plist}")
    fi
    if [[ "${SYSTEM_INSTALL}" == true || "$(id -u)" -eq 0 ]]; then
      local sys_plist="/Library/LaunchDaemons/com.bench-race.vllm.plist"
      if [[ -f "${sys_plist}" ]]; then
        run launchctl unload "${sys_plist}" >/dev/null 2>&1 || true
        run rm -f "${sys_plist}"
        removed+=("${sys_plist}")
      fi
    fi
  fi

  # Remove vLLM venv
  if [[ -d "${VENV_PATH}" ]]; then
    run rm -rf "${VENV_PATH}"
    removed+=("${VENV_PATH}")
  fi

  # Remove agent venv
  if [[ -d "${REPO_ROOT}/agent/.venv" ]]; then
    run rm -rf "${REPO_ROOT}/agent/.venv"
    removed+=("${REPO_ROOT}/agent/.venv")
  fi

  # Remove model directory
  if [[ -d "${MODEL_DIR}" ]]; then
    run rm -rf "${MODEL_DIR}"
    removed+=("${MODEL_DIR}")
  fi

  # Remove pid files
  if [[ -d "${REPO_ROOT}/run" ]]; then
    run rm -rf "${REPO_ROOT}/run"
    removed+=("${REPO_ROOT}/run")
  fi

  # Remove log files
  if [[ -d "${REPO_ROOT}/logs" ]]; then
    run rm -rf "${REPO_ROOT}/logs"
    removed+=("${REPO_ROOT}/logs")
  fi

  # Remove local-only marker
  if [[ -f "${REPO_ROOT}/.bench-race-local-only" ]]; then
    run rm -f "${REPO_ROOT}/.bench-race-local-only"
    removed+=("${REPO_ROOT}/.bench-race-local-only")
  fi

  # Remove conda env if created by this installer
  if command -v conda >/dev/null 2>&1; then
    if conda env list 2>/dev/null | grep -q "^bench-race-vllm[[:space:]]"; then
      log_info "Removing conda env bench-race-vllm"
      conda env remove -n bench-race-vllm -y >/dev/null 2>&1 || true
      removed+=("conda:bench-race-vllm")
    fi
  fi

  log_ok "Uninstall complete. Summary of removed items:"
  if [[ ${#removed[@]} -eq 0 ]]; then
    log_info "  (nothing found to remove)"
  else
    for item in "${removed[@]}"; do
      log_info "  removed: ${item}"
    done
  fi
}

create_or_update_agent_venv(){
  log_step "Ensuring agent venv is installed"
  local py
  py="$("${SCRIPT_DIR}/_python_pick.sh")"
  if [[ ! -d "${REPO_ROOT}/agent/.venv" ]]; then
    run "$py" -m venv "${REPO_ROOT}/agent/.venv"
  fi
  # Only upgrade pip and wheel — leave setuptools at its installed version unless pinned
  run "${REPO_ROOT}/agent/.venv/bin/python" -m pip install --upgrade pip wheel
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

# ---------------------------------------------------------------------------
# resolve_vllm_compat – platform-aware compat resolver
# Calls scripts/_resolve_compat.py and populates RESOLVED_* globals.
# Falls back gracefully when the resolver script is absent or fails.
# ---------------------------------------------------------------------------
resolve_vllm_compat() {
  local resolver="${SCRIPT_DIR}/_resolve_compat.py"
  if [[ ! -f "${resolver}" ]]; then
    log_warn "Compat resolver not found (${resolver}); skipping platform resolution"
    return 1
  fi

  local py
  py="$("${SCRIPT_DIR}/_python_pick.sh" 2>/dev/null)" || py="python3"

  local resolver_out
  # Pass detection overrides via env vars so tests can simulate any platform
  if ! resolver_out="$(BENCH_RACE_ARCH_PLATFORM="${ARCH_PLATFORM}" \
      BENCH_RACE_CUDA_VERSION="${CUDA_VERSION}" \
      "$py" "${resolver}" 2>/dev/null)"; then
    log_warn "Compat resolver returned non-zero for platform=${ARCH_PLATFORM} cuda=${CUDA_VERSION}"
    return 1
  fi

  if [[ -z "${resolver_out}" ]]; then
    log_warn "Compat resolver produced no output for platform=${ARCH_PLATFORM}"
    return 1
  fi

  # Parse key=value output into RESOLVED_* globals
  local key val
  while IFS='=' read -r key val; do
    [[ -z "${key}" ]] && continue
    case "${key}" in
      vllm)            RESOLVED_VLLM_VER="${val}" ;;
      torch)           RESOLVED_TORCH_VER="${val}" ;;
      torchvision)     RESOLVED_TORCHVISION_VER="${val}" ;;
      torchaudio)      RESOLVED_TORCHAUDIO_VER="${val}" ;;
      setuptools)      RESOLVED_SETUPTOOLS_VER="${val}" ;;
      torch_index_url) RESOLVED_TORCH_INDEX_URL="${val}" ;;
      torch_tag)       RESOLVED_TORCH_TAG="${val}" ;;
      nightly)         RESOLVED_NIGHTLY="${val}" ;;
      conda_preferred) RESOLVED_CONDA_PREFERRED="${val}" ;;
    esac
  done <<< "${resolver_out}"

  log_info "Resolved compat: vllm=${RESOLVED_VLLM_VER} torch=${RESOLVED_TORCH_VER} tag=${RESOLVED_TORCH_TAG} nightly=${RESOLVED_NIGHTLY}"
  if [[ "${RESOLVED_CONDA_PREFERRED}" == "true" && "${USE_CONDA}" == false ]]; then
    log_warn "Compat map recommends --use-conda for ${ARCH_PLATFORM}+CUDA ${CUDA_VERSION} (pip wheel availability may be limited)"
  fi
  return 0
}

# ---------------------------------------------------------------------------
# _ensure_conda_env – create/reuse a conda env for USE_CONDA=true installs
# Sets VENV_PATH to the conda env directory so downstream code (pip_bin, py_bin)
# continues to work unchanged.
# ---------------------------------------------------------------------------
_ensure_conda_env() {
  local env_name="bench-race-vllm"
  local py_ver="3.11"

  if [[ "${DRY_RUN}" == true ]]; then
    log_info "[DRY-RUN] Would create/reuse conda env '${env_name}'"
    VENV_PATH="(conda-env:${env_name})"
    return 0
  fi

  local conda_cmd=""
  if command -v mamba >/dev/null 2>&1; then
    conda_cmd="mamba"
  elif command -v conda >/dev/null 2>&1; then
    conda_cmd="conda"
  else
    log_error "--use-conda requested but neither conda nor mamba was found."
    log_error "Install Miniconda first, for example:"
    if [[ "${ARCH_PLATFORM}" == "linux-aarch64" ]]; then
      log_error "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
      log_error "  bash Miniconda3-latest-Linux-aarch64.sh -b -p \${HOME}/miniconda3"
    else
      log_error "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
      log_error "  bash Miniconda3-latest-Linux-x86_64.sh -b -p \${HOME}/miniconda3"
    fi
    log_error "  source \${HOME}/miniconda3/etc/profile.d/conda.sh"
    log_error "Then re-run the installer with --use-conda."
    return 1
  fi

  # Create env if it does not already exist
  if ! run_as_invoker "${conda_cmd}" env list 2>/dev/null | grep -q "^${env_name}[[:space:]]"; then
    log_step "Creating conda env '${env_name}' with Python ${py_ver}"
    run_as_invoker "${conda_cmd}" create -n "${env_name}" "python=${py_ver}" -y
  else
    log_info "Conda env '${env_name}' already exists"
  fi

  # Resolve env path
  local conda_env_path
  conda_env_path="$(run_as_invoker "${conda_cmd}" info --envs 2>/dev/null \
    | grep "^${env_name}[[:space:]]" | awk '{print $NF}' | head -n1 || true)"
  if [[ -z "${conda_env_path}" ]]; then
    log_error "Could not determine conda env path for '${env_name}'"
    return 1
  fi
  VENV_PATH="${conda_env_path}"
  log_ok "Conda env path: ${VENV_PATH}"

  # Install PyTorch via conda for reliable CUDA wheel availability
  log_step "Installing PyTorch via conda (CUDA ${CUDA_VERSION:-none})"
  if [[ -n "${CUDA_MAJOR}" ]]; then
    # Use a concrete cuda version conda package; for Blackwell / CUDA>=13 use 12.8 compat
    local cuda_pkg
    if [[ "${IS_BLACKWELL}" == true ]] || [[ "${CUDA_MAJOR}" -ge 13 ]]; then
      cuda_pkg="pytorch-cuda=12.8"
    else
      cuda_pkg="pytorch-cuda=${CUDA_VERSION}"
    fi
    run_as_invoker "${conda_cmd}" install -n "${env_name}" \
      pytorch torchvision torchaudio "${cuda_pkg}" \
      -c pytorch -c nvidia -y
  else
    run_as_invoker "${conda_cmd}" install -n "${env_name}" \
      pytorch torchvision torchaudio cpuonly \
      -c pytorch -y
  fi
}

# ---------------------------------------------------------------------------
# _install_torch_from_resolved – install torch/torchvision/torchaudio using
# the RESOLVED_* globals set by resolve_vllm_compat().
# ---------------------------------------------------------------------------
_install_torch_from_resolved() {
  local pip_bin="${VENV_PATH}/bin/pip"
  local torch_ver="${RESOLVED_TORCH_VER:-2.9.1}"
  local torchvision_ver="${RESOLVED_TORCHVISION_VER:-0.14.1}"
  local torchaudio_ver="${RESOLVED_TORCHAUDIO_VER:-2.9.1}"
  local torch_spec="torch==${torch_ver} torchvision==${torchvision_ver} torchaudio==${torchaudio_ver}"

  local use_nightly=false
  if [[ "${FORCE_NIGHTLY}" == "true" ]]; then
    use_nightly=true
  elif [[ "${PLATFORM}" == "linux-aarch64" || "${ARCH_PLATFORM}" == "linux-aarch64" ]]; then
    # GB10 (ARM) requires nightly or conda-nightly
    use_nightly=true
  else
    # Default for x86_64 (including Blackwell RTX Pro 6000)
    use_nightly=false
  fi

  if [[ "${DRY_RUN}" == true ]]; then
    log_info "[DRY-RUN] Would install torch first: ${torch_spec} nightly=${use_nightly}"
    return 0
  fi

  if [[ "${use_nightly}" == true ]]; then
    log_info "Installing torch first in nightly mode (linux-aarch64 only unless forced)"
    if [[ "${USE_CONDA}" == false ]] && { command -v conda >/dev/null 2>&1 || command -v mamba >/dev/null 2>&1; }; then
      log_info "Conda nightly recommendation:"
      log_info "  mamba create -n vllm-nightly python=3.12"
      log_info "  mamba activate vllm-nightly"
      log_info "  mamba install -c pytorch-nightly pytorch torchvision torchaudio"
      log_info "  pip install vllm"
    fi
    run_noabort "${pip_bin}" install --pre \
      --index-url https://download.pytorch.org/whl/nightly/cu128 \
      --upgrade --force-reinstall ${torch_spec}
    if [[ ${RC} -ne 0 ]]; then
      log_warn "Exact pinned nightly (${torch_spec}) failed (rc=${RC}). Falling back to latest nightly (unpinned) for torch/torchvision/torchaudio."
      run_noabort "${pip_bin}" install --pre \
        --index-url https://download.pytorch.org/whl/nightly/cu128 \
        --upgrade --force-reinstall torch torchvision torchaudio || true
    fi
# If exact pinned nightly isn't available on the index, fall back to unpinned nightly install
    if [[ $? -ne 0 ]]; then
      log_warn "Exact pinned nightly (${torch_spec}) failed. Falling back to latest nightly (unpinned) for torch/torchvision/torchaudio."
      run_as_invoker "${pip_bin}" install --pre \
        --index-url https://download.pytorch.org/whl/nightly/cu128 \
        --upgrade --force-reinstall torch torchvision torchaudio || true
    fi
  else
    log_info "Installing torch first in stable mode"
    # Try the pinned install first; if it fails (platform wheel missing), try installing torch alone.
    if ! run_noabort "${pip_bin}" install --upgrade --force-reinstall ${torch_spec}
    if [[ ${RC} -ne 0 ]]; then
      log_warn "Pinned stable torch spec (${torch_spec}) failed (rc=${RC}). Trying to install torch alone (un-pinned) as fallback."
      run_noabort "${pip_bin}" install --upgrade --force-reinstall torch || true
      run_noabort "${pip_bin}" install --upgrade --force-reinstall torchvision || true
      run_noabort "${pip_bin}" install --upgrade --force-reinstall torchaudio || true
    fi; then
      log_warn "Pinned stable torch spec (${torch_spec}) failed. Trying to install torch alone (un-pinned) as fallback."
      run_as_invoker "${pip_bin}" install --upgrade --force-reinstall torch || true
      # Try torchvision/torchaudio separately (they may be unavailable for this platform)
      run_noabort "${pip_bin}" install --upgrade --force-reinstall torchvision || true
      run_noabort "${pip_bin}" install --upgrade --force-reinstall torchaudio || true
    fi
  fi
}

# ---------------------------------------------------------------------------
# _pip_check_with_fixup – run pip check and attempt a setuptools repin when
# the initial check fails.  Prints full diagnostics and exits on persistent
# failure so the caller always gets a clear error message.
# ---------------------------------------------------------------------------
_pip_check_with_fixup() {
  local pip_bin="${VENV_PATH}/bin/pip"
  local torch_ver="${RESOLVED_TORCH_VER:-2.9.1}"
  local torchvision_ver="${RESOLVED_TORCHVISION_VER:-0.14.1}"
  local torchaudio_ver="${RESOLVED_TORCHAUDIO_VER:-2.9.1}"
  local torch_spec="torch==${torch_ver} torchvision==${torchvision_ver} torchaudio==${torchaudio_ver}"
  local vllm_spec="${VLLM_INSTALL_VERSION:+vllm==${VLLM_INSTALL_VERSION}}"
  [[ -z "${vllm_spec}" ]] && vllm_spec="vllm"

  local check_out
  if check_out="$(run_as_invoker "${pip_bin}" check 2>&1)"; then
    log_ok "pip check passed"
    echo "${check_out}"
    return 0
  fi

  log_error "pip check failed — dependency conflicts detected"
  log_error "=== pip check output ==="
  printf "%s\n" "${check_out}" >&2
  log_error ""
  log_error "=== pip list (torch/vllm filtered) ==="
  run_as_invoker "${pip_bin}" list --format=columns 2>/dev/null | grep -Ei '^(torch|torchaudio|torchvision|vllm)\b' >&2 || true
  log_error ""
  log_error "Manual repair commands:"
  log_error "  ${pip_bin} install --force-reinstall ${torch_spec}"
  log_error "  ${pip_bin} install --force-reinstall ${vllm_spec}"
  return 1
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
# Torch is installed first by the caller; this function only installs vLLM.
# On failure, writes diagnostics and returns non-zero.
_try_install_vllm_deterministic() {
  local v_spec="${1:-vllm}"
  local pip_bin="${VENV_PATH}/bin/pip"
  local logf="${VENV_PATH}/install_vllm.log"

  if [[ "${DRY_RUN}" == true ]]; then
    log_info "[DRY-RUN] Would install ${v_spec} after torch"
    return 0
  fi

  mkdir -p "$(dirname "${logf}")"
  echo "=== vLLM install attempt: $(date) ===" >> "${logf}"
  echo "spec=${v_spec} platform=${ARCH_PLATFORM} cuda=${CUDA_VERSION}" >> "${logf}"

  run_as_invoker "${pip_bin}" install --upgrade pip wheel >/dev/null 2>&1 || true

  log_info "Installing ${v_spec} (torch is always installed first)"
  run_noabort "${pip_bin}" install --force-reinstall "${v_spec}" >>"${logf}" 2>&1
  if [[ ${RC} -eq 0 ]]; then
    log_ok "Installed ${v_spec} successfully"
    return 0
  fi

  # If exact pinned vllm version not found (or pip failed), try installing latest vllm as a fallback.
  log_warn "Exact install of ${v_spec} failed; attempting fallback 'pip install vllm' (latest) to see if an unpinned install works."
  if run_as_invoker "${pip_bin}" install vllm >>"${logf}" 2>&1; then
    log_ok "Fallback 'pip install vllm' succeeded. Continuing with unpinned vllm."
    return 0
  fi

  log_error "vLLM install failed. Last 80 lines from ${logf}:"
  tail -n 80 "${logf}" >&2 || true
  return 2
}
# --- End vllm compatibility and deterministic install helpers ---

install_vllm_packages(){
  local pip="${VENV_PATH}/bin/pip"

  resolve_vllm_compat || log_info "Platform resolver unavailable; using defaults"

  if [[ -z "${VLLM_INSTALL_VERSION}" && -n "${RESOLVED_VLLM_VER}" ]]; then
    VLLM_INSTALL_VERSION="${RESOLVED_VLLM_VER}"
    log_info "Using resolver-selected vLLM version: ${VLLM_INSTALL_VERSION}"
  fi

  local vllm_install_spec
  if [[ -n "${VLLM_INSTALL_VERSION}" ]]; then
    vllm_install_spec="vllm==${VLLM_INSTALL_VERSION}"
  else
    vllm_install_spec="vllm"
  fi

  run_as_invoker "${pip}" install --upgrade pip wheel >/dev/null 2>&1 || true

  if [[ -n "${VLLM_INSTALL_VERSION:-}" ]]; then
    local compat_json
    compat_json="$(_vllm_compat_lookup "${VLLM_INSTALL_VERSION}" 2>/dev/null || true)"
    if [[ -n "${compat_json}" ]]; then
      RESOLVED_TORCH_VER="$(printf "%s\n" "${compat_json}" | jq -r '.torch // empty' 2>/dev/null || true)"
      RESOLVED_TORCHVISION_VER="$(printf "%s\n" "${compat_json}" | jq -r '.torchvision // empty' 2>/dev/null || true)"
      RESOLVED_TORCHAUDIO_VER="$(printf "%s\n" "${compat_json}" | jq -r '.torchaudio // empty' 2>/dev/null || true)"
      log_info "Using vllm_compat.json mapping for vLLM ${VLLM_INSTALL_VERSION}"
    else
      log_warn "No vllm_compat.json mapping found for ${VLLM_INSTALL_VERSION}; using torch defaults"
    fi
  fi

  # Always install torch before vLLM.
  _install_torch_from_resolved

  log_step "Installing ${vllm_install_spec}"
  if ! _try_install_vllm_deterministic "${vllm_install_spec}"; then
    log_error "vLLM install failed. Aborting installer."
    exit 1
  fi

  run_as_invoker "${pip}" install uvicorn 2>/dev/null || true

  log_step "Verifying installed vLLM and dependencies..."
  if [[ "${DRY_RUN}" == true ]]; then
    log_info "[DRY-RUN] Skipping pip check/import verification"
    return 0
  fi

  _pip_check_with_fixup || exit 1

  # Required final validation commands
  run_as_invoker "${VENV_PATH}/bin/pip" check
  run_as_invoker "${VENV_PATH}/bin/python" -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  run_as_invoker "${VENV_PATH}/bin/python" -c "import vllm; print(vllm.__version__)"
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
Environment=VLLM_VERSION=${VLLM_INSTALL_VERSION:-auto}
Environment=TORCH_VERSION=${RESOLVED_TORCH_VER:-auto}
Environment=BENCH_RACE_ARCH_PLATFORM=${ARCH_PLATFORM}
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
Environment=VLLM_VERSION=${VLLM_INSTALL_VERSION:-auto}
Environment=TORCH_VERSION=${RESOLVED_TORCH_VER:-auto}
Environment=BENCH_RACE_ARCH_PLATFORM=${ARCH_PLATFORM}
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
<key>EnvironmentVariables</key><dict><key>HF_HOME</key><string>${MODEL_DIR}</string><key>VLLM_VERSION</key><string>${VLLM_INSTALL_VERSION:-auto}</string><key>TORCH_VERSION</key><string>${RESOLVED_TORCH_VER:-auto}</string></dict>
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
<key>EnvironmentVariables</key><dict><key>HF_HOME</key><string>${MODEL_DIR}</string><key>VLLM_VERSION</key><string>${VLLM_INSTALL_VERSION:-auto}</string><key>TORCH_VERSION</key><string>${RESOLVED_TORCH_VER:-auto}</string></dict>
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
  # Handle --uninstall-all before any install logic
  if [[ "${UNINSTALL_ALL}" == true ]]; then
    uninstall_all
    exit 0
  fi

  log_info "Platform=${PLATFORM} arch_platform=${ARCH_PLATFORM} arch=${ARCH} cuda=${CUDA_VERSION:-none} gpu='${GPU_NAME:-none}' blackwell=${IS_BLACKWELL}"
  if [[ "${LOCAL_ONLY}" == true ]]; then
    log_info "Mode: local-only (no system-wide writes)"
  fi

  create_or_update_agent_venv
  write_agent_config_if_missing

  if [[ "${UNINSTALL_VLLM}" == true ]]; then
    uninstall_vllm
    exit 0
  fi

  if [[ "${INSTALL_VLLM}" == true ]]; then
    if [[ "${USE_CONDA}" == true ]]; then
      log_step "Conda mode: setting up conda env instead of venv"
      _ensure_conda_env || exit 1
    else
      ensure_vllm_venv
    fi
    install_vllm_packages
    ensure_vllm_config_block
    install_vllm_service
  else
    log_info "Skipping vLLM install (--no-vllm)"
  fi

  smoke_checks

  # Write local-only marker so scripts/agents can detect the install mode
  if [[ "${LOCAL_ONLY}" == true ]]; then
    if [[ "${DRY_RUN}" != true ]]; then
      touch "${REPO_ROOT}/.bench-race-local-only"
      log_info "Wrote local-only marker: ${REPO_ROOT}/.bench-race-local-only"
    else
      echo "[DRY-RUN] Would write local-only marker: ${REPO_ROOT}/.bench-race-local-only"
    fi
  fi

  log_ok "Install complete"
}

main "$@"

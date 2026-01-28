#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# pull_models.sh - Pull models from agent/config/agent.yaml
#
# Reads agent.yaml and pulls all required models:
# - LLM models via ollama pull
# - Whisper models (logs message if not implemented)
# - SDXL profiles (logs message if not implemented)
#
# Usage:
#   ./scripts/pull_models.sh              # Read from default agent/config/agent.yaml
#   AGENT_CONFIG=/path/to/agent.yaml ./scripts/pull_models.sh
#   ./scripts/pull_models.sh --skip-70b   # Skip 70B models
#   ./scripts/pull_models.sh --legacy     # Use hardcoded model list (old behavior)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

log() {
  echo "[pull_models] $*"
}

log_error() {
  echo "[pull_models] ERROR: $*" >&2
}

log_warn() {
  echo "[pull_models] WARNING: $*" >&2
}

# Check if ollama is installed
if ! command -v ollama >/dev/null 2>&1; then
  log_error "Ollama CLI not found. Install Ollama first: https://ollama.ai"
  exit 1
fi

# Parse arguments
skip_70b=false
legacy_mode=false
custom_models=()

for arg in "$@"; do
  case "$arg" in
    --skip-70b)
      skip_70b=true
      ;;
    --legacy)
      legacy_mode=true
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS] [MODELS...]"
      echo ""
      echo "Pull models specified in agent/config/agent.yaml"
      echo ""
      echo "Options:"
      echo "  --skip-70b     Skip models containing '70b' in the name"
      echo "  --legacy       Use hardcoded model list (old behavior)"
      echo "  --help, -h     Show this help message"
      echo ""
      echo "Environment:"
      echo "  AGENT_CONFIG   Path to agent.yaml (default: agent/config/agent.yaml)"
      echo ""
      echo "Examples:"
      echo "  $0                                  # Pull models from agent.yaml"
      echo "  $0 --skip-70b                       # Skip large 70B models"
      echo "  AGENT_CONFIG=/path/to/agent.yaml $0 # Use custom config path"
      exit 0
      ;;
    *)
      custom_models+=("$arg")
      ;;
  esac
done

# Determine which models to pull
llm_models=()
whisper_models=()
sdxl_profiles=()

if [ "$legacy_mode" = true ] || [ ${#custom_models[@]} -gt 0 ]; then
  # Legacy mode or custom models specified on command line
  if [ ${#custom_models[@]} -gt 0 ]; then
    llm_models=("${custom_models[@]}")
  else
    llm_models=("llama3.1:8b-instruct-q8_0" "llama3.1:70b-instruct-q4_K_M")
  fi
  log "Using $([ ${#custom_models[@]} -gt 0 ] && echo "command-line models" || echo "legacy hardcoded models")"
else
  # Read from agent.yaml using Python helper
  CONFIG_HELPER="$SCRIPT_DIR/_read_agent_config.py"

  if [ ! -f "$CONFIG_HELPER" ]; then
    log_error "Config helper not found: $CONFIG_HELPER"
    exit 1
  fi

  # Try to use agent venv's Python if available
  PYTHON="python3"
  if [ -f "$REPO_ROOT/agent/.venv/bin/python" ]; then
    PYTHON="$REPO_ROOT/agent/.venv/bin/python"
  fi

  log "Reading config from ${AGENT_CONFIG:-agent/config/agent.yaml}"

  # Get config as JSON
  config_json=$("$PYTHON" "$CONFIG_HELPER" 2>&1) || {
    log_error "Failed to read agent config: $config_json"
    exit 1
  }

  # Parse JSON using Python (portable, no jq dependency)
  llm_models_str=$("$PYTHON" -c "import json,sys; print('\\n'.join(json.loads(sys.argv[1]).get('llm_models', [])))" "$config_json")
  whisper_models_str=$("$PYTHON" -c "import json,sys; print('\\n'.join(json.loads(sys.argv[1]).get('whisper_models', [])))" "$config_json")
  sdxl_profiles_str=$("$PYTHON" -c "import json,sys; print('\\n'.join(json.loads(sys.argv[1]).get('sdxl_profiles', [])))" "$config_json")

  # Convert to arrays
  if [ -n "$llm_models_str" ]; then
    while IFS= read -r model; do
      [ -n "$model" ] && llm_models+=("$model")
    done <<< "$llm_models_str"
  fi

  if [ -n "$whisper_models_str" ]; then
    while IFS= read -r model; do
      [ -n "$model" ] && whisper_models+=("$model")
    done <<< "$whisper_models_str"
  fi

  if [ -n "$sdxl_profiles_str" ]; then
    while IFS= read -r profile; do
      [ -n "$profile" ] && sdxl_profiles+=("$profile")
    done <<< "$sdxl_profiles_str"
  fi
fi

# Filter out 70B models if requested
filtered_llm_models=()
for model in "${llm_models[@]}"; do
  if $skip_70b && [[ "$model" == *"70b"* ]]; then
    log "Skipping $model due to --skip-70b"
    continue
  fi
  filtered_llm_models+=("$model")
done

# Print planned pulls
echo ""
echo "=============================================="
echo "          PLANNED MODEL PULLS"
echo "=============================================="
echo ""
echo "LLM Models (via Ollama):"
if [ ${#filtered_llm_models[@]} -eq 0 ]; then
  echo "  (none)"
else
  for model in "${filtered_llm_models[@]}"; do
    echo "  - $model"
  done
fi
echo ""

echo "Whisper Models:"
if [ ${#whisper_models[@]} -eq 0 ]; then
  echo "  (none configured)"
else
  for model in "${whisper_models[@]}"; do
    echo "  - $model"
  done
fi
echo ""

echo "SDXL Profiles:"
if [ ${#sdxl_profiles[@]} -eq 0 ]; then
  echo "  (none configured)"
else
  for profile in "${sdxl_profiles[@]}"; do
    echo "  - $profile"
  done
fi
echo ""
echo "=============================================="
echo ""

# Track results
success_count=0
fail_count=0
failed_models=()

# Pull LLM models
if [ ${#filtered_llm_models[@]} -gt 0 ]; then
  log "Pulling ${#filtered_llm_models[@]} LLM model(s)..."
  echo ""

  for model in "${filtered_llm_models[@]}"; do
    log "Pulling LLM model: $model"
    if ollama pull "$model"; then
      ((success_count++))
      log "Successfully pulled: $model"
    else
      ((fail_count++))
      failed_models+=("$model")
      log_error "Failed to pull: $model"
    fi
    echo ""
  done
fi

# Handle Whisper models
if [ ${#whisper_models[@]} -gt 0 ]; then
  log_warn "Whisper model pulling not implemented yet"
  log_warn "Requested whisper models: ${whisper_models[*]}"
  log_warn "Please install whisper models manually if needed"
  echo ""
fi

# Handle SDXL profiles
if [ ${#sdxl_profiles[@]} -gt 0 ]; then
  log_warn "SDXL profile pulling not implemented yet"
  log_warn "Requested SDXL profiles: ${sdxl_profiles[*]}"
  log_warn "Please configure SDXL/ComfyUI profiles manually if needed"
  echo ""
fi

# Print summary
echo ""
echo "=============================================="
echo "               SUMMARY"
echo "=============================================="
echo ""
echo "LLM Models:"
echo "  Successfully pulled: $success_count"
echo "  Failed: $fail_count"
if [ ${#failed_models[@]} -gt 0 ]; then
  echo "  Failed models:"
  for model in "${failed_models[@]}"; do
    echo "    - $model"
  done
fi
echo ""

# Show installed models
log "Currently installed Ollama models:"
ollama list 2>/dev/null || log_warn "Could not list Ollama models"

echo ""
echo "=============================================="
echo ""

# Exit with error if any models failed
if [ $fail_count -gt 0 ]; then
  exit 1
fi

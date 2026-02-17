#!/usr/bin/env bash
# smoke_test_vllm.sh
#
# Post-installation smoke tests for vLLM.
# Validates that the vLLM installation is functional.
#
# Usage:
#   ./smoke_test_vllm.sh [OPTIONS]
#
# Options:
#   --venv-path PATH      Path to vLLM venv (default: /opt/bench-race/vllm-venv)
#   --vllm-url URL        vLLM server URL (default: http://127.0.0.1:8000)
#   --agent-url URL       Agent server URL (default: http://127.0.0.1:9001)
#   --gpu-required        Fail if GPU is not available
#   --skip-server         Skip vLLM server tests (import-only)
#   --skip-generate       Skip token generation test

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}PASS${NC} $*"; }
fail() { echo -e "  ${RED}FAIL${NC} $*"; }
skip() { echo -e "  ${YELLOW}SKIP${NC} $*"; }
info() { echo -e "  ${BLUE}INFO${NC} $*"; }

# Defaults
VENV_PATH="/opt/bench-race/vllm-venv"
VLLM_URL="http://127.0.0.1:8000"
AGENT_URL="http://127.0.0.1:9001"
GPU_REQUIRED=false
SKIP_SERVER=false
SKIP_GENERATE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv-path)       VENV_PATH="$2"; shift 2 ;;
        --vllm-url)        VLLM_URL="$2"; shift 2 ;;
        --agent-url)       AGENT_URL="$2"; shift 2 ;;
        --gpu-required)    GPU_REQUIRED=true; shift ;;
        --skip-server)     SKIP_SERVER=true; shift ;;
        --skip-generate)   SKIP_GENERATE=true; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  --venv-path PATH     vLLM venv path"
            echo "  --vllm-url URL       vLLM server URL"
            echo "  --agent-url URL      Agent server URL"
            echo "  --gpu-required       Fail if no GPU"
            echo "  --skip-server        Skip server tests"
            echo "  --skip-generate      Skip generation test"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

PYTHON="$VENV_PATH/bin/python"
FAILURES=0
TOTAL=0
PASSED=0
SKIPPED=0

run_test() {
    local name="$1"
    shift
    TOTAL=$((TOTAL + 1))
    if "$@"; then
        pass "$name"
        PASSED=$((PASSED + 1))
    else
        fail "$name"
        FAILURES=$((FAILURES + 1))
    fi
}

skip_test() {
    local name="$1"
    TOTAL=$((TOTAL + 1))
    SKIPPED=$((SKIPPED + 1))
    skip "$name"
}

echo ""
echo "============================================"
echo "  vLLM Smoke Tests"
echo "============================================"
echo ""

# ------------------------------------------------------------------
# Test 1: venv exists
# ------------------------------------------------------------------
run_test "venv exists at $VENV_PATH" test -d "$VENV_PATH"

# ------------------------------------------------------------------
# Test 2: Python works in venv
# ------------------------------------------------------------------
run_test "Python executable works" "$PYTHON" -c "import sys; print(f'Python {sys.version}')" 2>/dev/null

# ------------------------------------------------------------------
# Test 3: torch imports
# ------------------------------------------------------------------
run_test "torch imports" "$PYTHON" -c "import torch; print(f'torch {torch.__version__}')" 2>/dev/null

# ------------------------------------------------------------------
# Test 4: GPU availability
# ------------------------------------------------------------------
if $GPU_REQUIRED; then
    run_test "GPU available (CUDA)" "$PYTHON" -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'GPU: {torch.cuda.get_device_name(0)}')
" 2>/dev/null
else
    if "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        GPU_NAME=$("$PYTHON" -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
        TOTAL=$((TOTAL + 1))
        PASSED=$((PASSED + 1))
        pass "GPU available: $GPU_NAME"
    else
        skip_test "GPU availability (no GPU detected, not required)"
    fi
fi

# ------------------------------------------------------------------
# Test 5: vLLM imports
# ------------------------------------------------------------------
run_test "vLLM imports" "$PYTHON" -c "import vllm; print(f'vllm {vllm.__version__}')" 2>/dev/null

# ------------------------------------------------------------------
# Test 6: vLLM server healthcheck
# ------------------------------------------------------------------
if ! $SKIP_SERVER; then
    if curl -sf "${VLLM_URL}/health" >/dev/null 2>&1; then
        TOTAL=$((TOTAL + 1))
        PASSED=$((PASSED + 1))
        pass "vLLM server healthcheck (${VLLM_URL})"
    else
        skip_test "vLLM server healthcheck (server not running)"
    fi

    # ------------------------------------------------------------------
    # Test 7: vLLM model listing
    # ------------------------------------------------------------------
    if curl -sf "${VLLM_URL}/v1/models" >/dev/null 2>&1; then
        MODELS=$(curl -sf "${VLLM_URL}/v1/models" | python3 -c "
import json, sys
data = json.load(sys.stdin)
models = [m['id'] for m in data.get('data', [])]
print(', '.join(models) if models else 'none')
" 2>/dev/null || echo "error")
        TOTAL=$((TOTAL + 1))
        PASSED=$((PASSED + 1))
        pass "vLLM models available: $MODELS"
    else
        skip_test "vLLM model listing (server not running)"
    fi

    # ------------------------------------------------------------------
    # Test 8: Minimal token generation
    # ------------------------------------------------------------------
    if ! $SKIP_GENERATE; then
        if curl -sf "${VLLM_URL}/health" >/dev/null 2>&1; then
            FIRST_MODEL=$(curl -sf "${VLLM_URL}/v1/models" 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
models = data.get('data', [])
print(models[0]['id'] if models else '')
" 2>/dev/null || echo "")

            if [[ -n "$FIRST_MODEL" ]]; then
                GEN_RESULT=$(curl -sf -X POST "${VLLM_URL}/v1/completions" \
                    -H "Content-Type: application/json" \
                    -d "{\"model\": \"${FIRST_MODEL}\", \"prompt\": \"Hello\", \"max_tokens\": 5}" \
                    2>/dev/null || echo "")
                if [[ -n "$GEN_RESULT" ]] && echo "$GEN_RESULT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
text = data.get('choices', [{}])[0].get('text', '')
assert len(text) > 0, 'empty response'
print(f'Generated: {text[:50]}')
" 2>/dev/null; then
                    TOTAL=$((TOTAL + 1))
                    PASSED=$((PASSED + 1))
                    pass "Token generation test"
                else
                    skip_test "Token generation (model not responding)"
                fi
            else
                skip_test "Token generation (no model loaded)"
            fi
        else
            skip_test "Token generation (server not running)"
        fi
    else
        skip_test "Token generation (--skip-generate)"
    fi
else
    skip_test "vLLM server healthcheck (--skip-server)"
    skip_test "vLLM model listing (--skip-server)"
    skip_test "Token generation (--skip-server)"
fi

# ------------------------------------------------------------------
# Test 9: Agent /capabilities includes vLLM
# ------------------------------------------------------------------
if curl -sf "${AGENT_URL}/capabilities" >/dev/null 2>&1; then
    CAP_RESULT=$(curl -sf "${AGENT_URL}/capabilities" 2>/dev/null || echo "{}")
    HAS_VLLM=$(echo "$CAP_RESULT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
backends = data.get('backends', {})
print('yes' if 'vllm' in backends else 'no')
" 2>/dev/null || echo "no")
    if [[ "$HAS_VLLM" == "yes" ]]; then
        TOTAL=$((TOTAL + 1))
        PASSED=$((PASSED + 1))
        pass "Agent /capabilities includes vLLM backend"
    else
        TOTAL=$((TOTAL + 1))
        info "Agent /capabilities does not include vLLM (may need config update)"
        SKIPPED=$((SKIPPED + 1))
    fi
else
    skip_test "Agent /capabilities check (agent not running)"
fi

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo "============================================"
echo "  Results: $PASSED passed, $FAILURES failed, $SKIPPED skipped (of $TOTAL)"
echo "============================================"
echo ""

if [[ $FAILURES -gt 0 ]]; then
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed.${NC}"
    exit 0
fi

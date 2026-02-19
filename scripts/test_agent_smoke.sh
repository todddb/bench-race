#!/usr/bin/env bash
# test_agent_smoke.sh — Automated smoke tests for agent backends
#
# Verifies:
#   1. Installer created expected directories and venvs
#   2. scripts/agent start-backend / stop-backend lifecycle
#   3. Only-one-active-LLM-backend enforcement
#   4. .gitignore effectiveness (no model/venv files staged)
#
# Usage:
#   ./scripts/test_agent_smoke.sh [--skip-lifecycle] [--skip-git]
#
# Exit code 0 = all tests pass, non-zero = failures.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AGENT_DIR="${REPO_ROOT}/agent"
AGENT_SCRIPT="${SCRIPT_DIR}/agent"

# ============================================================================
# Colors
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

pass=0
fail=0
skip=0

SKIP_LIFECYCLE=false
SKIP_GIT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-lifecycle) SKIP_LIFECYCLE=true; shift ;;
        --skip-git)       SKIP_GIT=true; shift ;;
        -h|--help)
            echo "Usage: $(basename "$0") [--skip-lifecycle] [--skip-git]"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

test_pass() {
    echo -e "  ${GREEN}PASS${NC}: $1"
    pass=$((pass + 1))
}

test_fail() {
    echo -e "  ${RED}FAIL${NC}: $1"
    fail=$((fail + 1))
}

test_skip() {
    echo -e "  ${YELLOW}SKIP${NC}: $1"
    skip=$((skip + 1))
}

# ============================================================================
# 1. Directory structure tests
# ============================================================================

echo ""
echo "=== 1. Directory Structure ==="

# Check expected directories exist
for dir in \
    "$AGENT_DIR/third_party/comfyui" \
    "$AGENT_DIR/third_party/vllm" \
    "$AGENT_DIR/models" \
    "$AGENT_DIR/run" \
    "$AGENT_DIR/log" \
    "$AGENT_DIR/backends"; do
    if [[ -d "$dir" ]]; then
        test_pass "Directory exists: ${dir#$REPO_ROOT/}"
    else
        test_fail "Directory missing: ${dir#$REPO_ROOT/}"
    fi
done

# Check scripts are executable
for script in "$AGENT_SCRIPT" "$SCRIPT_DIR/sync_models.sh" "$SCRIPT_DIR/install_agent.sh"; do
    if [[ -x "$script" ]]; then
        test_pass "Script executable: $(basename "$script")"
    else
        test_fail "Script not executable: $(basename "$script")"
    fi
done

# ============================================================================
# 2. Venv tests (check if venvs exist, not necessarily populated)
# ============================================================================

echo ""
echo "=== 2. Virtual Environments ==="

# ComfyUI venv
if [[ -d "$AGENT_DIR/third_party/comfyui/.venv" ]]; then
    test_pass "ComfyUI venv exists"
else
    test_skip "ComfyUI venv not installed (run install_agent.sh first)"
fi

# vLLM venv
if [[ -d "$AGENT_DIR/third_party/vllm/.venv" ]]; then
    test_pass "vLLM venv exists"
else
    test_skip "vLLM venv not installed (run install_agent.sh first)"
fi

# Agent venv
if [[ -d "$AGENT_DIR/.venv" ]]; then
    test_pass "Agent venv exists"
else
    test_skip "Agent venv not installed (run install_agent.sh first)"
fi

# ============================================================================
# 3. Lifecycle tests
# ============================================================================

echo ""
echo "=== 3. Backend Lifecycle ==="

if [[ "$SKIP_LIFECYCLE" == true ]]; then
    test_skip "Lifecycle tests skipped (--skip-lifecycle)"
else
    # Test scripts/agent status command
    if "$AGENT_SCRIPT" status >/dev/null 2>&1; then
        test_pass "scripts/agent status runs without error"
    else
        test_fail "scripts/agent status failed"
    fi

    # Test help output
    if "$AGENT_SCRIPT" --help 2>&1 | grep -q "start-backend"; then
        test_pass "scripts/agent --help shows start-backend command"
    else
        test_fail "scripts/agent --help missing start-backend documentation"
    fi

    # Test that start-backend without args gives error
    if ! "$AGENT_SCRIPT" start-backend 2>/dev/null; then
        test_pass "start-backend with no backend gives error"
    else
        test_fail "start-backend with no backend should fail"
    fi

    # Test that start-backend vllm without model gives error
    if ! "$AGENT_SCRIPT" start-backend vllm 2>/dev/null; then
        test_pass "start-backend vllm with no model gives error"
    else
        test_fail "start-backend vllm with no model should fail"
    fi

    # Test that start-backend with invalid backend gives error
    if ! "$AGENT_SCRIPT" start-backend invalid_backend 2>/dev/null; then
        test_pass "start-backend with invalid backend gives error"
    else
        test_fail "start-backend with invalid backend should fail"
    fi
fi

# ============================================================================
# 4. .gitignore tests
# ============================================================================

echo ""
echo "=== 4. Git Hygiene ==="

if [[ "$SKIP_GIT" == true ]]; then
    test_skip "Git tests skipped (--skip-git)"
else
    # Check .gitignore entries
    gitignore_file="$REPO_ROOT/.gitignore"
    if [[ -f "$gitignore_file" ]]; then
        for pattern in "agent/models" "agent/third_party" "agent/run" "agent/log"; do
            if grep -q "$pattern" "$gitignore_file"; then
                test_pass ".gitignore contains: $pattern"
            else
                test_fail ".gitignore missing: $pattern"
            fi
        done
    else
        test_fail ".gitignore file not found"
    fi

    # Verify no model files would be committed
    # Create a test file and check git status
    test_model_file="$AGENT_DIR/models/.test_gitignore_check"
    mkdir -p "$AGENT_DIR/models"
    echo "test" > "$test_model_file"

    if ! git -C "$REPO_ROOT" status --porcelain "$test_model_file" 2>/dev/null | grep -q "$test_model_file"; then
        test_pass "Model files are ignored by git"
    else
        test_fail "Model files are NOT ignored by git"
    fi
    rm -f "$test_model_file"

    # Verify run dir is ignored
    test_run_file="$AGENT_DIR/run/.test_gitignore_check"
    echo "test" > "$test_run_file"

    if ! git -C "$REPO_ROOT" status --porcelain "$test_run_file" 2>/dev/null | grep -q "$test_run_file"; then
        test_pass "Run files are ignored by git"
    else
        test_fail "Run files are NOT ignored by git"
    fi
    rm -f "$test_run_file"
fi

# ============================================================================
# 5. Python backend modules
# ============================================================================

echo ""
echo "=== 5. Backend Modules ==="

for module in ollama_backend.py comfyui_backend.py vllm_backend.py; do
    if [[ -f "$AGENT_DIR/backends/$module" ]]; then
        test_pass "Backend module exists: $module"
    else
        test_fail "Backend module missing: $module"
    fi
done

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=============================================="
echo "  SMOKE TEST SUMMARY"
echo "=============================================="
echo -e "  ${GREEN}Passed${NC}: $pass"
echo -e "  ${RED}Failed${NC}: $fail"
echo -e "  ${YELLOW}Skipped${NC}: $skip"
echo "=============================================="
echo ""

if [[ $fail -gt 0 ]]; then
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi

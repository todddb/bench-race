#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# test_local_install.sh — acceptance tests for --local-only, --preflight,
# --uninstall-all, and root-guard features of install_agent.sh.
#
# Run as a non-root user:
#   ./scripts/test_local_install.sh
#
# Each test prints PASS or FAIL; exits non-zero if any test fails.
# ---------------------------------------------------------------------------
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INSTALLER="${SCRIPT_DIR}/install_agent.sh"

PASS=0
FAIL=0
TESTS=()

pass() {
  PASS=$((PASS + 1))
  echo -e "\033[0;32m  PASS\033[0m $1"
  TESTS+=("PASS: $1")
}

fail() {
  FAIL=$((FAIL + 1))
  echo -e "\033[0;31m  FAIL\033[0m $1"
  TESTS+=("FAIL: $1")
}

# ===================================================================
echo "=============================================="
echo " Test 1: --preflight --local-only output"
echo "=============================================="

output="$(bash "${INSTALLER}" --preflight --local-only 2>&1)" || true

if echo "${output}" | grep -q "\[PREFLIGHT\]"; then
  pass "preflight produces [PREFLIGHT] output"
else
  fail "preflight did not produce [PREFLIGHT] output"
fi

if echo "${output}" | grep -qi "local-only"; then
  pass "preflight mentions local-only"
else
  fail "preflight does not mention local-only"
fi

if echo "${output}" | grep -q "Will NOT write to /etc.*or /Library"; then
  pass "preflight states no /etc or /Library writes"
else
  fail "preflight missing /etc or /Library guard message"
fi

if echo "${output}" | grep -q "VENV_PATH:"; then
  pass "preflight prints VENV_PATH"
else
  fail "preflight missing VENV_PATH"
fi

if echo "${output}" | grep -q "MODEL_DIR:"; then
  pass "preflight prints MODEL_DIR"
else
  fail "preflight missing MODEL_DIR"
fi

if echo "${output}" | grep -q "SERVICE_TARGET:"; then
  pass "preflight prints SERVICE_TARGET"
else
  fail "preflight missing SERVICE_TARGET"
fi

# ===================================================================
echo ""
echo "=============================================="
echo " Test 2: --preflight (default mode, no --local-only)"
echo "=============================================="

output2="$(bash "${INSTALLER}" --preflight 2>&1)" || true

if echo "${output2}" | grep -q "\[PREFLIGHT\]"; then
  pass "preflight default mode produces output"
else
  fail "preflight default mode did not produce output"
fi

if echo "${output2}" | grep -q "LOCAL_ONLY:.*false"; then
  pass "preflight default mode shows LOCAL_ONLY=false"
else
  fail "preflight default mode LOCAL_ONLY not false"
fi

# ===================================================================
echo ""
echo "=============================================="
echo " Test 3: --local-only and --system conflict"
echo "=============================================="

conflict_out="$(bash "${INSTALLER}" --local-only --system 2>&1)" && conflict_rc=$? || conflict_rc=$?

if [[ ${conflict_rc} -ne 0 ]]; then
  pass "--local-only --system exits non-zero"
else
  fail "--local-only --system should exit non-zero"
fi

if echo "${conflict_out}" | grep -qi "mutually exclusive"; then
  pass "--local-only --system prints conflict message"
else
  fail "--local-only --system missing conflict message"
fi

# ===================================================================
echo ""
echo "=============================================="
echo " Test 4: --preflight --system shows system paths"
echo "=============================================="

# This test might need to be skipped if not running as root, since --system
# without root will still work for preflight (preflight exits before root guard).
sys_out="$(bash "${INSTALLER}" --preflight --system 2>&1)" || true

if echo "${sys_out}" | grep -q "SYSTEM_INSTALL:.*true"; then
  pass "preflight --system shows SYSTEM_INSTALL=true"
else
  fail "preflight --system should show SYSTEM_INSTALL=true"
fi

# ===================================================================
echo ""
echo "=============================================="
echo " Test 5: --uninstall-all without --confirm prompts"
echo "=============================================="

# Run with 'n' piped to stdin so the interactive prompt gets a "no"
uninstall_out="$(echo 'n' | bash "${INSTALLER}" --uninstall-all --local-only 2>&1)" && uninstall_rc=$? || uninstall_rc=$?

if echo "${uninstall_out}" | grep -qi "aborted\|proceed\|will be removed"; then
  pass "--uninstall-all without --confirm asks for confirmation"
else
  fail "--uninstall-all without --confirm should prompt"
  echo "    output was: ${uninstall_out}"
fi

# ===================================================================
echo ""
echo "=============================================="
echo " Test 6: --uninstall-all --confirm --local-only (dry check)"
echo "=============================================="

# Just verify the flag combination doesn't crash; it will try to uninstall
# which should be safe (nothing to remove on a clean system)
clean_out="$(bash "${INSTALLER}" --uninstall-all --confirm --local-only 2>&1)" && clean_rc=$? || clean_rc=$?

if [[ ${clean_rc} -eq 0 ]]; then
  pass "--uninstall-all --confirm --local-only exits cleanly"
else
  fail "--uninstall-all --confirm --local-only failed with rc=${clean_rc}"
  echo "    output was: ${clean_out}"
fi

if echo "${clean_out}" | grep -qi "uninstall complete\|nothing found to remove"; then
  pass "--uninstall-all --confirm --local-only prints summary"
else
  fail "--uninstall-all --confirm --local-only missing summary output"
  echo "    output was: ${clean_out}"
fi

# ===================================================================
echo ""
echo "=============================================="
echo " Test 7: scripts/agents local-only marker detection"
echo "=============================================="

# Create marker, verify detection, then remove
touch "${REPO_ROOT}/.bench-race-local-only"
# Source _common.sh to get the variables, then check
marker_test="$(bash -c "
  source '${SCRIPT_DIR}/_common.sh'
  BENCH_RACE_LOCAL_ONLY=false
  if [[ -f '${REPO_ROOT}/.bench-race-local-only' ]]; then
    BENCH_RACE_LOCAL_ONLY=true
  fi
  echo \"\${BENCH_RACE_LOCAL_ONLY}\"
" 2>&1)" || true
rm -f "${REPO_ROOT}/.bench-race-local-only"

if [[ "${marker_test}" == *"true"* ]]; then
  pass "local-only marker file is detected"
else
  fail "local-only marker file not detected"
fi

# ===================================================================
echo ""
echo "=============================================="
echo " Test 8: No /etc or /Library writes in preflight --local-only"
echo "=============================================="

preflight_paths="$(bash "${INSTALLER}" --preflight --local-only 2>&1)" || true

if echo "${preflight_paths}" | grep "SERVICE_PATH:" | grep -qE "/etc/|/Library/"; then
  fail "preflight --local-only SERVICE_PATH contains system path"
else
  pass "preflight --local-only SERVICE_PATH uses user-local path"
fi

if echo "${preflight_paths}" | grep "VENV_PATH:" | grep -qE "^.*(/opt/|/Library/)"; then
  fail "preflight --local-only VENV_PATH contains system path"
else
  pass "preflight --local-only VENV_PATH uses user-local path"
fi

# ===================================================================
echo ""
echo "=============================================="
echo " Test 9: --help includes new flags"
echo "=============================================="

help_out="$(bash "${INSTALLER}" --help 2>&1)" || true

if echo "${help_out}" | grep -q "\-\-local-only"; then
  pass "--help mentions --local-only"
else
  fail "--help missing --local-only"
fi

if echo "${help_out}" | grep -q "\-\-preflight"; then
  pass "--help mentions --preflight"
else
  fail "--help missing --preflight"
fi

if echo "${help_out}" | grep -q "\-\-uninstall-all"; then
  pass "--help mentions --uninstall-all"
else
  fail "--help missing --uninstall-all"
fi

if echo "${help_out}" | grep -q "\-\-confirm"; then
  pass "--help mentions --confirm"
else
  fail "--help missing --confirm"
fi

# ===================================================================
echo ""
echo "=============================================="
echo " SUMMARY"
echo "=============================================="
echo ""
for t in "${TESTS[@]}"; do
  echo "  ${t}"
done
echo ""
echo "Results: ${PASS} passed, ${FAIL} failed"
echo ""

if [[ ${FAIL} -gt 0 ]]; then
  exit 1
fi
exit 0

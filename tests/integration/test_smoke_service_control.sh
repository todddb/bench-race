#!/usr/bin/env bash
# Integration smoke test for service control
#
# This script tests the full start -> status -> stop cycle for services.
# It verifies that the control CLI works correctly with actual processes.
#
# Usage:
#   bash tests/integration/test_smoke_service_control.sh
#
# Exit codes:
#   0 - All tests passed
#   1 - Test failed

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTROL_CLI="${REPO_ROOT}/bin/control"

# Configuration
TIMEOUT=10
POLL_INTERVAL=0.5

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $*"
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $*"
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $*"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    if [[ ! -x "${CONTROL_CLI}" ]]; then
        log_fail "Control CLI not found or not executable: ${CONTROL_CLI}"
        exit 1
    fi

    if [[ ! -d "${REPO_ROOT}/agent/.venv" ]]; then
        log_skip "Agent venv not found. Skipping agent tests."
        SKIP_AGENT=true
    else
        SKIP_AGENT=false
    fi

    if [[ ! -d "${REPO_ROOT}/central/.venv" ]]; then
        log_skip "Central venv not found. Skipping central tests."
        SKIP_CENTRAL=true
    else
        SKIP_CENTRAL=false
    fi

    if $SKIP_AGENT && $SKIP_CENTRAL; then
        log_fail "No venvs found. Cannot run any tests."
        exit 1
    fi

    log_pass "Prerequisites check passed"
}

# Wait for service to be in expected state
wait_for_state() {
    local component="$1"
    local expected_running="$2"
    local timeout_seconds="${3:-$TIMEOUT}"

    local elapsed=0
    while (( elapsed < timeout_seconds )); do
        local status_json
        status_json=$("${CONTROL_CLI}" "${component}" status --json 2>/dev/null || echo '{"running": null}')

        local running
        running=$(echo "${status_json}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(str(d.get('running','')).lower())")

        if [[ "${running}" == "${expected_running}" ]]; then
            return 0
        fi

        sleep "${POLL_INTERVAL}"
        elapsed=$(echo "${elapsed} + ${POLL_INTERVAL}" | bc)
    done

    return 1
}

# Wait for port to be listening
wait_for_port() {
    local port="$1"
    local timeout_seconds="${2:-$TIMEOUT}"

    local elapsed=0
    while (( elapsed < timeout_seconds )); do
        if command -v lsof >/dev/null 2>&1; then
            if lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
                return 0
            fi
        elif command -v ss >/dev/null 2>&1; then
            if ss -tlnp | grep -q ":${port}"; then
                return 0
            fi
        elif command -v netstat >/dev/null 2>&1; then
            if netstat -tlnp 2>/dev/null | grep -q ":${port}"; then
                return 0
            fi
        else
            # Fallback: try to connect
            if python3 -c "import socket; s=socket.socket(); s.settimeout(1); exit(0 if s.connect_ex(('127.0.0.1', ${port}))==0 else 1)" 2>/dev/null; then
                return 0
            fi
        fi

        sleep "${POLL_INTERVAL}"
        elapsed=$(echo "${elapsed} + ${POLL_INTERVAL}" | bc)
    done

    return 1
}

# Test agent start/status/stop cycle
test_agent_cycle() {
    log "=== Testing Agent Start/Status/Stop Cycle ==="

    if $SKIP_AGENT; then
        log_skip "Agent venv not found"
        return 0
    fi

    # Ensure agent is stopped first
    log "Ensuring agent is stopped..."
    "${CONTROL_CLI}" agent stop --json >/dev/null 2>&1 || true
    sleep 1

    # Verify stopped
    if ! wait_for_state agent "false" 5; then
        log_fail "Agent did not stop in cleanup phase"
        return 1
    fi

    # Start agent
    log "Starting agent..."
    local start_output
    start_output=$("${CONTROL_CLI}" agent start --json 2>&1)
    log "Start output: ${start_output}"

    local start_result
    start_result=$(echo "${start_output}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('result',''))")

    if [[ "${start_result}" != "started" && "${start_result}" != "already_running" ]]; then
        log_fail "Agent start failed: ${start_output}"
        return 1
    fi

    log_pass "Agent start command succeeded"

    # Wait for agent to be running
    log "Waiting for agent to be running..."
    if ! wait_for_state agent "true" "${TIMEOUT}"; then
        log_fail "Agent did not start within ${TIMEOUT}s"
        "${CONTROL_CLI}" agent stop >/dev/null 2>&1 || true
        return 1
    fi

    log_pass "Agent is running"

    # Check status reports PID
    log "Checking status..."
    local status_output
    status_output=$("${CONTROL_CLI}" agent status --json 2>&1)

    local pid
    pid=$(echo "${status_output}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('pid',''))")

    if [[ -z "${pid}" || "${pid}" == "None" ]]; then
        log_fail "Status does not report PID: ${status_output}"
        "${CONTROL_CLI}" agent stop >/dev/null 2>&1 || true
        return 1
    fi

    log_pass "Status reports PID: ${pid}"

    # Wait for port 9001 to be listening
    log "Waiting for agent to listen on port 9001..."
    if wait_for_port 9001 "${TIMEOUT}"; then
        log_pass "Agent is listening on port 9001"
    else
        log "Agent not listening on 9001 (may be expected if Ollama is not running)"
    fi

    # Stop agent
    log "Stopping agent..."
    local stop_output
    stop_output=$("${CONTROL_CLI}" agent stop --json 2>&1)

    local stop_result
    stop_result=$(echo "${stop_output}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('result',''))")

    if [[ "${stop_result}" != "stopped" && "${stop_result}" != "not_running" ]]; then
        log_fail "Agent stop failed: ${stop_output}"
        return 1
    fi

    log_pass "Agent stop command succeeded"

    # Verify stopped
    log "Verifying agent is stopped..."
    if ! wait_for_state agent "false" "${TIMEOUT}"; then
        log_fail "Agent did not stop within ${TIMEOUT}s"
        return 1
    fi

    log_pass "Agent is stopped"

    log_pass "Agent cycle test passed!"
    return 0
}

# Test central start/status/stop cycle
test_central_cycle() {
    log "=== Testing Central Start/Status/Stop Cycle ==="

    if $SKIP_CENTRAL; then
        log_skip "Central venv not found"
        return 0
    fi

    # Ensure central is stopped first
    log "Ensuring central is stopped..."
    "${CONTROL_CLI}" central stop --json >/dev/null 2>&1 || true
    sleep 1

    # Verify stopped
    if ! wait_for_state central "false" 5; then
        log_fail "Central did not stop in cleanup phase"
        return 1
    fi

    # Start central
    log "Starting central..."
    local start_output
    start_output=$("${CONTROL_CLI}" central start --json 2>&1)
    log "Start output: ${start_output}"

    local start_result
    start_result=$(echo "${start_output}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('result',''))")

    if [[ "${start_result}" != "started" && "${start_result}" != "already_running" ]]; then
        log_fail "Central start failed: ${start_output}"
        return 1
    fi

    log_pass "Central start command succeeded"

    # Wait for central to be running
    log "Waiting for central to be running..."
    if ! wait_for_state central "true" "${TIMEOUT}"; then
        log_fail "Central did not start within ${TIMEOUT}s"
        "${CONTROL_CLI}" central stop >/dev/null 2>&1 || true
        return 1
    fi

    log_pass "Central is running"

    # Check status reports PID
    log "Checking status..."
    local status_output
    status_output=$("${CONTROL_CLI}" central status --json 2>&1)

    local pid
    pid=$(echo "${status_output}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('pid',''))")

    if [[ -z "${pid}" || "${pid}" == "None" ]]; then
        log_fail "Status does not report PID: ${status_output}"
        "${CONTROL_CLI}" central stop >/dev/null 2>&1 || true
        return 1
    fi

    log_pass "Status reports PID: ${pid}"

    # Wait for port 8080 to be listening
    log "Waiting for central to listen on port 8080..."
    if wait_for_port 8080 "${TIMEOUT}"; then
        log_pass "Central is listening on port 8080"

        # Test API endpoint
        log "Testing API status endpoint..."
        if command -v curl >/dev/null 2>&1; then
            local api_response
            api_response=$(curl -s "http://127.0.0.1:8080/api/service/agent/status" 2>/dev/null || echo '{}')
            if echo "${api_response}" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if 'component' in d else 1)" 2>/dev/null; then
                log_pass "API endpoint /api/service/agent/status works"
            else
                log "API endpoint returned unexpected response: ${api_response}"
            fi
        else
            log "curl not available, skipping API test"
        fi
    else
        log "Central not listening on 8080 (may be expected depending on config)"
    fi

    # Stop central
    log "Stopping central..."
    local stop_output
    stop_output=$("${CONTROL_CLI}" central stop --json 2>&1)

    local stop_result
    stop_result=$(echo "${stop_output}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('result',''))")

    if [[ "${stop_result}" != "stopped" && "${stop_result}" != "not_running" ]]; then
        log_fail "Central stop failed: ${stop_output}"
        return 1
    fi

    log_pass "Central stop command succeeded"

    # Verify stopped
    log "Verifying central is stopped..."
    if ! wait_for_state central "false" "${TIMEOUT}"; then
        log_fail "Central did not stop within ${TIMEOUT}s"
        return 1
    fi

    log_pass "Central is stopped"

    log_pass "Central cycle test passed!"
    return 0
}

# Test idempotency
test_idempotency() {
    log "=== Testing Idempotency ==="

    # Test stop when not running (should succeed)
    log "Testing stop when not running..."

    "${CONTROL_CLI}" agent stop --json >/dev/null 2>&1 || true
    local stop_output
    stop_output=$("${CONTROL_CLI}" agent stop --json 2>&1)

    local stop_result
    stop_result=$(echo "${stop_output}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('result',''))")

    if [[ "${stop_result}" != "not_running" && "${stop_result}" != "stopped" ]]; then
        log_fail "Stop when not running should succeed: ${stop_output}"
        return 1
    fi

    log_pass "Stop when not running is idempotent"

    log_pass "Idempotency tests passed!"
    return 0
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    "${CONTROL_CLI}" agent stop --json >/dev/null 2>&1 || true
    "${CONTROL_CLI}" central stop --json >/dev/null 2>&1 || true
}

# Main
main() {
    log "=========================================="
    log "Service Control Integration Smoke Test"
    log "=========================================="
    log ""

    # Set up cleanup trap
    trap cleanup EXIT

    cd "${REPO_ROOT}"

    local failed=0

    check_prerequisites

    echo ""
    if ! test_idempotency; then
        ((failed++))
    fi

    echo ""
    if ! test_agent_cycle; then
        ((failed++))
    fi

    echo ""
    if ! test_central_cycle; then
        ((failed++))
    fi

    echo ""
    log "=========================================="
    if [[ ${failed} -eq 0 ]]; then
        log_pass "All smoke tests passed!"
        exit 0
    else
        log_fail "${failed} test(s) failed"
        exit 1
    fi
}

main "$@"

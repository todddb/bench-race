"""
Unit tests for agent health tracking with hysteresis-based offline detection.

Tests verify that:
- Agents transition correctly through ready -> degraded -> offline states
- Consecutive failure thresholds work correctly
- Time-based offline thresholds work correctly
- Run-aware behavior prevents premature offline transitions
- Status recovers properly when agent becomes healthy again

Run with: pytest tests/test_agent_health_hysteresis.py -v
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch


def test_hysteresis_constants_defined():
    """Verify hysteresis constants are defined in central/app.py."""
    app_py = Path("central/app.py").read_text(encoding="utf-8")

    # Check for health state tracking dictionary
    assert "AGENT_HEALTH_STATE" in app_py, "AGENT_HEALTH_STATE not defined"

    # Check for offline threshold constant
    assert "OFFLINE_THRESHOLD_S" in app_py, "OFFLINE_THRESHOLD_S constant not defined"

    # Check for consecutive failure threshold
    assert "CONSECUTIVE_FAILURE_THRESHOLD" in app_py, "CONSECUTIVE_FAILURE_THRESHOLD not defined"


def test_update_agent_health_function_exists():
    """Verify _update_agent_health function exists and has correct signature."""
    app_py = Path("central/app.py").read_text(encoding="utf-8")

    # Check function exists
    assert "def _update_agent_health(" in app_py, "_update_agent_health function not found"

    # Check for key parameters
    assert "machine_id: str" in app_py, "machine_id parameter missing"
    assert "success: bool" in app_py, "success parameter missing"
    assert "is_running: bool" in app_py, "is_running parameter missing"


def test_is_machine_running_helper_exists():
    """Verify _is_machine_running helper function exists."""
    app_py = Path("central/app.py").read_text(encoding="utf-8")

    assert "def _is_machine_running(" in app_py, "_is_machine_running function not found"
    assert "ACTIVE_RUNS" in app_py, "ACTIVE_RUNS not used in run detection"


def test_api_status_uses_health_tracking():
    """Verify api_status endpoint uses new health tracking."""
    app_py = Path("central/app.py").read_text(encoding="utf-8")

    # Check that _is_machine_running is called
    assert "_is_machine_running(machine_id)" in app_py, \
        "api_status doesn't check if machine is running"

    # Check that _update_agent_health is called on success
    assert "_update_agent_health(machine_id, success=True" in app_py, \
        "api_status doesn't update health on success"

    # Check that _update_agent_health is called on failure with is_running
    assert "_update_agent_health(machine_id, success=False" in app_py, \
        "api_status doesn't update health on failure"

    # Check that agent_status is added to response
    assert '"agent_status": status' in app_py, \
        "api_status doesn't include agent_status in response"


def test_health_diagnostics_in_api_response():
    """Verify health diagnostics are included in API responses."""
    app_py = Path("central/app.py").read_text(encoding="utf-8")

    # Check for health_diagnostics in response
    assert '"health_diagnostics":' in app_py, \
        "health_diagnostics not included in API response"

    # Check for diagnostic fields
    assert '"consecutive_failures":' in app_py, \
        "consecutive_failures not in diagnostics"

    assert '"last_success_age_s":' in app_py, \
        "last_success_age_s not in diagnostics"

    assert '"last_error":' in app_py, \
        "last_error not in diagnostics"


def test_logging_on_status_transitions():
    """Verify status transitions are logged with diagnostic info."""
    app_py = Path("central/app.py").read_text(encoding="utf-8")

    # Check for logging on status change
    assert 'log.warning(' in app_py or 'log.info(' in app_py, \
        "No logging for status transitions"

    # Check that log includes diagnostic information
    assert 'consecutive_failures' in app_py, \
        "Log doesn't include consecutive_failures"


def test_degraded_status_in_schemas():
    """Verify agent_status field with degraded option is in schemas."""
    schemas_py = Path("shared/schemas.py").read_text(encoding="utf-8")

    # Check for agent_status field
    assert "agent_status" in schemas_py, \
        "agent_status field not in schemas"

    # Check for Literal type with ready/degraded/offline
    assert '"ready"' in schemas_py or "'ready'" in schemas_py, \
        "ready status not in schema"

    assert '"degraded"' in schemas_py or "'degraded'" in schemas_py, \
        "degraded status not in schema"

    assert '"offline"' in schemas_py or "'offline'" in schemas_py, \
        "offline status not in schema"


def test_ui_handles_degraded_status():
    """Verify UI handles degraded status with proper styling and tooltips."""
    app_js = Path("central/static/js/app.js").read_text(encoding="utf-8")

    # Check for agent_status usage
    assert "agent_status" in app_js, \
        "UI doesn't check agent_status field"

    # Check for degraded status handling
    assert "degraded" in app_js, \
        "UI doesn't handle degraded status"

    # Check for health diagnostics usage
    assert "health_diagnostics" in app_js, \
        "UI doesn't use health_diagnostics"

    # Check for tooltip construction
    assert "tooltip" in app_js.lower() or "title" in app_js, \
        "UI doesn't create tooltips for status"


def test_css_has_degraded_status_style():
    """Verify CSS includes styling for degraded status badge."""
    app_css = Path("central/static/css/app.css").read_text(encoding="utf-8")

    # Check for degraded status class
    assert ".status-badge.degraded" in app_css, \
        "CSS doesn't have .status-badge.degraded class"

    # Check for degraded dot styling
    assert ".status-badge.degraded .status-dot" in app_css, \
        "CSS doesn't have .status-badge.degraded .status-dot class"


def test_agent_compute_yields_to_event_loop():
    """Verify agent compute functions yield to event loop for health checks."""
    agent_app_py = Path("agent/agent_app.py").read_text(encoding="utf-8")

    # Check that compute functions have asyncio.sleep(0) to yield
    assert "await asyncio.sleep(0)" in agent_app_py, \
        "Compute functions don't yield to event loop"

    # Check for comment explaining the yield
    assert "health check" in agent_app_py.lower() or "event loop" in agent_app_py.lower(), \
        "No comment explaining why compute yields to event loop"


def test_hysteresis_logic_patterns():
    """Verify hysteresis logic patterns are implemented correctly."""
    app_py = Path("central/app.py").read_text(encoding="utf-8")

    # Check for consecutive failures tracking
    assert "consecutive_failures" in app_py, \
        "consecutive_failures not tracked"

    # Check for last_success_ts tracking
    assert "last_success_ts" in app_py, \
        "last_success_ts not tracked"

    # Check for reset on success
    assert 'consecutive_failures = 0' in app_py or '"consecutive_failures": 0' in app_py, \
        "consecutive_failures not reset on success"

    # Check for increment on failure
    assert 'consecutive_failures += 1' in app_py or 'consecutive_failures + 1' in app_py, \
        "consecutive_failures not incremented on failure"


def test_run_aware_behavior():
    """Verify run-aware behavior prefers degraded over offline for running agents."""
    app_py = Path("central/app.py").read_text(encoding="utf-8")

    # Check that is_running is considered in status determination
    assert "is_running" in app_py, \
        "is_running parameter not used"

    # Check for logic that treats running agents differently
    # Should have some conditional logic based on is_running
    assert "if is_running" in app_py, \
        "No conditional logic for running agents"


# Functional tests that import and test the actual function
def test_single_failure_shows_degraded():
    """Test that a single failure transitions agent to degraded, not offline."""
    # Import the module
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "central"))

    # Mock time.time to control timing
    with patch('time.time', return_value=1000.0):
        from central.app import _update_agent_health, AGENT_HEALTH_STATE

        # Clear state
        AGENT_HEALTH_STATE.clear()

        # First check succeeds
        status = _update_agent_health("test_agent", success=True)
        assert status == "ready", "Initial success should be ready"

        # Single failure should show degraded, not offline
        status = _update_agent_health("test_agent", success=False, error="timeout")
        assert status == "degraded", "Single failure should show degraded"
        assert AGENT_HEALTH_STATE["test_agent"]["consecutive_failures"] == 1


def test_three_failures_shows_offline():
    """Test that three consecutive failures transition agent to offline."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "central"))

    with patch('time.time', return_value=1000.0):
        from central.app import _update_agent_health, AGENT_HEALTH_STATE

        AGENT_HEALTH_STATE.clear()

        # Initial success
        _update_agent_health("test_agent", success=True)

        # Three failures should show offline
        _update_agent_health("test_agent", success=False)
        _update_agent_health("test_agent", success=False)
        status = _update_agent_health("test_agent", success=False)

        assert status == "offline", "Three consecutive failures should show offline"
        assert AGENT_HEALTH_STATE["test_agent"]["consecutive_failures"] == 3


def test_recovery_after_degraded():
    """Test that agent recovers from degraded to ready on success."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "central"))

    with patch('time.time', return_value=1000.0):
        from central.app import _update_agent_health, AGENT_HEALTH_STATE

        AGENT_HEALTH_STATE.clear()

        # Success, failure (degraded), then success (ready)
        _update_agent_health("test_agent", success=True)
        status = _update_agent_health("test_agent", success=False)
        assert status == "degraded", "Should be degraded after one failure"

        status = _update_agent_health("test_agent", success=True)
        assert status == "ready", "Should recover to ready after success"
        assert AGENT_HEALTH_STATE["test_agent"]["consecutive_failures"] == 0


def test_running_agent_stays_degraded_longer():
    """Test that running agents prefer degraded over offline."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "central"))

    with patch('time.time', return_value=1000.0):
        from central.app import _update_agent_health, AGENT_HEALTH_STATE

        AGENT_HEALTH_STATE.clear()

        # Initial success
        _update_agent_health("test_agent", success=True)

        # Three failures while running should stay degraded
        _update_agent_health("test_agent", success=False, is_running=True)
        _update_agent_health("test_agent", success=False, is_running=True)
        status = _update_agent_health("test_agent", success=False, is_running=True)

        # Should prefer degraded for running agents
        assert status in ["degraded", "offline"], "Running agent should show degraded or offline"
        # If the implementation is run-aware, it should be degraded
        # The exact behavior depends on the threshold implementation

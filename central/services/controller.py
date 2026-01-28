"""
Service controller - wraps bin/control CLI for programmatic access.

This module provides Python functions to control agent and central services
by invoking the unified control CLI.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Literal

# Resolve paths
SERVICE_DIR = Path(__file__).resolve().parent
CENTRAL_DIR = SERVICE_DIR.parent
REPO_ROOT = CENTRAL_DIR.parent
CONTROL_CLI = REPO_ROOT / "bin" / "control"

ComponentType = Literal["agent", "central"]
ActionType = Literal["start", "stop"]


def _run_control(component: ComponentType, action: str, extra_args: list = None) -> Dict[str, Any]:
    """Run the control CLI and parse JSON output."""
    cmd = [str(CONTROL_CLI), component, action, "--json"]
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(REPO_ROOT),
        )
        # Parse JSON output
        try:
            output = json.loads(result.stdout.strip()) if result.stdout.strip() else {}
        except json.JSONDecodeError:
            output = {
                "component": component,
                "action": action,
                "result": "error",
                "message": result.stdout or result.stderr or "Unknown error",
            }
        return output
    except subprocess.TimeoutExpired:
        return {
            "component": component,
            "action": action,
            "result": "error",
            "message": "Operation timed out",
        }
    except FileNotFoundError:
        return {
            "component": component,
            "action": action,
            "result": "error",
            "message": f"Control CLI not found: {CONTROL_CLI}",
        }
    except Exception as e:
        return {
            "component": component,
            "action": action,
            "result": "error",
            "message": str(e),
        }


def start_service(component: ComponentType) -> Dict[str, Any]:
    """
    Start a service component.

    Args:
        component: 'agent' or 'central'

    Returns:
        Dict with keys: component, action, result, pid (optional), message
    """
    return _run_control(component, "start")


def stop_service(component: ComponentType) -> Dict[str, Any]:
    """
    Stop a service component.

    Args:
        component: 'agent' or 'central'

    Returns:
        Dict with keys: component, action, result, message
    """
    return _run_control(component, "stop")


def get_status(component: ComponentType) -> Dict[str, Any]:
    """
    Get status of a service component.

    Args:
        component: 'agent' or 'central'

    Returns:
        Dict with keys: component, running, pid (optional), info (optional)
    """
    return _run_control(component, "status")


def perform_action(component: ComponentType, action: ActionType) -> Dict[str, Any]:
    """
    Perform an action on a service component.

    Args:
        component: 'agent' or 'central'
        action: 'start' or 'stop'

    Returns:
        Dict with action result
    """
    if action == "start":
        return start_service(component)
    elif action == "stop":
        return stop_service(component)
    else:
        return {
            "component": component,
            "action": action,
            "result": "error",
            "message": f"Unknown action: {action}",
        }

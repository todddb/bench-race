"""
Reset helper utilities for capturing subprocess output and persisting logs.

This module provides utilities for:
- Capturing stdout/stderr from subprocess operations
- Persisting logs to timestamped files
- Returning tails of log output for immediate diagnostics
"""

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import asyncio


# Friendly sudo failure message for non-interactive usage
SUDO_NONINTERACTIVE_HINT = (
    "Non-interactive sudo failed (sudo -n). "
    "Configure passwordless sudo for the benchrace-agent group or run the command as root. "
    "See /etc/sudoers.d/bench-race-agent."
)

# Configuration from environment variables
OLLAMA_START_TIMEOUT_S = float(os.getenv("OLLAMA_START_TIMEOUT_S", "120"))
COMFYUI_START_TIMEOUT_S = float(os.getenv("COMFYUI_START_TIMEOUT_S", "60"))
HEALTH_POLL_INTERVAL_S = float(os.getenv("HEALTH_POLL_INTERVAL_S", "1.5"))
RESET_LOG_DIR = Path(os.getenv("RESET_LOG_DIR", "./logs"))

# Ensure log directory exists
RESET_LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_timestamp() -> str:
    """Get ISO timestamp for log filenames."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")


def tail_lines(text: str, max_lines: int = 200, max_bytes: int = 32768) -> str:
    """
    Return the last N lines or max_bytes of text, whichever is smaller.

    Args:
        text: Full text content
        max_lines: Maximum number of lines to return
        max_bytes: Maximum bytes to return

    Returns:
        Tailed text content
    """
    if not text:
        return ""

    # First truncate by bytes if needed
    if len(text) > max_bytes:
        text = text[-max_bytes:]

    # Then truncate by lines
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]

    return "\n".join(lines)


def is_sudo_noninteractive_error(stderr_text: str) -> bool:
    """Return True if stderr looks like a sudo -n non-interactive failure."""
    if not stderr_text:
        return False
    lowered = stderr_text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "a password is required",
            "password is required",
            "no tty present",
            "no askpass program",
            "a terminal is required",
        )
    )


def run_subprocess_with_capture(
    cmd: List[str],
    log_prefix: str,
    timeout: float = 30,
    **kwargs
) -> Dict[str, any]:
    """
    Run a subprocess command and capture stdout/stderr to a log file.

    Args:
        cmd: Command and arguments as list
        log_prefix: Prefix for log filename (e.g., "ollama_start")
        timeout: Command timeout in seconds
        **kwargs: Additional subprocess.run() arguments

    Returns:
        Dict with:
        - returncode: int
        - stdout: str (full output)
        - stderr: str (full error output)
        - stdout_tail: str (last 200 lines)
        - stderr_tail: str (last 200 lines)
        - log_file: str (path to log file)
        - command: List[str] (command that was run)
    """
    timestamp = get_timestamp()
    log_file = RESET_LOG_DIR / f"{log_prefix}_{timestamp}.log"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            **kwargs
        )

        friendly_error = None
        if cmd and os.path.basename(cmd[0]) == "sudo" and result.returncode == 1:
            stderr_text = result.stderr or ""
            if is_sudo_noninteractive_error(stderr_text):
                friendly_error = SUDO_NONINTERACTIVE_HINT

        # Write full output to log file
        with open(log_file, "w") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("-" * 80 + "\n")
            f.write("STDOUT:\n")
            f.write(result.stdout or "(empty)\n")
            f.write("-" * 80 + "\n")
            f.write("STDERR:\n")
            f.write(result.stderr or "(empty)\n")
            if friendly_error:
                f.write("-" * 80 + "\n")
                f.write(f"FRIENDLY_ERROR:\n{friendly_error}\n")

        return {
            "returncode": result.returncode,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
            "stdout_tail": tail_lines(result.stdout or ""),
            "stderr_tail": tail_lines(result.stderr or ""),
            "log_file": str(log_file.absolute()),
            "command": cmd,
            "friendly_error": friendly_error,
            "error": friendly_error,
        }
    except subprocess.TimeoutExpired as e:
        # Write timeout info to log
        with open(log_file, "w") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Timeout: {timeout}s\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("-" * 80 + "\n")
            f.write("ERROR: Command timed out\n")
            if e.stdout:
                f.write("-" * 80 + "\n")
                f.write("STDOUT (partial):\n")
                f.write(e.stdout.decode() if isinstance(e.stdout, bytes) else str(e.stdout))
            if e.stderr:
                f.write("-" * 80 + "\n")
                f.write("STDERR (partial):\n")
                f.write(e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr))

        stdout_str = e.stdout.decode() if isinstance(e.stdout, bytes) else (str(e.stdout) if e.stdout else "")
        stderr_str = e.stderr.decode() if isinstance(e.stderr, bytes) else (str(e.stderr) if e.stderr else "")

        return {
            "returncode": -1,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "stdout_tail": tail_lines(stdout_str),
            "stderr_tail": tail_lines(stderr_str),
            "log_file": str(log_file.absolute()),
            "command": cmd,
            "error": f"Command timed out after {timeout}s",
        }
    except Exception as e:
        # Write error to log
        with open(log_file, "w") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("-" * 80 + "\n")
            f.write(f"ERROR: {str(e)}\n")

        return {
            "returncode": -1,
            "stdout": "",
            "stderr": "",
            "stdout_tail": "",
            "stderr_tail": "",
            "log_file": str(log_file.absolute()),
            "command": cmd,
            "error": str(e),
        }


class StreamingProcessLogger:
    """
    Capture stdout/stderr from a running process in real-time and save to log file.

    This is useful for long-running processes like Ollama/ComfyUI startup where
    we want to capture output as it happens.
    """

    def __init__(self, log_prefix: str, max_buffer_lines: int = 200):
        """
        Initialize the logger.

        Args:
            log_prefix: Prefix for log filename
            max_buffer_lines: Maximum lines to keep in memory buffer
        """
        timestamp = get_timestamp()
        self.log_file = RESET_LOG_DIR / f"{log_prefix}_{timestamp}.log"
        self.max_buffer_lines = max_buffer_lines
        self.stdout_buffer: List[str] = []
        self.stderr_buffer: List[str] = []
        self.file_handle = None

    def start(self, cmd: List[str]):
        """Start logging, write header to file."""
        self.file_handle = open(self.log_file, "w")
        self.file_handle.write(f"Command: {' '.join(cmd)}\n")
        self.file_handle.write(f"Timestamp: {get_timestamp()}\n")
        self.file_handle.write("-" * 80 + "\n")
        self.file_handle.flush()

    def log_stdout(self, line: str):
        """Log a line of stdout."""
        if self.file_handle:
            self.file_handle.write(f"[OUT] {line}\n")
            self.file_handle.flush()

        self.stdout_buffer.append(line)
        if len(self.stdout_buffer) > self.max_buffer_lines:
            self.stdout_buffer.pop(0)

    def log_stderr(self, line: str):
        """Log a line of stderr."""
        if self.file_handle:
            self.file_handle.write(f"[ERR] {line}\n")
            self.file_handle.flush()

        self.stderr_buffer.append(line)
        if len(self.stderr_buffer) > self.max_buffer_lines:
            self.stderr_buffer.pop(0)

    def close(self):
        """Close the log file."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def get_tails(self) -> Tuple[str, str]:
        """Get stdout and stderr tails."""
        return (
            "\n".join(self.stdout_buffer),
            "\n".join(self.stderr_buffer)
        )

    def get_result(self, pid: Optional[int] = None, returncode: Optional[int] = None) -> Dict[str, any]:
        """
        Get final result dictionary.

        Args:
            pid: Process ID if available
            returncode: Return code if process exited

        Returns:
            Dict with log information
        """
        stdout_tail, stderr_tail = self.get_tails()
        result = {
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "log_file": str(self.log_file.absolute()),
        }
        if pid is not None:
            result["pid"] = pid
        if returncode is not None:
            result["returncode"] = returncode
        return result


async def poll_http_health(
    url: str,
    timeout_s: float,
    poll_interval_s: float = None,
    expected_status: int = 200,
) -> Tuple[bool, float, Optional[str]]:
    """
    Poll an HTTP endpoint until it returns expected status or timeout.

    Args:
        url: URL to poll
        timeout_s: Maximum time to wait in seconds
        poll_interval_s: Interval between polls (uses HEALTH_POLL_INTERVAL_S if None)
        expected_status: Expected HTTP status code

    Returns:
        Tuple of (healthy: bool, time_to_ready_ms: float, error: Optional[str])
    """
    import httpx

    if poll_interval_s is None:
        poll_interval_s = HEALTH_POLL_INTERVAL_S

    t_start = time.time()
    last_error = None

    while (time.time() - t_start) < timeout_s:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(url)
                if resp.status_code == expected_status:
                    ms = (time.time() - t_start) * 1000
                    return (True, ms, None)
                else:
                    last_error = f"HTTP {resp.status_code}"
        except Exception as e:
            last_error = str(e)

        await asyncio.sleep(poll_interval_s)

    # Timeout
    ms = (time.time() - t_start) * 1000
    error_msg = f"Timeout after {timeout_s}s"
    if last_error:
        error_msg += f" (last error: {last_error})"
    return (False, ms, error_msg)

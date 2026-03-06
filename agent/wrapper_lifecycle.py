from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

AGENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = AGENT_DIR.parent
RUN_DIR = AGENT_DIR / "run"
LOG_DIR = AGENT_DIR / "log"
PID_FILE = RUN_DIR / "wrapper.pid"
LOG_FILE = LOG_DIR / "wrapper.log"
WRAPPER_HOST = "127.0.0.1"
WRAPPER_PORT = 9002
WRAPPER_HEALTH_URL = f"http://{WRAPPER_HOST}:{WRAPPER_PORT}/v1/health"

_WRAPPER_PROCESS: Optional[subprocess.Popen] = None


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_pid_file() -> Optional[int]:
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        PID_FILE.unlink(missing_ok=True)
        return None
    if not _pid_exists(pid):
        PID_FILE.unlink(missing_ok=True)
        return None
    return pid


def _write_pid_file(pid: int) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(pid), encoding="utf-8")


def _clear_pid_file() -> None:
    PID_FILE.unlink(missing_ok=True)


def is_wrapper_running() -> bool:
    global _WRAPPER_PROCESS
    if _WRAPPER_PROCESS is not None:
        if _WRAPPER_PROCESS.poll() is None:
            return True
        _WRAPPER_PROCESS = None
    return _read_pid_file() is not None


def wait_for_wrapper_health(timeout_s: float = 10.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(WRAPPER_HEALTH_URL, timeout=1.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def stop_wrapper() -> None:
    global _WRAPPER_PROCESS

    pid: Optional[int] = None
    proc = _WRAPPER_PROCESS
    if proc is not None and proc.poll() is None:
        pid = proc.pid
    if pid is None:
        pid = _read_pid_file()

    if pid is None:
        _WRAPPER_PROCESS = None
        _clear_pid_file()
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass

    end = time.time() + 5.0
    while time.time() < end:
        if not _pid_exists(pid):
            break
        time.sleep(0.1)

    if _pid_exists(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass

    try:
        subprocess.run(["pkill", "-f", "uvicorn agent.backends.wrapper.app"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    _WRAPPER_PROCESS = None
    _clear_pid_file()


def start_wrapper() -> int:
    global _WRAPPER_PROCESS

    stop_wrapper()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log_handle = open(LOG_FILE, "ab")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "agent.backends.wrapper.app:app",
        "--host",
        WRAPPER_HOST,
        "--port",
        str(WRAPPER_PORT),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT_DIR),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    _WRAPPER_PROCESS = proc
    _write_pid_file(proc.pid)

    if not wait_for_wrapper_health(10.0):
        stop_wrapper()
        raise RuntimeError("Wrapper failed health check on http://127.0.0.1:9002/v1/health")

    return proc.pid

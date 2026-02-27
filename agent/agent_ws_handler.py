from __future__ import annotations

import asyncio
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict

ROOT_DIR = Path(__file__).resolve().parent.parent

SendPayload = Callable[[Dict[str, Any]], Awaitable[None]]


async def _send_status(send: SendPayload, request_id: str, phase: str, detail: str) -> None:
    await send({
        "type": "backend_switch_status",
        "payload": {
            "request_id": request_id,
            "phase": phase,
            "detail": detail,
            "timestamp": int(time.time()),
        },
    })


async def _run_script_and_stream(send: SendPayload, request_id: str, phase: str, cmd: str) -> int:
    proc = await asyncio.create_subprocess_shell(
        cmd,
        cwd=str(ROOT_DIR),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    await _send_status(send, request_id, phase, f"Running: {cmd}")
    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        await _send_status(send, request_id, phase, line.decode(errors="replace").rstrip())
    return await proc.wait()


def _detect_platform_tag() -> str:
    env = (os.environ.get("BENCH_AGENT_PLATFORM") or "").strip().lower()
    if env in {"mac", "nvidia"}:
        return env
    sys_name = platform.system().lower()
    if "darwin" in sys_name:
        return "mac"
    if "linux" in sys_name:
        return "nvidia"
    return "unknown"


async def handle_backend_switch(payload: Dict[str, Any], send: SendPayload) -> None:
    backend = str(payload.get("backend") or "").strip().lower()
    target = str(payload.get("target") or "").strip().lower()
    request_id = str(payload.get("request_id") or "")
    if not request_id:
        return

    await _send_status(send, request_id, "offline", "Going offline to switch backend")

    rc = await _run_script_and_stream(send, request_id, "stopping", "./scripts/agent stop-ollama")
    if rc != 0:
        await _send_status(send, request_id, "error", f"./scripts/agent stop-ollama exited {rc}")
        return

    if backend == "ollama":
        start_cmd = "./scripts/agent start-ollama"
    elif backend == "custom":
        platform_tag = _detect_platform_tag()
        pick = target
        if pick not in {"mlx", "trtllm"}:
            pick = "mlx" if platform_tag == "mac" else "trtllm"
        start_cmd = "./scripts/agent start-mlx" if pick == "mlx" else "./scripts/agent start-trtllm"
    else:
        await _send_status(send, request_id, "error", f"Unsupported backend: {backend}")
        return

    rc = await _run_script_and_stream(send, request_id, "starting", start_cmd)
    if rc != 0:
        await _send_status(send, request_id, "error", f"{start_cmd} exited {rc}")
        return

    await _send_status(send, request_id, "running", f"Backend {backend} running")


async def maybe_handle_ws_command(raw: str, send: SendPayload) -> bool:
    try:
        msg = json.loads(raw)
    except Exception:
        return False
    if not isinstance(msg, dict):
        return False
    if msg.get("type") != "backend_switch":
        return False
    payload = msg.get("payload") or {}
    await handle_backend_switch(payload, send)
    return True

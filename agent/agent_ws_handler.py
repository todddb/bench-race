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
    request_id = str(payload.get("request_id") or "")
    model = str(payload.get("model") or "").strip()
    if not request_id:
        return

    if backend not in {"ollama", "custom"}:
        await _send_status(send, request_id, "error", f"Unsupported backend: {backend}")
        return

    await _send_status(send, request_id, "offline", "Going offline to switch backend")

    if backend == "custom":
        engine_type = str(payload.get("engine_type") or payload.get("target") or "").strip().lower()
        if engine_type not in {"mlx", "trtllm"}:
            await _send_status(send, request_id, "error", f"Missing or invalid engine_type for custom backend: '{engine_type}'. Central must provide engine_type.")
            return
        await _send_status(send, request_id, "starting", f"Starting custom backend ({engine_type})")
        rc = await _run_script_and_stream(send, request_id, "starting", f"./scripts/agent start-backend {engine_type} {model}" if model else f"./scripts/agent start-backend {engine_type}")
        if rc != 0:
            await _send_status(send, request_id, "error", f"start-backend {engine_type} exited {rc}")
            return
        rc = await _run_script_and_stream(send, request_id, "starting", "./scripts/agent start-wrapper")
        if rc != 0:
            await _send_status(send, request_id, "error", f"./scripts/agent start-wrapper exited {rc}")
            return
    else:
        await _send_status(send, request_id, "stopping", "Stopping wrapper and managed custom backends")
        for cmd in (
            "./scripts/agent stop-wrapper",
            "./scripts/agent stop-backend mlx",
            "./scripts/agent stop-backend trtllm",
            f"./scripts/agent stop-backend ollama {model}" if model else "./scripts/agent stop-backend ollama",
        ):
            rc = await _run_script_and_stream(send, request_id, "stopping", cmd)
            if rc != 0:
                await _send_status(send, request_id, "error", f"{cmd} exited {rc}")
                return
        await _send_status(send, request_id, "starting", "Using Ollama system service")

    await _send_status(send, request_id, "running", "Ready")



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

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional

import websockets


def _machine_target(machine: Dict[str, Any], backend: str) -> str:
    if backend == "ollama":
        return "ollama"
    machine_text = " ".join(
        str(machine.get(k, "")) for k in ("platform", "gpu_vendor", "vendor", "label", "machine_id")
    ).lower()
    if any(token in machine_text for token in ("mac", "apple", "darwin")):
        return "mlx"
    if any(token in machine_text for token in ("nvidia", "rtx", "cuda")):
        return "trtllm"
    return "custom"


def _base_to_ws_uri(agent_base_url: str) -> str:
    base = agent_base_url.rstrip("/")
    if base.startswith("https://"):
        return "wss://" + base[len("https://") :] + "/ws"
    if base.startswith("http://"):
        return "ws://" + base[len("http://") :] + "/ws"
    if base.startswith("ws://") or base.startswith("wss://"):
        return base + "/ws"
    return "ws://" + base + "/ws"


def build_backend_switch_message(machine: Dict[str, Any], backend: str, request_id: str) -> Dict[str, Any]:
    return {
        "type": "backend_switch",
        "payload": {
            "backend": backend,
            "target": _machine_target(machine, backend),
            "request_id": request_id,
            "timestamp": int(time.time()),
        },
    }


async def _send_to_agent(machine: Dict[str, Any], message: Dict[str, Any]) -> Optional[str]:
    base_url = machine.get("agent_base_url")
    if not base_url:
        return "missing agent_base_url"
    ws_uri = _base_to_ws_uri(str(base_url))
    try:
        async with websockets.connect(ws_uri, max_size=2 * 1024 * 1024) as ws:
            await ws.send(json.dumps(message))
        return None
    except Exception as exc:
        return str(exc)


def do_backend_switch(agents: Iterable[Dict[str, Any]], backend: str) -> Dict[str, Any]:
    request_id = str(uuid.uuid4())
    agent_list = list(agents)

    async def _runner() -> Dict[str, Any]:
        statuses: List[Dict[str, Any]] = []
        for machine in agent_list:
            msg = build_backend_switch_message(machine, backend, request_id)
            err = await _send_to_agent(machine, msg)
            statuses.append({
                "machine_id": machine.get("machine_id"),
                "ok": err is None,
                "error": err,
                "message": msg,
            })
        return {"request_id": request_id, "dispatch": statuses}

    return asyncio.run(_runner())

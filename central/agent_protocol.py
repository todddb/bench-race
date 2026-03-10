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
    if backend in {"mlx", "trtllm"}:
        return backend
    gpu_type = str(
        machine.get("gpu_type")
        or (machine.get("gpu") or {}).get("type")
        or ""
    ).strip().lower()
    if gpu_type == "apple":
        return "mlx"
    if gpu_type == "nvidia":
        return "trtllm"
    machine_id = machine.get("machine_id", "unknown")
    raise ValueError(
        f"Cannot determine target for machine '{machine_id}': "
        f"gpu type '{gpu_type}' is not 'nvidia' or 'apple'. "
        f"Set gpu.type in machines.yaml explicitly."
    )


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
    """Build a backend switch message for an agent.

    IMPORTANT ARCHITECTURAL RULE:
    Central is the sole authority for determining engine_type from
    machines.yaml gpu.type.  Agents must not infer engine_type.
    """
    target = _machine_target(machine, backend)
    payload: Dict[str, Any] = {
        "backend": backend,
        "target": target,
        "request_id": request_id,
        "timestamp": int(time.time()),
    }
    if backend == "custom":
        payload["engine_type"] = target
    return {"type": "backend_switch", "payload": payload}


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

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any, AsyncIterator

import websockets

from agent.backends.base import BackendType, BaseBackend


class MLXBackendWrapper(BaseBackend):
    backend_type = BackendType.MANAGED

    def __init__(self, host: str = "127.0.0.1", port: int = 8321) -> None:
        self.host = host
        self.port = port
        self.models_dir = Path(__file__).resolve().parents[1] / "models" / "mlx"
        self.repo_root = Path(__file__).resolve().parents[2]

    async def list_models(self) -> list[str]:
        if not self.models_dir.exists():
            return []
        return sorted([p.name for p in self.models_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])

    async def generate(self, model: str, messages: list[dict[str, Any]], stream: bool) -> AsyncIterator[str]:
        prompt = "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
        uri = f"ws://{self.host}:{self.port}/stream"
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({"model": model, "prompt": prompt, "params": {}}))
            while True:
                msg = await ws.recv()
                frame = json.loads(msg)
                ftype = frame.get("type")
                if ftype == "token":
                    token = frame.get("token") or frame.get("text", "")
                    if token:
                        yield str(token)
                elif ftype == "done":
                    break
                elif ftype == "error":
                    raise RuntimeError(str(frame.get("error") or "mlx stream error"))

    async def start(self, model: str):
        cmd = [str(self.repo_root / "scripts" / "agent"), "start-backend", "mlx", model]
        proc = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
        return {"ok": proc.returncode == 0, "stdout": proc.stdout, "stderr": proc.stderr}

    async def stop(self):
        cmd = [str(self.repo_root / "scripts" / "agent"), "stop-backend", "mlx"]
        proc = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
        return {"ok": proc.returncode == 0, "stdout": proc.stdout, "stderr": proc.stderr}

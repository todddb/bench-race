from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from agent.backends.base import BaseBackend


class TRTLLMBackendWrapper(BaseBackend):
    def __init__(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        self.base_url = f"http://{host}:{port}"
        self.models_dir = Path(__file__).resolve().parents[1] / "models" / "trtllm"
        self.repo_root = Path(__file__).resolve().parents[2]

    async def list_models(self) -> list[str]:
        if self.models_dir.exists():
            return sorted([p.name for p in self.models_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{self.base_url}/v1/models")
            resp.raise_for_status()
            payload = resp.json() if resp.content else {}
        data = payload.get("data", []) if isinstance(payload, dict) else []
        return [str(m.get("id", "")).strip() for m in data if isinstance(m, dict) and str(m.get("id", "")).strip()]

    async def generate(self, model: str, messages: list[dict[str, Any]], stream: bool) -> AsyncIterator[str]:
        payload = {"model": model, "messages": messages, "stream": True}
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=None, write=60.0, pool=60.0)) as client:
            async with client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield str(content)

    async def start(self, model: str):
        cmd = [str(self.repo_root / "scripts" / "agent"), "start-backend", "trtllm", model]
        proc = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
        return {"ok": proc.returncode == 0, "stdout": proc.stdout, "stderr": proc.stderr}

    async def stop(self):
        cmd = [str(self.repo_root / "scripts" / "agent"), "stop-backend", "trtllm"]
        proc = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
        return {"ok": proc.returncode == 0, "stdout": proc.stdout, "stderr": proc.stderr}

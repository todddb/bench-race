from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from agent.backends.base import BaseBackend


class OllamaBackendWrapper(BaseBackend):
    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None:
        self.base_url = base_url.rstrip("/")
        self.repo_root = Path(__file__).resolve().parents[2]

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
        payload = resp.json() if resp.content else {}
        models = payload.get("models", []) if isinstance(payload, dict) else []
        return [str(m.get("name", "")).strip() for m in models if isinstance(m, dict) and str(m.get("name", "")).strip()]

    async def generate(self, model: str, messages: list[dict[str, Any]], stream: bool) -> AsyncIterator[str]:
        prompt = "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
        payload = {"model": model, "prompt": prompt, "stream": True}
        timeout = httpx.Timeout(connect=5.0, read=None, write=60.0, pool=60.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = obj.get("response") if isinstance(obj, dict) else None
                    if isinstance(text, str) and text:
                        yield text
                    if isinstance(obj, dict) and obj.get("done") is True:
                        break

    async def start(self, model: str):
        cmd = [str(self.repo_root / "scripts" / "agent"), "start-backend", "ollama"]
        if model:
            cmd.append(model)
        proc = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
        return {"ok": proc.returncode == 0, "stdout": proc.stdout, "stderr": proc.stderr}

    async def stop(self):
        cmd = [str(self.repo_root / "scripts" / "agent"), "stop-backend", "ollama"]
        proc = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
        return {"ok": proc.returncode == 0, "stdout": proc.stdout, "stderr": proc.stderr}

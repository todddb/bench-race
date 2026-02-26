from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, List

import httpx


class MLXAdapter:
    backend_name = "mlx"

    def __init__(self, base_url: str = "http://127.0.0.1:8321") -> None:
        self.base_url = base_url.rstrip("/")

    async def list_models(self) -> List[Dict[str, Any]]:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self.base_url}/models")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            for model in models:
                model.setdefault("backend", self.backend_name)
            return models

    async def health(self) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            data = resp.json()
            data.setdefault("engine", self.backend_name)
            return data

    async def start_model(self, model_id: str, args: Dict[str, Any] | None = None) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{self.base_url}/start", json={"model_id": model_id, "args": args or {}})
            resp.raise_for_status()
            return resp.json()

    async def switch_model(self, model_id: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{self.base_url}/model/switch", json={"model_id": model_id})
            resp.raise_for_status()
            return resp.json()

    async def infer(self, model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = {
            "prompt": payload.get("prompt") or payload.get("inputs") or "",
            "max_tokens": int(payload.get("max_tokens", 256)),
            "temperature": float(payload.get("temperature", 0.7)),
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(f"{self.base_url}/infer", json=body)
            resp.raise_for_status()
            return resp.json()

    async def infer_stream(self, model_id: str, payload: Dict[str, Any]) -> AsyncGenerator[bytes, None]:
        prompt = payload.get("prompt") or payload.get("inputs") or ""
        req = {
            "prompt": prompt,
            "max_tokens": int(payload.get("max_tokens", 256)),
            "temperature": float(payload.get("temperature", 0.7)),
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("POST", f"{self.base_url}/infer", json=req) as resp:
                    resp.raise_for_status()
                    ctype = resp.headers.get("content-type", "")
                    if "text/event-stream" in ctype:
                        async for line in resp.aiter_lines():
                            if line:
                                yield line.encode("utf-8")
                        return
                    chunks = []
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            chunks.append(chunk)
                    if chunks:
                        merged = b"".join(chunks).decode("utf-8", errors="ignore")
                        try:
                            data = json.loads(merged)
                            text = data.get("text") or ""
                        except json.JSONDecodeError:
                            text = merged
                        for idx in range(0, len(text), 32):
                            yield text[idx : idx + 32].encode("utf-8")
                        return
            except Exception:
                resp = await client.post(f"{self.base_url}/infer", json={**req, "stream": False}, timeout=300)
                resp.raise_for_status()
                text = resp.json().get("text", "")
                for idx in range(0, len(text), 32):
                    yield text[idx : idx + 32].encode("utf-8")

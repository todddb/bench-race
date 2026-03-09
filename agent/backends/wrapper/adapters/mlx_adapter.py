from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Dict, List

import httpx

logger = logging.getLogger(__name__)

MLX_BASE_URL = "http://127.0.0.1:8321"


class MLXAdapter:
    backend_name = "mlx"

    def __init__(self) -> None:
        self.active_model_id: str | None = None

    async def list_models(self) -> List[Dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{MLX_BASE_URL}/v1/models")
                resp.raise_for_status()
                data = resp.json()
                models = data.get("data") or data.get("models") or []
                normalized: List[Dict[str, Any]] = []
                for model in models:
                    if isinstance(model, str):
                        normalized.append({"id": model, "backend": self.backend_name})
                    else:
                        model.setdefault("backend", self.backend_name)
                        normalized.append(model)
                return normalized
        except httpx.ConnectError:
            logger.warning("mlx_server_unreachable", extra={"backend": self.backend_name, "url": MLX_BASE_URL})
            return []
        except Exception as exc:
            logger.exception("mlx_list_models_failed", extra={"backend": self.backend_name, "error": str(exc)})
            return []

    async def health(self) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{MLX_BASE_URL}/health")
                resp.raise_for_status()
                data = resp.json()
                data.setdefault("engine", self.backend_name)
                return data
        except Exception:
            return {"ok": False, "engine": self.backend_name, "model": self.active_model_id}

    async def start_model(self, model_id: str, args: Dict[str, Any] | None = None) -> Dict[str, Any]:
        self.active_model_id = model_id
        return {"ok": True, "engine": self.backend_name, "model": model_id}

    async def switch_model(self, model_id: str) -> Dict[str, Any]:
        self.active_model_id = model_id
        return {"ok": True, "engine": self.backend_name, "model": model_id}

    async def stop_model(self) -> Dict[str, Any]:
        self.active_model_id = None
        return {"ok": True, "engine": self.backend_name}

    async def infer(self, model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = {
            "model": model_id or self.active_model_id or "",
            "prompt": payload.get("prompt") or payload.get("inputs") or "",
            "max_tokens": int(payload.get("max_tokens", 256)),
            "temperature": float(payload.get("temperature", 0.7)),
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(f"{MLX_BASE_URL}/v1/completions", json=body)
                resp.raise_for_status()
                return resp.json()
        except httpx.ConnectError:
            logger.error("mlx_server_unreachable", extra={"backend": self.backend_name, "url": MLX_BASE_URL})
            raise RuntimeError("MLX backend not reachable")

    async def infer_stream(self, model_id: str, payload: Dict[str, Any]) -> AsyncGenerator[bytes, None]:
        body = {
            "model": model_id or self.active_model_id or "",
            "prompt": payload.get("prompt") or payload.get("inputs") or "",
            "max_tokens": int(payload.get("max_tokens", 256)),
            "temperature": float(payload.get("temperature", 0.7)),
            "stream": True,
        }
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{MLX_BASE_URL}/v1/completions", json=body) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            yield chunk
        except httpx.ConnectError:
            logger.error("mlx_server_unreachable", extra={"backend": self.backend_name, "url": MLX_BASE_URL})
            raise RuntimeError("MLX backend not reachable")

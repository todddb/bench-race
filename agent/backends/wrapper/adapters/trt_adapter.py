from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict, List

import httpx

from ..service_manager import ServiceManager

logger = logging.getLogger(__name__)


class TRTAdapter:
    backend_name = "trt"

    def __init__(
        self,
        base_url: str,
        run_script: str = "agent/backends/trtllm_run.sh",
        service_manager: ServiceManager | None = None,
    ) -> None:
        try:
            self.base_url = base_url.rstrip("/")
            self.run_script = run_script
            self.service_manager = service_manager or ServiceManager()
        except Exception as exc:
            logger.exception("trt_engine_load_failed", extra={"backend": self.backend_name, "error": str(exc)})
            raise RuntimeError(f"TRT load failed: {exc}") from exc

    async def list_models(self) -> List[Dict[str, Any]]:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self.base_url}/v1/models")
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

    async def health(self) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                resp = await client.get(f"{self.base_url}/health")
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                data = {"status": "down"}
            data.setdefault("engine", self.backend_name)
            return data

    async def start_model(self, model_id: str, args: Dict[str, Any] | None = None) -> Dict[str, Any]:
        try:
            result = self.service_manager.start_backend("trt", model_id=model_id)
            if not result.get("ok", False):
                raise RuntimeError(result.get("stderr") or result.get("stdout") or "TRT backend start failed")
            result["engine_model"] = self.service_manager.trt_engine_id(model_id)
            return result
        except Exception as exc:
            logger.exception("trt_engine_load_failed", extra={"backend": self.backend_name, "model_id": model_id, "error": str(exc)})
            raise RuntimeError(f"TRT load failed: {exc}") from exc

    async def switch_model(self, model_id: str) -> Dict[str, Any]:
        return await self.start_model(model_id)

    async def infer(self, model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = {
            "model": model_id,
            "prompt": payload.get("prompt") or payload.get("inputs") or "",
            "max_tokens": int(payload.get("max_tokens", 256)),
            "temperature": float(payload.get("temperature", 0.7)),
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(f"{self.base_url}/v1/completions", json=body)
            resp.raise_for_status()
            return resp.json()

    async def infer_stream(self, model_id: str, payload: Dict[str, Any]) -> AsyncGenerator[bytes, None]:
        body = {
            "model": model_id,
            "prompt": payload.get("prompt") or payload.get("inputs") or "",
            "max_tokens": int(payload.get("max_tokens", 256)),
            "temperature": float(payload.get("temperature", 0.7)),
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("POST", f"{self.base_url}/v1/completions", json=body) as resp:
                    resp.raise_for_status()
                    ctype = resp.headers.get("content-type", "")
                    if "text/event-stream" in ctype:
                        async for line in resp.aiter_lines():
                            if line:
                                yield line.encode("utf-8")
                        return
                    raw = []
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            raw.append(chunk)
                    if raw:
                        txt = b"".join(raw).decode("utf-8", errors="ignore")
                        try:
                            data = json.loads(txt)
                            text = (data.get("choices") or [{}])[0].get("text", "")
                        except json.JSONDecodeError:
                            text = txt
                        for i in range(0, len(text), 32):
                            yield text[i:i + 32].encode("utf-8")
                        return
            except Exception:
                resp = await client.post(f"{self.base_url}/v1/completions", json={**body, "stream": False}, timeout=300)
                resp.raise_for_status()
                data = resp.json()
                text = (data.get("choices") or [{}])[0].get("text", "")
                for i in range(0, len(text), 32):
                    yield text[i:i + 32].encode("utf-8")

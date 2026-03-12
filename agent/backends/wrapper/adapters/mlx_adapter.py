from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List

import httpx

logger = logging.getLogger(__name__)

MLX_BASE_URL = "http://127.0.0.1:8321"


class MLXAdapter:
    backend_name = "mlx"

    def __init__(self) -> None:
        self.active_model_id: str | None = None

    # ------------------------------------------------------------------
    # 1. list_models – query /health for the active model
    # ------------------------------------------------------------------
    async def list_models(self) -> List[Dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{MLX_BASE_URL}/health")
                resp.raise_for_status()
                data = resp.json()
                model = data.get("model")
                if model:
                    return [
                        {
                            "id": model,
                            "object": "model",
                            "owned_by": "mlx",
                            "backend": self.backend_name,
                        }
                    ]
                return []
        except httpx.ConnectError:
            logger.warning("mlx_server_unreachable", extra={"backend": self.backend_name, "url": MLX_BASE_URL})
            return []
        except Exception as exc:
            logger.exception("mlx_list_models_failed", extra={"backend": self.backend_name, "error": str(exc)})
            return []

    # ------------------------------------------------------------------
    # 2. health – proxy /health and interpret status
    # ------------------------------------------------------------------
    async def health(self) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{MLX_BASE_URL}/health")
                resp.raise_for_status()
                data = resp.json()
                status = data.get("status", "")
                ok = resp.status_code == 200 and status != "error"
                data["ok"] = ok
                data.setdefault("engine", self.backend_name)
                return data
        except Exception:
            return {"ok": False, "engine": self.backend_name, "model": self.active_model_id}

    # ------------------------------------------------------------------
    # 3. start_model – POST /start
    # ------------------------------------------------------------------
    async def start_model(self, model_id: str, args: Dict[str, Any] | None = None) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{MLX_BASE_URL}/start",
                json={"model_id": model_id, "args": args or {}},
            )
            resp.raise_for_status()
            self.active_model_id = model_id
            return {"ok": True, "engine": self.backend_name, "model": model_id}

    # ------------------------------------------------------------------
    # 4. stop_model – POST /stop
    # ------------------------------------------------------------------
    async def stop_model(self) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(f"{MLX_BASE_URL}/stop")
                resp.raise_for_status()
        except Exception as exc:
            logger.warning("mlx_stop_model_failed", extra={"error": str(exc)})
        self.active_model_id = None
        return {"ok": True, "engine": self.backend_name}

    # ------------------------------------------------------------------
    # 5. switch_model – POST /model/switch
    # ------------------------------------------------------------------
    async def switch_model(self, model_id: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{MLX_BASE_URL}/model/switch",
                json={"model_id": model_id},
            )
            resp.raise_for_status()
            self.active_model_id = model_id
            return {"ok": True, "engine": self.backend_name, "model": model_id}

    # ------------------------------------------------------------------
    # 6. infer – translate OpenAI payload → MLX /infer, translate back
    # ------------------------------------------------------------------
    async def infer(self, model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_prompt(payload)
        body = {
            "prompt": prompt,
            "max_tokens": int(payload.get("max_tokens", 256)),
            "temperature": float(payload.get("temperature", 0.7)),
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(f"{MLX_BASE_URL}/infer", json=body)
                resp.raise_for_status()
                data = resp.json()
            return self._mlx_to_openai(model_id, data)
        except httpx.ConnectError:
            logger.error("mlx_server_unreachable", extra={"backend": self.backend_name, "url": MLX_BASE_URL})
            raise RuntimeError("MLX backend not reachable")

    # ------------------------------------------------------------------
    # 7. infer_stream – POST /infer with stream=true, yield SSE chunks
    # ------------------------------------------------------------------
    async def infer_stream(self, model_id: str, payload: Dict[str, Any]) -> AsyncGenerator[bytes, None]:
        prompt = self._build_prompt(payload)
        body = {
            "prompt": prompt,
            "max_tokens": int(payload.get("max_tokens", 256)),
            "temperature": float(payload.get("temperature", 0.7)),
            "stream": True,
        }
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=60.0)
            ) as client:
                async with client.stream("POST", f"{MLX_BASE_URL}/infer", json=body) as resp:
                    resp.raise_for_status()
                    ctype = resp.headers.get("content-type", "")

                    if "text/event-stream" in ctype:
                        # MLX emits SSE frames as "data: {...}\n\n".  Multiple frames
                        # can arrive in a single aiter_bytes() chunk when TCP coalesces
                        # them.  Buffer and split on the SSE frame boundary (\n\n) so
                        # the outer sse_generator receives exactly one frame per yield,
                        # preventing the wrapper from mangling multiple tokens into one.
                        buf = b""
                        async for chunk in resp.aiter_bytes():
                            if not chunk:
                                continue
                            buf += chunk
                            while b"\n\n" in buf:
                                frame, buf = buf.split(b"\n\n", 1)
                                if frame.strip():
                                    yield frame + b"\n\n"
                        if buf.strip():
                            yield buf
                        return

                    # Server returned a non-streaming JSON response — yield raw
                    # token bytes so the wrapper handles SSE framing (matches
                    # TRT adapter behaviour).
                    raw: list[bytes] = []
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            raw.append(chunk)
                    if raw:
                        txt = b"".join(raw).decode("utf-8", errors="ignore")
                        try:
                            data = json.loads(txt)
                            text = data.get("text", "")
                        except json.JSONDecodeError:
                            text = txt
                        for i in range(0, len(text), 32):
                            yield text[i:i + 32].encode("utf-8")
                        return
        except httpx.ConnectError:
            logger.error("mlx_server_unreachable", extra={"backend": self.backend_name, "url": MLX_BASE_URL})
            raise RuntimeError("MLX backend not reachable")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_prompt(payload: Dict[str, Any]) -> str:
        """Convert an OpenAI-style payload into a single prompt string.

        Handles both raw ``prompt`` fields and ``messages`` arrays.
        """
        messages = payload.get("messages")
        if messages:
            parts: list[str] = []
            for msg in messages:
                role = msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "role", "user")
                content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                if role == "system":
                    parts.append(f"System: {content}")
                elif role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
            return "\n".join(parts) + "\nAssistant:"

        return payload.get("prompt") or payload.get("inputs") or ""

    @staticmethod
    def _mlx_to_openai(model_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Translate an MLX /infer response into an OpenAI-compatible dict."""
        text = data.get("text", "")
        tokens = int(data.get("tokens", 0))
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": tokens,
                "total_tokens": tokens,
            },
        }

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Dict, List

logger = logging.getLogger(__name__)


class MLXAdapter:
    backend_name = "mlx"

    def __init__(self) -> None:
        self._mlx_lm = None
        self.model = None
        self.tokenizer = None
        self.active_model_id: str | None = None

    def _ensure_mlx(self) -> None:
        if self._mlx_lm is not None:
            return
        try:
            import mlx_lm  # type: ignore[import-untyped]
        except Exception as exc:
            logger.exception("mlx_engine_load_failed", extra={"backend": self.backend_name, "error": str(exc)})
            raise RuntimeError(f"MLX load failed: {exc}") from exc
        self._mlx_lm = mlx_lm

    async def list_models(self) -> List[Dict[str, Any]]:
        if not self.active_model_id:
            return []
        return [{"id": self.active_model_id, "backend": self.backend_name}]

    async def health(self) -> Dict[str, Any]:
        return {
            "ok": self.model is not None,
            "engine": self.backend_name,
            "model": self.active_model_id,
        }

    async def start_model(self, model_id: str, args: Dict[str, Any] | None = None) -> Dict[str, Any]:
        try:
            self._ensure_mlx()
            self.model, self.tokenizer = self._mlx_lm.load(model_id)
            self.active_model_id = model_id
            return {"ok": True, "engine": self.backend_name, "model": model_id}
        except Exception as exc:
            logger.exception("mlx_engine_load_failed", extra={"backend": self.backend_name, "model_id": model_id, "error": str(exc)})
            raise RuntimeError(f"MLX load failed: {exc}") from exc

    async def switch_model(self, model_id: str) -> Dict[str, Any]:
        return await self.start_model(model_id)

    async def stop_model(self) -> Dict[str, Any]:
        self.model = None
        self.tokenizer = None
        self.active_model_id = None
        return {"ok": True, "engine": self.backend_name}

    async def infer(self, model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No MLX model loaded")

        body = {
            "prompt": payload.get("prompt") or payload.get("inputs") or "",
            "max_tokens": int(payload.get("max_tokens", 256)),
            "temperature": float(payload.get("temperature", 0.7)),
        }
        text = self._mlx_lm.generate(
            self.model,
            self.tokenizer,
            prompt=body["prompt"],
            max_tokens=body["max_tokens"],
            temp=body["temperature"],
        )
        tokens = len(self.tokenizer.encode(text))
        return {"text": text, "tokens": tokens, "engine": self.backend_name, "model": self.active_model_id}

    async def infer_stream(self, model_id: str, payload: Dict[str, Any]) -> AsyncGenerator[bytes, None]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No MLX model loaded")

        prompt = payload.get("prompt") or payload.get("inputs") or ""
        max_tokens = int(payload.get("max_tokens", 256))
        temperature = float(payload.get("temperature", 0.7))
        full_text = ""
        for token_obj in self._mlx_lm.stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
        ):
            token_text = getattr(token_obj, "text", str(token_obj))
            if token_text.startswith(full_text):
                chunk = token_text[len(full_text):]
                full_text = token_text
            else:
                chunk = token_text
                full_text += token_text
            if chunk:
                yield chunk.encode("utf-8")

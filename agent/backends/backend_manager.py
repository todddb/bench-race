from __future__ import annotations

from typing import Dict

from agent.backends.base import BackendType, BaseBackend
from agent.backends.mlx_wrapper import MLXBackendWrapper
from agent.backends.ollama_wrapper import OllamaBackendWrapper
from agent.backends.trtllm_wrapper import TRTLLMBackendWrapper


class NullBackend:
    name = "null"
    backend_type = BackendType.EXTERNAL

    async def is_available(self):
        return False

    async def list_models(self):
        return []

    async def generate(self, *args, **kwargs):
        raise RuntimeError("No active backend")

    async def start(self, *args, **kwargs):
        pass

    async def stop(self):
        pass


class BackendManager:
    def __init__(self, ollama_base_url: str, mlx_host: str, mlx_port: int, trt_host: str, trt_port: int) -> None:
        self.backends: Dict[str, BaseBackend] = {
            "ollama": OllamaBackendWrapper(ollama_base_url),
            "mlx": MLXBackendWrapper(mlx_host, mlx_port),
            "trtllm": TRTLLMBackendWrapper(trt_host, trt_port),
        }
        self._active_backend: BaseBackend | None = None
        self._active_name: str | None = None

    def create_backend(self, backend_name: str) -> BaseBackend:
        key = (backend_name or "").strip().lower()
        backend = self.backends.get(key)
        if backend is None:
            raise ValueError(f"Unknown backend: {backend_name}")
        return backend

    def set_active_backend(self, backend: BaseBackend, backend_name: str) -> None:
        self._active_backend = backend
        self._active_name = (backend_name or "").strip().lower() or None

    def get_active_backend(self) -> BaseBackend:
        if self._active_backend is None:
            return NullBackend()
        return self._active_backend

    def clear_active_backend(self) -> None:
        self._active_backend = None
        self._active_name = None

    def get_active_backend_name(self) -> str | None:
        return self._active_name

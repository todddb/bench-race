from __future__ import annotations

from typing import Dict

from agent.backends.base import BaseBackend
from agent.backends.mlx_wrapper import MLXBackendWrapper
from agent.backends.ollama_wrapper import OllamaBackendWrapper
from agent.backends.trtllm_wrapper import TRTLLMBackendWrapper


class BackendManager:
    def __init__(self, ollama_base_url: str, mlx_host: str, mlx_port: int, trt_host: str, trt_port: int) -> None:
        self.backends: Dict[str, BaseBackend] = {
            "ollama": OllamaBackendWrapper(ollama_base_url),
            "mlx": MLXBackendWrapper(mlx_host, mlx_port),
            "trtllm": TRTLLMBackendWrapper(trt_host, trt_port),
        }

    def get_active_backend(self, active_backend: str | None) -> BaseBackend:
        key = (active_backend or "").strip().lower()
        if key not in self.backends:
            raise ValueError("No active backend")
        return self.backends[key]

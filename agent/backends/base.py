from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncIterator


class BackendType(str, Enum):
    """Categorises inference backends by lifecycle ownership.

    EXTERNAL — The backend runs independently (e.g. Ollama on port 11434).
        bench-race does NOT start, stop, or manage its process.
        It only calls the API and optionally checks availability.

    MANAGED — The backend is lifecycle-managed by bench-race (e.g. MLX, TRT-LLM).
        bench-race loads models, starts/stops the process, tracks running state,
        and owns the memory lifecycle.

    These two categories must never share lifecycle logic.
    """
    EXTERNAL = "external"
    MANAGED = "managed"


class BaseBackend(ABC):
    backend_type: BackendType = BackendType.MANAGED

    async def is_available(self) -> bool:
        return True

    @abstractmethod
    async def list_models(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    async def generate(self, model: str, messages: list[dict[str, Any]], stream: bool) -> AsyncIterator[str]:
        raise NotImplementedError

    @abstractmethod
    async def start(self, model: str):
        """Start the backend with the given model.

        For MANAGED backends, this initiates the actual process/model loading.
        For EXTERNAL backends, this is a no-op that returns success.
        """
        raise NotImplementedError

    @abstractmethod
    async def stop(self):
        """Stop the backend.

        For MANAGED backends, this unloads the model and stops the process.
        For EXTERNAL backends, this is a no-op — the agent never stops
        external services.
        """
        raise NotImplementedError

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncIterator


class BackendType(str, Enum):
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
        raise NotImplementedError

    @abstractmethod
    async def stop(self):
        raise NotImplementedError

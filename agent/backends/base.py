from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator


class BaseBackend(ABC):
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

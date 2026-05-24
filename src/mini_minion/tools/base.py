from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: dict


class Tool(ABC):
    @property
    @abstractmethod
    def schema(self) -> ToolSchema: ...

    @abstractmethod
    def execute(self, **kwargs: object) -> str: ...

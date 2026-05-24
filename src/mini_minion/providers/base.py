from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"


class LLMProvider(Protocol):
    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
    ) -> LLMResponse: ...

"""OpenAI-compatible provider (Chat Completions API)."""

from __future__ import annotations

import json

from openai import OpenAI

from .base import LLMResponse, ToolCall


class OpenAICompatibleProvider:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
    ) -> LLMResponse:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        if tools:
            kwargs["tools"] = tools

        response = self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments))
            for tc in (msg.tool_calls or [])
        ]
        return LLMResponse(
            text=msg.content or "",
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else (response.choices[0].finish_reason or "stop"),
        )

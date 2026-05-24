"""Anthropic provider (Claude models, native tool_use format)."""

from __future__ import annotations

import json

from .base import LLMResponse, ToolCall


class AnthropicProvider:
    def __init__(self, api_key: str, model: str) -> None:
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install mini-minion[anthropic]")
        self._client = _anthropic.Anthropic(api_key=api_key)
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
            "system": system,
            "messages": _to_anthropic_messages(messages),
        }
        if tools:
            kwargs["tools"] = _format_tools(tools)

        response = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

        return LLMResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else (response.stop_reason or "stop"),
        )


def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-format messages to Anthropic's messages format."""
    result: list[dict] = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        if role == "user":
            result.append({"role": "user", "content": content})

        elif role == "assistant":
            if tool_calls:
                blocks: list[dict] = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in tool_calls:
                    fn = tc["function"]
                    try:
                        input_args = json.loads(fn["arguments"])
                    except (json.JSONDecodeError, KeyError):
                        input_args = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": fn["name"],
                        "input": input_args,
                    })
                result.append({"role": "assistant", "content": blocks})
            else:
                result.append({"role": "assistant", "content": content or ""})

        elif role == "tool":
            # Anthropic requires tool_result inside a "user" turn; merge consecutive ones.
            block: dict = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": content or "",
            }
            if (
                result
                and result[-1]["role"] == "user"
                and isinstance(result[-1]["content"], list)
                and result[-1]["content"]
                and result[-1]["content"][-1].get("type") == "tool_result"
            ):
                result[-1]["content"].append(block)
            else:
                result.append({"role": "user", "content": [block]})

    return result


def _format_tools(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-format tool dicts to Anthropic's tool format."""
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "input_schema": t["function"]["parameters"],
        }
        for t in tools
    ]

"""Tests for the agent TAO loop (run_turn)."""

import json
from unittest.mock import Mock

from mini_minion.agents.runner import run_turn
from mini_minion.providers.base import LLMResponse, ToolCall
from mini_minion.tools.base import Tool, ToolSchema
from mini_minion.tools.registry import ToolRegistry


class _EchoTool(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="echo",
            description="Echo text back.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        return str(kwargs.get("text", ""))


def _provider(*responses: LLMResponse) -> Mock:
    p = Mock()
    p.chat = Mock(side_effect=list(responses))
    return p


def test_simple_text_response(capsys):
    messages: list[dict] = []
    provider = _provider(LLMResponse(text="Hello!", finish_reason="stop"))

    run_turn(provider, "Ada", "system", 100, ToolRegistry(), messages)

    assert len(messages) == 1
    assert messages[0] == {"role": "assistant", "content": "Hello!"}
    assert "Ada: Hello!" in capsys.readouterr().out


def test_tool_call_then_stop(capsys):
    registry = ToolRegistry()
    registry.register(_EchoTool())
    messages: list[dict] = [{"role": "user", "content": "hi"}]

    provider = _provider(
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id="tc1", name="echo", arguments={"text": "pong"})],
            finish_reason="tool_calls",
        ),
        LLMResponse(text="Done.", finish_reason="stop"),
    )

    run_turn(provider, "Ada", "system", 100, registry, messages)

    # user + assistant(tool_call) + tool_result + assistant(final)
    assert len(messages) == 4
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["function"]["name"] == "echo"
    assert messages[2] == {"role": "tool", "tool_call_id": "tc1", "content": "pong"}
    assert messages[3] == {"role": "assistant", "content": "Done."}
    assert "Ada: Done." in capsys.readouterr().out


def test_multiple_tool_calls_in_one_response(capsys):
    """All tool calls from a single response must be executed before the next LLM call."""
    registry = ToolRegistry()
    registry.register(_EchoTool())
    messages: list[dict] = []

    provider = _provider(
        LLMResponse(
            text="",
            tool_calls=[
                ToolCall(id="a", name="echo", arguments={"text": "first"}),
                ToolCall(id="b", name="echo", arguments={"text": "second"}),
            ],
            finish_reason="tool_calls",
        ),
        LLMResponse(text="all done", finish_reason="stop"),
    )

    run_turn(provider, "Ada", "system", 100, registry, messages)

    tool_results = [m for m in messages if m["role"] == "tool"]
    assert len(tool_results) == 2
    assert tool_results[0]["content"] == "first"
    assert tool_results[1]["content"] == "second"


def test_unknown_tool_returns_error_and_continues(capsys):
    """An unknown tool call must not crash the loop; the error is fed back as a tool result."""
    messages: list[dict] = []
    provider = _provider(
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id="x", name="nonexistent", arguments={})],
            finish_reason="tool_calls",
        ),
        LLMResponse(text="recovered", finish_reason="stop"),
    )

    run_turn(provider, "Ada", "system", 100, ToolRegistry(), messages)

    tool_result = next(m for m in messages if m["role"] == "tool")
    assert "nonexistent" in tool_result["content"].lower() or "unknown" in tool_result["content"].lower()
    assert "Ada: recovered" in capsys.readouterr().out


def test_tool_call_arguments_serialised(capsys):
    """Tool call arguments in the assistant message must be JSON-serialised strings."""
    registry = ToolRegistry()
    registry.register(_EchoTool())
    messages: list[dict] = []

    provider = _provider(
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id="x", name="echo", arguments={"text": "hi"})],
            finish_reason="tool_calls",
        ),
        LLMResponse(text="ok", finish_reason="stop"),
    )

    run_turn(provider, "Ada", "system", 100, registry, messages)

    raw_args = messages[0]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(raw_args) == {"text": "hi"}


def test_empty_response_no_print(capsys):
    messages: list[dict] = []
    provider = _provider(LLMResponse(text="", finish_reason="stop"))

    run_turn(provider, "Ada", "system", 100, ToolRegistry(), messages)

    assert capsys.readouterr().out == ""


def test_provider_called_with_correct_args():
    registry = ToolRegistry()
    registry.register(_EchoTool())
    messages: list[dict] = []
    provider = _provider(LLMResponse(text="hi", finish_reason="stop"))

    run_turn(provider, "Ada", "my system prompt", 512, registry, messages)

    call_args = provider.chat.call_args[0]
    assert call_args[0] == "my system prompt"
    assert call_args[3] == 512
    assert any(d["function"]["name"] == "echo" for d in call_args[2])

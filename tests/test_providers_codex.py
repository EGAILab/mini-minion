"""Tests for CodexProvider and the Responses API format helpers.

Uses fake clients/responses injected via object.__new__ — no real API
connection is made.
"""

from unittest.mock import MagicMock, patch

import pytest

from minion_assist.providers.codex import (
    CodexProvider,
    _convert_messages,
    _convert_tools,
    _extract_usage,
    _parse_arguments,
)
from minion_assist.providers.base import LLMResponse, TokenUsage, ToolCall


# ---------------------------------------------------------------------------
# _convert_messages — Chat Completions → Responses API input format
# ---------------------------------------------------------------------------


def test_convert_user_message():
    result = _convert_messages([{"role": "user", "content": "hello"}])
    assert result == [{"role": "user", "content": "hello"}]


def test_convert_assistant_text():
    result = _convert_messages([{"role": "assistant", "content": "hi there"}])
    assert result == [
        {"role": "assistant", "content": [{"type": "output_text", "text": "hi there"}]}
    ]


def test_convert_assistant_no_content_no_tool_calls():
    # Edge case: assistant message with neither content nor tool calls.
    result = _convert_messages([{"role": "assistant", "content": None}])
    assert result == []


def test_convert_assistant_tool_calls():
    msgs = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path": "/tmp/x"}'},
                }
            ],
        }
    ]
    result = _convert_messages(msgs)
    assert result == [
        {
            "type": "function_call",
            "call_id": "call_abc",
            "name": "read",
            "arguments": '{"path": "/tmp/x"}',
        }
    ]


def test_convert_assistant_text_and_tool_calls():
    # Text plus tool call: should produce two separate items.
    msgs = [
        {
            "role": "assistant",
            "content": "I'll read the file.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        }
    ]
    result = _convert_messages(msgs)
    assert len(result) == 2
    assert result[0] == {
        "role": "assistant",
        "content": [{"type": "output_text", "text": "I'll read the file."}],
    }
    assert result[1] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "read",
        "arguments": "{}",
    }


def test_convert_tool_result():
    msgs = [{"role": "tool", "tool_call_id": "call_abc", "content": "file contents here"}]
    result = _convert_messages(msgs)
    assert result == [
        {
            "type": "function_call_output",
            "call_id": "call_abc",
            "output": "file contents here",
        }
    ]


def test_convert_mixed_conversation():
    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! Let me use a tool."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "file.txt"},
        {"role": "user", "content": "Thanks"},
    ]
    result = _convert_messages(msgs)
    assert len(result) == 5
    assert result[0] == {"role": "user", "content": "Hello"}
    assert result[1]["role"] == "assistant"
    assert result[2]["type"] == "function_call"
    assert result[3]["type"] == "function_call_output"
    assert result[4] == {"role": "user", "content": "Thanks"}


def test_convert_empty_messages():
    assert _convert_messages([]) == []


# ---------------------------------------------------------------------------
# _convert_tools — Chat Completions → Responses API tool format
# ---------------------------------------------------------------------------


def test_convert_tools_basic():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]
    result = _convert_tools(tools)
    assert result == [
        {
            "type": "function",
            "name": "read",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]


def test_convert_tools_empty():
    assert _convert_tools([]) == []


def test_convert_tools_non_function_ignored():
    # Non-function tool types (hypothetical) should be skipped.
    tools = [{"type": "unknown", "stuff": "value"}]
    assert _convert_tools(tools) == []


def test_convert_tools_multiple():
    tools = [
        {"type": "function", "function": {"name": "a", "description": "A", "parameters": {}}},
        {"type": "function", "function": {"name": "b", "description": "B", "parameters": {}}},
    ]
    result = _convert_tools(tools)
    assert len(result) == 2
    assert result[0]["name"] == "a"
    assert result[1]["name"] == "b"


# ---------------------------------------------------------------------------
# _parse_arguments
# ---------------------------------------------------------------------------


def test_parse_arguments_valid():
    args, err = _parse_arguments('{"path": "/tmp/x"}', "read")
    assert args == {"path": "/tmp/x"}
    assert err is None


def test_parse_arguments_empty_string():
    args, err = _parse_arguments("", "tool")
    assert args == {}
    assert err is None


def test_parse_arguments_invalid_json():
    args, err = _parse_arguments("{not json}", "tool")
    assert args == {}
    assert err is not None
    assert "tool" in err


def test_parse_arguments_non_object():
    args, err = _parse_arguments("[1, 2, 3]", "tool")
    assert args == {}
    assert err is not None
    assert "JSON object" in err


# ---------------------------------------------------------------------------
# _extract_usage
# ---------------------------------------------------------------------------


def test_extract_usage_present():
    raw = MagicMock()
    raw.usage.input_tokens = 10
    raw.usage.output_tokens = 20
    result = _extract_usage(raw)
    assert result == TokenUsage(input_tokens=10, output_tokens=20)


def test_extract_usage_missing():
    raw = MagicMock(spec=[])  # no 'usage' attribute
    result = _extract_usage(raw)
    assert result is None


# ---------------------------------------------------------------------------
# CodexProvider — blocking path
# ---------------------------------------------------------------------------


def _provider() -> CodexProvider:
    """Build a provider with a mock client, bypassing __init__."""
    p = object.__new__(CodexProvider)
    p._client = MagicMock()
    p._model = "codex-mini-latest"
    return p


def _make_message_item(text: str):
    part = MagicMock()
    part.type = "output_text"
    part.text = text
    item = MagicMock()
    item.type = "message"
    item.content = [part]
    return item


def _make_function_call_item(call_id: str, name: str, arguments: str):
    item = MagicMock()
    item.type = "function_call"
    item.call_id = call_id
    item.name = name
    item.arguments = arguments
    return item


def _make_response(output_items, input_tokens=5, output_tokens=10):
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    response = MagicMock()
    response.output = output_items
    response.usage = usage
    return response


def test_blocking_text_response():
    p = _provider()
    p._client.responses.create.return_value = _make_response(
        [_make_message_item("Hello world")]
    )

    result = p.chat(
        system="You are helpful.",
        messages=[{"role": "user", "content": "Hi"}],
        tools=[],
        max_tokens=100,
    )

    assert result.text == "Hello world"
    assert result.tool_calls == []
    assert result.finish_reason == "stop"
    assert result.was_streamed is False
    assert result.usage == TokenUsage(input_tokens=5, output_tokens=10)


def test_blocking_instructions_sent_as_field():
    """System prompt should be passed as 'instructions', not prepended as a message."""
    p = _provider()
    p._client.responses.create.return_value = _make_response(
        [_make_message_item("ok")]
    )

    p.chat(system="Be concise.", messages=[], tools=[], max_tokens=50)

    call_kwargs = p._client.responses.create.call_args[1]
    assert call_kwargs["instructions"] == "Be concise."
    assert "instructions" in call_kwargs
    # The system prompt must NOT appear as a message in 'input'.
    for item in call_kwargs.get("input", []):
        assert item.get("role") != "system"


def test_blocking_tool_call_response():
    p = _provider()
    p._client.responses.create.return_value = _make_response([
        _make_function_call_item("call_1", "read", '{"path": "/tmp/x"}')
    ])

    result = p.chat(
        system="",
        messages=[{"role": "user", "content": "read file"}],
        tools=[],
        max_tokens=100,
    )

    assert result.text == ""
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.name == "read"
    assert tc.arguments == {"path": "/tmp/x"}
    assert tc.error is None
    assert result.finish_reason == "tool_calls"


def test_blocking_tool_call_invalid_json():
    p = _provider()
    p._client.responses.create.return_value = _make_response([
        _make_function_call_item("call_bad", "tool", "not_json{")
    ])

    result = p.chat(system="", messages=[], tools=[], max_tokens=50)

    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.error is not None
    assert "Invalid JSON" in tc.error


def test_blocking_tools_converted_before_call():
    """Tool definitions should be flattened from Chat Completions format."""
    p = _provider()
    p._client.responses.create.return_value = _make_response([_make_message_item("ok")])

    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a bash command",
                "parameters": {"type": "object"},
            },
        }
    ]
    p.chat(system="", messages=[], tools=tools, max_tokens=50)

    call_kwargs = p._client.responses.create.call_args[1]
    assert "tools" in call_kwargs
    sent_tool = call_kwargs["tools"][0]
    # Must be flat (no nested "function" key).
    assert sent_tool["name"] == "bash"
    assert "function" not in sent_tool


def test_blocking_no_tools_omits_field():
    """When tools=[], the 'tools' key should not be sent to the API."""
    p = _provider()
    p._client.responses.create.return_value = _make_response([_make_message_item("ok")])

    p.chat(system="", messages=[], tools=[], max_tokens=50)

    call_kwargs = p._client.responses.create.call_args[1]
    assert "tools" not in call_kwargs


def test_blocking_mixed_output_items():
    """Text and tool call in the same response should both be captured."""
    p = _provider()
    p._client.responses.create.return_value = _make_response([
        _make_message_item("Let me read that."),
        _make_function_call_item("c1", "read", "{}"),
    ])

    result = p.chat(system="", messages=[], tools=[], max_tokens=100)

    assert result.text == "Let me read that."
    assert len(result.tool_calls) == 1
    assert result.finish_reason == "tool_calls"


def test_blocking_reasoning_items_ignored():
    """Reasoning (thinking) output items should be silently skipped."""
    reasoning_item = MagicMock()
    reasoning_item.type = "reasoning"

    p = _provider()
    p._client.responses.create.return_value = _make_response([
        reasoning_item,
        _make_message_item("Final answer"),
    ])

    result = p.chat(system="", messages=[], tools=[], max_tokens=100)
    assert result.text == "Final answer"
    assert result.tool_calls == []


# ---------------------------------------------------------------------------
# CodexProvider — streaming path
# ---------------------------------------------------------------------------


def test_streaming_delivers_tokens_via_callback():
    p = _provider()

    # Build a mock stream context manager.
    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.text_stream = iter(["Hello", " world"])
    mock_stream.get_final_response.return_value = _make_response(
        [_make_message_item("Hello world")]
    )
    p._client.responses.stream.return_value = mock_stream

    tokens: list[str] = []
    result = p.chat(
        system="",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        max_tokens=100,
        on_token=tokens.append,
    )

    assert tokens == ["Hello", " world"]
    assert result.text == "Hello world"
    assert result.was_streamed is True


def test_streaming_tool_call_from_final_response():
    """Tool calls should be extracted from the final response after streaming."""
    p = _provider()

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.text_stream = iter([])  # no text, only tool call
    mock_stream.get_final_response.return_value = _make_response([
        _make_function_call_item("c1", "bash", '{"cmd": "ls"}')
    ])
    p._client.responses.stream.return_value = mock_stream

    tokens: list[str] = []
    result = p.chat(system="", messages=[], tools=[], max_tokens=100, on_token=tokens.append)

    assert result.text == ""
    assert result.was_streamed is False
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "bash"
    assert result.finish_reason == "tool_calls"


def test_streaming_was_streamed_false_when_no_text():
    """was_streamed must be False when no text tokens were delivered."""
    p = _provider()

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.text_stream = iter([])
    mock_stream.get_final_response.return_value = _make_response([_make_message_item("")])
    p._client.responses.stream.return_value = mock_stream

    result = p.chat(system="", messages=[], tools=[], max_tokens=50, on_token=lambda t: None)
    assert result.was_streamed is False

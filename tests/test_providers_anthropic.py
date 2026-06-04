"""Tests for Anthropic message conversion (pure functions) and provider paths.

Pure-function tests (_to_anthropic_messages, _format_tools) need no mocking.
Provider path tests (_chat_blocking, _chat_streaming) use object.__new__ to
inject a fake client — the anthropic package does not need to be installed.
"""

from unittest.mock import MagicMock, Mock

from mini_minion.providers.anthropic import (
    AnthropicProvider,
    _format_tools,
    _to_anthropic_messages,
)

# ---------------------------------------------------------------------------
# _to_anthropic_messages — pure function, no mock needed
# ---------------------------------------------------------------------------


def test_plain_user_and_assistant_messages_pass_through():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert _to_anthropic_messages(msgs) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_assistant_tool_calls_become_tool_use_blocks():
    msgs = [{"role": "assistant", "content": "", "tool_calls": [{
        "id": "tc1", "type": "function",
        "function": {"name": "read", "arguments": '{"path": "/tmp"}'},
    }]}]
    result = _to_anthropic_messages(msgs)
    blocks = result[0]["content"]
    assert blocks[0] == {
        "type": "tool_use", "id": "tc1", "name": "read", "input": {"path": "/tmp"},
    }


def test_assistant_text_before_tool_call_preserved_as_text_block():
    msgs = [{"role": "assistant", "content": "thinking...", "tool_calls": [{
        "id": "tc1", "type": "function",
        "function": {"name": "bash", "arguments": "{}"},
    }]}]
    blocks = _to_anthropic_messages(msgs)[0]["content"]
    assert blocks[0] == {"type": "text", "text": "thinking..."}
    assert blocks[1]["type"] == "tool_use"


def test_consecutive_tool_results_merged_into_single_user_message():
    msgs = [
        {"role": "tool", "tool_call_id": "a", "content": "res_a"},
        {"role": "tool", "tool_call_id": "b", "content": "res_b"},
    ]
    result = _to_anthropic_messages(msgs)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"][0]["tool_use_id"] == "a"
    assert result[0]["content"][1]["tool_use_id"] == "b"


def test_tool_result_not_merged_after_plain_user_message():
    """A tool result after a plain user message must start a new user turn."""
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "tool_call_id": "x", "content": "result"},
    ]
    result = _to_anthropic_messages(msgs)
    assert len(result) == 2
    assert result[0]["content"] == "hi"
    assert isinstance(result[1]["content"], list)
    assert result[1]["content"][0]["type"] == "tool_result"


def test_tool_call_arguments_parsed_from_json_string():
    """JSON-string arguments must be parsed into a dict for Anthropic's 'input' field."""
    msgs = [{"role": "assistant", "content": "", "tool_calls": [{
        "id": "tc1", "type": "function",
        "function": {"name": "bash", "arguments": '{"command": "ls"}'},
    }]}]
    blocks = _to_anthropic_messages(msgs)[0]["content"]
    assert blocks[0]["input"] == {"command": "ls"}


# ---------------------------------------------------------------------------
# _format_tools — pure function, no mock needed
# ---------------------------------------------------------------------------


def test_format_tools_renames_parameters_to_input_schema():
    tools = [{"type": "function", "function": {
        "name": "bash",
        "description": "Run a command.",
        "parameters": {"type": "object", "properties": {}},
    }}]
    assert _format_tools(tools) == [{
        "name": "bash",
        "description": "Run a command.",
        "input_schema": {"type": "object", "properties": {}},
    }]


def test_format_tools_handles_multiple_tools():
    tools = [
        {"type": "function", "function": {"name": "a", "description": "A.", "parameters": {}}},
        {"type": "function", "function": {"name": "b", "description": "B.", "parameters": {}}},
    ]
    result = _format_tools(tools)
    assert len(result) == 2
    assert result[0]["name"] == "a"
    assert result[1]["name"] == "b"


# ---------------------------------------------------------------------------
# Provider path helpers
# ---------------------------------------------------------------------------


def _provider() -> AnthropicProvider:
    """Build a provider with a mock client, bypassing __init__ entirely."""
    p = object.__new__(AnthropicProvider)
    p._client = Mock()
    p._model = "claude-3"
    return p


def _block(type_: str, **kwargs):
    b = Mock()
    b.type = type_
    for k, v in kwargs.items():
        setattr(b, k, v)
    return b


# ---------------------------------------------------------------------------
# _chat_blocking with fake client
# ---------------------------------------------------------------------------


def test_blocking_text_response():
    p = _provider()
    resp = Mock(content=[_block("text", text="Hi!")], stop_reason="end_turn")
    p._client.messages.create.return_value = resp

    result = p._chat_blocking({})

    assert result.text == "Hi!"
    assert result.tool_calls == []
    # Anthropic stop_reason is passed through as-is; runner only checks != "tool_calls"
    assert result.finish_reason == "end_turn"
    assert result.was_streamed is False


def test_blocking_tool_call_response():
    p = _provider()
    resp = Mock(
        content=[_block("tool_use", id="tc1", name="read", input={"path": "/tmp"})],
        stop_reason="tool_use",
    )
    p._client.messages.create.return_value = resp

    result = p._chat_blocking({})

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "tc1"
    assert result.tool_calls[0].name == "read"
    assert result.tool_calls[0].arguments == {"path": "/tmp"}
    assert result.finish_reason == "tool_calls"


def test_blocking_mixed_text_and_tool_call():
    """Text before a tool call must be preserved in the response text."""
    p = _provider()
    resp = Mock(
        content=[
            _block("text", text="Let me check."),
            _block("tool_use", id="tc1", name="bash", input={"command": "ls"}),
        ],
        stop_reason="tool_use",
    )
    p._client.messages.create.return_value = resp

    result = p._chat_blocking({})

    assert result.text == "Let me check."
    assert result.tool_calls[0].name == "bash"


# ---------------------------------------------------------------------------
# _chat_streaming with fake client
# ---------------------------------------------------------------------------


def _make_stream(text_tokens, final_content_blocks, stop_reason="end_turn"):
    final = Mock(content=final_content_blocks, stop_reason=stop_reason)
    stream = MagicMock()
    stream.__enter__.return_value = stream
    stream.__exit__.return_value = False
    stream.text_stream = text_tokens
    stream.get_final_message.return_value = final
    return stream


def test_streaming_delivers_tokens_and_assembles_text():
    p = _provider()
    p._client.messages.stream.return_value = _make_stream(
        text_tokens=["Hel", "lo!"],
        final_content_blocks=[_block("text", text="Hello!")],
    )

    tokens: list[str] = []
    result = p._chat_streaming({}, on_token=tokens.append)

    assert tokens == ["Hel", "lo!"]
    assert result.text == "Hello!"
    assert result.was_streamed is True
    assert result.finish_reason == "end_turn"


def test_streaming_tool_call_extracted_from_final_message():
    p = _provider()
    p._client.messages.stream.return_value = _make_stream(
        text_tokens=[],
        final_content_blocks=[_block("tool_use", id="tc1", name="bash", input={"command": "ls"})],
        stop_reason="tool_use",
    )

    result = p._chat_streaming({}, on_token=lambda t: None)

    assert result.tool_calls[0].name == "bash"
    assert result.tool_calls[0].arguments == {"command": "ls"}
    assert result.finish_reason == "tool_calls"
    assert result.was_streamed is False


def test_streaming_was_streamed_false_when_no_text_tokens():
    p = _provider()
    p._client.messages.stream.return_value = _make_stream(
        text_tokens=[],
        final_content_blocks=[_block("tool_use", id="x", name="read", input={})],
        stop_reason="tool_use",
    )

    result = p._chat_streaming({}, on_token=lambda t: None)

    assert result.was_streamed is False


# ---------------------------------------------------------------------------
# IMP-07: Usage population
# ---------------------------------------------------------------------------


def test_blocking_populates_usage_when_api_returns_it():
    from mini_minion.providers.base import TokenUsage
    p = _provider()
    usage = Mock()
    usage.input_tokens = 120
    usage.output_tokens = 40
    resp = Mock(content=[_block("text", text="ok")], stop_reason="end_turn", usage=usage)
    p._client.messages.create.return_value = resp

    result = p._chat_blocking({})

    assert isinstance(result.usage, TokenUsage)
    assert result.usage.input_tokens == 120
    assert result.usage.output_tokens == 40

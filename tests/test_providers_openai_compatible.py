"""Behavioral tests for OpenAICompatibleProvider blocking and streaming paths.

Uses a fake _client injected via object.__new__ — no real API connection needed.
"""

from unittest.mock import Mock

from mini_minion.providers.openai_compatible import OpenAICompatibleProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _provider() -> OpenAICompatibleProvider:
    """Build a provider with a mock client, bypassing __init__."""
    p = object.__new__(OpenAICompatibleProvider)
    p._client = Mock()
    p._model = "test-model"
    return p


def _blocking_response(content=None, tool_calls=None, finish_reason="stop"):
    msg = Mock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    choice = Mock()
    choice.message = msg
    choice.finish_reason = finish_reason
    return Mock(choices=[choice])


def _tc_mock(id: str, name: str, arguments_json: str):
    fn = Mock()
    fn.name = name
    fn.arguments = arguments_json
    tc = Mock()
    tc.id = id
    tc.function = fn
    return tc


def _stream_chunk(content=None, tc_deltas=None, finish_reason=None):
    delta = Mock()
    delta.content = content
    delta.tool_calls = tc_deltas or []
    choice = Mock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    return Mock(choices=[choice])


def _tc_delta(index: int, id=None, name=None, arguments=None):
    fn = Mock()
    fn.name = name
    fn.arguments = arguments
    tc = Mock()
    tc.index = index
    tc.id = id
    tc.function = fn
    return tc


# ---------------------------------------------------------------------------
# Blocking path
# ---------------------------------------------------------------------------


def test_blocking_text_response():
    p = _provider()
    p._client.chat.completions.create.return_value = _blocking_response(content="Hello!")

    result = p._chat_blocking({})

    assert result.text == "Hello!"
    assert result.tool_calls == []
    assert result.finish_reason == "stop"
    assert result.was_streamed is False


def test_blocking_tool_call_extracted():
    p = _provider()
    tc = _tc_mock("tc1", "read", '{"path": "/tmp"}')
    p._client.chat.completions.create.return_value = _blocking_response(
        tool_calls=[tc], finish_reason="tool_calls"
    )

    result = p._chat_blocking({})

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "tc1"
    assert result.tool_calls[0].name == "read"
    assert result.tool_calls[0].arguments == {"path": "/tmp"}
    assert result.tool_calls[0].error is None
    assert result.finish_reason == "tool_calls"


def test_blocking_malformed_tool_args_sets_error():
    p = _provider()
    tc = _tc_mock("tc1", "bash", "{broken json")
    p._client.chat.completions.create.return_value = _blocking_response(tool_calls=[tc])

    result = p._chat_blocking({})

    assert result.tool_calls[0].error is not None
    assert "Invalid JSON" in result.tool_calls[0].error


def test_blocking_finish_reason_overridden_when_tool_calls_present():
    """finish_reason must be 'tool_calls' whenever tool calls are returned."""
    p = _provider()
    tc = _tc_mock("tc1", "glob", "{}")
    # API returns "stop" but we have tool calls — finish_reason must be overridden.
    p._client.chat.completions.create.return_value = _blocking_response(
        tool_calls=[tc], finish_reason="stop"
    )

    result = p._chat_blocking({})

    assert result.finish_reason == "tool_calls"


# ---------------------------------------------------------------------------
# Streaming path
# ---------------------------------------------------------------------------


def test_streaming_assembles_text_and_calls_on_token():
    p = _provider()
    chunks = [
        _stream_chunk(content="Hel"),
        _stream_chunk(content="lo!"),
        _stream_chunk(finish_reason="stop"),
    ]
    p._client.chat.completions.create.return_value = iter(chunks)

    tokens: list[str] = []
    result = p._chat_streaming({}, on_token=tokens.append)

    assert tokens == ["Hel", "lo!"]
    assert result.text == "Hello!"
    assert result.was_streamed is True
    assert result.finish_reason == "stop"


def test_streaming_assembles_tool_call_fragments():
    """Tool-call argument JSON arriving in fragments must be concatenated correctly."""
    p = _provider()
    chunks = [
        _stream_chunk(tc_deltas=[_tc_delta(0, id="tc1", name="read", arguments=None)]),
        _stream_chunk(tc_deltas=[_tc_delta(0, arguments='{"path":')]),
        _stream_chunk(tc_deltas=[_tc_delta(0, arguments='"/tmp"}')]),
        _stream_chunk(finish_reason="tool_calls"),
    ]
    p._client.chat.completions.create.return_value = iter(chunks)

    result = p._chat_streaming({}, on_token=lambda t: None)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "tc1"
    assert result.tool_calls[0].name == "read"
    assert result.tool_calls[0].arguments == {"path": "/tmp"}
    assert result.finish_reason == "tool_calls"


def test_streaming_was_streamed_false_when_only_tool_calls():
    """was_streamed must be False when no text tokens were emitted."""
    p = _provider()
    chunks = [
        _stream_chunk(tc_deltas=[_tc_delta(0, id="x", name="bash", arguments="{}")]),
        _stream_chunk(finish_reason="tool_calls"),
    ]
    p._client.chat.completions.create.return_value = iter(chunks)

    result = p._chat_streaming({}, on_token=lambda t: None)

    assert result.was_streamed is False
    assert result.finish_reason == "tool_calls"


def test_streaming_multiple_tool_calls_assembled():
    """Multiple tool calls in parallel must each accumulate to their own index."""
    p = _provider()
    chunks = [
        _stream_chunk(tc_deltas=[
            _tc_delta(0, id="a", name="read", arguments=None),
            _tc_delta(1, id="b", name="glob", arguments=None),
        ]),
        _stream_chunk(tc_deltas=[
            _tc_delta(0, arguments='{"path":"/a"}'),
            _tc_delta(1, arguments='{"pattern":"*.py"}'),
        ]),
        _stream_chunk(finish_reason="tool_calls"),
    ]
    p._client.chat.completions.create.return_value = iter(chunks)

    result = p._chat_streaming({}, on_token=lambda t: None)

    assert len(result.tool_calls) == 2
    names = {tc.name for tc in result.tool_calls}
    assert names == {"read", "glob"}

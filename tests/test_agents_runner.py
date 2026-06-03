"""Tests for the agent TAO loop (run_turn)."""

import json
from unittest.mock import Mock

import pytest
from mini_minion.agents.events import (
    FinalAnswer,
    MaxRoundsReached,
    StreamingStarted,
    ThoughtEmitted,
    TokenStreamed,
    ToolCalled,
)
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


def _run(provider, messages, stream=False):
    """Run a turn and return collected events."""
    events: list[object] = []
    run_turn(provider, "Ada", "system", 100, ToolRegistry(), messages, on_event=events.append, stream=stream)
    return events


def test_simple_text_response():
    messages: list[dict] = []
    provider = _provider(LLMResponse(text="Hello!", finish_reason="stop"))

    events = _run(provider, messages)

    assert len(messages) == 1
    assert messages[0] == {"role": "assistant", "content": "Hello!"}
    final = next(e for e in events if isinstance(e, FinalAnswer))
    assert final.text == "Hello!"
    assert final.agent_name == "Ada"


def test_tool_call_then_stop():
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
    events: list[object] = []
    run_turn(provider, "Ada", "system", 100, registry, messages, on_event=events.append)

    # user + assistant(tool_call) + tool_result + assistant(final)
    assert len(messages) == 4
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["function"]["name"] == "echo"
    assert messages[2] == {"role": "tool", "tool_call_id": "tc1", "content": "pong"}
    assert messages[3] == {"role": "assistant", "content": "Done."}

    tool_events = [e for e in events if isinstance(e, ToolCalled)]
    assert len(tool_events) == 1
    assert tool_events[0].name == "echo"

    final = next(e for e in events if isinstance(e, FinalAnswer))
    assert final.text == "Done."


def test_preamble_text_before_tools_emits_thought_event():
    """Text emitted alongside tool calls must surface as ThoughtEmitted (non-streaming)."""
    registry = ToolRegistry()
    registry.register(_EchoTool())
    messages: list[dict] = []

    provider = _provider(
        LLMResponse(
            text="Let me look that up for you.",
            tool_calls=[ToolCall(id="tc1", name="echo", arguments={"text": "ping"})],
            finish_reason="tool_calls",
            was_streamed=False,
        ),
        LLMResponse(text="Done.", finish_reason="stop"),
    )
    events = _run(provider, messages)

    thought_events = [e for e in events if isinstance(e, ThoughtEmitted)]
    assert len(thought_events) == 1
    assert thought_events[0].text == "Let me look that up for you."
    assert thought_events[0].agent_name == "Ada"

    # ThoughtEmitted must appear before the ToolCalled event.
    thought_idx = next(i for i, e in enumerate(events) if isinstance(e, ThoughtEmitted))
    tool_idx = next(i for i, e in enumerate(events) if isinstance(e, ToolCalled))
    assert thought_idx < tool_idx


def test_no_thought_event_when_preamble_was_streamed():
    """When text was already streamed token-by-token, no ThoughtEmitted is emitted."""
    registry = ToolRegistry()
    registry.register(_EchoTool())
    messages: list[dict] = []
    _call = [0]

    def _chat(system, msgs, tools, max_tokens, on_token=None):
        _call[0] += 1
        if _call[0] == 1:
            if on_token:
                on_token("Let me check.")
            return LLMResponse(
                text="Let me check.",
                tool_calls=[ToolCall(id="tc1", name="echo", arguments={"text": "x"})],
                finish_reason="tool_calls",
                was_streamed=True,
            )
        return LLMResponse(text="Done.", finish_reason="stop")

    provider = Mock()
    provider.chat = Mock(side_effect=_chat)
    events: list[object] = []
    run_turn(provider, "Ada", "system", 100, registry, messages, on_event=events.append, stream=True)

    assert not any(isinstance(e, ThoughtEmitted) for e in events)


def test_no_thought_event_when_tool_call_has_no_text():
    """Tool calls with an empty preamble must not emit ThoughtEmitted."""
    registry = ToolRegistry()
    registry.register(_EchoTool())
    messages: list[dict] = []

    provider = _provider(
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id="tc1", name="echo", arguments={"text": "x"})],
            finish_reason="tool_calls",
        ),
        LLMResponse(text="Done.", finish_reason="stop"),
    )
    events = _run(provider, messages)

    assert not any(isinstance(e, ThoughtEmitted) for e in events)


def test_multiple_tool_calls_in_one_response():
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

    events: list[object] = []
    run_turn(provider, "Ada", "system", 100, registry, messages, on_event=events.append)

    tool_results = [m for m in messages if m["role"] == "tool"]
    assert len(tool_results) == 2
    assert tool_results[0]["content"] == "first"
    assert tool_results[1]["content"] == "second"

    tool_events = [e for e in events if isinstance(e, ToolCalled)]
    assert len(tool_events) == 2


def test_unknown_tool_returns_error_and_continues():
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

    events = _run(provider, messages)

    tool_result = next(m for m in messages if m["role"] == "tool")
    assert "nonexistent" in tool_result["content"].lower() or "unknown" in tool_result["content"].lower()
    final = next(e for e in events if isinstance(e, FinalAnswer))
    assert final.text == "recovered"


def test_tool_call_arguments_serialised():
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

    _run(provider, messages)

    raw_args = messages[0]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(raw_args) == {"text": "hi"}


def test_empty_response_emits_final_answer_on_last_round():
    """On the final allowed round, an empty response emits FinalAnswer (not a recovery nudge)."""
    messages: list[dict] = []
    provider = _provider(LLMResponse(text="", finish_reason="stop"))
    events: list[object] = []
    # max_tool_rounds=1 means round 0 is the last — no recovery budget, goes to FinalAnswer.
    run_turn(provider, "Ada", "system", 100, ToolRegistry(), messages,
             on_event=events.append, max_tool_rounds=1)
    final_events = [e for e in events if isinstance(e, FinalAnswer)]
    assert len(final_events) == 1
    assert final_events[0].text == ""


def test_no_events_when_on_event_is_none():
    """When on_event is None, run_turn completes silently without error."""
    messages: list[dict] = []
    provider = _provider(LLMResponse(text="Hello!", finish_reason="stop"))

    # Should not raise and should still mutate messages
    run_turn(provider, "Ada", "system", 100, ToolRegistry(), messages)

    assert messages == [{"role": "assistant", "content": "Hello!"}]


def test_tool_call_with_parse_error_feeds_error_to_model():
    """When tc.error is set, the parse error is fed back as the tool result instead of executing."""
    messages: list[dict] = []
    provider = _provider(
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id="x", name="echo", arguments={}, error="Invalid JSON arguments for 'echo': ...")],
            finish_reason="tool_calls",
        ),
        LLMResponse(text="I see the error, let me retry.", finish_reason="stop"),
    )

    events = _run(provider, messages)

    tool_result = next(m for m in messages if m["role"] == "tool")
    assert "Invalid JSON" in tool_result["content"]
    assert tool_result["tool_call_id"] == "x"
    final = next(e for e in events if isinstance(e, FinalAnswer))
    assert "I see the error" in final.text


def test_tool_round_limit_stops_loop():
    """Loop must stop after _MAX_TOOL_ROUNDS calls when the model never stops using tools."""
    from mini_minion.agents.runner import _MAX_TOOL_ROUNDS  # noqa: PLC0415

    registry = ToolRegistry()
    registry.register(_EchoTool())
    messages: list[dict] = []

    never_stops = [
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id=f"tc{i}", name="echo", arguments={"text": "x"})],
            finish_reason="tool_calls",
        )
        for i in range(_MAX_TOOL_ROUNDS + 5)
    ]
    provider = _provider(*never_stops)
    events: list[object] = []
    run_turn(provider, "Ada", "system", 100, registry, messages, on_event=events.append)

    assert provider.chat.call_count == _MAX_TOOL_ROUNDS
    assert messages[-1]["role"] == "assistant"
    assert "Stopped" in messages[-1]["content"]

    max_events = [e for e in events if isinstance(e, MaxRoundsReached)]
    assert len(max_events) == 1
    assert "Stopped" in max_events[0].message


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


def test_streaming_emits_token_events():
    """With stream=True and on_event provided, StreamingStarted + TokenStreamed events are emitted."""
    messages: list[dict] = []

    # Mock provider that calls on_token to simulate streaming
    def _streaming_chat(system, msgs, tools, max_tokens, on_token=None):
        if on_token:
            on_token("Hel")
            on_token("lo!")
        return LLMResponse(text="Hello!", finish_reason="stop", was_streamed=True)

    provider = Mock()
    provider.chat = Mock(side_effect=_streaming_chat)
    events: list[object] = []

    run_turn(provider, "Ada", "system", 100, ToolRegistry(), messages,
             on_event=events.append, stream=True)

    stream_start = [e for e in events if isinstance(e, StreamingStarted)]
    tokens = [e for e in events if isinstance(e, TokenStreamed)]
    assert len(stream_start) == 1
    assert stream_start[0].agent_name == "Ada"
    assert len(tokens) == 2
    assert tokens[0].token == "Hel"
    assert tokens[1].token == "lo!"


def test_no_streaming_without_stream_flag():
    """With stream=False (default), no StreamingStarted or TokenStreamed events are emitted."""
    messages: list[dict] = []

    def _streaming_chat(system, msgs, tools, max_tokens, on_token=None):
        # on_token should not be passed when stream=False
        assert on_token is None, "on_token should not be passed with stream=False"
        return LLMResponse(text="Hello!", finish_reason="stop")

    provider = Mock()
    provider.chat = Mock(side_effect=_streaming_chat)
    events: list[object] = []

    run_turn(provider, "Ada", "system", 100, ToolRegistry(), messages,
             on_event=events.append, stream=False)

    stream_start = [e for e in events if isinstance(e, StreamingStarted)]
    tokens = [e for e in events if isinstance(e, TokenStreamed)]
    assert stream_start == []
    assert tokens == []
    final = next(e for e in events if isinstance(e, FinalAnswer))
    assert final.text == "Hello!"


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------

def test_retry_on_transient_error():
    """run_turn must retry transient provider errors and succeed on a later attempt."""
    import time
    from unittest.mock import patch
    from mini_minion.providers.base import LLMResponse as R

    call_count = [0]

    def _flaky(system, msgs, tools, max_tokens, on_token=None):
        call_count[0] += 1
        if call_count[0] < 3:
            err = Exception("connection timeout")
            raise err
        return R(text="Finally!", finish_reason="stop")

    provider = Mock()
    provider.chat = Mock(side_effect=_flaky)
    messages: list[dict] = []

    # Patch sleep so the test runs instantly.
    with patch("mini_minion.agents.runner.time.sleep"):
        events = _run(provider, messages)

    assert call_count[0] == 3
    final = next(e for e in events if isinstance(e, FinalAnswer))
    assert final.text == "Finally!"


def test_no_retry_on_permanent_error():
    """Non-retryable errors (e.g. HTTP 400) must raise immediately without retry."""
    import time
    from unittest.mock import patch

    class _PermanentError(Exception):
        class response:
            status_code = 400

    call_count = [0]

    def _bad_request(system, msgs, tools, max_tokens, on_token=None):
        call_count[0] += 1
        raise _PermanentError("bad request")

    provider = Mock()
    provider.chat = Mock(side_effect=_bad_request)
    messages: list[dict] = []

    with patch("mini_minion.agents.runner.time.sleep"):
        with pytest.raises(_PermanentError):
            run_turn(provider, "Ada", "system", 100, ToolRegistry(), messages)

    assert call_count[0] == 1  # no retry


# ---------------------------------------------------------------------------
# Empty response recovery
# ---------------------------------------------------------------------------

def test_empty_response_recovery():
    """An empty response (no text, no tool calls) must inject a nudge and loop."""
    messages: list[dict] = []
    provider = _provider(
        LLMResponse(text="", finish_reason="stop"),   # empty — triggers nudge
        LLMResponse(text="Here I am!", finish_reason="stop"),  # real answer
    )
    events = _run(provider, messages)
    final = next(e for e in events if isinstance(e, FinalAnswer))
    assert final.text == "Here I am!"
    # Two provider calls made
    assert provider.chat.call_count == 2


def test_empty_response_on_last_round_emits_final_answer():
    """On the last round, empty response (no recovery budget) falls through to FinalAnswer."""
    # max_tool_rounds=1 means round 0 is last — recovery is skipped, FinalAnswer emitted.
    messages: list[dict] = []
    provider = _provider(LLMResponse(text="", finish_reason="stop"))
    events: list[object] = []
    run_turn(provider, "Ada", "system", 100, ToolRegistry(), messages,
             on_event=events.append, max_tool_rounds=1)
    final_events = [e for e in events if isinstance(e, FinalAnswer)]
    assert len(final_events) == 1
    assert final_events[0].text == ""


# ---------------------------------------------------------------------------
# Length (truncation) recovery
# ---------------------------------------------------------------------------

def test_length_recovery_injects_continuation():
    """finish_reason='length' must inject a continuation prompt and loop."""
    messages: list[dict] = []
    provider = _provider(
        LLMResponse(text="The answer is...", finish_reason="length"),
        LLMResponse(text="...42.", finish_reason="stop"),
    )
    events = _run(provider, messages)
    final = next(e for e in events if isinstance(e, FinalAnswer))
    assert "42" in final.text
    assert provider.chat.call_count == 2


# ---------------------------------------------------------------------------
# Configurable max_tool_rounds
# ---------------------------------------------------------------------------

def test_custom_max_tool_rounds():
    """max_tool_rounds parameter must override the module-level default."""
    registry = ToolRegistry()
    registry.register(_EchoTool())
    messages: list[dict] = []

    never_stops = [
        LLMResponse(
            text="",
            tool_calls=[ToolCall(id=f"t{i}", name="echo", arguments={"text": "x"})],
            finish_reason="tool_calls",
        )
        for i in range(30)
    ]
    provider = _provider(*never_stops)
    events: list[object] = []
    run_turn(provider, "Ada", "system", 100, registry, messages,
             on_event=events.append, max_tool_rounds=5)

    assert provider.chat.call_count == 5
    max_events = [e for e in events if isinstance(e, MaxRoundsReached)]
    assert len(max_events) == 1
    assert "5" in max_events[0].message

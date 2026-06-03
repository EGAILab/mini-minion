"""Tests for AgentSession — the reusable, headless agent execution unit."""

import pytest
from unittest.mock import Mock, patch

from mini_minion.agents.definitions import AgentConfig
from mini_minion.agents.events import (
    CompactionFailed,
    CompactionStarted,
    FinalAnswer,
    ToolCalled,
)
from mini_minion.agents.session import AgentSession
from mini_minion.context import Compactor
from mini_minion.memory.long_term import LongTermMemory
from mini_minion.memory.short_term import ShortTermMemory
from mini_minion.providers.base import LLMResponse
from mini_minion.session import SessionStore
from mini_minion.tools import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_provider(text="response text", finish_reason="stop"):
    provider = Mock()
    provider.chat = Mock(return_value=LLMResponse(text=text, finish_reason=finish_reason))
    return provider


def _make_session(tmp_path, provider=None, agent_id="main"):
    """Build an AgentSession wired to a tmp_path workspace."""
    if provider is None:
        provider = _mock_provider()
    short_term = ShortTermMemory(tmp_path / "sessions")
    session_store = SessionStore(tmp_path / "sessions.json")
    compactor = Compactor(context_window=100_000, preserve_tokens=2_000)
    return AgentSession(
        agent_id=agent_id,
        agent=AgentConfig(name="Ada", soul="You are Ada."),
        provider=provider,
        max_output_tokens=512,
        tools=ToolRegistry(),
        compactor=compactor,
        short_term=short_term,
        session_store=session_store,
    )


# ---------------------------------------------------------------------------
# Basic send behaviour
# ---------------------------------------------------------------------------

def test_send_returns_response_text(tmp_path):
    """send() must return the text of the model's final answer."""
    provider = _mock_provider(text="Hello from Ada!")
    session = _make_session(tmp_path, provider=provider)

    result = session.send("hi")

    assert result == "Hello from Ada!"


def test_send_appends_user_and_assistant_to_history(tmp_path):
    """After send(), history contains both the user message and assistant reply."""
    session = _make_session(tmp_path)
    session.send("hello")

    history = session.history
    assert any(m["role"] == "user" and m["content"] == "hello" for m in history)
    assert any(m["role"] == "assistant" for m in history)


def test_send_persists_history_to_disk(tmp_path):
    """History is saved to disk after each send() so it survives restarts."""
    session = _make_session(tmp_path)
    session.send("persist me")

    on_disk = ShortTermMemory(tmp_path / "sessions").load("main")
    assert any(m["role"] == "user" and m["content"] == "persist me" for m in on_disk)


def test_send_emits_final_answer_event(tmp_path):
    """send() with on_event must emit a FinalAnswer event."""
    provider = _mock_provider(text="Ada here.")
    session = _make_session(tmp_path, provider=provider)
    events: list[object] = []

    session.send("hi", on_event=events.append)

    final_events = [e for e in events if isinstance(e, FinalAnswer)]
    assert len(final_events) == 1
    assert final_events[0].text == "Ada here."
    assert final_events[0].agent_name == "Ada"


def test_send_none_response_when_no_text(tmp_path):
    """send() returns None when the model produces no text (only tool calls)."""
    provider = _mock_provider(text="")
    session = _make_session(tmp_path, provider=provider)

    result = session.send("hi")

    # Empty text → no non-empty assistant content → None
    assert result is None


# ---------------------------------------------------------------------------
# Exception handling and rollback
# ---------------------------------------------------------------------------

def test_send_raises_on_provider_exception(tmp_path):
    """send() re-raises provider exceptions after recording the error in history."""
    provider = Mock()
    provider.chat = Mock(side_effect=RuntimeError("connection refused"))
    session = _make_session(tmp_path, provider=provider)

    with pytest.raises(RuntimeError, match="connection refused"):
        session.send("will fail")


def test_send_records_error_in_history_on_exception(tmp_path):
    """After a provider exception, history contains user message + error record."""
    provider = Mock()
    provider.chat = Mock(side_effect=RuntimeError("timeout"))
    session = _make_session(tmp_path, provider=provider)

    try:
        session.send("hello")
    except RuntimeError:
        pass

    history = session.history
    assert any(m["role"] == "user" and m["content"] == "hello" for m in history)
    assert any(
        m["role"] == "assistant" and "Provider error" in m["content"]
        for m in history
    )


def test_send_rolls_back_partial_messages_on_exception(tmp_path):
    """Partial messages appended by run_turn before a crash are removed."""

    def _crash_after_partial(provider, name, soul, max_tokens, tools, messages, **kwargs):
        messages.append({"role": "assistant", "content": "", "tool_calls": []})
        raise RuntimeError("mid-turn crash")

    session = _make_session(tmp_path)
    with patch("mini_minion.agents.session.run_turn", side_effect=_crash_after_partial):
        try:
            session.send("hello")
        except RuntimeError:
            pass

    history = session.history
    # Must not contain the partial assistant message with empty tool_calls
    assert not any("tool_calls" in m for m in history)
    # User message + error record only
    assert history[0] == {"role": "user", "content": "hello"}
    assert "Provider error" in history[1]["content"]


def test_send_persists_user_message_before_run_turn(tmp_path):
    """User message is on disk even when the provider raises before returning."""
    provider = Mock()
    provider.chat = Mock(side_effect=RuntimeError("crash"))
    session = _make_session(tmp_path, provider=provider)

    try:
        session.send("my question")
    except RuntimeError:
        pass

    on_disk = ShortTermMemory(tmp_path / "sessions").load("main")
    assert on_disk[0] == {"role": "user", "content": "my question"}


# ---------------------------------------------------------------------------
# Compaction integration
# ---------------------------------------------------------------------------

def test_send_emits_compaction_started_event(tmp_path):
    """When compaction occurs, a CompactionStarted event must be emitted."""
    provider = _mock_provider()
    session = _make_session(tmp_path, provider=provider)
    # Populate history so _select has at least 2 messages to split.
    session._history = [
        {"role": "user", "content": "old message"},
        {"role": "assistant", "content": "old reply"},
    ]
    events: list[object] = []

    # Force compaction by overriding needs_compaction and the summarise call.
    with patch.object(session._compactor, "needs_compaction", return_value=True):
        with patch.object(session._compactor, "_summarise", return_value="Summary."):
            session.send("trigger compaction", on_event=events.append)

    compaction_events = [e for e in events if isinstance(e, CompactionStarted)]
    assert len(compaction_events) == 1


# ---------------------------------------------------------------------------
# Session store integration
# ---------------------------------------------------------------------------

def test_send_increments_turn_count_on_success(tmp_path):
    """Successful turn must increment the turn counter in the session store."""
    session = _make_session(tmp_path)
    store = SessionStore(tmp_path / "sessions.json")

    before = store.get_or_create("main").turn_count
    session.send("hello")
    after = store.get_or_create("main").turn_count

    assert after == before + 1


def test_send_does_not_increment_turn_count_on_exception(tmp_path):
    """Failed turn must NOT increment the turn counter."""
    provider = Mock()
    provider.chat = Mock(side_effect=RuntimeError("fail"))
    session = _make_session(tmp_path, provider=provider)
    store = SessionStore(tmp_path / "sessions.json")

    before = store.get_or_create("main").turn_count
    try:
        session.send("hello")
    except RuntimeError:
        pass
    after = store.get_or_create("main").turn_count

    assert after == before


# ---------------------------------------------------------------------------
# history property
# ---------------------------------------------------------------------------

def test_history_property_returns_defensive_copy(tmp_path):
    """Mutating the returned history list must not affect the session's internal state."""
    session = _make_session(tmp_path)
    session.send("hello")

    h = session.history
    h.clear()

    assert len(session.history) > 0


# ---------------------------------------------------------------------------
# max_tool_rounds forwarding
# ---------------------------------------------------------------------------

def test_max_tool_rounds_forwarded_to_run_turn(tmp_path):
    """AgentSession must forward agent.max_tool_rounds to run_turn."""
    provider = _mock_provider()
    short_term = ShortTermMemory(tmp_path / "sessions")
    session_store = SessionStore(tmp_path / "sessions.json")
    compactor = Compactor(context_window=100_000, preserve_tokens=2_000)
    session = AgentSession(
        agent_id="main",
        agent=AgentConfig(name="Ada", soul="You are Ada.", max_tool_rounds=7),
        provider=provider,
        max_output_tokens=512,
        tools=ToolRegistry(),
        compactor=compactor,
        short_term=short_term,
        session_store=session_store,
    )
    captured: dict = {}

    def _fake_run_turn(*args, **kwargs):
        captured["max_tool_rounds"] = kwargs.get("max_tool_rounds")

    with patch("mini_minion.agents.session.run_turn", side_effect=_fake_run_turn):
        try:
            session.send("hello")
        except Exception:
            pass

    assert captured.get("max_tool_rounds") == 7


# ---------------------------------------------------------------------------
# CompactionFailed event
# ---------------------------------------------------------------------------

def test_send_emits_compaction_failed_event_on_summarisation_error(tmp_path):
    """When compaction summarisation fails, CompactionFailed event must be emitted."""
    provider = _mock_provider()
    session = _make_session(tmp_path, provider=provider)
    session._history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "reply"},
    ]
    events: list[object] = []

    with patch.object(session._compactor, "needs_compaction", return_value=True):
        with patch.object(
            session._compactor,
            "_summarise",
            side_effect=RuntimeError("summarisation failed"),
        ):
            session.send("new message", on_event=events.append)

    failed = [e for e in events if isinstance(e, CompactionFailed)]
    assert len(failed) == 1
    assert "RuntimeError" in failed[0].error


# ---------------------------------------------------------------------------
# User context injection
# ---------------------------------------------------------------------------

def test_user_context_injected_into_system_prompt(tmp_path):
    """When user_context.md exists, its content must appear in the system prompt."""
    long_term = LongTermMemory(tmp_path / "memory")
    long_term.save("user_context", "User is a Python expert.")

    short_term = ShortTermMemory(tmp_path / "sessions")
    session_store = SessionStore(tmp_path / "sessions.json")
    compactor = Compactor(context_window=100_000, preserve_tokens=2_000)
    provider = _mock_provider()

    session = AgentSession(
        agent_id="main",
        agent=AgentConfig(name="Ada", soul="Base soul."),
        provider=provider,
        max_output_tokens=512,
        tools=ToolRegistry(),
        compactor=compactor,
        short_term=short_term,
        session_store=session_store,
        long_term=long_term,
    )

    captured_system: list[str] = []

    def _capture(system, msgs, tools, max_tokens, on_token=None):
        captured_system.append(system)
        return LLMResponse(text="ok", finish_reason="stop")

    provider.chat = Mock(side_effect=_capture)
    session.send("hello")

    assert len(captured_system) == 1
    assert "User is a Python expert." in captured_system[0]
    assert "user_context" in captured_system[0].lower()


def test_no_user_context_when_file_absent(tmp_path):
    """When user_context.md is absent, the system prompt must not contain the block."""
    long_term = LongTermMemory(tmp_path / "memory")
    # Do NOT save user_context — file absent

    short_term = ShortTermMemory(tmp_path / "sessions")
    session_store = SessionStore(tmp_path / "sessions.json")
    compactor = Compactor(context_window=100_000, preserve_tokens=2_000)
    provider = _mock_provider()

    session = AgentSession(
        agent_id="main",
        agent=AgentConfig(name="Ada", soul="Base soul."),
        provider=provider,
        max_output_tokens=512,
        tools=ToolRegistry(),
        compactor=compactor,
        short_term=short_term,
        session_store=session_store,
        long_term=long_term,
    )

    captured_system: list[str] = []

    def _capture(system, msgs, tools, max_tokens, on_token=None):
        captured_system.append(system)
        return LLMResponse(text="ok", finish_reason="stop")

    provider.chat = Mock(side_effect=_capture)
    session.send("hello")

    assert "<user_context>" not in captured_system[0]

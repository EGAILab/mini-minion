"""Tests for AgentSession — the reusable, headless agent execution unit."""

import pytest
from unittest.mock import Mock, patch

from minion_assist.agents.definitions import AgentConfig
from minion_assist.agents.events import (
    CompactionFailed,
    CompactionStarted,
    FinalAnswer,
    ToolCalled,
)
from minion_assist.agents.session import AgentSession
from minion_assist.context import Compactor
from minion_assist.memory.long_term import LongTermMemory
from minion_assist.memory.short_term import ShortTermMemory
from minion_assist.providers.base import LLMResponse
from minion_assist.session import SessionStore
from minion_assist.tools import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_provider(text="response text", finish_reason="stop"):
    provider = Mock()
    provider.chat = Mock(return_value=LLMResponse(text=text, finish_reason=finish_reason))
    return provider


def _make_session(tmp_path, provider=None, agent_id="main", session_id="test-session"):
    """Build an AgentSession wired to a tmp_path workspace."""
    if provider is None:
        provider = _mock_provider()
    short_term = ShortTermMemory(tmp_path / "sessions")
    session_store = SessionStore(tmp_path / "sessions.json")
    compactor = Compactor(context_window=100_000, preserve_tokens=2_000)
    return AgentSession(
        agent_id=agent_id,
        session_id=session_id,
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

    on_disk = ShortTermMemory(tmp_path / "sessions").load("main", "test-session")
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
    with patch("minion_assist.agents.session.run_turn", side_effect=_crash_after_partial):
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

    on_disk = ShortTermMemory(tmp_path / "sessions").load("main", "test-session")
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
    # Create fresh instances before and after to verify disk persistence.
    # The in-memory cache is per-instance, so a new instance always reads from disk.
    before = SessionStore(tmp_path / "sessions.json").get_or_create("main").turn_count
    session.send("hello")
    after = SessionStore(tmp_path / "sessions.json").get_or_create("main").turn_count

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
        session_id="test-session",
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

    with patch("minion_assist.agents.session.run_turn", side_effect=_fake_run_turn):
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
        session_id="test-session",
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
        session_id="test-session",
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


# ---------------------------------------------------------------------------
# Task context auto-injection (B2 architectural fix)
# ---------------------------------------------------------------------------

def _make_task_file(tasks_dir, agent_id, goal, steps):
    """Write a minimal task JSON file for testing."""
    import json
    path = tasks_dir / f"{agent_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"goal": goal, "steps": steps}), encoding="utf-8")
    return path


def _make_session_with_tasks(tmp_path, tasks_dir, provider=None, agent_id="main"):
    if provider is None:
        provider = _mock_provider()
    short_term = ShortTermMemory(tmp_path / "sessions")
    session_store = SessionStore(tmp_path / "sessions.json")
    compactor = Compactor(context_window=100_000, preserve_tokens=2_000)
    return AgentSession(
        agent_id=agent_id,
        session_id="test-session",
        agent=AgentConfig(name="Ada", soul="You are Ada."),
        provider=provider,
        max_output_tokens=512,
        tools=ToolRegistry(),
        compactor=compactor,
        short_term=short_term,
        session_store=session_store,
        tasks_dir=tasks_dir,
    )


def test_active_task_injected_into_system_prompt(tmp_path):
    """When a task file exists, <active_task> block must appear in the system prompt."""
    tasks_dir = tmp_path / "tasks"
    _make_task_file(tasks_dir, "main", "Build the API", [
        {"id": 1, "description": "Design schema", "status": "done"},
        {"id": 2, "description": "Write endpoints", "status": "pending"},
    ])
    provider = _mock_provider()
    session = _make_session_with_tasks(tmp_path, tasks_dir, provider=provider)

    captured: list[str] = []
    provider.chat = Mock(side_effect=lambda s, *a, **kw: (captured.append(s), LLMResponse(text="ok", finish_reason="stop"))[1])
    session.send("hello")

    assert len(captured) == 1
    assert "<active_task>" in captured[0]
    assert "Build the API" in captured[0]
    assert "Design schema" in captured[0]


def test_no_active_task_when_file_absent(tmp_path):
    """When no task file exists, <active_task> must not appear in the system prompt."""
    tasks_dir = tmp_path / "tasks"  # directory exists but no file inside
    tasks_dir.mkdir(parents=True)
    provider = _mock_provider()
    session = _make_session_with_tasks(tmp_path, tasks_dir, provider=provider)

    captured: list[str] = []
    provider.chat = Mock(side_effect=lambda s, *a, **kw: (captured.append(s), LLMResponse(text="ok", finish_reason="stop"))[1])
    session.send("hello")

    assert "<active_task>" not in captured[0]


def test_in_progress_step_injects_update_task_reminder(tmp_path):
    """When a step is in_progress, the <active_task> block must include the update reminder."""
    tasks_dir = tmp_path / "tasks"
    _make_task_file(tasks_dir, "main", "Write tests", [
        {"id": 1, "description": "Set up pytest", "status": "done"},
        {"id": 2, "description": "Write unit tests", "status": "in_progress"},
    ])
    provider = _mock_provider()
    session = _make_session_with_tasks(tmp_path, tasks_dir, provider=provider)

    captured: list[str] = []
    provider.chat = Mock(side_effect=lambda s, *a, **kw: (captured.append(s), LLMResponse(text="ok", finish_reason="stop"))[1])
    session.send("hello")

    assert "update_task" in captured[0]
    assert "Step 2" in captured[0]


def test_no_update_task_reminder_when_no_step_in_progress(tmp_path):
    """When no step is in_progress, the update reminder must NOT appear in the prompt."""
    tasks_dir = tmp_path / "tasks"
    _make_task_file(tasks_dir, "main", "Deploy", [
        {"id": 1, "description": "Build image", "status": "done"},
        {"id": 2, "description": "Push image", "status": "pending"},
    ])
    provider = _mock_provider()
    session = _make_session_with_tasks(tmp_path, tasks_dir, provider=provider)

    captured: list[str] = []
    provider.chat = Mock(side_effect=lambda s, *a, **kw: (captured.append(s), LLMResponse(text="ok", finish_reason="stop"))[1])
    session.send("hello")

    assert "update_task" not in captured[0]


# ---------------------------------------------------------------------------
# Context budget warning (D1+K1 architectural fix)
# ---------------------------------------------------------------------------

def test_budget_warning_injected_into_system_prompt(tmp_path):
    """When _format_budget_context returns a block, send() must include it in the system prompt."""
    # Test the injection path independently — we patch _format_budget_context to return
    # a known string so the test doesn't depend on token-counting thresholds.
    from minion_assist.agents import session as session_module

    provider = _mock_provider()
    sess = _make_session(tmp_path, provider=provider)

    fake_block = "<context_budget>\nApproximately 65% used.\n</context_budget>"
    captured: list[str] = []

    with patch.object(session_module, "_format_budget_context", return_value=fake_block):
        provider.chat = Mock(side_effect=lambda s, *a, **kw: (captured.append(s), LLMResponse(text="ok", finish_reason="stop"))[1])
        sess.send("hello")

    assert fake_block in captured[0]


def test_budget_warning_absent_when_history_small(tmp_path):
    """<context_budget> must NOT appear when history is well within the usable budget."""
    from minion_assist.agents.session import _format_budget_context

    compactor = Compactor(context_window=100_000, preserve_tokens=2_000)
    tiny_history = [{"role": "user", "content": "hello"}]
    # usable = 98_000 tokens; tiny history is well below 50% threshold
    assert _format_budget_context(tiny_history, compactor) == ""


def test_budget_warning_function_fires_at_threshold():
    """_format_budget_context must return a non-empty block when history exceeds 50% usable."""
    from minion_assist.agents.session import _format_budget_context

    compactor = Compactor(context_window=10_000, preserve_tokens=2_000)
    # usable = 8_000 tokens; 50% = 4_000.  Fill with enough text (varied content
    # avoids BPE merging) to exceed the threshold without triggering compaction.
    large_history = [{"role": "user", "content": f"word{i} " * 300} for i in range(20)]
    result = _format_budget_context(large_history, compactor)
    assert "<context_budget>" in result
    assert "%" in result


# ---------------------------------------------------------------------------
# IMP-08: TurnCompleted event
# ---------------------------------------------------------------------------


def test_turn_completed_event_emitted(tmp_path):
    """TurnCompleted must be emitted after every successful turn."""
    import uuid
    from minion_assist.agents.events import TurnCompleted

    provider = _mock_provider(text="done")
    session = _make_session(tmp_path, provider=provider)
    events: list[object] = []
    session.send("hello", on_event=events.append)

    completed = [e for e in events if isinstance(e, TurnCompleted)]
    assert len(completed) == 1
    tc = completed[0]
    assert tc.agent_name == "Ada"
    assert uuid.UUID(tc.trace_id)   # must be a valid UUID
    assert tc.turn_number == 1
    assert tc.elapsed_ms >= 0
    assert isinstance(tc.compacted, bool)


def test_turn_completed_not_emitted_on_failure(tmp_path):
    """TurnCompleted must NOT be emitted when the turn raises."""
    from minion_assist.agents.events import TurnCompleted
    provider = Mock()
    provider.chat = Mock(side_effect=RuntimeError("fail"))
    session = _make_session(tmp_path, provider=provider)
    events: list[object] = []
    try:
        session.send("hello", on_event=events.append)
    except RuntimeError:
        pass
    assert not any(isinstance(e, TurnCompleted) for e in events)


# ---------------------------------------------------------------------------
# reload() — restores history from disk
# ---------------------------------------------------------------------------

def test_reload_restores_history_from_disk(tmp_path):
    """reload() must replace in-memory history with what is on disk."""
    session = _make_session(tmp_path)
    session.send("persisted message")

    # Corrupt the in-memory history to simulate drift.
    session._history = []
    assert session.history == []

    session.reload()

    # After reload, history should match what send() persisted.
    assert any(m["content"] == "persisted message" for m in session.history)


def test_reload_on_empty_disk_clears_history(tmp_path):
    """reload() on a session that was never sent clears in-memory history too."""
    session = _make_session(tmp_path)
    # Inject some in-memory history without persisting it.
    session._history = [{"role": "user", "content": "phantom"}]

    session.reload()

    # Nothing was ever persisted, so reload loads empty history.
    assert session.history == []


def test_reload_is_idempotent(tmp_path):
    """Calling reload() twice returns the same result both times."""
    session = _make_session(tmp_path)
    session.send("hello")

    session.reload()
    h1 = list(session.history)
    session.reload()
    h2 = list(session.history)

    assert h1 == h2


# ---------------------------------------------------------------------------
# enable_memory_extraction=False — suppresses background extraction
# ---------------------------------------------------------------------------

def test_enable_memory_extraction_false_suppresses_extraction(tmp_path):
    """When enable_memory_extraction=False, the extraction daemon thread is never started."""
    long_term = LongTermMemory(tmp_path / "memory")
    short_term = ShortTermMemory(tmp_path / "sessions")
    session_store = SessionStore(tmp_path / "sessions.json")
    compactor = Compactor(context_window=100_000, preserve_tokens=2_000)
    provider = _mock_provider()

    session = AgentSession(
        agent_id="main",
        session_id="test-session",
        agent=AgentConfig(name="Ada", soul="You are Ada."),
        provider=provider,
        max_output_tokens=512,
        tools=ToolRegistry(),
        compactor=compactor,
        short_term=short_term,
        session_store=session_store,
        long_term=long_term,
        enable_memory_extraction=False,
    )

    extraction_calls: list[bool] = []

    def fake_extract(*args, **kwargs):
        extraction_calls.append(True)

    with patch("minion_assist.memory.extractor.extract_and_save_async", side_effect=fake_extract):
        session.send("hello")

    # The extraction function must NOT have been called.
    assert not extraction_calls


def test_enable_memory_extraction_true_allows_extraction(tmp_path):
    """When enable_memory_extraction=True (default), extraction is triggered after a turn."""
    long_term = LongTermMemory(tmp_path / "memory")
    short_term = ShortTermMemory(tmp_path / "sessions")
    session_store = SessionStore(tmp_path / "sessions.json")
    compactor = Compactor(context_window=100_000, preserve_tokens=2_000)
    provider = _mock_provider()

    session = AgentSession(
        agent_id="main",
        session_id="test-session",
        agent=AgentConfig(name="Ada", soul="You are Ada."),
        provider=provider,
        max_output_tokens=512,
        tools=ToolRegistry(),
        compactor=compactor,
        short_term=short_term,
        session_store=session_store,
        long_term=long_term,
        enable_memory_extraction=True,
    )

    extraction_calls: list[bool] = []

    def fake_extract(*args, **kwargs):
        extraction_calls.append(True)

    with patch("minion_assist.memory.extractor.extract_and_save_async", side_effect=fake_extract):
        session.send("hello")

    assert extraction_calls


# ---------------------------------------------------------------------------
# Reseed context injection
# ---------------------------------------------------------------------------

def test_reseed_context_injected_on_first_send(tmp_path):
    """reseed_context must appear in the system prompt on the first send only."""
    provider = _mock_provider()
    session = AgentSession(
        agent_id="main",
        session_id="new-session",
        reseed_context="<prior_session_history>\nUser: hi\nAssistant: hello\n</prior_session_history>",
        agent=AgentConfig(name="Ada", soul="You are Ada."),
        provider=provider,
        max_output_tokens=512,
        tools=ToolRegistry(),
        compactor=Compactor(context_window=100_000, preserve_tokens=2_000),
        short_term=ShortTermMemory(tmp_path / "sessions"),
        session_store=SessionStore(tmp_path / "sessions.json"),
    )

    captured: list[str] = []

    def _capture(system, msgs, tools, max_tokens, on_token=None):
        captured.append(system)
        return LLMResponse(text="ok", finish_reason="stop")

    provider.chat = Mock(side_effect=_capture)

    session.send("first message")
    session.send("second message")

    assert "<prior_session_history>" in captured[0]
    assert "<prior_session_history>" not in captured[1]


def test_reseed_context_none_does_not_pollute_system(tmp_path):
    """When reseed_context is None, the system prompt must not contain history tags."""
    provider = _mock_provider()
    session = AgentSession(
        agent_id="main",
        session_id="fresh",
        reseed_context=None,
        agent=AgentConfig(name="Ada", soul="You are Ada."),
        provider=provider,
        max_output_tokens=512,
        tools=ToolRegistry(),
        compactor=Compactor(context_window=100_000, preserve_tokens=2_000),
        short_term=ShortTermMemory(tmp_path / "sessions"),
        session_store=SessionStore(tmp_path / "sessions.json"),
    )

    captured: list[str] = []

    def _capture(system, msgs, tools, max_tokens, on_token=None):
        captured.append(system)
        return LLMResponse(text="ok", finish_reason="stop")

    provider.chat = Mock(side_effect=_capture)
    session.send("hello")

    assert "<prior_session_history>" not in captured[0]

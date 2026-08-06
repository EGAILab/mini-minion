"""Tests for AgentSession — the reusable, headless agent execution unit."""

import pytest
from unittest.mock import Mock, patch

from minion_assist.agents.definitions import AgentConfig
from minion_assist.agents.events import (
    CompactionFailed,
    CompactionStarted,
    FinalAnswer,
    MemoryInjected,
    ToolCalled,
)
from minion_assist.agents.session import AgentSession, build_prompt_section
from minion_assist.context import Compactor
from minion_assist.memory.files import MemoryFileRepository
from minion_assist.memory.models import MemoryHit
from minion_assist.memory.service import MemoryService
from minion_assist.memory.short_term import ShortTermMemory
from minion_assist.messages import EVENT_ID_KEY
from minion_assist.providers.base import LLMResponse
from minion_assist.session import SessionStore
from minion_assist.tools import ToolRegistry
from minion_assist.worker_health import WorkerHealth


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
    # User message + error record only. _event_id (Phase 2 slice A's mirroring
    # identity) is expected on every message now — check role/content only.
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "hello"
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
    # _event_id (Phase 2 slice A's mirroring identity) is expected now —
    # check role/content only.
    assert on_disk[0]["role"] == "user"
    assert on_disk[0]["content"] == "my question"


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
# Pre-compaction flush (Stage One Phase 2, slice B)
# ---------------------------------------------------------------------------

def _make_session_with_memory(tmp_path, memory):
    short_term = ShortTermMemory(tmp_path / "sessions")
    session_store = SessionStore(tmp_path / "sessions.json")
    compactor = Compactor(context_window=100_000, preserve_tokens=2_000)
    return AgentSession(
        agent_id="main",
        session_id="test-session",
        agent=AgentConfig(name="Ada", soul="You are Ada."),
        provider=_mock_provider(),
        max_output_tokens=512,
        tools=ToolRegistry(),
        compactor=compactor,
        short_term=short_term,
        session_store=session_store,
        memory=memory,
    )


def test_memory_property_returns_the_configured_memory_service(tmp_path):
    # MEM-GAP-003: /delete-session reaches a session's MemoryService via
    # this property to clean up proposal-derived index/evidence data.
    memory = MemoryService(MemoryFileRepository(tmp_path / "workspace"))
    session = _make_session_with_memory(tmp_path, memory)

    assert session.memory is memory


def test_memory_property_is_none_when_no_memory_service_was_configured(tmp_path):
    session = _make_session(tmp_path)

    assert session.memory is None


def test_flush_head_writes_daily_note_before_compaction(tmp_path):
    """Compaction's about-to-be-summarized head is flushed to a daily note first."""
    memory = MemoryService(MemoryFileRepository(tmp_path / "workspace"))
    session = _make_session_with_memory(tmp_path, memory)
    session._history = [
        {"role": "user", "content": "important context to preserve"},
        {"role": "assistant", "content": "old reply"},
    ]

    with patch.object(session._compactor, "needs_compaction", return_value=True):
        with patch.object(session._compactor, "_summarise", return_value="Summary."):
            session.send("trigger compaction")

    from datetime import date as _date
    daily_file = tmp_path / "workspace" / "memory" / f"{_date.today().isoformat()}.md"
    assert daily_file.exists()
    assert "important context to preserve" in daily_file.read_text(encoding="utf-8")


def test_memory_flushed_event_emitted_when_memory_configured(tmp_path):
    memory = MemoryService(MemoryFileRepository(tmp_path / "workspace"))
    session = _make_session_with_memory(tmp_path, memory)
    session._history = [
        {"role": "user", "content": "old message"},
        {"role": "assistant", "content": "old reply"},
    ]
    events: list[object] = []

    with patch.object(session._compactor, "needs_compaction", return_value=True):
        with patch.object(session._compactor, "_summarise", return_value="Summary."):
            session.send("trigger compaction", on_event=events.append)

    from minion_assist.agents.events import MemoryFlushed
    flush_events = [e for e in events if isinstance(e, MemoryFlushed)]
    assert len(flush_events) == 1
    assert flush_events[0].status == "flushed"


def test_no_memory_flushed_event_without_memory_configured(tmp_path):
    """Without memory=, no flush is attempted — nothing to flush to."""
    session = _make_session(tmp_path)
    session._history = [
        {"role": "user", "content": "old message"},
        {"role": "assistant", "content": "old reply"},
    ]
    events: list[object] = []

    with patch.object(session._compactor, "needs_compaction", return_value=True):
        with patch.object(session._compactor, "_summarise", return_value="Summary."):
            session.send("trigger compaction", on_event=events.append)

    from minion_assist.agents.events import MemoryFlushed
    assert not any(isinstance(e, MemoryFlushed) for e in events)


def test_turn_completes_even_if_flush_fails(tmp_path):
    """A failed flush must not block or fail the turn — it's reported, not raised."""
    memory = MemoryService(MemoryFileRepository(tmp_path / "workspace"))
    session = _make_session_with_memory(tmp_path, memory)
    session._history = [
        {"role": "user", "content": "old message"},
        {"role": "assistant", "content": "old reply"},
    ]
    events: list[object] = []

    from minion_assist.memory.models import FlushOutcome
    with patch.object(memory, "flush_head", return_value=FlushOutcome(status="failed", detail="disk full")):
        with patch.object(session._compactor, "needs_compaction", return_value=True):
            with patch.object(session._compactor, "_summarise", return_value="Summary."):
                result = session.send("trigger compaction", on_event=events.append)

    assert result is not None  # turn completed normally
    from minion_assist.agents.events import MemoryFlushed
    flush_events = [e for e in events if isinstance(e, MemoryFlushed)]
    assert flush_events[0].status == "failed"
    assert flush_events[0].detail == "disk full"


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
# Relevant-memory injection
# ---------------------------------------------------------------------------
# Stable user-profile injection (formerly a separate <user_context> block
# loaded once at __init__ from a "user_context" note) was retired in Phase 1
# — bootstrap.py's live USER.md handling already covers it, every turn, with
# no restart needed. See docs/adr/0003-per-agent-memory-scope.md.

def test_relevant_memory_injected_into_system_prompt(tmp_path):
    """A note matching the user's message must appear in the system prompt."""
    memory = MemoryService(MemoryFileRepository(tmp_path))
    memory.remember("python-facts", "User is a Python expert.")

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
        memory=memory,
    )

    captured_system: list[str] = []

    def _capture(system, msgs, tools, max_tokens, on_token=None):
        captured_system.append(system)
        return LLMResponse(text="ok", finish_reason="stop")

    provider.chat = Mock(side_effect=_capture)
    session.send("Tell me about Python")

    assert len(captured_system) == 1
    assert "User is a Python expert." in captured_system[0]
    assert "<relevant_memories>" in captured_system[0]


def test_no_relevant_memories_block_when_nothing_matches(tmp_path):
    """When no note matches the message, the system prompt has no memories block."""
    memory = MemoryService(MemoryFileRepository(tmp_path))
    # No notes saved at all.

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
        memory=memory,
    )

    captured_system: list[str] = []

    def _capture(system, msgs, tools, max_tokens, on_token=None):
        captured_system.append(system)
        return LLMResponse(text="ok", finish_reason="stop")

    provider.chat = Mock(side_effect=_capture)
    session.send("hello")

    assert "<relevant_memories>" not in captured_system[0]


# ---------------------------------------------------------------------------
# build_prompt_section (Stage One Phase 4, slice D)
# ---------------------------------------------------------------------------

def test_build_prompt_section_returns_empty_for_no_results(tmp_path):
    memory = MemoryService(MemoryFileRepository(tmp_path))
    text, injected_hits, tokens = build_prompt_section(memory, "anything", max_tokens=1000)
    assert text == ""
    assert injected_hits == ()
    assert tokens == 0


def test_build_prompt_section_returns_injected_hits_in_order(tmp_path):
    memory = MemoryService(MemoryFileRepository(tmp_path))
    memory.remember("python-facts", "User is a Python expert.")

    text, injected_hits, tokens = build_prompt_section(memory, "Python", max_tokens=1000)

    assert "python-facts" in [h.key for h in injected_hits]
    assert "<relevant_memories>" in text
    assert tokens > 0


def test_build_prompt_section_includes_source_label(tmp_path):
    memory = MemoryService(MemoryFileRepository(tmp_path))
    memory.remember("python-facts", "User is a Python expert.")

    text, _injected_hits, _tokens = build_prompt_section(memory, "Python", max_tokens=1000)

    assert "[topic]" in text  # linear-scan source tag, no index configured


def test_build_prompt_section_omits_citation_without_an_index(tmp_path):
    # A Phase 1 linear-scan hit never carries rel_path/start_line/end_line.
    memory = MemoryService(MemoryFileRepository(tmp_path))
    memory.remember("python-facts", "User is a Python expert.")

    text, _injected_hits, _tokens = build_prompt_section(memory, "Python", max_tokens=1000)

    assert "python-facts.md" not in text


def test_build_prompt_section_renders_a_boundary_annotation_when_present(tmp_path):
    # Stage One Phase 6, slice A. Only an index-backed hit ever carries
    # hit.boundary (see MemoryService._apply_boundaries) -- a Mock stands
    # in for that here rather than standing up a real Postgres index.
    memory = Mock()
    memory.search.return_value = [
        MemoryHit(
            key="deploy-note", content="Some content.", source="durable",
            boundary="[Boundary — advisory only, does not itself grant permission — Owner: main]",
        )
    ]

    text, _injected_hits, _tokens = build_prompt_section(memory, "deploy", max_tokens=1000)

    assert "Boundary" in text
    assert "Owner: main" in text


def test_build_prompt_section_omits_boundary_text_when_absent(tmp_path):
    memory = Mock()
    memory.search.return_value = [
        MemoryHit(key="plain-note", content="Some content.", source="durable")
    ]

    text, _injected_hits, _tokens = build_prompt_section(memory, "query", max_tokens=1000)

    assert "Boundary" not in text


def test_build_prompt_section_respects_a_tiny_token_budget(tmp_path):
    memory = MemoryService(MemoryFileRepository(tmp_path))
    memory.remember("python-facts", "User is a Python expert. " * 50)

    # A budget too small to fit even the header must return nothing rather
    # than a truncated/broken block.
    text, injected_hits, tokens = build_prompt_section(memory, "Python", max_tokens=1)

    assert text == ""
    assert injected_hits == ()
    assert tokens == 0


def test_build_prompt_section_stops_once_the_budget_is_exhausted(tmp_path):
    memory = MemoryService(MemoryFileRepository(tmp_path))
    for i in range(5):
        memory.remember(f"note-{i}", "shared keyword " * 30)

    # A budget big enough for the header and roughly one entry, but not five.
    text, injected_hits, tokens = build_prompt_section(memory, "shared keyword", max_tokens=40)

    assert len(injected_hits) < 5
    assert tokens <= 40


# ---------------------------------------------------------------------------
# MemoryInjected event and context-generation tracking (Stage One Phase 4, slice D)
# ---------------------------------------------------------------------------

def test_memory_injected_event_fired_when_memory_is_injected(tmp_path):
    memory = MemoryService(MemoryFileRepository(tmp_path))
    memory.remember("python-facts", "User is a Python expert.")
    session = _make_session_with_memory(tmp_path, memory)

    events: list[object] = []
    session.send("Tell me about Python", on_event=events.append)

    [injected] = [e for e in events if isinstance(e, MemoryInjected)]
    assert "python-facts" in injected.keys
    assert injected.context_generation == 0
    assert injected.token_count > 0


def test_memory_injected_event_not_fired_when_nothing_matches(tmp_path):
    memory = MemoryService(MemoryFileRepository(tmp_path))
    session = _make_session_with_memory(tmp_path, memory)

    events: list[object] = []
    session.send("hello", on_event=events.append)

    assert not [e for e in events if isinstance(e, MemoryInjected)]


def test_send_marks_injected_recall_telemetry_on_the_index(tmp_path):
    """Stage One Phase 5, slice A: send() must tell the index which surfaced result was injected."""
    mock_index = Mock()
    mock_index.hybrid_search.return_value = [{
        "id": 1, "rel_path": "memory/topics/python-facts.md", "source_kind": "durable",
        "chunk_index": 0, "heading_path": "", "content": "User is a Python expert.",
        "start_line": 1, "end_line": 1, "score": 0.9,
    }]
    mock_index.get_boundary.return_value = None  # Stage One Phase 6, slice A
    memory = MemoryService(
        MemoryFileRepository(tmp_path), index=mock_index, agent_id="main"
    )
    session = _make_session_with_memory(tmp_path, memory)

    session.send("Tell me about Python")

    mock_index.mark_injected.assert_called_once()
    call_args = mock_index.mark_injected.call_args.args
    assert call_args[0] == "main"
    assert call_args[1] == ["memory/topics/python-facts.md"]


def test_send_does_not_mark_injected_when_the_provider_call_fails(tmp_path):
    # MEM-GAP-012: a failed turn must not count its surfaced memory as
    # successfully "used" — mark_injected() must only fire once run_turn()
    # has actually returned, not at prompt-build time.
    mock_index = Mock()
    mock_index.hybrid_search.return_value = [{
        "id": 1, "rel_path": "memory/topics/python-facts.md", "source_kind": "durable",
        "chunk_index": 0, "heading_path": "", "content": "User is a Python expert.",
        "start_line": 1, "end_line": 1, "score": 0.9,
    }]
    mock_index.get_boundary.return_value = None
    memory = MemoryService(MemoryFileRepository(tmp_path), index=mock_index, agent_id="main")
    provider = Mock()
    provider.chat = Mock(side_effect=RuntimeError("provider unavailable"))
    short_term = ShortTermMemory(tmp_path / "sessions")
    session_store = SessionStore(tmp_path / "sessions.json")
    compactor = Compactor(context_window=100_000, preserve_tokens=2_000)
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
        memory=memory,
    )

    try:
        session.send("Tell me about Python")
    except RuntimeError:
        pass

    mock_index.mark_injected.assert_not_called()


def test_context_generation_starts_at_zero(tmp_path):
    session = _make_session(tmp_path)
    assert session._context_generation == 0


def test_context_generation_increments_on_reset(tmp_path):
    session = _make_session(tmp_path)
    session.reset()
    assert session._context_generation == 1
    session.reset()
    assert session._context_generation == 2


def test_context_generation_increments_on_manual_compact(tmp_path):
    session = _make_session(tmp_path)
    session._history = [
        {"role": "user", "content": "old message"},
        {"role": "assistant", "content": "old reply"},
    ]
    with patch.object(session._compactor, "_summarise", return_value="Summary."):
        changed = session.compact_now()

    assert changed is True
    assert session._context_generation == 1


def test_context_generation_unchanged_when_manual_compact_does_nothing(tmp_path):
    session = _make_session(tmp_path)
    # No history at all -- Compactor needs at least 2 messages to compact.
    changed = session.compact_now()

    assert changed is False
    assert session._context_generation == 0


def test_context_generation_increments_on_automatic_compaction(tmp_path):
    provider = _mock_provider()
    session = _make_session(tmp_path, provider=provider)
    session._history = [
        {"role": "user", "content": "old message"},
        {"role": "assistant", "content": "old reply"},
    ]

    with patch.object(session._compactor, "needs_compaction", return_value=True):
        with patch.object(session._compactor, "_summarise", return_value="Summary."):
            session.send("trigger compaction")

    assert session._context_generation == 1


def test_fork_does_not_change_the_forking_sessions_context_generation(tmp_path):
    session = _make_session(tmp_path)
    session.reset()  # generation is now 1
    session.fork("forked-agent")
    assert session._context_generation == 1


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
    memory = MemoryService(MemoryFileRepository(tmp_path))
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
        memory=memory,
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
    memory = MemoryService(MemoryFileRepository(tmp_path))
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
        memory=memory,
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


# ---------------------------------------------------------------------------
# Prompt date position — OpenAI prompt-caching alignment
# ---------------------------------------------------------------------------

def test_date_appears_in_system_prompt(tmp_path):
    """The system prompt must include today's date somewhere."""
    from datetime import date

    provider = _mock_provider()
    session = _make_session(tmp_path, provider=provider)

    captured: list[str] = []

    def _capture(system, msgs, tools, max_tokens, on_token=None):
        captured.append(system)
        return LLMResponse(text="ok", finish_reason="stop")

    provider.chat = Mock(side_effect=_capture)
    session.send("hello")

    today = date.today().isoformat()
    assert today in captured[0], "system prompt must contain today's ISO date"


def test_date_does_not_lead_system_prompt(tmp_path):
    """The date must NOT be the very first thing in the system prompt.

    The stable soul text must come before the date so that OpenAI's automatic
    prompt caching can cache the large soul+bootstrap prefix across turns.
    Prepending the date would change byte 0 daily and invalidate the cache.
    """
    from datetime import date

    provider = _mock_provider()
    session = _make_session(tmp_path, provider=provider)

    captured: list[str] = []

    def _capture(system, msgs, tools, max_tokens, on_token=None):
        captured.append(system)
        return LLMResponse(text="ok", finish_reason="stop")

    provider.chat = Mock(side_effect=_capture)
    session.send("hello")

    today = date.today().isoformat()
    date_pos = captured[0].index(today)
    soul_pos = captured[0].index("You are Ada.")
    assert soul_pos < date_pos, (
        "soul text must appear before the date in the system prompt "
        "so the stable prefix is byte-identical across turns (OpenAI prompt caching)"
    )


# ---------------------------------------------------------------------------
# PostgreSQL mirroring (Stage One Phase 2, slice A)
# ---------------------------------------------------------------------------

def _make_session_with_mock_db(tmp_path, mock_db, enable_commitments=False, health=None):
    """Build an AgentSession wired to a mock SessionDB, for mirroring tests."""
    short_term = ShortTermMemory(tmp_path / "sessions")
    session_store = SessionStore(tmp_path / "sessions.json")
    compactor = Compactor(context_window=100_000, preserve_tokens=2_000)
    return AgentSession(
        agent_id="main",
        session_id="test-session",
        agent=AgentConfig(name="Ada", soul="You are Ada."),
        provider=_mock_provider(),
        max_output_tokens=512,
        tools=ToolRegistry(),
        compactor=compactor,
        short_term=short_term,
        session_store=session_store,
        db=mock_db,
        enable_commitments=enable_commitments,
        health=health,
    )


def test_no_event_ids_assigned_without_a_database(tmp_path):
    """Without db=, nothing will ever mirror — assigning ids would just be JSONL noise."""
    session = _make_session(tmp_path)
    session.send("hello")

    assert not any(EVENT_ID_KEY in m for m in session.history)


def test_every_history_message_gets_an_event_id_when_db_configured(tmp_path):
    """Every message (user + assistant) has _event_id assigned when db is set."""
    mock_db = Mock()
    mock_db.mirror_message = Mock(return_value=1)
    session = _make_session_with_mock_db(tmp_path, mock_db)

    session.send("hello")

    assert all(EVENT_ID_KEY in m for m in session.history)


def test_event_ids_are_unique_per_message(tmp_path):
    mock_db = Mock()
    mock_db.mirror_message = Mock(return_value=1)
    session = _make_session_with_mock_db(tmp_path, mock_db)

    session.send("hello")

    ids = [m[EVENT_ID_KEY] for m in session.history]
    assert len(ids) == len(set(ids))


def test_db_mirror_message_called_for_user_and_assistant(tmp_path):
    """Mirroring uses mirror_message() (idempotent), never the raw add_message()."""
    mock_db = Mock()
    mock_db.mirror_message = Mock(return_value=1)
    session = _make_session_with_mock_db(tmp_path, mock_db)

    session.send("hello")

    assert mock_db.mirror_message.called
    assert not mock_db.add_message.called
    # The event_id passed to mirror_message must match what's on the message.
    mirrored_event_ids = {call.args[1] for call in mock_db.mirror_message.call_args_list}
    history_event_ids = {m[EVENT_ID_KEY] for m in session.history}
    assert mirrored_event_ids == history_event_ids


def test_second_turn_does_not_reassign_event_ids_from_first_turn(tmp_path):
    """Historical messages keep their event_id across subsequent turns."""
    mock_db = Mock()
    mock_db.mirror_message = Mock(return_value=1)
    session = _make_session_with_mock_db(tmp_path, mock_db)

    session.send("first")
    ids_after_first = {m[EVENT_ID_KEY] for m in session.history}

    session.send("second")
    ids_after_second = {m[EVENT_ID_KEY] for m in session.history}

    assert ids_after_first.issubset(ids_after_second)


# ---------------------------------------------------------------------------
# WorkerHealth wiring for mirror/enqueue failures (MEM-GAP-007)
# ---------------------------------------------------------------------------

def _make_session_with_mock_db_and_memory(tmp_path, mock_db, health=None):
    """Like _make_session_with_mock_db, but with a real MemoryService so
    capture-job enqueueing (gated on self._memory is not None) actually fires."""
    memory = MemoryService(MemoryFileRepository(tmp_path / "workspace"))
    short_term = ShortTermMemory(tmp_path / "sessions")
    session_store = SessionStore(tmp_path / "sessions.json")
    compactor = Compactor(context_window=100_000, preserve_tokens=2_000)
    return AgentSession(
        agent_id="main",
        session_id="test-session",
        agent=AgentConfig(name="Ada", soul="You are Ada."),
        provider=_mock_provider(),
        max_output_tokens=512,
        tools=ToolRegistry(),
        compactor=compactor,
        short_term=short_term,
        session_store=session_store,
        db=mock_db,
        memory=memory,
        health=health,
    )


def test_successful_mirror_records_health_success(tmp_path):
    mock_db = Mock()
    mock_db.mirror_message = Mock(return_value=1)
    health = WorkerHealth("session_writes:main")
    session = _make_session_with_mock_db(tmp_path, mock_db, health=health)

    session.send("hello")

    snap = health.snapshot()
    assert snap["last_success_at"] is not None
    assert snap["consecutive_failures"] == 0


def test_mirror_failure_is_recorded_but_does_not_break_the_turn(tmp_path):
    mock_db = Mock()
    mock_db.mirror_message = Mock(side_effect=RuntimeError("connection refused"))
    health = WorkerHealth("session_writes:main")
    session = _make_session_with_mock_db(tmp_path, mock_db, health=health)

    result = session.send("hello")  # must not raise

    assert result  # the turn still completed and returned a response
    snap = health.snapshot()
    assert snap["consecutive_failures"] >= 1
    assert "connection refused" in snap["last_error"]


def test_mirror_failure_without_health_configured_still_does_not_raise(tmp_path):
    mock_db = Mock()
    mock_db.mirror_message = Mock(side_effect=RuntimeError("boom"))
    session = _make_session_with_mock_db(tmp_path, mock_db)  # health=None (default)

    session.send("hello")  # must not raise — matches pre-MEM-GAP-007 behavior


def test_capture_job_enqueue_failure_is_recorded(tmp_path):
    mock_db = Mock()
    mock_db.mirror_message = Mock(side_effect=[1, 2])
    mock_db.enqueue_capture_job = Mock(side_effect=RuntimeError("db unavailable"))
    # No vector lane configured — isolates this assertion to the
    # capture-job enqueue failure, not the (also-shared-health)
    # message-embedding enqueue site's own success/failure signal.
    mock_db.has_vector_lane = False
    health = WorkerHealth("session_writes:main")
    session = _make_session_with_mock_db_and_memory(tmp_path, mock_db, health=health)

    session.send("hello")  # must not raise

    snap = health.snapshot()
    assert snap["consecutive_failures"] >= 1
    assert "db unavailable" in snap["last_error"]


def test_commitment_job_enqueue_failure_is_recorded(tmp_path):
    mock_db = Mock()
    mock_db.mirror_message = Mock(side_effect=[1, 2])
    mock_db.enqueue_commitment_job = Mock(side_effect=RuntimeError("db unavailable"))
    # No vector lane configured — isolates this assertion to the
    # commitment-job enqueue failure, not the (also-shared-health)
    # message-embedding enqueue site's own success/failure signal.
    mock_db.has_vector_lane = False
    health = WorkerHealth("session_writes:main")
    short_term = ShortTermMemory(tmp_path / "sessions")
    session_store = SessionStore(tmp_path / "sessions.json")
    compactor = Compactor(context_window=100_000, preserve_tokens=2_000)
    session = AgentSession(
        agent_id="main",
        session_id="test-session",
        agent=AgentConfig(name="Ada", soul="You are Ada."),
        provider=_mock_provider(),
        max_output_tokens=512,
        tools=ToolRegistry(),
        compactor=compactor,
        short_term=short_term,
        session_store=session_store,
        db=mock_db,
        enable_commitments=True,
        health=health,
    )

    session.send("hello")  # must not raise

    snap = health.snapshot()
    assert snap["consecutive_failures"] >= 1
    assert "db unavailable" in snap["last_error"]


# ---------------------------------------------------------------------------
# Commitment-job enqueueing (Stage One Phase 6, slice B)
# ---------------------------------------------------------------------------

def test_commitment_job_enqueued_when_enabled(tmp_path):
    mock_db = Mock()
    mock_db.mirror_message = Mock(side_effect=[1, 2])
    session = _make_session_with_mock_db(tmp_path, mock_db, enable_commitments=True)

    session.send("hello")

    mock_db.enqueue_commitment_job.assert_called_once()
    call_args = mock_db.enqueue_commitment_job.call_args.args
    assert call_args[0] == "main"
    assert call_args[1] == "test-session"
    assert call_args[2] == "cli"  # channel defaults to "cli" when none is passed


def test_commitment_job_not_enqueued_when_disabled(tmp_path):
    mock_db = Mock()
    mock_db.mirror_message = Mock(side_effect=[1, 2])
    session = _make_session_with_mock_db(tmp_path, mock_db, enable_commitments=False)

    session.send("hello")

    mock_db.enqueue_commitment_job.assert_not_called()


def test_commitment_job_uses_the_passed_channel(tmp_path):
    mock_db = Mock()
    mock_db.mirror_message = Mock(side_effect=[1, 2])
    session = _make_session_with_mock_db(tmp_path, mock_db, enable_commitments=True)

    session.send("hello", channel="!room:example.org")

    call_args = mock_db.enqueue_commitment_job.call_args.args
    assert call_args[2] == "!room:example.org"


def test_commitment_job_not_enqueued_without_a_database(tmp_path):
    short_term = ShortTermMemory(tmp_path / "sessions")
    session_store = SessionStore(tmp_path / "sessions.json")
    compactor = Compactor(context_window=100_000, preserve_tokens=2_000)
    session = AgentSession(
        agent_id="main",
        session_id="test-session",
        agent=AgentConfig(name="Ada", soul="You are Ada."),
        provider=_mock_provider(),
        max_output_tokens=512,
        tools=ToolRegistry(),
        compactor=compactor,
        short_term=short_term,
        session_store=session_store,
        db=None,
        enable_commitments=True,
    )

    session.send("hello")  # must not raise even though enable_commitments=True


def test_commitment_job_idempotency_key_includes_the_channel(tmp_path):
    # Two different channels for the "same" message range must not collide
    # on the same idempotency key.
    mock_db = Mock()
    mock_db.mirror_message = Mock(side_effect=[1, 2])
    session = _make_session_with_mock_db(tmp_path, mock_db, enable_commitments=True)

    session.send("hello", channel="room-a")
    key_a = mock_db.enqueue_commitment_job.call_args.args[5]

    mock_db.mirror_message = Mock(side_effect=[3, 4])
    session.send("hello again", channel="room-b")
    key_b = mock_db.enqueue_commitment_job.call_args.args[5]

    assert key_a != key_b


# ---------------------------------------------------------------------------
# Message-embedding job enqueue (MEM-GAP-006)
# ---------------------------------------------------------------------------

def test_message_embedding_job_enqueued_when_vector_lane_is_configured(tmp_path):
    mock_db = Mock()
    mock_db.mirror_message = Mock(side_effect=[1, 2])
    mock_db.has_vector_lane = True
    mock_db.embedding_model_identity = "test-endpoint::test-model"
    session = _make_session_with_mock_db(tmp_path, mock_db)

    session.send("hello")

    # One job per newly-mirrored user+assistant message (both ids 1 and 2).
    assert mock_db.enqueue_message_embedding_job.call_count == 2
    call_args = [c.args for c in mock_db.enqueue_message_embedding_job.call_args_list]
    assert call_args[0] == ("main", "test-session", 1, "main:1:test-endpoint::test-model")
    assert call_args[1] == ("main", "test-session", 2, "main:2:test-endpoint::test-model")


def test_message_embedding_job_not_enqueued_without_a_vector_lane(tmp_path):
    mock_db = Mock()
    mock_db.mirror_message = Mock(side_effect=[1, 2])
    mock_db.has_vector_lane = False
    session = _make_session_with_mock_db(tmp_path, mock_db)

    session.send("hello")

    mock_db.enqueue_message_embedding_job.assert_not_called()


def test_message_embedding_job_not_enqueued_without_a_database(tmp_path):
    session = _make_session(tmp_path)  # db=None

    session.send("hello")  # must not raise


def test_message_embedding_job_enqueue_failure_is_recorded(tmp_path):
    mock_db = Mock()
    mock_db.mirror_message = Mock(side_effect=[1, 2])
    mock_db.has_vector_lane = True
    mock_db.embedding_model_identity = "test-endpoint::test-model"
    mock_db.enqueue_message_embedding_job = Mock(side_effect=RuntimeError("db unavailable"))
    health = WorkerHealth("session_writes:main")
    session = _make_session_with_mock_db(tmp_path, mock_db, health=health)

    session.send("hello")  # must not raise

    snap = health.snapshot()
    assert snap["consecutive_failures"] >= 1
    assert "db unavailable" in snap["last_error"]

"""Tests for REPL exception handling and history persistence (minion.py)."""

import pytest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from minion_assist.memory.short_term import ShortTermMemory
from minion_assist.session import SessionStore


def _load_history(tmp_path, agent_id: str) -> list[dict]:
    """Load history for an agent by resolving the session_id from the store."""
    store = SessionStore(tmp_path / "sessions.json")
    info = store.get_or_create(agent_id)
    return ShortTermMemory(tmp_path / "sessions").load(agent_id, info.session_id)


def _run_main(tmp_path, inputs, run_turn_effect=None):
    """Run main() with a controlled workspace, input sequence, and run_turn behaviour.

    After the refactor, main() uses AgentSession which calls run_turn internally.
    We patch minion_assist.agents.session.run_turn (where AgentSession imports it)
    rather than minion_assist.minion.run_turn (which no longer exists there).

    Args:
        tmp_path: Pytest temporary directory used as the workspace root.
        inputs: Sequence of strings returned by successive ``input()`` calls.
        run_turn_effect: Passed as ``side_effect`` to the ``run_turn`` mock.
            Pass an exception instance to make it raise, a callable to replace
            the implementation, or ``None`` for a no-op mock.

    Returns:
        The ``run_turn`` Mock so callers can inspect call counts / args.
    """
    import minion_assist.minion as minion_mod

    rt_mock = Mock(side_effect=run_turn_effect) if run_turn_effect is not None else Mock()

    with (
        patch("minion_assist.minion.workspace", tmp_path),
        patch("minion_assist.minion.mcp_cfg", SimpleNamespace(servers=())),
        patch("minion_assist.minion.channels_cfg", SimpleNamespace(matrix=None)),
        patch("minion_assist.agents.session.run_turn", rt_mock),
        patch("minion_assist.minion.create_provider", return_value=Mock()),
        patch("builtins.input", side_effect=iter(inputs)),
    ):
        minion_mod.main()

    return rt_mock


def test_repl_survives_provider_exception(tmp_path, capsys):
    """REPL loop continues after run_turn raises; user sees a friendly error line."""
    _run_main(tmp_path, ["hello", "quit"], run_turn_effect=RuntimeError("connection refused"))

    captured = capsys.readouterr()
    assert "[Error]" in captured.err


def test_provider_exception_records_error_in_history(tmp_path):
    """After a provider exception, history has the user message and an assistant error entry."""
    _run_main(tmp_path, ["hello", "quit"], run_turn_effect=RuntimeError("timeout"))

    history = _load_history(tmp_path, "main")
    assert any(m["role"] == "user" and m["content"] == "hello" for m in history)
    assert any(
        m["role"] == "assistant" and "Provider error" in m["content"]
        for m in history
    )


def test_provider_exception_rolls_back_partial_messages(tmp_path):
    """Partial messages appended by run_turn before a crash are stripped from saved history."""

    def _crash_after_partial(provider, name, soul, max_tokens, tools, messages, **kwargs):
        # Simulate run_turn appending a partial assistant message before crashing.
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        })
        raise RuntimeError("mid-turn crash")

    _run_main(tmp_path, ["hello", "quit"], run_turn_effect=_crash_after_partial)

    history = _load_history(tmp_path, "main")
    # The partial assistant+tool_calls message must not appear in saved history.
    assert not any("tool_calls" in m for m in history)
    # Only the user message and the assistant error message should remain.
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_user_message_persisted_on_exception(tmp_path):
    """User message is on disk after a provider crash (early save before run_turn)."""
    _run_main(tmp_path, ["my question", "quit"], run_turn_effect=RuntimeError("boom"))

    history = _load_history(tmp_path, "main")
    # _event_id (Phase 2 slice A's mirroring identity) is expected now —
    # check role/content only.
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "my question"


def test_successful_turn_persists_history(tmp_path):
    """On a successful turn, history is saved and contains the user message."""
    _run_main(tmp_path, ["hello", "quit"])

    history = _load_history(tmp_path, "main")
    assert any(m["role"] == "user" and m["content"] == "hello" for m in history)


def test_route_prefix_message_is_not_treated_as_unknown_command(tmp_path, capsys):
    """A routed chat message like '/research hello' must reach the routed agent."""
    rt_mock = _run_main(tmp_path, ["/research how to prepare", "quit"])

    assert rt_mock.call_count == 1
    assert rt_mock.call_args.args[1] == "Elizabeth"

    history = _load_history(tmp_path, "researcher")
    assert any(
        m["role"] == "user" and m["content"] == "how to prepare"
        for m in history
    )
    assert "Unknown command '/research'" not in capsys.readouterr().out


def test_main_exits_on_agent_identity_mismatch(tmp_path):
    """main() must raise SystemExit when AGENTS and config.json agent keys differ."""
    import minion_assist.minion as minion_mod
    from minion_assist.agents.definitions import AgentConfig

    # Replace AGENTS with a key that doesn't exist in config.json.
    wrong_agents = {"unknown_agent": AgentConfig(name="X", soul="x")}

    with (
        patch("minion_assist.minion.workspace", tmp_path),
        patch("minion_assist.minion.AGENTS", wrong_agents),
        patch("builtins.input", side_effect=iter([])),
    ):
        with pytest.raises(SystemExit):
            minion_mod.main()


def test_compaction_receives_user_message(tmp_path):
    """compact() must be called after the user message is appended to the list."""
    import minion_assist.minion as minion_mod
    from minion_assist.context import Compactor

    seen: list[list[dict]] = []

    def spy_compact(self, messages, provider, on_compaction=None, on_compaction_failed=None):
        seen.append(list(messages))
        return messages  # pass-through, no actual compaction

    with (
        patch("minion_assist.minion.workspace", tmp_path),
        patch("minion_assist.agents.session.run_turn", Mock()),
        patch("minion_assist.minion.create_provider", return_value=Mock()),
        patch("builtins.input", side_effect=iter(["hello", "quit"])),
        patch.object(Compactor, "compact", spy_compact),
    ):
        minion_mod.main()

    # Every compaction call must have seen the user message already appended.
    user_msgs_seen = [
        m for call in seen for m in call
        if m.get("role") == "user" and m.get("content") == "hello"
    ]
    assert user_msgs_seen, "compact() was called before the user message was appended"


def test_streaming_response_printed_once(tmp_path, capsys):
    """Streamed response text must appear exactly once — not duplicated by FinalAnswer handler."""
    import minion_assist.minion as minion_mod
    from minion_assist.agents.events import FinalAnswer, StreamingStarted, TokenStreamed

    def _emit_streaming_events(provider, name, soul, max_tokens, tools, messages, on_event=None, **kwargs):
        # Simulate the runner emitting streaming events then FinalAnswer.
        if on_event:
            on_event(StreamingStarted(agent_name=name))
            on_event(TokenStreamed(token="Hello"))
            on_event(TokenStreamed(token=" world"))
            on_event(FinalAnswer(agent_name=name, text="Hello world"))

    with (
        patch("minion_assist.minion.workspace", tmp_path),
        patch("minion_assist.agents.session.run_turn", side_effect=_emit_streaming_events),
        patch("minion_assist.minion.create_provider", return_value=Mock()),
        patch("minion_assist.minion.streaming") as mock_streaming,
        patch("builtins.input", side_effect=iter(["hi", "quit"])),
    ):
        mock_streaming.chat_mode = True
        minion_mod.main()

    out = capsys.readouterr().out
    # "Hello world" should appear exactly once in the output, not twice.
    assert out.count("Hello world") == 1


# ---------------------------------------------------------------------------
# IMP-15: Graceful shutdown
# ---------------------------------------------------------------------------


def test_keyboard_interrupt_at_prompt_exits_cleanly(tmp_path, capsys):
    """KeyboardInterrupt at the input() prompt causes a clean exit, not a traceback."""
    import minion_assist.minion as minion_mod
    with (
        patch("minion_assist.minion.workspace", tmp_path),
        patch("minion_assist.agents.session.run_turn", Mock()),
        patch("minion_assist.minion.create_provider", return_value=Mock()),
        patch("builtins.input", side_effect=KeyboardInterrupt),
    ):
        minion_mod.main()  # must return without raising

    out = capsys.readouterr().out
    assert "Goodbye" in out


def test_keyboard_interrupt_during_turn_continues_repl(tmp_path, capsys):
    """KeyboardInterrupt mid-turn prints a message and continues the REPL."""
    import minion_assist.minion as minion_mod
    inputs = iter(["hello", "quit"])
    with (
        patch("minion_assist.minion.workspace", tmp_path),
        patch("minion_assist.agents.session.run_turn", side_effect=KeyboardInterrupt),
        patch("minion_assist.minion.create_provider", return_value=Mock()),
        patch("builtins.input", side_effect=inputs),
    ):
        minion_mod.main()  # must return without raising

    out = capsys.readouterr().out
    assert "interrupted" in out.lower()


# ---------------------------------------------------------------------------
# Session ID resolution (_resolve_session_id)
# ---------------------------------------------------------------------------


def _fake_agents_cfg(reset_mode="daily", at_hour=4, idle_minutes=0):
    """Build a minimal fake agents_cfg dict matching the real AGENTS keys."""
    _provider = SimpleNamespace(base_url="", api="lmstudio", api_key="", name="lmstudio")
    _model = SimpleNamespace(id="test", context_window=8192, max_output_tokens=512)
    _cfg = SimpleNamespace(
        provider=_provider, model=_model,
        session_reset_mode=reset_mode, session_reset_at_hour=at_hour, session_idle_minutes=idle_minutes,
    )
    return {
        "main": SimpleNamespace(**{**vars(_cfg), "route_prefix": None}),
        "researcher": SimpleNamespace(**{**vars(_cfg), "route_prefix": "/research"}),
    }


def test_idle_zero_always_generates_new_session_id(tmp_path):
    """idle mode with idle_minutes=0 must rotate to a new UUID on every startup."""
    from minion_assist.minion import _resolve_session_id
    from minion_assist.memory.short_term import ShortTermMemory as _STM
    store = SessionStore(tmp_path / "sessions.json")
    stm = _STM(tmp_path / "sessions")
    id1, _ = _resolve_session_id("main", store, stm, reset_mode="idle", idle_minutes=0)
    id2, _ = _resolve_session_id("main", store, stm, reset_mode="idle", idle_minutes=0)
    assert id1 != id2


def test_idle_long_window_reuses_session_id(tmp_path):
    """idle mode within the window must reuse the same UUID."""
    from minion_assist.minion import _resolve_session_id
    from minion_assist.memory.short_term import ShortTermMemory as _STM
    store = SessionStore(tmp_path / "sessions.json")
    stm = _STM(tmp_path / "sessions")
    id1, _ = _resolve_session_id("main", store, stm, reset_mode="idle", idle_minutes=60)
    id2, _ = _resolve_session_id("main", store, stm, reset_mode="idle", idle_minutes=60)
    assert id1 == id2


def test_idle_stale_session_rotates_to_new_uuid(tmp_path):
    """idle mode: when last_active exceeds idle_minutes a new UUID is issued."""
    from minion_assist.minion import _resolve_session_id
    from minion_assist.memory.short_term import ShortTermMemory as _STM
    from minion_assist.session.store import SessionStore as _Store
    import json, datetime as dt

    old_ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2)).isoformat()
    raw = {
        "main": {
            "agent_id": "main",
            "created_at": old_ts,
            "last_active": old_ts,
            "turn_count": 0,
            "parent_id": None,
            "session_id": "old-uuid",
        }
    }
    (tmp_path / "sessions.json").write_text(json.dumps(raw), encoding="utf-8")
    store = _Store(tmp_path / "sessions.json")
    stm = _STM(tmp_path / "sessions")
    new_id, _ = _resolve_session_id("main", store, stm, reset_mode="idle", idle_minutes=60)
    assert new_id != "old-uuid"


def test_daily_reuses_session_started_after_reset_hour(tmp_path):
    """daily mode: session started after today's reset hour must be reused."""
    from minion_assist.minion import _resolve_session_id
    from minion_assist.memory.short_term import ShortTermMemory as _STM
    from minion_assist.session.store import SessionStore as _Store
    import json, datetime as dt

    now = dt.datetime.now(dt.timezone.utc)
    recent_ts = (now - dt.timedelta(hours=2)).isoformat()
    boundary = now.replace(hour=4, minute=0, second=0, microsecond=0)
    if now - dt.timedelta(hours=2) < boundary:
        pytest.skip("too early in the day for this test to be meaningful")

    raw = {
        "main": {
            "agent_id": "main",
            "created_at": recent_ts,
            "last_active": recent_ts,
            "turn_count": 0,
            "parent_id": None,
            "session_id": "same-uuid",
        }
    }
    (tmp_path / "sessions.json").write_text(json.dumps(raw), encoding="utf-8")
    store = _Store(tmp_path / "sessions.json")
    stm = _STM(tmp_path / "sessions")
    reused, reseed = _resolve_session_id("main", store, stm, reset_mode="daily", reset_at_hour=4)
    assert reused == "same-uuid"
    assert reseed is None


def test_daily_rotates_session_started_before_reset_hour(tmp_path):
    """daily mode: session with last_active before today's reset hour must rotate."""
    from minion_assist.minion import _resolve_session_id
    from minion_assist.memory.short_term import ShortTermMemory as _STM
    from minion_assist.session.store import SessionStore as _Store
    import json, datetime as dt

    old_ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).isoformat()
    raw = {
        "main": {
            "agent_id": "main",
            "created_at": old_ts,
            "last_active": old_ts,
            "turn_count": 0,
            "parent_id": None,
            "session_id": "yesterday-uuid",
        }
    }
    (tmp_path / "sessions.json").write_text(json.dumps(raw), encoding="utf-8")
    store = _Store(tmp_path / "sessions.json")
    stm = _STM(tmp_path / "sessions")
    new_id, _ = _resolve_session_id("main", store, stm, reset_mode="daily", reset_at_hour=4)
    assert new_id != "yesterday-uuid"


def test_reseed_context_included_on_rotation(tmp_path):
    """On session rotation, reseed_context must contain prior history text."""
    from minion_assist.minion import _resolve_session_id
    from minion_assist.memory.short_term import ShortTermMemory as _STM
    from minion_assist.session.store import SessionStore as _Store
    import json, datetime as dt

    old_sid = "old-session"
    old_ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).isoformat()
    raw = {
        "main": {
            "agent_id": "main",
            "created_at": old_ts,
            "last_active": old_ts,
            "turn_count": 2,
            "parent_id": None,
            "session_id": old_sid,
        }
    }
    (tmp_path / "sessions.json").write_text(json.dumps(raw), encoding="utf-8")
    stm = _STM(tmp_path / "sessions")
    stm.save("main", old_sid, [
        {"role": "user", "content": "what is 2+2"},
        {"role": "assistant", "content": "4"},
    ])
    store = _Store(tmp_path / "sessions.json")
    new_id, reseed = _resolve_session_id("main", store, stm, reset_mode="daily", reset_at_hour=4)
    assert new_id != old_sid
    assert reseed is not None
    assert "what is 2+2" in reseed
    assert "4" in reseed
    assert "<prior_session_history>" in reseed


def test_reseed_context_none_when_no_prior_history(tmp_path):
    """Rotation with empty prior history must return None reseed_context."""
    from minion_assist.minion import _resolve_session_id
    from minion_assist.memory.short_term import ShortTermMemory as _STM
    from minion_assist.session.store import SessionStore as _Store
    import json, datetime as dt

    old_sid = "empty-session"
    old_ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).isoformat()
    raw = {
        "main": {
            "agent_id": "main",
            "created_at": old_ts,
            "last_active": old_ts,
            "turn_count": 0,
            "parent_id": None,
            "session_id": old_sid,
        }
    }
    (tmp_path / "sessions.json").write_text(json.dumps(raw), encoding="utf-8")
    store = _Store(tmp_path / "sessions.json")
    stm = _STM(tmp_path / "sessions")
    _, reseed = _resolve_session_id("main", store, stm, reset_mode="daily", reset_at_hour=4)
    assert reseed is None


def test_session_files_stored_per_agent_subdirectory(tmp_path):
    """After a turn, history lands in sessions/{agent_id}/{session_id}.jsonl."""
    import minion_assist.minion as minion_mod

    with (
        patch("minion_assist.minion.workspace", tmp_path),
        patch("minion_assist.minion.agents_cfg", _fake_agents_cfg()),
        patch("minion_assist.minion.mcp_cfg", SimpleNamespace(servers=())),
        patch("minion_assist.minion.channels_cfg", SimpleNamespace(matrix=None)),
        patch("minion_assist.agents.session.run_turn", Mock()),
        patch("minion_assist.minion.create_provider", return_value=Mock()),
        patch("builtins.input", side_effect=iter(["hello", "quit"])),
    ):
        minion_mod.main()

    sessions_dir = tmp_path / "sessions" / "main"
    assert sessions_dir.is_dir(), "agent subdirectory must exist"
    jsonl_files = list(sessions_dir.glob("*.jsonl"))
    assert len(jsonl_files) == 1, "exactly one session file must exist"
    # File must contain the user message.
    history = ShortTermMemory(tmp_path / "sessions").load("main", jsonl_files[0].stem)
    assert any(m["role"] == "user" and m["content"] == "hello" for m in history)


# ---------------------------------------------------------------------------
# `minion-assist memory ...` CLI dispatch guard
# ---------------------------------------------------------------------------
# main() checks sys.argv[1] == "memory" before any REPL setup runs (agent
# identity validation, skill discovery, session/provider construction, etc.)
# and hands off to memory/cli.py instead. These tests verify the guard fires
# and forwards the right arguments — memory/cli.py's own behavior is covered
# by tests/memory/test_cli.py.

def test_main_dispatches_memory_subcommand_before_repl_setup():
    """`minion-assist memory migrate --apply` calls memory.cli.main and exits
    with its return code, without touching any REPL setup (skills, sessions,
    providers) along the way."""
    import minion_assist.minion as minion_mod

    with (
        patch("sys.argv", ["minion-assist", "memory", "migrate", "--apply"]),
        patch("minion_assist.memory.cli.main", return_value=0) as cli_main_mock,
    ):
        with pytest.raises(SystemExit) as exc_info:
            minion_mod.main()

    cli_main_mock.assert_called_once_with(["migrate", "--apply"])
    assert exc_info.value.code == 0


def test_main_propagates_memory_cli_exit_code():
    """A non-zero exit code from the memory CLI propagates through SystemExit."""
    import minion_assist.minion as minion_mod

    with (
        patch("sys.argv", ["minion-assist", "memory", "migrate", "--rollback", "missing.json"]),
        patch("minion_assist.memory.cli.main", return_value=1),
    ):
        with pytest.raises(SystemExit) as exc_info:
            minion_mod.main()

    assert exc_info.value.code == 1


def test_main_does_not_dispatch_for_non_memory_args(tmp_path):
    """Without a leading 'memory' argv token, main() proceeds to the normal REPL path."""
    _run_main(tmp_path, ["hello", "quit"])
    # No SystemExit escaped and no exception — REPL setup ran as usual, already
    # covered by the assertions in _run_main's other callers. This test exists
    # to document that the guard is argv[1]-specific, not a blanket short-circuit.


# ---------------------------------------------------------------------------
# `minion-assist config` CLI dispatch guard (MEM-GAP-019)
# ---------------------------------------------------------------------------
# Same shape as the `memory` dispatch guard above — verifies main() hands off
# to config_report.py before any REPL setup, without re-testing
# config_report.py's own report content (tests/test_config_report.py's job).

def test_main_dispatches_config_subcommand_before_repl_setup():
    import minion_assist.minion as minion_mod

    with (
        patch("sys.argv", ["minion-assist", "config"]),
        patch("minion_assist.config_report.main", return_value=0) as cli_main_mock,
    ):
        with pytest.raises(SystemExit) as exc_info:
            minion_mod.main()

    cli_main_mock.assert_called_once_with([])
    assert exc_info.value.code == 0


def test_main_propagates_config_cli_exit_code():
    import minion_assist.minion as minion_mod

    with (
        patch("sys.argv", ["minion-assist", "config"]),
        patch("minion_assist.config_report.main", return_value=2),
    ):
        with pytest.raises(SystemExit) as exc_info:
            minion_mod.main()

    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# MEM-GAP-018: production-wiring coverage for WorkerHealth / MessageEmbeddingWorker
# ---------------------------------------------------------------------------
# Every WorkerHealth/MessageEmbeddingWorker unit test elsewhere in the suite
# exercises the class directly — none of them prove minion.py's real startup
# path actually constructs and threads these into the real AgentSession/
# worker call sites. That's exactly the "forgot to wire it up" failure mode
# already self-caught once this session (MEM-GAP-007's `health=` kwarg was
# initially missing from the AgentSession(...) construction). These two
# tests run minion.py's real main() against the real dev config.json
# (database + embeddings), so they only pass when the wiring genuinely
# exists — not when it's mocked away. Skipped, not failed, when the dev
# database isn't reachable, matching tests/test_session_db.py's convention.

from minion_assist.config import database as _database_cfg
from minion_assist.config import embeddings as _embeddings_cfg

try:
    import psycopg as _psycopg_wiring

    _wiring_conn = _psycopg_wiring.connect(_database_cfg.url, connect_timeout=2)
    _wiring_conn.close()
    _DB_AVAILABLE_FOR_WIRING = bool(_database_cfg.url)
except Exception:
    _DB_AVAILABLE_FOR_WIRING = False

_requires_live_db_and_embeddings = pytest.mark.skipif(
    not (_DB_AVAILABLE_FOR_WIRING and _embeddings_cfg is not None),
    reason="requires a live PostgreSQL instance and an 'embeddings' section in config.json",
)


@_requires_live_db_and_embeddings
def test_main_wires_message_embedding_worker_and_per_agent_health(tmp_path, capsys):
    """`/status deep`, driven through the real REPL command path, must show
    MessageEmbeddingWorker and the 'main' agent's memory_search/
    session_writes health as actually running (not 'not running', which is
    what a dropped construction/wiring line would silently produce) — and
    must show memory_extractor as 'not running', since that tracker only
    ever applies to the no-database branch (MEM-GAP-013's inverse gate)."""
    _run_main(tmp_path, ["/status deep", "quit"])

    out = capsys.readouterr().out
    assert "message_embedding_worker: not running" not in out
    assert "session_writes:main: not running" not in out
    assert "memory_search:main: not running" not in out
    # Inverse gate (MEM-GAP-013): a database IS configured here, so the
    # degraded-mode extractor tracker must be absent, not present.
    assert "memory_extractor:main: not running" in out


@_requires_live_db_and_embeddings
def test_main_wires_health_into_matrix_session_factory_closure(tmp_path):
    """The Matrix room-scoped AgentSession factory closure (MEM-GAP-001)
    must carry the exact same WorkerHealth instances as the agent's main
    REPL session. This closure duplicates the health=/extraction_health=
    kwargs a few lines below the main AgentSession(...) construction in
    minion.py — precisely the kind of second call site a future edit could
    silently forget to update."""
    import minion_assist.minion as minion_mod

    captured_kwargs: dict = {}

    class _FakeMatrixChannel:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self, *_args, **kwargs):
            captured_kwargs.update(kwargs)

        def stop(self, *_args, **_kwargs):
            pass

    with (
        patch("minion_assist.minion.workspace", tmp_path),
        patch("minion_assist.minion.mcp_cfg", SimpleNamespace(servers=())),
        patch("minion_assist.minion.channels_cfg", SimpleNamespace(matrix=SimpleNamespace())),
        patch("minion_assist.matrix.channel.MatrixChannel", _FakeMatrixChannel),
        patch("minion_assist.agents.session.run_turn", Mock()),
        patch("minion_assist.minion.create_provider", return_value=Mock()),
        patch("builtins.input", side_effect=iter(["quit"])),
    ):
        minion_mod.main()

    factories = captured_kwargs["session_factories"]
    worker_health = captured_kwargs["worker_health"]

    room_session = factories["main"]("fake-room-session-id")

    assert room_session._health is not None
    assert room_session._health is worker_health.get("session_writes:main")
    # A database is configured for "main" in this environment, so the
    # degraded-mode extraction tracker must be None here — not silently
    # wired to the wrong object.
    assert room_session._extraction_health is None


# ---------------------------------------------------------------------------
# MEM-GAP-015: production-wiring coverage for MemoryRetentionScheduler
# ---------------------------------------------------------------------------
# Only requires a database (not embeddings), unlike the two tests above —
# a narrower skip guard than _requires_live_db_and_embeddings.

_requires_live_db = pytest.mark.skipif(
    not _DB_AVAILABLE_FOR_WIRING, reason="requires a live PostgreSQL instance"
)


@_requires_live_db
def test_main_wires_memory_retention_scheduler_when_enabled(tmp_path, capsys):
    """minion.py's real startup path must actually construct and start a
    MemoryRetentionScheduler and register its WorkerHealth when
    memory_retention.enabled=True and a database is configured — the same
    'did minion.py forget to wire it up' check the MEM-GAP-018 round added
    for the other schedulers, applied to this new one."""
    import minion_assist.minion as minion_mod
    from minion_assist.config import MemoryRetentionConfig

    enabled_cfg = MemoryRetentionConfig(
        enabled=True, hour=4, minute=45, timezone="UTC", retention_days=30
    )

    with (
        patch("minion_assist.minion.workspace", tmp_path),
        patch("minion_assist.minion.mcp_cfg", SimpleNamespace(servers=())),
        patch("minion_assist.minion.channels_cfg", SimpleNamespace(matrix=None)),
        patch("minion_assist.minion.memory_retention_cfg", enabled_cfg),
        patch("minion_assist.agents.session.run_turn", Mock()),
        patch("minion_assist.minion.create_provider", return_value=Mock()),
        patch("builtins.input", side_effect=iter(["/status deep", "quit"])),
    ):
        minion_mod.main()

    out = capsys.readouterr().out
    assert "memory_retention: not running" not in out

"""Tests for the /session, /rename, and /delete-session slash commands."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from minion_assist.commands import CommandContext, dispatch_command
from minion_assist.memory.short_term import ShortTermMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_session(stm: ShortTermMemory, agent_id: str, session_id: str, messages: list[dict]) -> None:
    stm.save(agent_id, session_id, messages)


def _make_session_mock(session_id: str) -> MagicMock:
    session = MagicMock()
    type(session).session_id = property(lambda s: session_id)
    session.switch_session.return_value = 4
    return session


def _ctx(args: str, stm: ShortTermMemory, agent_id: str = "main", session_id: str = "aaa") -> CommandContext:
    session = _make_session_mock(session_id)
    return CommandContext(
        raw=f"/session {args}".strip(),
        command="/session",
        args=args,
        target_agent_id=agent_id,
        sessions={agent_id: session},
        agents_cfg={},
        short_term=stm,
    )


# ---------------------------------------------------------------------------
# Tests: bare /session (list)
# ---------------------------------------------------------------------------

def test_history_no_sessions(tmp_path):
    stm = ShortTermMemory(tmp_path / "sessions")
    result = dispatch_command(_ctx("", stm))
    assert result.handled
    assert "No session history" in result.message


def test_history_lists_sessions(tmp_path):
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "aaa-000", [
        {"role": "user", "content": "Hello world"},
        {"role": "assistant", "content": "Hi!"},
    ])
    time.sleep(0.01)  # ensure different mtime
    _write_session(stm, "main", "bbb-111", [
        {"role": "user", "content": "Second session"},
    ])

    result = dispatch_command(_ctx("", stm, session_id="bbb-111"))
    assert result.handled
    msg = result.message
    assert "History for main" in msg
    assert "[1]" in msg
    assert "[2]" in msg
    # Most recent (bbb) is [1], oldest (aaa) is [2]
    assert msg.index("[1]") < msg.index("bbb")
    assert "Hello world" in msg or "Second session" in msg
    assert "Use /session" in msg


def test_history_marks_current_session(tmp_path):
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "cur-001", [{"role": "user", "content": "current"}])
    time.sleep(0.01)
    _write_session(stm, "main", "old-002", [{"role": "user", "content": "older"}])

    # Current is old-002 (most recent in list = [1])
    result = dispatch_command(_ctx("", stm, session_id="old-002"))
    msg = result.message
    lines = msg.splitlines()
    # The [1] line should be marked with *
    entry_1 = next(l for l in lines if "[1]" in l)
    assert entry_1.startswith("*")
    # The [2] line should not be marked
    entry_2 = next(l for l in lines if "[2]" in l)
    assert not entry_2.startswith("*")


def test_history_no_short_term_returns_error():
    ctx = CommandContext(
        raw="/session",
        command="/session",
        args="",
        target_agent_id="main",
        sessions={"main": MagicMock()},
        agents_cfg={},
        short_term=None,
    )
    result = dispatch_command(ctx)
    assert result.handled
    assert "not available" in result.message


# ---------------------------------------------------------------------------
# Tests: /session <N> (load by index)
# ---------------------------------------------------------------------------

def test_history_load_by_index(tmp_path):
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "aaa-000", [{"role": "user", "content": "a"}])
    time.sleep(0.01)
    _write_session(stm, "main", "bbb-111", [{"role": "user", "content": "b"}])

    ctx = _ctx("2", stm, session_id="bbb-111")
    # [2] is the older one (aaa-000)
    result = dispatch_command(ctx)
    assert result.handled
    assert "aaa" in result.message
    ctx.sessions["main"].switch_session.assert_called_once_with("aaa-000")


def test_history_load_index_out_of_range(tmp_path):
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "aaa-000", [{"role": "user", "content": "a"}])

    result = dispatch_command(_ctx("5", stm))
    assert result.handled
    assert "out of range" in result.message


def test_history_already_on_current(tmp_path):
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "aaa-000", [{"role": "user", "content": "a"}])

    result = dispatch_command(_ctx("1", stm, session_id="aaa-000"))
    assert result.handled
    assert "Already" in result.message


# ---------------------------------------------------------------------------
# Tests: /session <uuid-prefix> (load by prefix)
# ---------------------------------------------------------------------------

def test_history_load_by_uuid_prefix(tmp_path):
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "aaa-000", [{"role": "user", "content": "a"}])
    time.sleep(0.01)
    _write_session(stm, "main", "bbb-111", [{"role": "user", "content": "b"}])

    result = dispatch_command(_ctx("aaa", stm, session_id="bbb-111"))
    assert result.handled
    assert "aaa" in result.message
    result.message  # loaded message confirms


def test_history_ambiguous_prefix(tmp_path):
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "abc-001", [{"role": "user", "content": "a"}])
    _write_session(stm, "main", "abc-002", [{"role": "user", "content": "b"}])

    result = dispatch_command(_ctx("abc", stm))
    assert result.handled
    assert "Ambiguous" in result.message


def test_history_no_match_prefix(tmp_path):
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "aaa-000", [{"role": "user", "content": "a"}])

    result = dispatch_command(_ctx("zzz", stm))
    assert result.handled
    assert "No session matching" in result.message


# ---------------------------------------------------------------------------
# Tests: /session load — transcript display
# ---------------------------------------------------------------------------

def test_session_load_shows_transcript(tmp_path):
    """Loading a session should render the conversation in the message."""
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "old-001", [
        {"role": "user", "content": "Hello from old session"},
        {"role": "assistant", "content": "Hi there!"},
    ])
    time.sleep(0.01)
    _write_session(stm, "main", "cur-001", [{"role": "user", "content": "current"}])

    # Use a real session mock that returns proper history after switch
    from unittest.mock import PropertyMock
    session = MagicMock()
    type(session).session_id = property(lambda s: "cur-001")
    loaded = []

    def _switch(sid):
        loaded[:] = stm.load("main", sid)
    session.switch_session.side_effect = _switch
    type(session).history = property(lambda s: loaded)

    ctx = CommandContext(
        raw="/session 2",
        command="/session",
        args="2",
        target_agent_id="main",
        sessions={"main": session},
        agents_cfg={},
        short_term=stm,
    )
    result = dispatch_command(ctx)
    assert result.handled
    assert "Hello from old session" in result.message
    assert "Hi there!" in result.message
    assert "User:" in result.message
    assert "Assistant:" in result.message


def test_session_load_shows_name_in_header(tmp_path):
    """Header should show the session name when one is set."""
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "old-001", [{"role": "user", "content": "hi"}])
    stm.set_name("main", "old-001", "Auth work")
    time.sleep(0.01)
    _write_session(stm, "main", "cur-001", [])

    from unittest.mock import PropertyMock
    session = MagicMock()
    type(session).session_id = property(lambda s: "cur-001")
    loaded = []

    def _switch(sid):
        loaded[:] = stm.load("main", sid)
    session.switch_session.side_effect = _switch
    type(session).history = property(lambda s: loaded)

    ctx = CommandContext(
        raw="/session 2",
        command="/session",
        args="2",
        target_agent_id="main",
        sessions={"main": session},
        agents_cfg={},
        short_term=stm,
    )
    result = dispatch_command(ctx)
    assert "[Auth work]" in result.message


# ---------------------------------------------------------------------------
# Tests: /rename
# ---------------------------------------------------------------------------

def _rename_ctx(args: str, stm: ShortTermMemory, session_id: str = "cur-001") -> CommandContext:
    session = _make_session_mock(session_id)
    return CommandContext(
        raw=f"/rename {args}".strip(),
        command="/rename",
        args=args,
        target_agent_id="main",
        sessions={"main": session},
        agents_cfg={},
        short_term=stm,
    )


def test_rename_current_session(tmp_path):
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "cur-001", [{"role": "user", "content": "hi"}])

    result = dispatch_command(_rename_ctx("Auth debugging", stm))
    assert result.handled
    assert "Auth debugging" in result.message
    assert stm.get_name("main", "cur-001") == "Auth debugging"


def test_rename_by_index(tmp_path):
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "old-001", [{"role": "user", "content": "old"}])
    time.sleep(0.01)
    _write_session(stm, "main", "cur-001", [{"role": "user", "content": "new"}])

    # [2] is the older session (old-001)
    result = dispatch_command(_rename_ctx("2 Old session", stm, session_id="cur-001"))
    assert result.handled
    assert "Old session" in result.message
    assert stm.get_name("main", "old-001") == "Old session"


def test_rename_empty_name_returns_error(tmp_path):
    stm = ShortTermMemory(tmp_path / "sessions")
    result = dispatch_command(_rename_ctx("", stm))
    assert result.handled
    assert "Usage" in result.message


def test_rename_shows_in_history_listing(tmp_path):
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "aaa-001", [{"role": "user", "content": "first message"}])
    stm.set_name("main", "aaa-001", "My named session")

    result = dispatch_command(_ctx("", stm, session_id="aaa-001"))
    assert result.handled
    assert "[My named session]" in result.message
    # Name takes priority — first message preview should not appear
    assert "first message" not in result.message


def test_rename_no_short_term_returns_error():
    ctx = CommandContext(
        raw="/rename foo",
        command="/rename",
        args="foo",
        target_agent_id="main",
        sessions={"main": MagicMock()},
        agents_cfg={},
        short_term=None,
    )
    result = dispatch_command(ctx)
    assert result.handled
    assert "not available" in result.message


def test_rename_index_out_of_range(tmp_path):
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "aaa-001", [{"role": "user", "content": "a"}])

    result = dispatch_command(_rename_ctx("5 Name", stm))
    assert result.handled
    assert "out of range" in result.message


def test_set_name_empty_clears_name(tmp_path):
    stm = ShortTermMemory(tmp_path / "sessions")
    stm.save("main", "ses-001", [])
    stm.set_name("main", "ses-001", "Temp name")
    assert stm.get_name("main", "ses-001") == "Temp name"
    stm.set_name("main", "ses-001", "")
    assert stm.get_name("main", "ses-001") is None


# ---------------------------------------------------------------------------
# Helpers for /delete-session
# ---------------------------------------------------------------------------

def _delete_ctx(
    args: str,
    stm: ShortTermMemory,
    agent_id: str = "main",
    session_id: str = "cur-001",
) -> CommandContext:
    """Build a CommandContext targeting /delete-session."""
    session = _make_session_mock(session_id)
    return CommandContext(
        raw=f"/delete-session {args}".strip(),
        command="/delete-session",
        args=args,
        target_agent_id=agent_id,
        sessions={agent_id: session},
        agents_cfg={},
        short_term=stm,
    )


# ---------------------------------------------------------------------------
# Tests: /delete-session
# ---------------------------------------------------------------------------

def test_delete_session_no_sessions(tmp_path):
    """/delete-session with no history should report no sessions found."""
    stm = ShortTermMemory(tmp_path / "sessions")
    result = dispatch_command(_delete_ctx("", stm))
    assert result.handled
    assert "No session history" in result.message


def test_delete_session_no_short_term_returns_error():
    """/delete-session should fail gracefully when short_term is None."""
    ctx = CommandContext(
        raw="/delete-session",
        command="/delete-session",
        args="",
        target_agent_id="main",
        sessions={"main": MagicMock()},
        agents_cfg={},
        short_term=None,
    )
    result = dispatch_command(ctx)
    assert result.handled
    assert "not available" in result.message


def test_delete_session_no_arg_shows_listing(tmp_path):
    """Bare /delete-session shows the session list with a usage hint."""
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "aaa-001", [{"role": "user", "content": "hello"}])
    time.sleep(0.01)
    _write_session(stm, "main", "bbb-002", [{"role": "user", "content": "world"}])

    result = dispatch_command(_delete_ctx("", stm, session_id="bbb-002"))
    assert result.handled
    assert "History for main" in result.message
    assert "[1]" in result.message
    assert "[2]" in result.message
    assert "Use /delete-session" in result.message


def test_delete_session_by_index(tmp_path):
    """/delete-session N removes the .jsonl file for that session."""
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "old-001", [{"role": "user", "content": "old"}])
    time.sleep(0.01)
    _write_session(stm, "main", "cur-001", [{"role": "user", "content": "current"}])

    # [2] is the older session (old-001) in newest-first order
    result = dispatch_command(_delete_ctx("2", stm, session_id="cur-001"))
    assert result.handled
    assert "Deleted session" in result.message
    assert "old" in result.message  # uuid hint or name
    # The .jsonl file should be gone
    assert not (tmp_path / "sessions" / "main" / "old-001.jsonl").exists()


def test_delete_session_by_uuid_prefix(tmp_path):
    """/delete-session <prefix> resolves by UUID prefix and removes the file."""
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "xyz-999", [{"role": "user", "content": "x"}])
    time.sleep(0.01)
    _write_session(stm, "main", "cur-001", [{"role": "user", "content": "current"}])

    result = dispatch_command(_delete_ctx("xyz", stm, session_id="cur-001"))
    assert result.handled
    assert "Deleted session" in result.message
    assert not (tmp_path / "sessions" / "main" / "xyz-999.jsonl").exists()


def test_delete_session_removes_name_sidecar(tmp_path):
    """Deleting a session also removes its .name sidecar file."""
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "old-001", [{"role": "user", "content": "old"}])
    stm.set_name("main", "old-001", "Work session")
    time.sleep(0.01)
    _write_session(stm, "main", "cur-001", [{"role": "user", "content": "current"}])

    result = dispatch_command(_delete_ctx("2", stm, session_id="cur-001"))
    assert result.handled
    assert "Work session" in result.message  # name shown in confirmation
    assert not (tmp_path / "sessions" / "main" / "old-001.jsonl").exists()
    assert not (tmp_path / "sessions" / "main" / "old-001.name").exists()


def test_delete_session_active_rejected(tmp_path):
    """Attempting to delete the currently active session is refused."""
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "cur-001", [{"role": "user", "content": "active"}])

    # [1] is the only (and active) session
    result = dispatch_command(_delete_ctx("1", stm, session_id="cur-001"))
    assert result.handled
    assert "Cannot delete the active session" in result.message
    # The file must still exist
    assert (tmp_path / "sessions" / "main" / "cur-001.jsonl").exists()


def test_delete_session_index_out_of_range(tmp_path):
    """/delete-session with an out-of-range index reports an error."""
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "aaa-001", [{"role": "user", "content": "a"}])

    result = dispatch_command(_delete_ctx("5", stm))
    assert result.handled
    assert "out of range" in result.message


def test_delete_session_ambiguous_prefix(tmp_path):
    """/delete-session refuses when multiple sessions share the same prefix."""
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "abc-001", [{"role": "user", "content": "a"}])
    _write_session(stm, "main", "abc-002", [{"role": "user", "content": "b"}])

    result = dispatch_command(_delete_ctx("abc", stm))
    assert result.handled
    assert "Ambiguous" in result.message


def test_delete_session_no_match_prefix(tmp_path):
    """/delete-session reports an error when the prefix matches nothing."""
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "aaa-001", [{"role": "user", "content": "a"}])

    result = dispatch_command(_delete_ctx("zzz", stm))
    assert result.handled
    assert "No session matching" in result.message


# ---------------------------------------------------------------------------
# Tests: /delete-session cross-store cleanup (MEM-GAP-003)
# ---------------------------------------------------------------------------

def _delete_ctx_with_db(
    args: str,
    stm: ShortTermMemory,
    db: MagicMock,
    agent_id: str = "main",
    session_id: str = "cur-001",
    memory: MagicMock | None = None,
) -> CommandContext:
    """Like _delete_ctx, but with a mock SessionDB (and optionally a mock
    MemoryService reachable via sessions[agent_id].memory) wired in."""
    session = _make_session_mock(session_id)
    session.memory = memory  # plain instance attribute — no class-level property tricks
    return CommandContext(
        raw=f"/delete-session {args}".strip(),
        command="/delete-session",
        args=args,
        target_agent_id=agent_id,
        sessions={agent_id: session},
        agents_cfg={},
        short_term=stm,
        db=db,
    )


def test_delete_session_without_db_has_no_database_note(tmp_path):
    """No ctx.db configured (the default) — behavior is unchanged from before MEM-GAP-003."""
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "old-001", [{"role": "user", "content": "old"}])
    time.sleep(0.01)
    _write_session(stm, "main", "cur-001", [{"role": "user", "content": "current"}])

    result = dispatch_command(_delete_ctx("2", stm, session_id="cur-001"))

    assert result.message.rstrip(".") == "Deleted session old-001"
    assert "database" not in result.message.lower()


def test_delete_session_with_db_reports_counts_and_forgets_proposals(tmp_path):
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "old-001", [{"role": "user", "content": "old"}])
    time.sleep(0.01)
    _write_session(stm, "main", "cur-001", [{"role": "user", "content": "current"}])

    db = MagicMock()
    db.delete_session.return_value = {"messages": 5, "proposal_ids": [10, 11]}
    memory = MagicMock()

    result = dispatch_command(
        _delete_ctx_with_db("2", stm, db, session_id="cur-001", memory=memory)
    )

    assert result.handled
    db.delete_session.assert_called_once_with("main", "old-001")
    memory.forget_proposals.assert_called_once_with([10, 11])
    assert "5 database message" in result.message
    assert "2 proposal" in result.message
    # The JSONL file is still gone regardless of the database cleanup outcome.
    assert not (tmp_path / "sessions" / "main" / "old-001.jsonl").exists()


def test_delete_session_with_db_and_no_proposals_skips_memory_call(tmp_path):
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "old-001", [{"role": "user", "content": "old"}])
    time.sleep(0.01)
    _write_session(stm, "main", "cur-001", [{"role": "user", "content": "current"}])

    db = MagicMock()
    db.delete_session.return_value = {"messages": 3, "proposal_ids": []}
    memory = MagicMock()

    result = dispatch_command(
        _delete_ctx_with_db("2", stm, db, session_id="cur-001", memory=memory)
    )

    assert result.handled
    memory.forget_proposals.assert_not_called()
    assert "3 database message" in result.message
    assert "0 proposal" in result.message


def test_delete_session_db_returns_none_reports_only_jsonl_deletion(tmp_path):
    """delete_session() returns None when this session was never mirrored to PostgreSQL."""
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "old-001", [{"role": "user", "content": "old"}])
    time.sleep(0.01)
    _write_session(stm, "main", "cur-001", [{"role": "user", "content": "current"}])

    db = MagicMock()
    db.delete_session.return_value = None
    memory = MagicMock()

    result = dispatch_command(
        _delete_ctx_with_db("2", stm, db, session_id="cur-001", memory=memory)
    )

    assert result.handled
    memory.forget_proposals.assert_not_called()
    assert "Deleted session" in result.message
    assert "database" not in result.message.lower()


def test_delete_session_db_failure_still_deletes_jsonl_and_warns(tmp_path):
    """A PostgreSQL cleanup failure must not hide that the JSONL file is already gone."""
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "old-001", [{"role": "user", "content": "old"}])
    time.sleep(0.01)
    _write_session(stm, "main", "cur-001", [{"role": "user", "content": "current"}])

    db = MagicMock()
    db.delete_session.side_effect = Exception("connection refused")

    result = dispatch_command(_delete_ctx_with_db("2", stm, db, session_id="cur-001"))

    assert result.handled
    assert "WARNING" in result.message
    assert "connection refused" in result.message
    assert not (tmp_path / "sessions" / "main" / "old-001.jsonl").exists()


def test_delete_session_forget_proposals_failure_is_reported_not_swallowed(tmp_path):
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "old-001", [{"role": "user", "content": "old"}])
    time.sleep(0.01)
    _write_session(stm, "main", "cur-001", [{"role": "user", "content": "current"}])

    db = MagicMock()
    db.delete_session.return_value = {"messages": 1, "proposal_ids": [42]}
    memory = MagicMock()
    memory.forget_proposals.side_effect = Exception("index unreachable")

    result = dispatch_command(
        _delete_ctx_with_db("2", stm, db, session_id="cur-001", memory=memory)
    )

    assert result.handled
    assert "index unreachable" in result.message
    # The database-level deletion itself still succeeded and is reported.
    assert "1 database message" in result.message


def test_delete_session_with_db_but_no_memory_available_skips_forget_silently(tmp_path):
    """If sessions[agent_id].memory is None, forget_proposals is simply not called."""
    stm = ShortTermMemory(tmp_path / "sessions")
    _write_session(stm, "main", "old-001", [{"role": "user", "content": "old"}])
    time.sleep(0.01)
    _write_session(stm, "main", "cur-001", [{"role": "user", "content": "current"}])

    db = MagicMock()
    db.delete_session.return_value = {"messages": 2, "proposal_ids": [1]}

    result = dispatch_command(
        _delete_ctx_with_db("2", stm, db, session_id="cur-001", memory=None)
    )

    assert result.handled
    assert "2 database message" in result.message

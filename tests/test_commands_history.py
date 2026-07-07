"""Tests for the /history slash command."""

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
        raw=f"/history {args}".strip(),
        command="/history",
        args=args,
        target_agent_id=agent_id,
        sessions={agent_id: session},
        agents_cfg={},
        short_term=stm,
    )


# ---------------------------------------------------------------------------
# Tests: bare /history (list)
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
    assert "Use /history" in msg


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
        raw="/history",
        command="/history",
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
# Tests: /history <N> (load by index)
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
# Tests: /history <uuid-prefix> (load by prefix)
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

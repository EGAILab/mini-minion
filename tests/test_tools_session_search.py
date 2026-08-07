"""Tests for SessionSearchTool's agent-scoping (MEM-GAP-002).

Uses a Mock in place of SessionDB so these tests run without a live
PostgreSQL instance (the SQL-level isolation itself is proven separately in
test_session_db.py, against a real database) — here we only need to prove
the tool always forwards its own agent_id into every DB call, and never
lets a caller override it.
"""

from __future__ import annotations

from unittest.mock import Mock

from minion_assist.tools.session_search import SessionSearchTool


def _tool(agent_id: str = "main") -> tuple[SessionSearchTool, Mock]:
    db = Mock()
    db.hybrid_search_messages.return_value = []
    db.get_sessions_by_ids.return_value = {}
    db.get_session_bookends.return_value = ([], [])
    db.get_messages_around.return_value = []
    db.list_sessions.return_value = []
    return SessionSearchTool(db, agent_id), db


def test_discover_passes_the_owning_agent_id():
    tool, db = _tool("researcher")

    tool.execute(mode="DISCOVER", query="hunter2")

    db.hybrid_search_messages.assert_called_once_with("hunter2", "researcher", limit=15)


def test_scroll_passes_the_owning_agent_id():
    tool, db = _tool("researcher")

    tool.execute(mode="SCROLL", session_id="some-session", anchor_message_id=5)

    db.get_messages_around.assert_called_once_with("some-session", "researcher", 5, window=5)


def test_browse_passes_the_owning_agent_id():
    tool, db = _tool("researcher")

    tool.execute(mode="BROWSE")

    db.list_sessions.assert_called_once_with("researcher", limit=20)


def test_discover_uses_the_agent_id_for_session_metadata_and_bookends():
    tool, db = _tool("researcher")
    db.hybrid_search_messages.return_value = [
        {"session_id": "s1", "id": 1, "rank": 0.5, "content": "hi", "snippet": "hi"},
    ]
    db.get_sessions_by_ids.return_value = {"s1": {"title": "t", "agent_id": "researcher"}}

    tool.execute(mode="DISCOVER", query="hi")

    db.get_sessions_by_ids.assert_called_once_with(["s1"], "researcher")
    db.get_session_bookends.assert_called_once_with("s1", "researcher", n=2)
    db.get_messages_around.assert_called_once_with("s1", "researcher", 1, window=3)


# ---------------------------------------------------------------------------
# Real-database cross-agent isolation (MEM-GAP-018)
# ---------------------------------------------------------------------------
# Everything above proves the tool always *forwards* its own agent_id — but
# only against a Mock, which can't prove PostgreSQL's WHERE agent_id = ...
# clause actually blocks the query. These tests drive
# SessionSearchTool.execute() through a REAL SessionDB with two different
# agents' data present, proving genuine end-to-end isolation through the
# actual tool call path (not the DB layer directly, which is already
# covered by tests/test_session_db.py). Skipped, not failed, without a
# reachable dev PostgreSQL instance — same convention as test_session_db.py.

import uuid

import pytest

# _DB_URL sourced from minion_assist.config.database.url (patched to an
# isolated per-session schema by tests/conftest.py — R2-GAP-015), not a
# literal — see that file's module docstring.
from minion_assist.config import database as _database_cfg

_DB_URL = _database_cfg.url or "postgresql://minion:minion@localhost:5433/minion_assist"

try:
    import psycopg as _psycopg

    _test_conn = _psycopg.connect(_DB_URL, connect_timeout=2)
    _test_conn.close()
    _DB_AVAILABLE = True
except Exception:
    _DB_AVAILABLE = False

_requires_live_db = pytest.mark.skipif(
    not _DB_AVAILABLE, reason="requires a live PostgreSQL instance"
)


@pytest.fixture
def _real_db():
    from minion_assist.session.db import SessionDB
    return SessionDB(_DB_URL)


def _cleanup_sessions(real_db, *session_ids: str) -> None:
    conn = real_db._conn()
    for sid in session_ids:
        conn.execute("DELETE FROM message_mirrors WHERE session_id = %s", (sid,))
        conn.execute("DELETE FROM messages WHERE session_id = %s", (sid,))
        conn.execute("DELETE FROM sessions WHERE id = %s", (sid,))


@_requires_live_db
def test_discover_never_surfaces_another_agents_session_through_real_db(_real_db):
    marker = f"zzqisolationmarker{uuid.uuid4().hex[:8]}"
    owner_session = f"test-{uuid.uuid4()}"
    other_session = f"test-{uuid.uuid4()}"
    try:
        _real_db.upsert_session(owner_session, "main")
        _real_db.upsert_session(other_session, "researcher")
        _real_db.mirror_message(owner_session, "e1", "user", f"my note about {marker}")
        _real_db.mirror_message(other_session, "e2", "user", f"other agent note about {marker}")

        owner_tool = SessionSearchTool(_real_db, "main")
        result = owner_tool.execute(mode="DISCOVER", query=marker)

        assert owner_session[:8] in result
        assert other_session[:8] not in result
        assert "[researcher]" not in result
    finally:
        _cleanup_sessions(_real_db, owner_session, other_session)


@_requires_live_db
def test_browse_never_lists_another_agents_sessions_through_real_db(_real_db):
    owner_session = f"test-{uuid.uuid4()}"
    other_session = f"test-{uuid.uuid4()}"
    try:
        _real_db.upsert_session(owner_session, "main")
        _real_db.upsert_session(other_session, "researcher")
        _real_db.mirror_message(owner_session, "e1", "user", "hello")
        _real_db.mirror_message(other_session, "e2", "user", "hello")

        owner_tool = SessionSearchTool(_real_db, "main")
        result = owner_tool.execute(mode="BROWSE")

        assert owner_session[:8] in result
        assert other_session[:8] not in result
    finally:
        _cleanup_sessions(_real_db, owner_session, other_session)

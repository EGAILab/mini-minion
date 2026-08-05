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
    db.search_messages.return_value = []
    db.get_sessions_by_ids.return_value = {}
    db.get_session_bookends.return_value = ([], [])
    db.get_messages_around.return_value = []
    db.list_sessions.return_value = []
    return SessionSearchTool(db, agent_id), db


def test_discover_passes_the_owning_agent_id():
    tool, db = _tool("researcher")

    tool.execute(mode="DISCOVER", query="hunter2")

    db.search_messages.assert_called_once_with("hunter2", "researcher", limit=15)


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
    db.search_messages.return_value = [
        {"session_id": "s1", "id": 1, "rank": 0.5, "content": "hi", "snippet": "hi"},
    ]
    db.get_sessions_by_ids.return_value = {"s1": {"title": "t", "agent_id": "researcher"}}

    tool.execute(mode="DISCOVER", query="hi")

    db.get_sessions_by_ids.assert_called_once_with(["s1"], "researcher")
    db.get_session_bookends.assert_called_once_with("s1", "researcher", n=2)
    db.get_messages_around.assert_called_once_with("s1", "researcher", 1, window=3)

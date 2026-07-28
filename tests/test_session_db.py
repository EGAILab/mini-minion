"""Tests for session/db.py: SessionDB — Stage One Phase 2, slice A.

Focuses on idempotent mirroring (mirror_message/is_mirrored) and
reconciliation (reconcile_session/reconcile_all_sessions), since those are
what slice A adds. This module had zero test coverage before this change.

Requires a live PostgreSQL instance matching config.json's configured URL
(``postgresql://minion:minion@localhost:5433/minion_assist`` — see
docker-compose.yml). The whole module is skipped if one isn't reachable, so
this doesn't break test runs in environments without Docker/Postgres.

Every test uses a fresh, random session_id and cleans up its own rows in an
autouse fixture, so tests never collide with each other or leave rows behind
in the shared dev database.
"""

from __future__ import annotations

import uuid

import pytest

from minion_assist.memory.short_term import ShortTermMemory
from minion_assist.messages import EVENT_ID_KEY, ensure_event_id

_DB_URL = "postgresql://minion:minion@localhost:5433/minion_assist"

try:
    import psycopg as _psycopg

    _test_conn = _psycopg.connect(_DB_URL, connect_timeout=2)
    _test_conn.close()
    _DB_AVAILABLE = True
except Exception:
    _DB_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _DB_AVAILABLE, reason="requires a live PostgreSQL instance")

from minion_assist.session.db import SessionDB  # noqa: E402


@pytest.fixture
def db():
    return SessionDB(_DB_URL)


@pytest.fixture
def session_id():
    """A fresh, unique session_id per test so tests never collide."""
    return f"test-{uuid.uuid4()}"


@pytest.fixture(autouse=True)
def _cleanup_after(db, session_id):
    yield
    conn = db._conn()
    conn.execute("DELETE FROM message_mirrors WHERE session_id = %s", (session_id,))
    conn.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id = %s", (session_id,))


def _message_count(db, session_id) -> int:
    return db._conn().execute(
        "SELECT count(*) FROM messages WHERE session_id = %s", (session_id,)
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# is_mirrored / mirror_message
# ---------------------------------------------------------------------------

def test_is_mirrored_false_before_mirroring(db, session_id):
    assert db.is_mirrored(session_id, "event-1") is False


def test_mirror_message_inserts_and_marks_mirrored(db, session_id):
    db.upsert_session(session_id, "main")
    message_id = db.mirror_message(session_id, "event-1", "user", "hello")
    assert message_id is not None
    assert db.is_mirrored(session_id, "event-1") is True


def test_mirror_message_is_idempotent(db, session_id):
    db.upsert_session(session_id, "main")
    first = db.mirror_message(session_id, "event-1", "user", "hello")
    second = db.mirror_message(session_id, "event-1", "user", "hello")
    assert first is not None
    assert second is None  # no-op, not a duplicate


def test_mirror_message_idempotent_does_not_duplicate_row(db, session_id):
    db.upsert_session(session_id, "main")
    db.mirror_message(session_id, "event-1", "user", "hello")
    db.mirror_message(session_id, "event-1", "user", "hello")
    assert _message_count(db, session_id) == 1


def test_mirror_message_different_event_ids_both_inserted(db, session_id):
    db.upsert_session(session_id, "main")
    db.mirror_message(session_id, "event-1", "user", "first")
    db.mirror_message(session_id, "event-2", "assistant", "second")
    assert _message_count(db, session_id) == 2


# ---------------------------------------------------------------------------
# reconcile_session
# ---------------------------------------------------------------------------

def test_reconcile_session_mirrors_all_messages(db, session_id):
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    mirrored = db.reconcile_session(session_id, "main", messages, mtime=1234.0)
    assert mirrored == 2


def test_reconcile_session_assigns_event_ids(db, session_id):
    messages = [{"role": "user", "content": "hi"}]
    db.reconcile_session(session_id, "main", messages, mtime=1234.0)
    assert EVENT_ID_KEY in messages[0]


def test_reconcile_session_second_call_mirrors_nothing_new(db, session_id):
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    db.reconcile_session(session_id, "main", messages, mtime=1234.0)
    second = db.reconcile_session(session_id, "main", messages, mtime=1234.0)
    assert second == 0


def test_reconcile_session_completes_a_partial_mirror(db, session_id):
    """Simulates a crash that mirrored message 1 but never reached message 2."""
    messages = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "second"}]
    ensure_event_id(messages[0])
    ensure_event_id(messages[1])
    db.upsert_session(session_id, "main")
    db.mirror_message(session_id, messages[0][EVENT_ID_KEY], "user", "first", timestamp=1234.0)

    mirrored = db.reconcile_session(session_id, "main", messages, mtime=1234.0)

    assert mirrored == 1  # only the missing one
    assert db.is_mirrored(session_id, messages[1][EVENT_ID_KEY])
    assert _message_count(db, session_id) == 2  # no duplicate of message 1


def test_reconcile_session_skips_messages_without_role(db, session_id):
    messages = [{"content": "no role"}, {"role": "user", "content": "has role"}]
    mirrored = db.reconcile_session(session_id, "main", messages, mtime=1234.0)
    assert mirrored == 1


# ---------------------------------------------------------------------------
# reconcile_all_sessions
# ---------------------------------------------------------------------------

def test_reconcile_all_sessions_mirrors_and_persists_event_ids(db, tmp_path, session_id):
    short_term = ShortTermMemory(tmp_path)
    short_term.save("main", session_id, [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])

    total = db.reconcile_all_sessions(short_term, ["main"])

    assert total == 2
    reloaded = short_term.load("main", session_id)
    assert all(EVENT_ID_KEY in m for m in reloaded)


def test_reconcile_all_sessions_second_call_is_a_no_op(db, tmp_path, session_id):
    short_term = ShortTermMemory(tmp_path)
    short_term.save("main", session_id, [{"role": "user", "content": "hi"}])

    db.reconcile_all_sessions(short_term, ["main"])
    second_total = db.reconcile_all_sessions(short_term, ["main"])

    assert second_total == 0


def test_reconcile_all_sessions_skips_agents_with_no_sessions(db, tmp_path):
    short_term = ShortTermMemory(tmp_path)
    total = db.reconcile_all_sessions(short_term, ["nonexistent-agent"])
    assert total == 0

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

import time
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


@pytest.fixture(scope="session", autouse=True)
def _purge_stale_test_rows():
    """Purge leftover rows before this file's tests run at all.

    claim_next_capture_job() is a genuinely global queue operation (by
    design — one worker services every session), so any *other* pending row
    can be claimed instead of the one a test just enqueued, since both are
    "the oldest pending job" from the query's point of view. Per-test
    cleanup alone can't fix this — it only runs after a test that actually
    completes, and it only ever touches rows the *same* test created.

    In practice this isn't limited to interrupted-run leftovers: other test
    files (e.g. test_minion.py) build real AgentSessions against the actual
    configured database and enqueue genuine capture jobs under plain
    session-id UUIDs (not prefixed 'test-') as a side effect of exercising
    send(). memory_capture_jobs/memory_proposals are new tables introduced
    by this feature — nothing pre-existing could have meaningful data in
    them — so it's safe to wipe them unconditionally rather than filtering
    by session_id prefix, which would otherwise miss exactly this case.
    messages/message_mirrors/sessions predate this feature, so those stay
    scoped to this file's own 'test-' prefixed rows.
    """
    conn = _psycopg.connect(_DB_URL, autocommit=True)
    conn.execute("DELETE FROM memory_proposals")
    conn.execute("DELETE FROM memory_capture_jobs")
    conn.execute("DELETE FROM message_mirrors WHERE session_id LIKE 'test-%'")
    conn.execute("DELETE FROM messages WHERE session_id LIKE 'test-%'")
    conn.execute("DELETE FROM sessions WHERE id LIKE 'test-%'")
    conn.close()


@pytest.fixture(autouse=True)
def _cleanup_after(db, session_id):
    yield
    conn = db._conn()
    conn.execute(
        """
        DELETE FROM memory_proposals
        WHERE job_id IN (SELECT id FROM memory_capture_jobs WHERE session_id = %s)
        """,
        (session_id,),
    )
    conn.execute("DELETE FROM memory_capture_jobs WHERE session_id = %s", (session_id,))
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


# ---------------------------------------------------------------------------
# get_messages_in_range
# ---------------------------------------------------------------------------

def test_get_messages_in_range_returns_ordered_messages(db, session_id):
    db.upsert_session(session_id, "main")
    id1 = db.mirror_message(session_id, "e1", "user", "first", timestamp=1.0)
    id2 = db.mirror_message(session_id, "e2", "assistant", "second", timestamp=2.0)

    messages = db.get_messages_in_range(session_id, id1, id2)

    assert [m["content"] for m in messages] == ["first", "second"]


def test_get_messages_in_range_excludes_outside_range(db, session_id):
    db.upsert_session(session_id, "main")
    id1 = db.mirror_message(session_id, "e1", "user", "first", timestamp=1.0)
    db.mirror_message(session_id, "e2", "assistant", "second", timestamp=2.0)

    messages = db.get_messages_in_range(session_id, id1, id1)

    assert [m["content"] for m in messages] == ["first"]


# ---------------------------------------------------------------------------
# Durable capture jobs (Stage One Phase 2, slice C)
# ---------------------------------------------------------------------------

def test_enqueue_capture_job_returns_job_id(db, session_id):
    job_id = db.enqueue_capture_job(
        "main", session_id, source_from_message_id=1, source_to_message_id=2,
        idempotency_key=f"key-{session_id}",
    )
    assert job_id is not None


def test_enqueue_capture_job_idempotent(db, session_id):
    key = f"key-{session_id}"
    first = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=key)
    second = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=key)

    assert first is not None
    assert second is None  # no-op, not a duplicate job


def test_claim_next_capture_job_returns_a_pending_job(db, session_id):
    db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")

    job = db.claim_next_capture_job()

    assert job is not None
    assert job["agent_id"] == "main"
    assert job["session_id"] == session_id
    assert job["source_from_message_id"] == 1
    assert job["source_to_message_id"] == 2
    assert job["attempts"] == 0


def test_claim_next_capture_job_does_not_reclaim_running_job(db, session_id):
    db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    first_claim = db.claim_next_capture_job()

    second_claim = db.claim_next_capture_job()

    assert first_claim is not None
    assert second_claim is None  # already claimed (state='running'), not due again


def test_complete_capture_job_records_proposals_and_marks_done(db, session_id):
    job_id = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    db.claim_next_capture_job()

    db.complete_capture_job(job_id, ["User prefers dark mode.", "User's dog is named Biscuit."])

    state_row = db._conn().execute(
        "SELECT state FROM memory_capture_jobs WHERE id = %s", (job_id,)
    ).fetchone()
    assert state_row[0] == "done"
    proposal_rows = db._conn().execute(
        "SELECT claim_text FROM memory_proposals WHERE job_id = %s ORDER BY id", (job_id,)
    ).fetchall()
    assert [r[0] for r in proposal_rows] == [
        "User prefers dark mode.", "User's dog is named Biscuit.",
    ]


def test_complete_capture_job_with_no_proposals_still_marks_done(db, session_id):
    job_id = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    db.claim_next_capture_job()

    db.complete_capture_job(job_id, [])

    state_row = db._conn().execute(
        "SELECT state FROM memory_capture_jobs WHERE id = %s", (job_id,)
    ).fetchone()
    assert state_row[0] == "done"


def test_complete_capture_job_returns_new_proposal_ids_and_text(db, session_id):
    # Stage One Phase 5, slice B: CaptureWorker needs each new proposal's id
    # and claim text (without a second query) to index it as searchable
    # right after this call returns.
    job_id = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    db.claim_next_capture_job()

    new_proposals = db.complete_capture_job(
        job_id, ["User prefers dark mode.", "User's dog is named Biscuit."]
    )

    assert [p["claim_text"] for p in new_proposals] == [
        "User prefers dark mode.", "User's dog is named Biscuit.",
    ]
    assert all(p["agent_id"] == "main" for p in new_proposals)
    assert all(isinstance(p["id"], int) for p in new_proposals)
    # ids must match what was actually inserted, not just be present
    proposal_rows = db._conn().execute(
        "SELECT id, claim_text FROM memory_proposals WHERE job_id = %s ORDER BY id", (job_id,)
    ).fetchall()
    assert [p["id"] for p in new_proposals] == [r[0] for r in proposal_rows]


def test_complete_capture_job_with_no_proposals_returns_empty_list(db, session_id):
    job_id = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    db.claim_next_capture_job()

    new_proposals = db.complete_capture_job(job_id, [])

    assert new_proposals == []


def test_list_pending_proposals_returns_only_pending_ones(db, session_id):
    job_id = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    db.claim_next_capture_job()
    new_proposals = db.complete_capture_job(
        job_id, ["User prefers dark mode.", "User's dog is named Biscuit."]
    )
    db._conn().execute(
        "UPDATE memory_proposals SET status = 'promoted' WHERE id = %s", (new_proposals[0]["id"],)
    )

    pending = db.list_pending_proposals("main")

    pending_ids = {p["id"] for p in pending}
    assert new_proposals[0]["id"] not in pending_ids
    assert new_proposals[1]["id"] in pending_ids


def test_list_pending_proposals_scopes_to_the_given_agent(db, session_id):
    job_id = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    db.claim_next_capture_job()
    db.complete_capture_job(job_id, ["A fact about main."])

    assert db.list_pending_proposals(f"other-{session_id}") == []


def test_get_proposal_returns_the_matching_row(db, session_id):
    job_id = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    db.claim_next_capture_job()
    new_proposals = db.complete_capture_job(job_id, ["User prefers dark mode."])

    proposal = db.get_proposal(new_proposals[0]["id"])

    assert proposal["claim_text"] == "User prefers dark mode."
    assert proposal["status"] == "pending"
    assert proposal["job_id"] == job_id


def test_get_proposal_returns_none_for_an_unknown_id(db):
    assert db.get_proposal(-1) is None


def test_new_proposal_defaults_to_pending_status(db, session_id):
    # Stage One Phase 5, slice B: the status column (used by Phase 5 slice
    # D's consolidation review) defaults to "pending" for every existing
    # and newly-created proposal.
    job_id = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    db.claim_next_capture_job()
    db.complete_capture_job(job_id, ["User prefers dark mode."])

    status_row = db._conn().execute(
        "SELECT status FROM memory_proposals WHERE job_id = %s", (job_id,)
    ).fetchone()
    assert status_row[0] == "pending"


def test_fail_capture_job_reschedules_with_backoff(db, session_id):
    job_id = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    db.claim_next_capture_job()

    db.fail_capture_job(job_id, "boom", backoff_seconds=100.0, max_attempts=5)

    row = db._conn().execute(
        "SELECT state, attempts, run_after, last_error FROM memory_capture_jobs WHERE id = %s",
        (job_id,),
    ).fetchone()
    assert row[0] == "pending"  # eligible for retry
    assert row[1] == 1
    assert row[2] > time.time()  # scheduled in the future
    assert row[3] == "boom"


def test_fail_capture_job_gives_up_after_max_attempts(db, session_id):
    job_id = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")

    for _ in range(3):
        db.claim_next_capture_job()
        db.fail_capture_job(job_id, "boom", backoff_seconds=-1.0, max_attempts=3)

    row = db._conn().execute(
        "SELECT state, attempts FROM memory_capture_jobs WHERE id = %s", (job_id,)
    ).fetchone()
    assert row[0] == "failed"
    assert row[1] == 3


def test_failed_job_is_reclaimable_after_backoff_expires(db, session_id):
    job_id = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    db.claim_next_capture_job()
    db.fail_capture_job(job_id, "boom", backoff_seconds=-1.0, max_attempts=5)  # already due

    job = db.claim_next_capture_job()

    assert job is not None
    assert job["id"] == job_id
    assert job["attempts"] == 1

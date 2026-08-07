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
in the shared dev database. As of R2-GAP-015, ``_DB_URL`` also isn't
literally the shared dev database at all when run through the normal
``pytest`` entry point — ``tests/conftest.py``'s ``pytest_configure`` hook
patches ``minion_assist.config.database`` to point at a fresh, isolated
PostgreSQL schema before this module is even imported, so table-name
collisions with a real running bot process or another concurrent test run
are structurally impossible, not just conventionally avoided. Falls back to
the literal dev URL only if that hook didn't run (e.g. this file executed
standalone, bypassing the root conftest.py) or config.json genuinely has no
``database.url`` configured.
"""

from __future__ import annotations

import time
import uuid

import pytest

from minion_assist.config import database as _database_cfg
from minion_assist.memory.short_term import ShortTermMemory
from minion_assist.messages import EVENT_ID_KEY, ensure_event_id

_DB_URL = _database_cfg.url or "postgresql://minion:minion@localhost:5433/minion_assist"

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

    Constructing a throwaway ``SessionDB`` first (rather than connecting
    directly via psycopg) runs ``_ensure_schema()`` — necessary the very
    first time a brand-new table (e.g. ``commitments``, Stage One Phase 6)
    is introduced: this fixture is session-scoped and autouse, so it runs
    before any test's own ``db`` fixture on a database that has never run
    this code before, and a bare ``DELETE FROM`` a table that doesn't
    exist yet would fail outright.
    """
    SessionDB(_DB_URL)
    conn = _psycopg.connect(_DB_URL, autocommit=True)
    conn.execute("DELETE FROM memory_proposals")
    conn.execute("DELETE FROM memory_capture_jobs")
    conn.execute("DELETE FROM commitments")
    conn.execute("DELETE FROM memory_commitment_jobs")
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
    conn.execute(
        "DELETE FROM commitments WHERE session_id = %s", (session_id,)
    )
    conn.execute(
        "DELETE FROM memory_commitment_jobs WHERE session_id = %s", (session_id,)
    )
    conn.execute("DELETE FROM message_embedding_jobs WHERE session_id = %s", (session_id,))
    if db.has_vector_lane:
        conn.execute(
            """
            DELETE FROM message_embeddings WHERE message_id IN (
                SELECT id FROM messages WHERE session_id = %s
            )
            """,
            (session_id,),
        )
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
    db.upsert_session(session_id, "main")
    job_id = db.enqueue_capture_job(
        "main", session_id, source_from_message_id=1, source_to_message_id=2,
        idempotency_key=f"key-{session_id}",
    )
    assert job_id is not None


def test_enqueue_capture_job_idempotent(db, session_id):
    db.upsert_session(session_id, "main")
    key = f"key-{session_id}"
    first = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=key)
    second = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=key)

    assert first is not None
    assert second is None  # no-op, not a duplicate job


def test_claim_next_capture_job_returns_a_pending_job(db, session_id):
    db.upsert_session(session_id, "main")
    db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")

    job = db.claim_next_capture_job()

    assert job is not None
    assert job["agent_id"] == "main"
    assert job["session_id"] == session_id
    assert job["source_from_message_id"] == 1
    assert job["source_to_message_id"] == 2
    assert job["attempts"] == 0


def test_claim_next_capture_job_does_not_reclaim_running_job(db, session_id):
    db.upsert_session(session_id, "main")
    db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    first_claim = db.claim_next_capture_job()

    second_claim = db.claim_next_capture_job()

    assert first_claim is not None
    assert second_claim is None  # already claimed (state='running'), not due again


def test_complete_capture_job_records_proposals_and_marks_done(db, session_id):
    db.upsert_session(session_id, "main")  # MEM-GAP-011: FK now requires a real session row
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
    db.upsert_session(session_id, "main")  # MEM-GAP-011: FK now requires a real session row
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
    db.upsert_session(session_id, "main")  # MEM-GAP-011: FK now requires a real session row
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
    db.upsert_session(session_id, "main")  # MEM-GAP-011: FK now requires a real session row
    job_id = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    db.claim_next_capture_job()

    new_proposals = db.complete_capture_job(job_id, [])

    assert new_proposals == []


def test_list_pending_proposals_returns_only_pending_ones(db, session_id):
    db.upsert_session(session_id, "main")  # MEM-GAP-011: FK now requires a real session row
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
    db.upsert_session(session_id, "main")  # MEM-GAP-011: FK now requires a real session row
    job_id = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    db.claim_next_capture_job()
    db.complete_capture_job(job_id, ["A fact about main."])

    assert db.list_pending_proposals(f"other-{session_id}") == []


def test_get_proposal_returns_the_matching_row(db, session_id):
    db.upsert_session(session_id, "main")  # MEM-GAP-011: FK now requires a real session row
    job_id = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    db.claim_next_capture_job()
    new_proposals = db.complete_capture_job(job_id, ["User prefers dark mode."])

    proposal = db.get_proposal(new_proposals[0]["id"])

    assert proposal["claim_text"] == "User prefers dark mode."
    assert proposal["status"] == "pending"
    assert proposal["job_id"] == job_id


def test_get_proposal_returns_none_for_an_unknown_id(db):
    assert db.get_proposal(-1) is None


def test_set_proposal_status_updates_status_and_reason(db, session_id):
    db.upsert_session(session_id, "main")  # MEM-GAP-011: FK now requires a real session row
    job_id = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    db.claim_next_capture_job()
    new_proposals = db.complete_capture_job(job_id, ["User prefers dark mode."])

    db.set_proposal_status(new_proposals[0]["id"], "rejected", reason="Not useful.")

    proposal = db.get_proposal(new_proposals[0]["id"])
    assert proposal["status"] == "rejected"
    assert proposal["rejected_reason"] == "Not useful."


def test_set_proposal_status_clears_a_stale_reason_on_a_later_transition(db, session_id):
    db.upsert_session(session_id, "main")  # MEM-GAP-011: FK now requires a real session row
    job_id = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    db.claim_next_capture_job()
    new_proposals = db.complete_capture_job(job_id, ["User prefers dark mode."])
    db.set_proposal_status(new_proposals[0]["id"], "rejected", reason="Not useful.")

    db.set_proposal_status(new_proposals[0]["id"], "pending")  # e.g. a rollback

    proposal = db.get_proposal(new_proposals[0]["id"])
    assert proposal["status"] == "pending"
    assert proposal["rejected_reason"] == ""


def test_new_proposal_defaults_to_pending_status(db, session_id):
    # Stage One Phase 5, slice B: the status column (used by Phase 5 slice
    # D's consolidation review) defaults to "pending" for every existing
    # and newly-created proposal.
    db.upsert_session(session_id, "main")  # MEM-GAP-011: FK now requires a real session row
    job_id = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    db.claim_next_capture_job()
    db.complete_capture_job(job_id, ["User prefers dark mode."])

    status_row = db._conn().execute(
        "SELECT status FROM memory_proposals WHERE job_id = %s", (job_id,)
    ).fetchone()
    assert status_row[0] == "pending"


def test_fail_capture_job_reschedules_with_backoff(db, session_id):
    db.upsert_session(session_id, "main")  # MEM-GAP-011: FK now requires a real session row
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
    db.upsert_session(session_id, "main")  # MEM-GAP-011: FK now requires a real session row
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
    db.upsert_session(session_id, "main")  # MEM-GAP-011: FK now requires a real session row
    job_id = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    db.claim_next_capture_job()
    db.fail_capture_job(job_id, "boom", backoff_seconds=-1.0, max_attempts=5)  # already due

    job = db.claim_next_capture_job()

    assert job is not None
    assert job["id"] == job_id
    assert job["attempts"] == 1


# ---------------------------------------------------------------------------
# Backfill primitives (Stage One Phase 5, slice D)
# ---------------------------------------------------------------------------

def test_list_message_ids_returns_every_id_ascending(db, session_id):
    db.upsert_session(session_id, "main")
    id1 = db.add_message(session_id, "user", "hi")
    id2 = db.add_message(session_id, "assistant", "hello")

    assert db.list_message_ids(session_id) == sorted([id1, id2])


def test_list_message_ids_is_empty_for_a_session_with_no_messages(db, session_id):
    assert db.list_message_ids(session_id) == []


def test_list_capture_job_ranges_returns_every_enqueued_range(db, session_id):
    db.upsert_session(session_id, "main")
    db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-a-{session_id}")
    db.enqueue_capture_job("main", session_id, 5, 6, idempotency_key=f"key-b-{session_id}")

    ranges = db.list_capture_job_ranges(session_id)

    assert sorted(ranges) == [(1, 2), (5, 6)]


def test_list_capture_job_ranges_includes_failed_jobs(db, session_id):
    db.upsert_session(session_id, "main")  # MEM-GAP-011: FK now requires a real session row
    job_id = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    db.claim_next_capture_job()
    db.fail_capture_job(job_id, "boom", backoff_seconds=999.0, max_attempts=1)  # marks 'failed'

    assert db.list_capture_job_ranges(session_id) == [(1, 2)]


def test_list_capture_job_ranges_is_empty_for_a_session_with_no_jobs(db, session_id):
    assert db.list_capture_job_ranges(session_id) == []


def test_list_session_ids_for_agent_finds_an_upserted_session(db, session_id):
    db.upsert_session(session_id, "main")

    assert session_id in db.list_session_ids_for_agent("main")


def test_list_session_ids_for_agent_excludes_a_different_agents_session(db, session_id):
    db.upsert_session(session_id, "researcher")

    assert session_id not in db.list_session_ids_for_agent("main")


# ---------------------------------------------------------------------------
# Durable commitment-extraction jobs (Stage One Phase 6, slice B)
# ---------------------------------------------------------------------------

def _candidate(**overrides) -> dict:
    base = {
        "kind": "open_loop", "sensitivity": "routine", "source": "inferred_user_context",
        "reason": "User mentioned an interview.", "suggested_text": "How did the interview go?",
        "dedupe_key": "interview:2026-08-01", "confidence": 0.8,
        "due_earliest": 2000000000.0, "due_latest": 2000010000.0,
    }
    base.update(overrides)
    return base


def test_enqueue_commitment_job_returns_job_id(db, session_id):
    db.upsert_session(session_id, "main")
    job_id = db.enqueue_commitment_job(
        "main", session_id, "cli", 1, 2, idempotency_key=f"commit-key-{session_id}"
    )
    assert job_id is not None


def test_enqueue_commitment_job_idempotent(db, session_id):
    db.upsert_session(session_id, "main")
    key = f"commit-key-{session_id}"
    first = db.enqueue_commitment_job("main", session_id, "cli", 1, 2, idempotency_key=key)
    second = db.enqueue_commitment_job("main", session_id, "cli", 1, 2, idempotency_key=key)

    assert first is not None
    assert second is None


def test_claim_next_commitment_job_returns_a_pending_job(db, session_id):
    db.upsert_session(session_id, "main")
    db.enqueue_commitment_job(
        "main", session_id, "!room:example.org", 1, 2, idempotency_key=f"commit-key-{session_id}"
    )

    job = db.claim_next_commitment_job()

    assert job is not None
    assert job["agent_id"] == "main"
    assert job["channel"] == "!room:example.org"
    assert job["source_from_message_id"] == 1
    assert job["source_to_message_id"] == 2
    assert job["attempts"] == 0


def test_claim_next_commitment_job_returns_none_when_queue_empty(db):
    assert db.claim_next_commitment_job() is None


def test_claim_next_commitment_job_does_not_reclaim_a_running_job(db, session_id):
    db.upsert_session(session_id, "main")
    db.enqueue_commitment_job(
        "main", session_id, "cli", 1, 2, idempotency_key=f"commit-key-{session_id}"
    )
    first_claim = db.claim_next_commitment_job()

    second_claim = db.claim_next_commitment_job()

    assert first_claim is not None
    assert second_claim is None


def test_complete_commitment_job_inserts_a_new_commitment(db, session_id):
    db.upsert_session(session_id, "main")
    job_id = db.enqueue_commitment_job(
        "main", session_id, "cli", 1, 2, idempotency_key=f"commit-key-{session_id}"
    )
    db.claim_next_commitment_job()

    results = db.complete_commitment_job(job_id, [_candidate()])

    assert len(results) == 1
    assert results[0]["created"] is True
    row = db._conn().execute(
        "SELECT kind, sensitivity, source, reason, suggested_text, dedupe_key, "
        "confidence, due_earliest, due_latest, status, source_job_id "
        "FROM commitments WHERE id = %s",
        (results[0]["id"],),
    ).fetchone()
    assert row[0] == "open_loop"
    assert row[1] == "routine"
    assert row[2] == "inferred_user_context"
    assert row[3] == "User mentioned an interview."
    assert row[4] == "How did the interview go?"
    assert row[5] == "interview:2026-08-01"
    assert row[6] == 0.8
    assert row[9] == "pending"
    assert row[10] == job_id


def test_complete_commitment_job_marks_the_job_done(db, session_id):
    db.upsert_session(session_id, "main")
    job_id = db.enqueue_commitment_job(
        "main", session_id, "cli", 1, 2, idempotency_key=f"commit-key-{session_id}"
    )
    db.claim_next_commitment_job()

    db.complete_commitment_job(job_id, [])

    state = db._conn().execute(
        "SELECT state FROM memory_commitment_jobs WHERE id = %s", (job_id,)
    ).fetchone()[0]
    assert state == "done"


def test_complete_commitment_job_with_no_candidates_returns_empty_list(db, session_id):
    db.upsert_session(session_id, "main")
    job_id = db.enqueue_commitment_job(
        "main", session_id, "cli", 1, 2, idempotency_key=f"commit-key-{session_id}"
    )
    db.claim_next_commitment_job()

    assert db.complete_commitment_job(job_id, []) == []


def test_complete_commitment_job_upserts_a_matching_dedupe_key_instead_of_duplicating(
    db, session_id
):
    db.upsert_session(session_id, "main")
    job_id = db.enqueue_commitment_job(
        "main", session_id, "cli", 1, 2, idempotency_key=f"commit-key-{session_id}"
    )
    db.claim_next_commitment_job()
    db.complete_commitment_job(job_id, [_candidate(due_earliest=2000000000.0, due_latest=2000001000.0)])

    job_id_2 = db.enqueue_commitment_job(
        "main", session_id, "cli", 3, 4, idempotency_key=f"commit-key-2-{session_id}"
    )
    db.claim_next_commitment_job()
    results = db.complete_commitment_job(
        job_id_2, [_candidate(due_earliest=1999999000.0, due_latest=2000010000.0, confidence=0.95)]
    )

    assert results[0]["created"] is False  # merged, not a new row
    count = db._conn().execute(
        "SELECT count(*) FROM commitments WHERE agent_id = %s AND channel = 'cli' "
        "AND dedupe_key = 'interview:2026-08-01'",
        ("main",),
    ).fetchone()[0]
    assert count == 1
    row = db._conn().execute(
        "SELECT due_earliest, due_latest, confidence FROM commitments WHERE id = %s",
        (results[0]["id"],),
    ).fetchone()
    assert row[0] == 1999999000.0  # widened to the earlier of the two
    assert row[1] == 2000010000.0  # widened to the later of the two
    assert row[2] == 0.95  # kept the higher confidence


def test_complete_commitment_job_does_not_upsert_against_a_non_pending_commitment(db, session_id):
    db.upsert_session(session_id, "main")
    job_id = db.enqueue_commitment_job(
        "main", session_id, "cli", 1, 2, idempotency_key=f"commit-key-{session_id}"
    )
    db.claim_next_commitment_job()
    first = db.complete_commitment_job(job_id, [_candidate()])
    db._conn().execute(
        "UPDATE commitments SET status = 'sent' WHERE id = %s", (first[0]["id"],)
    )

    job_id_2 = db.enqueue_commitment_job(
        "main", session_id, "cli", 3, 4, idempotency_key=f"commit-key-2-{session_id}"
    )
    db.claim_next_commitment_job()
    results = db.complete_commitment_job(job_id_2, [_candidate()])

    assert results[0]["created"] is True  # a fresh row, since the old one is no longer pending


def test_complete_commitment_job_scopes_dedupe_to_the_same_channel(db, session_id):
    db.upsert_session(session_id, "main")
    job_id = db.enqueue_commitment_job(
        "main", session_id, "room-a", 1, 2, idempotency_key=f"commit-key-a-{session_id}"
    )
    db.claim_next_commitment_job()
    db.complete_commitment_job(job_id, [_candidate()])

    job_id_2 = db.enqueue_commitment_job(
        "main", session_id, "room-b", 3, 4, idempotency_key=f"commit-key-b-{session_id}"
    )
    db.claim_next_commitment_job()
    results = db.complete_commitment_job(job_id_2, [_candidate()])

    assert results[0]["created"] is True  # different channel, not the same commitment


def test_fail_commitment_job_reschedules_with_backoff(db, session_id):
    db.upsert_session(session_id, "main")
    job_id = db.enqueue_commitment_job(
        "main", session_id, "cli", 1, 2, idempotency_key=f"commit-key-{session_id}"
    )
    db.claim_next_commitment_job()

    db.fail_commitment_job(job_id, "boom", backoff_seconds=100.0, max_attempts=5)

    row = db._conn().execute(
        "SELECT state, attempts, run_after, last_error FROM memory_commitment_jobs WHERE id = %s",
        (job_id,),
    ).fetchone()
    assert row[0] == "pending"
    assert row[1] == 1
    assert row[2] > time.time()
    assert row[3] == "boom"


def test_fail_commitment_job_gives_up_after_max_attempts(db, session_id):
    db.upsert_session(session_id, "main")
    job_id = db.enqueue_commitment_job(
        "main", session_id, "cli", 1, 2, idempotency_key=f"commit-key-{session_id}"
    )
    for _ in range(3):
        db.claim_next_commitment_job()
        db.fail_commitment_job(job_id, "boom", backoff_seconds=-1.0, max_attempts=3)

    state = db._conn().execute(
        "SELECT state FROM memory_commitment_jobs WHERE id = %s", (job_id,)
    ).fetchone()[0]
    assert state == "failed"


def test_list_pending_commitments_for_scope_returns_pending_only(db, session_id):
    db.upsert_session(session_id, "main")
    job_id = db.enqueue_commitment_job(
        "main", session_id, "cli", 1, 2, idempotency_key=f"commit-key-{session_id}"
    )
    db.claim_next_commitment_job()
    results = db.complete_commitment_job(
        job_id, [_candidate(dedupe_key="a"), _candidate(dedupe_key="b")]
    )
    db._conn().execute(
        "UPDATE commitments SET status = 'sent' WHERE id = %s", (results[0]["id"],)
    )

    pending = db.list_pending_commitments_for_scope("main", "cli")

    assert [p["dedupe_key"] for p in pending] == ["b"]


def test_list_pending_commitments_for_scope_scoped_to_channel(db, session_id):
    db.upsert_session(session_id, "main")
    job_id = db.enqueue_commitment_job(
        "main", session_id, "room-a", 1, 2, idempotency_key=f"commit-key-{session_id}"
    )
    db.claim_next_commitment_job()
    db.complete_commitment_job(job_id, [_candidate()])

    assert db.list_pending_commitments_for_scope("main", "room-b") == []


def test_list_pending_commitments_for_scope_respects_limit(db, session_id):
    db.upsert_session(session_id, "main")
    job_id = db.enqueue_commitment_job(
        "main", session_id, "cli", 1, 2, idempotency_key=f"commit-key-{session_id}"
    )
    db.claim_next_commitment_job()
    db.complete_commitment_job(
        job_id, [_candidate(dedupe_key=f"key-{i}") for i in range(5)]
    )

    pending = db.list_pending_commitments_for_scope("main", "cli", limit=2)

    assert len(pending) == 2


# ---------------------------------------------------------------------------
# Commitment lifecycle (Stage One Phase 6, slice C)
# ---------------------------------------------------------------------------

def _create_commitment(db, session_id, agent_id="main", channel="cli", **overrides) -> dict:
    """Create a real commitment row via the enqueue/claim/complete flow and return it in full."""
    db.upsert_session(session_id, agent_id)  # MEM-GAP-011: FK now requires a real session row
    job_id = db.enqueue_commitment_job(
        agent_id, session_id, channel, 1, 2,
        idempotency_key=f"commit-key-{session_id}-{overrides.get('dedupe_key', 'x')}",
    )
    db.claim_next_commitment_job()
    results = db.complete_commitment_job(job_id, [_candidate(**overrides)])
    return db.get_commitment(results[0]["id"])


def test_get_commitment_returns_the_full_row(db, session_id):
    created = _create_commitment(db, session_id)

    fetched = db.get_commitment(created["id"])

    assert fetched["status"] == "pending"
    assert fetched["kind"] == "open_loop"
    assert fetched["channel"] == "cli"


def test_get_commitment_returns_none_for_an_unknown_id(db):
    assert db.get_commitment(-1) is None


def test_mark_commitment_sent_updates_status_and_sent_at(db, session_id):
    created = _create_commitment(db, session_id)

    db.mark_commitment_sent(created["id"])

    updated = db.get_commitment(created["id"])
    assert updated["status"] == "sent"
    sent_at = db._conn().execute(
        "SELECT sent_at FROM commitments WHERE id = %s", (created["id"],)
    ).fetchone()[0]
    assert sent_at is not None


def test_mark_commitment_dismissed_updates_status_and_dismissed_at(db, session_id):
    created = _create_commitment(db, session_id)

    db.mark_commitment_dismissed(created["id"])

    updated = db.get_commitment(created["id"])
    assert updated["status"] == "dismissed"
    dismissed_at = db._conn().execute(
        "SELECT dismissed_at FROM commitments WHERE id = %s", (created["id"],)
    ).fetchone()[0]
    assert dismissed_at is not None


def test_expire_stale_commitments_expires_ones_past_the_grace_period(db, session_id):
    long_ago = time.time() - 100 * 3600.0  # well past the 72h grace period
    created = _create_commitment(
        db, session_id, due_earliest=long_ago, due_latest=long_ago + 60
    )

    expired_count = db.expire_stale_commitments("main", time.time())

    assert expired_count >= 1
    assert db.get_commitment(created["id"])["status"] == "expired"


def test_expire_stale_commitments_leaves_recent_pending_commitments_alone(db, session_id):
    created = _create_commitment(
        db, session_id, due_earliest=time.time() + 10, due_latest=time.time() + 20
    )

    db.expire_stale_commitments("main", time.time())

    assert db.get_commitment(created["id"])["status"] == "pending"


def test_expire_stale_commitments_never_touches_an_already_sent_commitment(db, session_id):
    long_ago = time.time() - 100 * 3600.0
    created = _create_commitment(
        db, session_id, due_earliest=long_ago, due_latest=long_ago + 60
    )
    db.mark_commitment_sent(created["id"])

    db.expire_stale_commitments("main", time.time())

    assert db.get_commitment(created["id"])["status"] == "sent"


def test_list_due_commitments_for_agent_returns_a_due_commitment(db, session_id):
    now = time.time()
    created = _create_commitment(
        db, session_id, due_earliest=now - 10, due_latest=now + 3600
    )

    due = db.list_due_commitments_for_agent("main", now)

    assert [c["id"] for c in due] == [created["id"]]


def test_list_due_commitments_for_agent_excludes_a_not_yet_due_commitment(db, session_id):
    now = time.time()
    _create_commitment(db, session_id, due_earliest=now + 3600, due_latest=now + 7200)

    due = db.list_due_commitments_for_agent("main", now)

    assert due == []


def test_list_due_commitments_for_agent_spans_every_channel(db, session_id):
    now = time.time()
    a = _create_commitment(
        db, session_id, channel="room-a", due_earliest=now - 10, due_latest=now + 3600,
        dedupe_key="a",
    )
    b = _create_commitment(
        db, session_id, channel="room-b", due_earliest=now - 10, due_latest=now + 3600,
        dedupe_key="b",
    )

    due = db.list_due_commitments_for_agent("main", now)

    assert {c["id"] for c in due} == {a["id"], b["id"]}


def test_list_due_commitments_for_agent_respects_max_per_heartbeat(db, session_id):
    now = time.time()
    for i in range(5):
        _create_commitment(
            db, session_id, due_earliest=now - 10, due_latest=now + 3600, dedupe_key=f"k{i}",
        )

    due = db.list_due_commitments_for_agent("main", now, max_per_heartbeat=2)

    assert len(due) == 2


def test_list_due_commitments_for_agent_respects_max_per_day(db, session_id):
    now = time.time()
    created = [
        _create_commitment(
            db, session_id, due_earliest=now - 10, due_latest=now + 3600, dedupe_key=f"k{i}",
        )
        for i in range(3)
    ]
    for c in created[:2]:
        db.mark_commitment_sent(c["id"], now=now - 60)  # 2 already sent today

    due = db.list_due_commitments_for_agent("main", now, max_per_day=2, max_per_heartbeat=10)

    assert due == []  # today's quota is already used up


def test_list_due_commitments_for_agent_orders_earliest_due_first(db, session_id):
    now = time.time()
    later = _create_commitment(
        db, session_id, due_earliest=now - 5, due_latest=now + 3600, dedupe_key="later",
    )
    earlier = _create_commitment(
        db, session_id, due_earliest=now - 50, due_latest=now + 3600, dedupe_key="earlier",
    )

    due = db.list_due_commitments_for_agent("main", now)

    assert [c["id"] for c in due] == [earlier["id"], later["id"]]


def test_list_due_commitments_for_agent_scoped_to_the_given_agent(db, session_id):
    now = time.time()
    _create_commitment(
        db, session_id, agent_id="researcher", due_earliest=now - 10, due_latest=now + 3600,
    )

    due = db.list_due_commitments_for_agent("main", now)

    assert due == []


def test_list_commitments_returns_every_status_by_default(db, session_id):
    created = _create_commitment(db, session_id)
    db.mark_commitment_sent(created["id"])

    results = db.list_commitments("main")

    assert created["id"] in [c["id"] for c in results]


def test_list_commitments_filters_by_status(db, session_id):
    pending_one = _create_commitment(db, session_id, dedupe_key="p")
    sent_one = _create_commitment(db, session_id, dedupe_key="s")
    db.mark_commitment_sent(sent_one["id"])

    results = db.list_commitments("main", status="pending")

    result_ids = [c["id"] for c in results]
    assert pending_one["id"] in result_ids
    assert sent_one["id"] not in result_ids


def test_list_commitments_filters_by_channel(db, session_id):
    a = _create_commitment(db, session_id, channel="room-a", dedupe_key="a")
    _create_commitment(db, session_id, channel="room-b", dedupe_key="b")

    results = db.list_commitments("main", channel="room-a")

    assert [c["id"] for c in results] == [a["id"]]


def test_delete_commitment_removes_the_row(db, session_id):
    created = _create_commitment(db, session_id)

    deleted = db.delete_commitment("main", created["id"])

    assert deleted is True
    assert db.get_commitment(created["id"]) is None


def test_delete_commitment_returns_false_for_an_unknown_id(db):
    assert db.delete_commitment("main", -1) is False


def test_delete_commitment_is_scoped_to_the_given_agent(db, session_id):
    created = _create_commitment(db, session_id, agent_id="main")

    deleted = db.delete_commitment("researcher", created["id"])

    assert deleted is False
    assert db.get_commitment(created["id"]) is not None


# ---------------------------------------------------------------------------
# Agent isolation for session-search reads (MEM-GAP-002)
#
# SessionSearchTool lets an agent search/browse/scroll its own past
# conversations. Before this fix, none of the methods below took an
# agent_id, so any agent (or a guessed session_id) could read any other
# agent's session history. Every test here plants a "main" session and
# proves a "researcher"-scoped call can't see it — including via an id it
# already knows, not just via listing/searching.
# ---------------------------------------------------------------------------

def test_search_messages_excludes_another_agents_session(db, session_id):
    db.upsert_session(session_id, "main")
    db.mirror_message(session_id, "e1", "user", "the secret password is hunter2")

    results = db.search_messages("hunter2", "researcher")

    assert results == []


def test_search_messages_finds_the_owning_agents_own_session(db, session_id):
    db.upsert_session(session_id, "main")
    db.mirror_message(session_id, "e1", "user", "the secret password is hunter2")

    results = db.search_messages("hunter2", "main")

    assert [r["session_id"] for r in results] == [session_id]


def test_search_vector_uses_the_language_neutral_simple_config(db):
    # R2-GAP-012: messages.search_vector must be generated with PostgreSQL's
    # 'simple' text-search config (migration 008), not 'english' — the
    # actual schema-level proof the migration took effect, independent of
    # any single word's stemming/stopword behavior under either config.
    row = db._conn().execute(
        "SELECT pg_get_expr(ad.adbin, ad.adrelid) FROM pg_attrdef ad "
        "JOIN pg_attribute a ON a.attrelid = ad.adrelid AND a.attnum = ad.adnum "
        "WHERE a.attrelid = 'messages'::regclass AND a.attname = 'search_vector'"
    ).fetchone()
    assert row is not None
    definition = row[0]
    assert "'simple'" in definition
    assert "'english'" not in definition


def test_list_sessions_excludes_another_agents_session(db, session_id):
    # This is a shared dev database, so "researcher" may already have real
    # sessions of its own — the assertion is membership, not an empty list.
    db.upsert_session(session_id, "main")

    ids = {s["id"] for s in db.list_sessions("researcher", limit=1000)}

    assert session_id not in ids


def test_list_sessions_includes_the_owning_agents_own_session(db, session_id):
    db.upsert_session(session_id, "main")

    ids = {s["id"] for s in db.list_sessions("main", limit=1000)}

    assert session_id in ids


def test_get_sessions_by_ids_excludes_another_agents_session(db, session_id):
    db.upsert_session(session_id, "main")

    assert db.get_sessions_by_ids([session_id], "researcher") == {}


def test_get_sessions_by_ids_includes_the_owning_agents_own_session(db, session_id):
    db.upsert_session(session_id, "main")

    result = db.get_sessions_by_ids([session_id], "main")

    assert session_id in result


def test_get_messages_around_denies_a_guessed_session_id_from_another_agent(db, session_id):
    db.upsert_session(session_id, "main")
    db.mirror_message(session_id, "e1", "user", "hello")

    # "researcher" already knows the exact session_id (e.g. from a prior
    # DISCOVER result shown to a different agent) — ownership must still
    # be checked, not just discoverability.
    assert db.get_messages_around(session_id, "researcher", 0) == []


def test_get_messages_around_returns_messages_for_the_owning_agent(db, session_id):
    db.upsert_session(session_id, "main")
    db.mirror_message(session_id, "e1", "user", "hello")

    result = db.get_messages_around(session_id, "main", 0)

    assert [m["content"] for m in result] == ["hello"]


def test_get_session_bookends_denies_a_guessed_session_id_from_another_agent(db, session_id):
    db.upsert_session(session_id, "main")
    db.mirror_message(session_id, "e1", "user", "hello")

    first, last = db.get_session_bookends(session_id, "researcher")

    assert first == []
    assert last == []


def test_get_session_bookends_returns_messages_for_the_owning_agent(db, session_id):
    db.upsert_session(session_id, "main")
    db.mirror_message(session_id, "e1", "user", "hello")

    first, last = db.get_session_bookends(session_id, "main")

    assert len(first) == 1


# ---------------------------------------------------------------------------
# delete_session (MEM-GAP-003)
# ---------------------------------------------------------------------------

def test_delete_session_returns_none_for_a_session_owned_by_another_agent(db, session_id):
    db.upsert_session(session_id, "researcher")

    assert db.delete_session("main", session_id) is None
    # Untouched — still owned by researcher and still readable.
    assert db._session_owned_by(session_id, "researcher") is True


def test_delete_session_returns_none_for_an_unknown_session_id(db):
    assert db.delete_session("main", "does-not-exist") is None


def test_delete_session_removes_the_sessions_row(db, session_id):
    db.upsert_session(session_id, "main")

    db.delete_session("main", session_id)

    assert db._session_owned_by(session_id, "main") is False


def test_delete_session_removes_messages_and_returns_the_count(db, session_id):
    db.upsert_session(session_id, "main")
    db.mirror_message(session_id, "e1", "user", "hello")
    db.mirror_message(session_id, "e2", "assistant", "hi there")

    result = db.delete_session("main", session_id)

    assert result["messages"] == 2
    assert _message_count(db, session_id) == 0


def test_delete_session_removes_message_mirrors(db, session_id):
    db.upsert_session(session_id, "main")
    db.mirror_message(session_id, "e1", "user", "hello")

    db.delete_session("main", session_id)

    assert db.is_mirrored(session_id, "e1") is False


def test_delete_session_removes_capture_jobs(db, session_id):
    db.upsert_session(session_id, "main")
    db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")

    db.delete_session("main", session_id)

    row = db._conn().execute(
        "SELECT count(*) FROM memory_capture_jobs WHERE session_id = %s", (session_id,)
    ).fetchone()
    assert row[0] == 0


def test_delete_session_removes_commitment_jobs(db, session_id):
    db.upsert_session(session_id, "main")
    db.enqueue_commitment_job(
        "main", session_id, "cli", 1, 2, idempotency_key=f"commit-key-{session_id}"
    )

    db.delete_session("main", session_id)

    row = db._conn().execute(
        "SELECT count(*) FROM memory_commitment_jobs WHERE session_id = %s", (session_id,)
    ).fetchone()
    assert row[0] == 0


def test_delete_session_removes_message_embedding_jobs(db, session_id):
    db.upsert_session(session_id, "main")
    message_id = db.add_message(session_id, "user", "hello")
    db.enqueue_message_embedding_job("main", session_id, message_id, idempotency_key=f"emb-{session_id}")

    db.delete_session("main", session_id)

    row = db._conn().execute(
        "SELECT count(*) FROM message_embedding_jobs WHERE session_id = %s", (session_id,)
    ).fetchone()
    assert row[0] == 0


def test_delete_session_removes_commitments(db, session_id):
    db.upsert_session(session_id, "main")
    _create_commitment(db, session_id)

    db.delete_session("main", session_id)

    row = db._conn().execute(
        "SELECT count(*) FROM commitments WHERE session_id = %s", (session_id,)
    ).fetchone()
    assert row[0] == 0


def test_delete_session_removes_proposals_and_returns_their_ids(db, session_id):
    db.upsert_session(session_id, "main")
    db.upsert_session(session_id, "main")  # MEM-GAP-011: FK now requires a real session row
    job_id = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    db.claim_next_capture_job()
    new_proposals = db.complete_capture_job(
        job_id, ["User prefers dark mode.", "User's dog is named Biscuit."]
    )
    proposal_ids = [p["id"] for p in new_proposals]

    result = db.delete_session("main", session_id)

    assert sorted(result["proposal_ids"]) == sorted(proposal_ids)
    row = db._conn().execute(
        "SELECT count(*) FROM memory_proposals WHERE id = ANY(%s)", (proposal_ids,)
    ).fetchone()
    assert row[0] == 0


def test_delete_session_with_no_proposals_returns_empty_proposal_ids(db, session_id):
    db.upsert_session(session_id, "main")
    db.mirror_message(session_id, "e1", "user", "hello")

    result = db.delete_session("main", session_id)

    assert result["proposal_ids"] == []


def test_delete_session_does_not_touch_a_different_sessions_data(db, session_id):
    other_session_id = f"{session_id}-other"
    db.upsert_session(session_id, "main")
    db.upsert_session(other_session_id, "main")
    db.mirror_message(other_session_id, "e1", "user", "should survive")

    db.delete_session("main", session_id)

    assert db._session_owned_by(other_session_id, "main") is True
    assert _message_count(db, other_session_id) == 1

    # Manual cleanup since other_session_id isn't the fixture's own session_id
    # (the autouse _cleanup_after fixture only scopes to the `session_id` fixture value).
    db._conn().execute("DELETE FROM messages WHERE session_id = %s", (other_session_id,))
    db._conn().execute("DELETE FROM sessions WHERE id = %s", (other_session_id,))


def test_delete_session_is_a_harmless_no_op_the_second_time(db, session_id):
    db.upsert_session(session_id, "main")
    db.mirror_message(session_id, "e1", "user", "hello")

    first = db.delete_session("main", session_id)
    second = db.delete_session("main", session_id)

    assert first is not None
    assert second is None


# ---------------------------------------------------------------------------
# Deletion tombstones (R2-GAP-007)
# ---------------------------------------------------------------------------

def test_start_deletion_tombstone_creates_an_all_pending_row(db, session_id):
    tombstone = db.start_deletion_tombstone("main", session_id)

    assert tombstone == {
        "jsonl_deleted": False, "db_deleted": False, "evidence_cleaned": False,
        "proposal_ids": [], "completed_at": None,
    }


def test_start_deletion_tombstone_is_idempotent_and_preserves_progress(db, session_id):
    db.start_deletion_tombstone("main", session_id)
    db.mark_deletion_jsonl_done("main", session_id)

    tombstone = db.start_deletion_tombstone("main", session_id)  # called again

    assert tombstone["jsonl_deleted"] is True  # not reset


def test_get_deletion_tombstone_returns_none_when_never_started(db, session_id):
    assert db.get_deletion_tombstone("main", session_id) is None


def test_mark_deletion_jsonl_done_updates_only_that_flag(db, session_id):
    db.start_deletion_tombstone("main", session_id)
    db.mark_deletion_jsonl_done("main", session_id)

    tombstone = db.get_deletion_tombstone("main", session_id)

    assert tombstone["jsonl_deleted"] is True
    assert tombstone["db_deleted"] is False
    assert tombstone["evidence_cleaned"] is False


def test_mark_deletion_db_done_records_proposal_ids(db, session_id):
    db.start_deletion_tombstone("main", session_id)

    db.mark_deletion_db_done("main", session_id, [3, 7, 9])

    tombstone = db.get_deletion_tombstone("main", session_id)
    assert tombstone["db_deleted"] is True
    assert tombstone["proposal_ids"] == [3, 7, 9]


def test_mark_deletion_evidence_done_sets_completed_at(db, session_id):
    db.start_deletion_tombstone("main", session_id)
    db.mark_deletion_jsonl_done("main", session_id)
    db.mark_deletion_db_done("main", session_id, [])

    db.mark_deletion_evidence_done("main", session_id)

    tombstone = db.get_deletion_tombstone("main", session_id)
    assert tombstone["evidence_cleaned"] is True
    assert tombstone["completed_at"] is not None


def test_list_incomplete_deletion_tombstones_excludes_completed_ones(db, session_id):
    other_session_id = f"{session_id}-other"
    db.start_deletion_tombstone("main", session_id)
    db.mark_deletion_jsonl_done("main", session_id)
    db.mark_deletion_db_done("main", session_id, [])
    db.mark_deletion_evidence_done("main", session_id)  # fully complete

    db.start_deletion_tombstone("main", other_session_id)  # still incomplete
    db.mark_deletion_jsonl_done("main", other_session_id)

    incomplete = db.list_incomplete_deletion_tombstones()

    ids = {(t["agent_id"], t["session_id"]) for t in incomplete}
    assert ("main", session_id) not in ids
    assert ("main", other_session_id) in ids
    matching = next(t for t in incomplete if t["session_id"] == other_session_id)
    assert matching["jsonl_deleted"] is True
    assert matching["db_deleted"] is False


def test_list_incomplete_deletion_tombstones_empty_when_nothing_started(db):
    # No tombstone at all for a brand new, never-attempted session_id.
    result = [
        t for t in db.list_incomplete_deletion_tombstones()
        if t["session_id"] == "no-such-session-id"
    ]
    assert result == []


# ---------------------------------------------------------------------------
# queue_lag_summary (MEM-GAP-016)
#
# Uses a fresh, unique agent_id per test (not "main") since this method
# aggregates across every session for an agent, and "main" may carry real
# pending jobs from actual usage of this shared dev database. Cleaned up
# manually since the file's autouse fixture only scopes to the `session_id`
# fixture value, not an arbitrary agent_id.
# ---------------------------------------------------------------------------

@pytest.fixture
def lag_agent_id():
    return f"test-lag-{uuid.uuid4()}"


@pytest.fixture(autouse=True)
def _cleanup_lag_agent(db, lag_agent_id):
    yield
    conn = db._conn()
    conn.execute("DELETE FROM memory_capture_jobs WHERE agent_id = %s", (lag_agent_id,))
    conn.execute("DELETE FROM memory_commitment_jobs WHERE agent_id = %s", (lag_agent_id,))
    conn.execute("DELETE FROM message_embedding_jobs WHERE agent_id = %s", (lag_agent_id,))


def test_queue_lag_summary_is_zero_for_an_agent_with_no_jobs(db, lag_agent_id):
    summary = db.queue_lag_summary(lag_agent_id)

    _empty_lane = {
        "pending_count": 0, "oldest_pending_age_s": None,
        "running_count": 0, "oldest_running_age_s": None,
        "failed_count": 0,
    }
    assert summary == {
        "capture": _empty_lane,
        "commitment": _empty_lane,
        "message_embedding": _empty_lane,
    }


def test_queue_lag_summary_counts_pending_capture_jobs(db, session_id, lag_agent_id):
    db.upsert_session(session_id, lag_agent_id)
    db.enqueue_capture_job(lag_agent_id, session_id, 1, 2, idempotency_key=f"key-{session_id}")

    summary = db.queue_lag_summary(lag_agent_id)

    assert summary["capture"]["pending_count"] == 1
    assert summary["capture"]["oldest_pending_age_s"] >= 0


def test_queue_lag_summary_counts_pending_commitment_jobs(db, session_id, lag_agent_id):
    db.upsert_session(session_id, lag_agent_id)
    db.enqueue_commitment_job(
        lag_agent_id, session_id, "cli", 1, 2, idempotency_key=f"commit-key-{session_id}"
    )

    summary = db.queue_lag_summary(lag_agent_id)

    assert summary["commitment"]["pending_count"] == 1
    assert summary["commitment"]["oldest_pending_age_s"] >= 0


def test_queue_lag_summary_counts_pending_message_embedding_jobs(db, session_id, lag_agent_id):
    db.upsert_session(session_id, lag_agent_id)
    message_id = db.add_message(session_id, "user", "hello")
    db.enqueue_message_embedding_job(
        lag_agent_id, session_id, message_id, idempotency_key=f"emb-{session_id}"
    )

    summary = db.queue_lag_summary(lag_agent_id)

    assert summary["message_embedding"]["pending_count"] == 1
    assert summary["message_embedding"]["oldest_pending_age_s"] >= 0


def test_queue_lag_summary_excludes_claimed_jobs(db, session_id, lag_agent_id):
    db.upsert_session(session_id, lag_agent_id)
    db.enqueue_capture_job(lag_agent_id, session_id, 1, 2, idempotency_key=f"key-{session_id}")
    db.claim_next_capture_job()  # moves state 'pending' -> 'running'

    summary = db.queue_lag_summary(lag_agent_id)

    assert summary["capture"]["pending_count"] == 0


def test_queue_lag_summary_excludes_completed_jobs(db, session_id, lag_agent_id):
    db.upsert_session(session_id, lag_agent_id)
    job_id = db.enqueue_capture_job(
        lag_agent_id, session_id, 1, 2, idempotency_key=f"key-{session_id}"
    )
    db.claim_next_capture_job()
    db.complete_capture_job(job_id, [])

    summary = db.queue_lag_summary(lag_agent_id)

    assert summary["capture"]["pending_count"] == 0


def test_queue_lag_summary_is_scoped_to_the_given_agent(db, session_id, lag_agent_id):
    db.upsert_session(session_id, lag_agent_id)
    db.enqueue_capture_job(lag_agent_id, session_id, 1, 2, idempotency_key=f"key-{session_id}")

    summary = db.queue_lag_summary(f"other-{lag_agent_id}")

    assert summary["capture"]["pending_count"] == 0


def test_queue_lag_summary_reports_the_oldest_pending_jobs_age(db, session_id, lag_agent_id):
    db.upsert_session(session_id, lag_agent_id)
    old_job_id = db.enqueue_capture_job(
        lag_agent_id, session_id, 1, 2, idempotency_key=f"old-key-{session_id}"
    )
    db._conn().execute(
        "UPDATE memory_capture_jobs SET created_at = %s WHERE id = %s",
        (time.time() - 3600.0, old_job_id),
    )
    db.enqueue_capture_job(lag_agent_id, session_id, 3, 4, idempotency_key=f"new-key-{session_id}")

    summary = db.queue_lag_summary(lag_agent_id)

    assert summary["capture"]["pending_count"] == 2
    assert summary["capture"]["oldest_pending_age_s"] >= 3600.0


# ---------------------------------------------------------------------------
# prune_operational_tables (MEM-GAP-015)
# ---------------------------------------------------------------------------

def _old_done_capture_job(db, session_id, agent_id, days_old: float) -> int:
    """Create a capture job, complete it, then backdate updated_at."""
    job_id = db.enqueue_capture_job(
        agent_id, session_id, 1, 2, idempotency_key=f"key-{session_id}-{days_old}"
    )
    db.claim_next_capture_job()
    db.complete_capture_job(job_id, [])
    db._conn().execute(
        "UPDATE memory_capture_jobs SET updated_at = %s WHERE id = %s",
        (time.time() - days_old * 86400, job_id),
    )
    return job_id


def test_prune_operational_tables_deletes_old_done_capture_jobs(db, session_id, lag_agent_id):
    db.upsert_session(session_id, lag_agent_id)
    _old_done_capture_job(db, session_id, lag_agent_id, days_old=40)

    counts = db.prune_operational_tables(30)

    assert counts["capture_jobs"] == 1


def test_prune_operational_tables_dry_run_counts_without_deleting(db, session_id, lag_agent_id):
    db.upsert_session(session_id, lag_agent_id)
    job_id = _old_done_capture_job(db, session_id, lag_agent_id, days_old=40)

    counts = db.prune_operational_tables(30, dry_run=True)

    assert counts["capture_jobs"] == 1
    row = db._conn().execute(
        "SELECT 1 FROM memory_capture_jobs WHERE id = %s", (job_id,)
    ).fetchone()
    assert row is not None  # still there — dry run must not delete


def test_prune_operational_tables_never_deletes_a_recent_done_job(db, session_id, lag_agent_id):
    db.upsert_session(session_id, lag_agent_id)
    job_id = db.enqueue_capture_job(
        lag_agent_id, session_id, 1, 2, idempotency_key=f"key-{session_id}"
    )
    db.claim_next_capture_job()
    db.complete_capture_job(job_id, [])  # updated_at is "now" — well within the window

    counts = db.prune_operational_tables(30)

    assert counts["capture_jobs"] == 0
    row = db._conn().execute(
        "SELECT 1 FROM memory_capture_jobs WHERE id = %s", (job_id,)
    ).fetchone()
    assert row is not None


def test_prune_operational_tables_never_deletes_a_pending_job_regardless_of_age(
    db, session_id, lag_agent_id
):
    db.upsert_session(session_id, lag_agent_id)
    job_id = db.enqueue_capture_job(
        lag_agent_id, session_id, 1, 2, idempotency_key=f"key-{session_id}"
    )
    db._conn().execute(
        "UPDATE memory_capture_jobs SET updated_at = %s WHERE id = %s",
        (time.time() - 365 * 86400, job_id),
    )

    counts = db.prune_operational_tables(30)

    assert counts["capture_jobs"] == 0
    row = db._conn().execute(
        "SELECT 1 FROM memory_capture_jobs WHERE id = %s", (job_id,)
    ).fetchone()
    assert row is not None  # still pending, untouched no matter how old


def test_prune_operational_tables_deletes_old_failed_commitment_jobs(db, session_id, lag_agent_id):
    db.upsert_session(session_id, lag_agent_id)
    job_id = db.enqueue_commitment_job(
        lag_agent_id, session_id, "cli", 1, 2, idempotency_key=f"commit-{session_id}"
    )
    db.claim_next_commitment_job()
    db.fail_commitment_job(job_id, "boom", 0.0, max_attempts=1)  # -> terminal 'failed'
    db._conn().execute(
        "UPDATE memory_commitment_jobs SET updated_at = %s WHERE id = %s",
        (time.time() - 40 * 86400, job_id),
    )

    counts = db.prune_operational_tables(30)

    assert counts["commitment_jobs"] == 1


def test_prune_operational_tables_deletes_old_failed_message_embedding_jobs(
    db, session_id, lag_agent_id
):
    db.upsert_session(session_id, lag_agent_id)
    message_id = db.add_message(session_id, "user", "hello")
    job_id = db.enqueue_message_embedding_job(
        lag_agent_id, session_id, message_id, idempotency_key=f"emb-{session_id}"
    )
    db.claim_next_message_embedding_job()
    db.fail_message_embedding_job(job_id, "boom", 0.0, max_attempts=1)  # -> terminal 'failed'
    db._conn().execute(
        "UPDATE message_embedding_jobs SET updated_at = %s WHERE id = %s",
        (time.time() - 40 * 86400, job_id),
    )

    counts = db.prune_operational_tables(30)

    assert counts["message_embedding_jobs"] == 1


# ---------------------------------------------------------------------------
# R2-GAP-001: retention must not destroy outputs or erase coverage
# ---------------------------------------------------------------------------
#
# Before this fix, memory_proposals.job_id / commitments.source_job_id were
# ON DELETE CASCADE against the job tables prune_operational_tables() deletes
# rows from — pruning an old job silently deleted its already-materialized
# proposal/commitment too. Separately, find_uncovered_capture_ranges /
# find_uncovered_commitment_ranges / find_uncovered_message_ids_for_embedding
# derived their "already covered" cursor from job-table state, so pruning
# also reset reconciliation's notion of what had already been enqueued.
# These tests prove both are fixed: outputs survive a job being pruned (or
# deleted directly), and reconciliation still sees the range as covered
# afterward.

def test_prune_operational_tables_detaches_but_preserves_a_proposal_from_its_pruned_job(
    db, session_id, lag_agent_id
):
    db.upsert_session(session_id, lag_agent_id)
    job_id = db.enqueue_capture_job(
        lag_agent_id, session_id, 1, 2, idempotency_key=f"key-{session_id}"
    )
    db.claim_next_capture_job()
    proposals = db.complete_capture_job(job_id, ["remember this"])
    proposal_id = proposals[0]["id"]
    db._conn().execute(
        "UPDATE memory_capture_jobs SET updated_at = %s WHERE id = %s",
        (time.time() - 40 * 86400, job_id),
    )

    counts = db.prune_operational_tables(30)

    assert counts["capture_jobs"] == 1
    row = db._conn().execute(
        "SELECT job_id, claim_text FROM memory_proposals WHERE id = %s", (proposal_id,)
    ).fetchone()
    assert row is not None  # the proposal itself must survive
    assert row[0] is None  # but its now-deleted job's back-pointer is detached
    assert row[1] == "remember this"


def test_prune_operational_tables_detaches_but_preserves_a_commitment_from_its_pruned_job(
    db, session_id, lag_agent_id
):
    commitment = _create_commitment(db, session_id, agent_id=lag_agent_id)
    commitment_id = commitment["id"]
    job_id = db._conn().execute(
        "SELECT source_job_id FROM commitments WHERE id = %s", (commitment_id,)
    ).fetchone()[0]
    db._conn().execute(
        "UPDATE memory_commitment_jobs SET updated_at = %s WHERE id = %s",
        (time.time() - 40 * 86400, job_id),
    )

    counts = db.prune_operational_tables(30)

    assert counts["commitment_jobs"] == 1
    row = db._conn().execute(
        "SELECT source_job_id FROM commitments WHERE id = %s", (commitment_id,)
    ).fetchone()
    assert row is not None  # the commitment itself must survive
    assert row[0] is None  # detached from its now-deleted job


def test_prune_operational_tables_never_erases_capture_coverage(db, session_id, lag_agent_id):
    db.upsert_session(session_id, lag_agent_id)
    m1 = db.mirror_message(session_id, "e1", "user", "hello")
    m2 = db.mirror_message(session_id, "e2", "assistant", "hi")
    job_id = db.enqueue_capture_job(
        lag_agent_id, session_id, m1, m2, idempotency_key=f"key-{session_id}"
    )
    db.claim_next_capture_job()
    db.complete_capture_job(job_id, [])
    db._conn().execute(
        "UPDATE memory_capture_jobs SET updated_at = %s WHERE id = %s",
        (time.time() - 40 * 86400, job_id),
    )

    db.prune_operational_tables(30)  # deletes the only capture job for this session

    # Before R2-GAP-001/R2-GAP-002's fix, this would now re-surface
    # messages 1-2 as "uncovered" because the cursor was derived from the
    # very job row just deleted. It must still report nothing to catch up
    # on — coverage lives in session_job_coverage_ranges, never touched by
    # pruning.
    assert db.find_uncovered_capture_ranges(lag_agent_id, session_id) == []


def test_prune_operational_tables_never_erases_message_embedding_coverage(
    db_with_vector, session_id
):
    db_with_vector.upsert_session(session_id, "main")
    message_id = db_with_vector.mirror_message(session_id, "e1", "user", "hello")
    job_id = db_with_vector.enqueue_message_embedding_job(
        "main", session_id, message_id, idempotency_key=f"emb-{session_id}"
    )
    db_with_vector.complete_message_embedding_job(
        job_id, message_id, db_with_vector.embedding_model_identity, [0.1, 0.2, 0.3]
    )
    db_with_vector._conn().execute(
        "UPDATE message_embedding_jobs SET updated_at = %s WHERE id = %s",
        (time.time() - 40 * 86400, job_id),
    )

    db_with_vector.prune_operational_tables(30)  # deletes the (done) embedding job row

    # Coverage is checked directly against message_embeddings, never a job
    # table (R2-GAP-005), so pruning the already-terminal job row can't
    # affect it.
    assert db_with_vector.find_uncovered_message_ids_for_embedding("main", session_id) == []


def test_prune_operational_tables_never_erases_commitment_coverage_or_its_channel(
    db, session_id, lag_agent_id
):
    db.upsert_session(session_id, lag_agent_id)
    m1 = db.mirror_message(session_id, "e1", "user", "hello")
    m2 = db.mirror_message(session_id, "e2", "assistant", "hi")
    job_id = db.enqueue_commitment_job(
        lag_agent_id, session_id, "!room:example.org", m1, m2,
        idempotency_key=f"commit-{session_id}",
    )
    db.claim_next_commitment_job()
    db.fail_commitment_job(job_id, "boom", 0.0, max_attempts=1)  # -> terminal 'failed'
    db._conn().execute(
        "UPDATE memory_commitment_jobs SET updated_at = %s WHERE id = %s",
        (time.time() - 40 * 86400, job_id),
    )

    db.prune_operational_tables(30)  # deletes the only commitment job for this session

    # Both "nothing new to catch up on" AND the remembered channel must
    # survive — without session_job_coverage_ranges.channel, a session
    # whose only commitment job was pruned would look indistinguishable
    # from one that never had commitments enabled, permanently exiting
    # reconciliation.
    assert db.find_uncovered_commitment_ranges(lag_agent_id, session_id) == []
    row = db._conn().execute(
        "SELECT channel FROM session_job_coverage_ranges "
        "WHERE session_id = %s AND lane = 'commitment'",
        (session_id,),
    ).fetchone()
    assert row == ("!room:example.org",)


def test_delete_session_removes_proposals_whose_job_was_already_pruned(
    db, session_id, lag_agent_id
):
    db.upsert_session(session_id, lag_agent_id)
    job_id = db.enqueue_capture_job(
        lag_agent_id, session_id, 1, 2, idempotency_key=f"key-{session_id}"
    )
    db.claim_next_capture_job()
    proposals = db.complete_capture_job(job_id, ["remember this"])
    proposal_id = proposals[0]["id"]
    db._conn().execute(
        "UPDATE memory_capture_jobs SET updated_at = %s WHERE id = %s",
        (time.time() - 40 * 86400, job_id),
    )
    db.prune_operational_tables(30)  # job_id gone; the proposal now has job_id = NULL

    result = db.delete_session(lag_agent_id, session_id)

    # Before R2-GAP-001's fix, delete_session found a session's proposals by
    # walking memory_capture_jobs.session_id — with the job already pruned,
    # this proposal would have been silently left behind (orphaned).
    assert proposal_id in result["proposal_ids"]
    row = db._conn().execute(
        "SELECT 1 FROM memory_proposals WHERE id = %s", (proposal_id,)
    ).fetchone()
    assert row is None


# ---------------------------------------------------------------------------
# R2-GAP-003: stale-running job reclaim
# ---------------------------------------------------------------------------

def _mark_running(db, table: str, job_id: int, age_seconds: float) -> None:
    """Force one job row into 'running' at a given age, bypassing claim_next_*_job().

    Deliberately doesn't go through claim_next_*_job(): that claims
    whatever the globally oldest *eligible* pending row happens to be
    across this shared dev database (potentially a leftover row from
    unrelated earlier manual testing, not the one this test just created)
    — see R2-GAP-015. Setting state/updated_at directly by this test's own
    job_id is what actually pins down "this specific job."
    """
    db._conn().execute(
        f"UPDATE {table} SET state = 'running', updated_at = %s WHERE id = %s",
        (time.time() - age_seconds, job_id),
    )


def test_reclaim_stale_running_jobs_resets_an_old_running_capture_job(
    db, session_id, lag_agent_id
):
    db.upsert_session(session_id, lag_agent_id)
    job_id = db.enqueue_capture_job(
        lag_agent_id, session_id, 1, 2, idempotency_key=f"key-{session_id}"
    )
    _mark_running(db, "memory_capture_jobs", job_id, 7200)  # 2h old — a crashed worker

    counts = db.reclaim_stale_running_jobs()

    # reclaim_stale_running_jobs() is deliberately global, not per-agent
    # (see its docstring), so >= rather than == — this shared dev database
    # may have other genuinely stale rows at the same moment. The precise
    # assertion is that THIS job specifically transitioned back to pending.
    assert counts["capture_jobs"] >= 1
    row = db._conn().execute(
        "SELECT state FROM memory_capture_jobs WHERE id = %s", (job_id,)
    ).fetchone()
    assert row[0] == "pending"


def test_reclaim_stale_running_jobs_resets_an_old_running_commitment_job(
    db, session_id, lag_agent_id
):
    db.upsert_session(session_id, lag_agent_id)
    job_id = db.enqueue_commitment_job(
        lag_agent_id, session_id, "cli", 1, 2, idempotency_key=f"commit-{session_id}"
    )
    _mark_running(db, "memory_commitment_jobs", job_id, 7200)

    counts = db.reclaim_stale_running_jobs()

    assert counts["commitment_jobs"] >= 1
    row = db._conn().execute(
        "SELECT state FROM memory_commitment_jobs WHERE id = %s", (job_id,)
    ).fetchone()
    assert row[0] == "pending"


def test_reclaim_stale_running_jobs_resets_an_old_running_embedding_job(
    db, session_id, lag_agent_id
):
    db.upsert_session(session_id, lag_agent_id)
    message_id = db.mirror_message(session_id, "e1", "user", "hello")
    job_id = db.enqueue_message_embedding_job(
        lag_agent_id, session_id, message_id, idempotency_key=f"emb-{session_id}"
    )
    _mark_running(db, "message_embedding_jobs", job_id, 7200)

    counts = db.reclaim_stale_running_jobs()

    assert counts["message_embedding_jobs"] >= 1
    row = db._conn().execute(
        "SELECT state FROM message_embedding_jobs WHERE id = %s", (job_id,)
    ).fetchone()
    assert row[0] == "pending"


def test_reclaim_stale_running_jobs_leaves_a_recently_claimed_job_alone(
    db, session_id, lag_agent_id
):
    db.upsert_session(session_id, lag_agent_id)
    job_id = db.enqueue_capture_job(
        lag_agent_id, session_id, 1, 2, idempotency_key=f"key-{session_id}"
    )
    _mark_running(db, "memory_capture_jobs", job_id, 5)  # 5s old — well within the default threshold

    db.reclaim_stale_running_jobs()

    row = db._conn().execute(
        "SELECT state FROM memory_capture_jobs WHERE id = %s", (job_id,)
    ).fetchone()
    assert row[0] == "running"


def test_reclaim_stale_running_jobs_respects_a_custom_threshold(db, session_id, lag_agent_id):
    db.upsert_session(session_id, lag_agent_id)
    job_id = db.enqueue_capture_job(
        lag_agent_id, session_id, 1, 2, idempotency_key=f"key-{session_id}"
    )
    _mark_running(db, "memory_capture_jobs", job_id, 30)  # 30 seconds ago

    counts = db.reclaim_stale_running_jobs(stale_after_seconds=10)

    assert counts["capture_jobs"] >= 1
    row = db._conn().execute(
        "SELECT state FROM memory_capture_jobs WHERE id = %s", (job_id,)
    ).fetchone()
    assert row[0] == "pending"


def test_queue_lag_summary_reports_a_stuck_running_job(db, session_id, lag_agent_id):
    db.upsert_session(session_id, lag_agent_id)
    job_id = db.enqueue_capture_job(
        lag_agent_id, session_id, 1, 2, idempotency_key=f"key-{session_id}"
    )
    _mark_running(db, "memory_capture_jobs", job_id, 7200)

    summary = db.queue_lag_summary(lag_agent_id)

    assert summary["capture"]["running_count"] == 1
    assert summary["capture"]["oldest_running_age_s"] >= 7200.0


def test_queue_lag_summary_reports_a_permanently_failed_job(db, session_id, lag_agent_id):
    db.upsert_session(session_id, lag_agent_id)
    job_id = db.enqueue_capture_job(
        lag_agent_id, session_id, 1, 2, idempotency_key=f"key-{session_id}"
    )
    db.claim_next_capture_job()
    db.fail_capture_job(job_id, "boom", backoff_seconds=0.0, max_attempts=1)  # -> terminal 'failed'

    summary = db.queue_lag_summary(lag_agent_id)

    assert summary["capture"]["failed_count"] == 1


def test_queue_lag_summary_never_counts_pending_jobs_as_failed(db, session_id, lag_agent_id):
    db.upsert_session(session_id, lag_agent_id)
    db.enqueue_capture_job(lag_agent_id, session_id, 1, 2, idempotency_key=f"key-{session_id}")

    summary = db.queue_lag_summary(lag_agent_id)

    assert summary["capture"]["failed_count"] == 0


# ---------------------------------------------------------------------------
# find_uncovered_capture_ranges / find_uncovered_commitment_ranges
# (MEM-GAP-007, R2-GAP-002)
# ---------------------------------------------------------------------------

def test_find_uncovered_capture_ranges_empty_without_any_messages(db, session_id):
    db.upsert_session(session_id, "main")

    assert db.find_uncovered_capture_ranges("main", session_id) == []


def test_find_uncovered_capture_ranges_empty_for_an_unowned_session(db, session_id):
    db.upsert_session(session_id, "researcher")
    db.mirror_message(session_id, "e1", "user", "hello")

    assert db.find_uncovered_capture_ranges("main", session_id) == []


def test_find_uncovered_capture_ranges_returns_the_full_range_with_no_prior_job(db, session_id):
    db.upsert_session(session_id, "main")
    id1 = db.mirror_message(session_id, "e1", "user", "hello")
    id2 = db.mirror_message(session_id, "e2", "assistant", "hi there")

    assert db.find_uncovered_capture_ranges("main", session_id) == [(id1, id2)]


def test_find_uncovered_capture_ranges_empty_when_fully_covered(db, session_id):
    db.upsert_session(session_id, "main")
    id1 = db.mirror_message(session_id, "e1", "user", "hello")
    id2 = db.mirror_message(session_id, "e2", "assistant", "hi there")
    db.enqueue_capture_job(
        "main", session_id, id1, id2, idempotency_key=f"key-{session_id}"
    )

    assert db.find_uncovered_capture_ranges("main", session_id) == []


def test_find_uncovered_capture_ranges_returns_only_the_gap_after_the_last_job(db, session_id):
    db.upsert_session(session_id, "main")
    id1 = db.mirror_message(session_id, "e1", "user", "first turn")
    id2 = db.mirror_message(session_id, "e2", "assistant", "first reply")
    db.enqueue_capture_job(
        "main", session_id, id1, id2, idempotency_key=f"key1-{session_id}"
    )
    id3 = db.mirror_message(session_id, "e3", "user", "second turn")
    id4 = db.mirror_message(session_id, "e4", "assistant", "second reply")

    assert db.find_uncovered_capture_ranges("main", session_id) == [(id3, id4)]


def test_find_uncovered_capture_ranges_finds_a_sparse_gap_below_a_later_covered_range(
    db, session_id
):
    # R2-GAP-002: the whole point of moving off a high-water-mark cursor —
    # a missed enqueue for an EARLIER exchange must still surface even
    # after a LATER exchange was successfully covered.
    db.upsert_session(session_id, "main")
    id1 = db.mirror_message(session_id, "e1", "user", "first turn — enqueue missed")
    id2 = db.mirror_message(session_id, "e2", "assistant", "first reply — enqueue missed")
    id3 = db.mirror_message(session_id, "e3", "user", "second turn — enqueued fine")
    id4 = db.mirror_message(session_id, "e4", "assistant", "second reply — enqueued fine")
    db.enqueue_capture_job(
        "main", session_id, id3, id4, idempotency_key=f"key-{session_id}"
    )

    assert db.find_uncovered_capture_ranges("main", session_id) == [(id1, id2)]


def test_find_uncovered_capture_ranges_finds_a_sparse_gap_and_a_tail_gap_together(
    db, session_id
):
    db.upsert_session(session_id, "main")
    id1 = db.mirror_message(session_id, "e1", "user", "first turn — missed")
    id2 = db.mirror_message(session_id, "e2", "assistant", "first reply — missed")
    id3 = db.mirror_message(session_id, "e3", "user", "second turn — covered")
    id4 = db.mirror_message(session_id, "e4", "assistant", "second reply — covered")
    db.enqueue_capture_job(
        "main", session_id, id3, id4, idempotency_key=f"key-{session_id}"
    )
    id5 = db.mirror_message(session_id, "e5", "user", "third turn — never enqueued")
    id6 = db.mirror_message(session_id, "e6", "assistant", "third reply — never enqueued")

    assert db.find_uncovered_capture_ranges("main", session_id) == [(id1, id2), (id5, id6)]


def test_find_uncovered_capture_ranges_treats_a_failed_job_as_still_covering(db, session_id):
    db.upsert_session(session_id, "main")
    id1 = db.mirror_message(session_id, "e1", "user", "hello")
    id2 = db.mirror_message(session_id, "e2", "assistant", "hi there")
    job_id = db.enqueue_capture_job(
        "main", session_id, id1, id2, idempotency_key=f"key-{session_id}"
    )
    db.claim_next_capture_job()
    db.fail_capture_job(job_id, "boom", backoff_seconds=999.0, max_attempts=1)

    # Still "covered" -- a failed extraction attempt is not an unenqueued gap.
    assert db.find_uncovered_capture_ranges("main", session_id) == []


def test_find_uncovered_capture_ranges_excludes_tool_and_empty_messages(db, session_id):
    db.upsert_session(session_id, "main")
    id1 = db.mirror_message(session_id, "e1", "user", "hello")
    db.mirror_message(session_id, "e2", "tool", "tool output", tool_name="bash")
    id3 = db.mirror_message(session_id, "e3", "assistant", "hi there")

    assert db.find_uncovered_capture_ranges("main", session_id) == [(id1, id3)]


def test_find_uncovered_capture_ranges_survives_the_covering_job_being_pruned(
    db, session_id, lag_agent_id
):
    # R2-GAP-001/R2-GAP-002 together: coverage lives in
    # session_job_coverage_ranges, which retention pruning never touches —
    # deleting the job row itself must not make an already-covered range
    # look uncovered again.
    db.upsert_session(session_id, lag_agent_id)
    id1 = db.mirror_message(session_id, "e1", "user", "hello")
    id2 = db.mirror_message(session_id, "e2", "assistant", "hi there")
    job_id = db.enqueue_capture_job(
        lag_agent_id, session_id, id1, id2, idempotency_key=f"key-{session_id}"
    )
    db._conn().execute("DELETE FROM memory_capture_jobs WHERE id = %s", (job_id,))

    assert db.find_uncovered_capture_ranges(lag_agent_id, session_id) == []


def test_find_uncovered_commitment_ranges_empty_without_any_prior_job(db, session_id):
    # A session with zero commitment jobs ever is left alone -- most likely
    # means commitments were disabled, not a missed enqueue (see docstring).
    db.upsert_session(session_id, "main")
    db.mirror_message(session_id, "e1", "user", "hello")

    assert db.find_uncovered_commitment_ranges("main", session_id) == []


def test_find_uncovered_commitment_ranges_empty_for_an_unowned_session(db, session_id):
    db.upsert_session(session_id, "researcher")
    db.mirror_message(session_id, "e1", "user", "hello")

    assert db.find_uncovered_commitment_ranges("main", session_id) == []


def test_find_uncovered_commitment_ranges_returns_the_gap_after_the_last_job(db, session_id):
    db.upsert_session(session_id, "main")
    id1 = db.mirror_message(session_id, "e1", "user", "first turn")
    id2 = db.mirror_message(session_id, "e2", "assistant", "first reply")
    db.enqueue_commitment_job(
        "main", session_id, "cli", id1, id2, idempotency_key=f"key1-{session_id}"
    )
    id3 = db.mirror_message(session_id, "e3", "user", "second turn")
    id4 = db.mirror_message(session_id, "e4", "assistant", "second reply")

    assert db.find_uncovered_commitment_ranges("main", session_id) == [("cli", id3, id4)]


def test_find_uncovered_commitment_ranges_finds_a_sparse_gap_below_a_later_covered_range(
    db, session_id
):
    db.upsert_session(session_id, "main")
    id1 = db.mirror_message(session_id, "e1", "user", "first turn — missed")
    id2 = db.mirror_message(session_id, "e2", "assistant", "first reply — missed")
    id3 = db.mirror_message(session_id, "e3", "user", "second turn — covered")
    id4 = db.mirror_message(session_id, "e4", "assistant", "second reply — covered")
    db.enqueue_commitment_job(
        "main", session_id, "cli", id3, id4, idempotency_key=f"key-{session_id}"
    )

    assert db.find_uncovered_commitment_ranges("main", session_id) == [("cli", id1, id2)]


def test_find_uncovered_commitment_ranges_uses_the_most_recently_covered_jobs_channel(
    db, session_id
):
    db.upsert_session(session_id, "main")
    id1 = db.mirror_message(session_id, "e1", "user", "first turn")
    id2 = db.mirror_message(session_id, "e2", "assistant", "first reply")
    db.enqueue_commitment_job(
        "main", session_id, "!room:example.org", id1, id2, idempotency_key=f"key1-{session_id}"
    )
    id3 = db.mirror_message(session_id, "e3", "user", "second turn")
    id4 = db.mirror_message(session_id, "e4", "assistant", "second reply")

    gaps = db.find_uncovered_commitment_ranges("main", session_id)

    assert gaps == [("!room:example.org", id3, id4)]


def test_find_uncovered_commitment_ranges_empty_when_fully_covered(db, session_id):
    db.upsert_session(session_id, "main")
    id1 = db.mirror_message(session_id, "e1", "user", "hello")
    id2 = db.mirror_message(session_id, "e2", "assistant", "hi there")
    db.enqueue_commitment_job(
        "main", session_id, "cli", id1, id2, idempotency_key=f"key-{session_id}"
    )

    assert db.find_uncovered_commitment_ranges("main", session_id) == []


def test_find_uncovered_commitment_ranges_survives_the_covering_job_being_pruned(
    db, session_id, lag_agent_id
):
    db.upsert_session(session_id, lag_agent_id)
    id1 = db.mirror_message(session_id, "e1", "user", "hello")
    id2 = db.mirror_message(session_id, "e2", "assistant", "hi there")
    job_id = db.enqueue_commitment_job(
        lag_agent_id, session_id, "cli", id1, id2, idempotency_key=f"key-{session_id}"
    )
    db._conn().execute("DELETE FROM memory_commitment_jobs WHERE id = %s", (job_id,))

    assert db.find_uncovered_commitment_ranges(lag_agent_id, session_id) == []


# ---------------------------------------------------------------------------
# list_capture_job_ranges / list_commitment_job_ranges (R2-GAP-002)
# ---------------------------------------------------------------------------

def test_list_capture_job_ranges_survives_the_job_being_pruned(db, session_id, lag_agent_id):
    db.upsert_session(session_id, lag_agent_id)
    job_id = db.enqueue_capture_job(
        lag_agent_id, session_id, 5, 9, idempotency_key=f"key-{session_id}"
    )
    db._conn().execute("DELETE FROM memory_capture_jobs WHERE id = %s", (job_id,))

    assert db.list_capture_job_ranges(session_id) == [(5, 9)]


def test_list_commitment_job_ranges_survives_the_job_being_pruned(db, session_id, lag_agent_id):
    db.upsert_session(session_id, lag_agent_id)
    job_id = db.enqueue_commitment_job(
        lag_agent_id, session_id, "cli", 5, 9, idempotency_key=f"key-{session_id}"
    )
    db._conn().execute("DELETE FROM memory_commitment_jobs WHERE id = %s", (job_id,))

    assert db.list_commitment_job_ranges(session_id) == [(5, 9)]


# ---------------------------------------------------------------------------
# Message-embedding pipeline (MEM-GAP-006)
# ---------------------------------------------------------------------------
# db_with_vector uses a small 3-dim embedding (matching
# tests/memory/test_postgres_index.py's own vector_index/embedding_index
# convention for this shared dev database) rather than a real model's real
# dimensions — only the plumbing is under test here, not embedding quality.

@pytest.fixture
def mock_embedding_provider():
    """A fake EmbeddingProvider returning one fixed 3-dim vector per input text."""
    from unittest.mock import Mock

    provider = Mock()
    provider.model_identity = "test-endpoint::test-model"
    provider.embed = Mock(side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts])
    return provider


@pytest.fixture
def db_with_vector(mock_embedding_provider):
    """A SessionDB fully wired for embeddings — dimensions + a mock provider."""
    return SessionDB(
        _DB_URL, embedding_dimensions=3, embedding_provider=mock_embedding_provider
    )


@pytest.fixture(autouse=True)
def _cleanup_message_embeddings(db_with_vector, session_id):
    yield
    conn = db_with_vector._conn()
    conn.execute(
        "DELETE FROM message_embeddings WHERE message_id IN "
        "(SELECT id FROM messages WHERE session_id = %s)",
        (session_id,),
    )
    conn.execute("DELETE FROM message_embedding_jobs WHERE session_id = %s", (session_id,))


def test_has_vector_lane_is_false_without_configured_dimensions(db):
    assert db.has_vector_lane is False


def test_has_vector_lane_is_true_with_configured_dimensions(db_with_vector):
    assert db_with_vector.has_vector_lane is True


def test_embedding_model_identity_is_none_without_a_provider(db):
    assert db.embedding_model_identity is None


def test_embedding_model_identity_reflects_the_configured_provider(db_with_vector):
    assert db_with_vector.embedding_model_identity == "test-endpoint::test-model"


# --- job queue ---

def test_enqueue_message_embedding_job_returns_a_job_id(db, session_id):
    db.upsert_session(session_id, "main")
    message_id = db.add_message(session_id, "user", "hello")

    job_id = db.enqueue_message_embedding_job(
        "main", session_id, message_id, idempotency_key=f"key-{session_id}"
    )

    assert job_id is not None


def test_enqueue_message_embedding_job_is_idempotent(db, session_id):
    db.upsert_session(session_id, "main")
    message_id = db.add_message(session_id, "user", "hello")
    key = f"key-{session_id}"

    first = db.enqueue_message_embedding_job("main", session_id, message_id, idempotency_key=key)
    second = db.enqueue_message_embedding_job("main", session_id, message_id, idempotency_key=key)

    assert first is not None
    assert second is None


def test_claim_next_message_embedding_job_returns_the_enqueued_job(db, session_id):
    db.upsert_session(session_id, "main")
    message_id = db.add_message(session_id, "user", "hello")
    job_id = db.enqueue_message_embedding_job(
        "main", session_id, message_id, idempotency_key=f"key-{session_id}"
    )

    claimed = db.claim_next_message_embedding_job()

    # The queue is global (like capture/commitment jobs) — some other
    # pending row from the shared dev database could be claimed first, so
    # only assert on our own job if it happens to be the one claimed.
    if claimed is not None and claimed["id"] == job_id:
        assert claimed["agent_id"] == "main"
        assert claimed["session_id"] == session_id
        assert claimed["message_id"] == message_id
        assert claimed["attempts"] == 0


def test_complete_message_embedding_job_stores_the_embedding_and_marks_done(db, session_id):
    db.upsert_session(session_id, "main")
    message_id = db.add_message(session_id, "user", "hello")
    job_id = db.enqueue_message_embedding_job(
        "main", session_id, message_id, idempotency_key=f"key-{session_id}"
    )

    db.complete_message_embedding_job(job_id, message_id, "test-endpoint::test-model", [0.1, 0.2, 0.3])

    row = db._conn().execute(
        "SELECT state FROM message_embedding_jobs WHERE id = %s", (job_id,)
    ).fetchone()
    assert row[0] == "done"
    stored = db._conn().execute(
        "SELECT model_identity FROM message_embeddings WHERE message_id = %s", (message_id,)
    ).fetchone()
    assert stored[0] == "test-endpoint::test-model"


def test_complete_message_embedding_job_upserts_on_retry(db, session_id):
    db.upsert_session(session_id, "main")
    message_id = db.add_message(session_id, "user", "hello")
    job_id = db.enqueue_message_embedding_job(
        "main", session_id, message_id, idempotency_key=f"key-{session_id}"
    )

    db.complete_message_embedding_job(job_id, message_id, "test-endpoint::test-model", [0.1, 0.2, 0.3])
    db.complete_message_embedding_job(job_id, message_id, "test-endpoint::test-model", [0.4, 0.5, 0.6])

    count = db._conn().execute(
        "SELECT count(*) FROM message_embeddings WHERE message_id = %s", (message_id,)
    ).fetchone()[0]
    assert count == 1  # upsert, not a second row


def test_fail_message_embedding_job_retries_with_backoff(db, session_id):
    db.upsert_session(session_id, "main")
    message_id = db.add_message(session_id, "user", "hello")
    job_id = db.enqueue_message_embedding_job(
        "main", session_id, message_id, idempotency_key=f"key-{session_id}"
    )

    db.fail_message_embedding_job(job_id, "boom", backoff_seconds=60.0, max_attempts=5)

    row = db._conn().execute(
        "SELECT state, attempts, last_error FROM message_embedding_jobs WHERE id = %s", (job_id,)
    ).fetchone()
    assert row[0] == "pending"  # still under max_attempts, retried
    assert row[1] == 1
    assert row[2] == "boom"


def test_fail_message_embedding_job_gives_up_after_max_attempts(db, session_id):
    db.upsert_session(session_id, "main")
    message_id = db.add_message(session_id, "user", "hello")
    job_id = db.enqueue_message_embedding_job(
        "main", session_id, message_id, idempotency_key=f"key-{session_id}"
    )

    db.fail_message_embedding_job(job_id, "boom", backoff_seconds=0.0, max_attempts=1)

    row = db._conn().execute(
        "SELECT state FROM message_embedding_jobs WHERE id = %s", (job_id,)
    ).fetchone()
    assert row[0] == "failed"


# --- find_uncovered_message_ids_for_embedding (R2-GAP-002/R2-GAP-005) ---
# Checks message_embeddings directly for the active model, not a job-table
# cursor -- needs db_with_vector (a configured provider), not the plain
# db fixture, for anything but the "no provider configured" case.

def test_find_uncovered_message_ids_for_embedding_returns_empty_for_an_unowned_session(
    db_with_vector, session_id
):
    db_with_vector.upsert_session(session_id, "researcher")
    db_with_vector.mirror_message(session_id, "e1", "user", "hello")

    assert db_with_vector.find_uncovered_message_ids_for_embedding("main", session_id) == []


def test_find_uncovered_message_ids_for_embedding_returns_empty_without_a_provider(db, session_id):
    db.upsert_session(session_id, "main")
    db.mirror_message(session_id, "e1", "user", "hello")

    assert db.find_uncovered_message_ids_for_embedding("main", session_id) == []


def test_find_uncovered_message_ids_for_embedding_returns_all_new_messages(
    db_with_vector, session_id
):
    db_with_vector.upsert_session(session_id, "main")
    id1 = db_with_vector.mirror_message(session_id, "e1", "user", "first")
    id2 = db_with_vector.mirror_message(session_id, "e2", "assistant", "second")

    assert db_with_vector.find_uncovered_message_ids_for_embedding("main", session_id) == [id1, id2]


def test_find_uncovered_message_ids_for_embedding_excludes_tool_and_empty_messages(
    db_with_vector, session_id
):
    db_with_vector.upsert_session(session_id, "main")
    id1 = db_with_vector.mirror_message(session_id, "e1", "user", "question")
    db_with_vector.mirror_message(session_id, "e2", "tool", "raw output", tool_name="read")
    db_with_vector.mirror_message(session_id, "e3", "assistant", None)

    assert db_with_vector.find_uncovered_message_ids_for_embedding("main", session_id) == [id1]


def test_find_uncovered_message_ids_for_embedding_still_reports_a_merely_pending_job(
    db_with_vector, session_id
):
    # Coverage means "has an actual embedding," not "has a job enqueued" —
    # a pending (not yet completed) job doesn't hide a message from
    # reconciliation. Harmless: enqueue_message_embedding_job is
    # idempotency-keyed, so re-enqueuing the same pending job no-ops.
    db_with_vector.upsert_session(session_id, "main")
    id1 = db_with_vector.mirror_message(session_id, "e1", "user", "first")
    db_with_vector.enqueue_message_embedding_job(
        "main", session_id, id1, idempotency_key=f"key-{session_id}"
    )

    assert db_with_vector.find_uncovered_message_ids_for_embedding("main", session_id) == [id1]


def test_find_uncovered_message_ids_for_embedding_excludes_a_completed_embedding(
    db_with_vector, session_id
):
    db_with_vector.upsert_session(session_id, "main")
    id1 = db_with_vector.mirror_message(session_id, "e1", "user", "first")
    id2 = db_with_vector.mirror_message(session_id, "e2", "assistant", "second")
    job_id = db_with_vector.enqueue_message_embedding_job(
        "main", session_id, id1, idempotency_key=f"key-{session_id}"
    )
    db_with_vector.complete_message_embedding_job(
        job_id, id1, db_with_vector.embedding_model_identity, [0.1, 0.2, 0.3]
    )

    assert db_with_vector.find_uncovered_message_ids_for_embedding("main", session_id) == [id2]


def test_find_uncovered_message_ids_for_embedding_empty_when_fully_covered(
    db_with_vector, session_id
):
    db_with_vector.upsert_session(session_id, "main")
    id1 = db_with_vector.mirror_message(session_id, "e1", "user", "hello")
    job_id = db_with_vector.enqueue_message_embedding_job(
        "main", session_id, id1, idempotency_key=f"key-{session_id}"
    )
    db_with_vector.complete_message_embedding_job(
        job_id, id1, db_with_vector.embedding_model_identity, [0.1, 0.2, 0.3]
    )

    assert db_with_vector.find_uncovered_message_ids_for_embedding("main", session_id) == []


def test_find_uncovered_message_ids_for_embedding_is_model_aware(db_with_vector, session_id):
    # R2-GAP-005: a message embedded under an OLD model must re-surface as
    # uncovered once the active model changes -- this is what makes
    # periodic reconciliation the automatic backfill mechanism after a
    # model switch, with no separate manual "rebuild" command needed.
    db_with_vector.upsert_session(session_id, "main")
    id1 = db_with_vector.mirror_message(session_id, "e1", "user", "hello")
    job_id = db_with_vector.enqueue_message_embedding_job(
        "main", session_id, id1, idempotency_key=f"key-{session_id}"
    )
    db_with_vector.complete_message_embedding_job(
        job_id, id1, "old-endpoint::old-model", [0.1, 0.2, 0.3]
    )

    # Embedded under the OLD model; db_with_vector's active model
    # (mock_embedding_provider) is "test-endpoint::test-model" — still uncovered.
    assert db_with_vector.find_uncovered_message_ids_for_embedding("main", session_id) == [id1]


# ---------------------------------------------------------------------------
# embedding_coverage_summary (R2-GAP-014)
# ---------------------------------------------------------------------------

def test_embedding_coverage_summary_counts_messages_missing_the_active_model(
    db_with_vector, session_id, lag_agent_id
):
    db_with_vector.upsert_session(session_id, lag_agent_id)
    id1 = db_with_vector.mirror_message(session_id, "e1", "user", "first")
    db_with_vector.mirror_message(session_id, "e2", "assistant", "second — never embedded")
    job_id = db_with_vector.enqueue_message_embedding_job(
        lag_agent_id, session_id, id1, idempotency_key=f"key-{session_id}"
    )
    db_with_vector.complete_message_embedding_job(
        job_id, id1, db_with_vector.embedding_model_identity, [0.1, 0.2, 0.3]
    )

    summary = db_with_vector.embedding_coverage_summary(lag_agent_id)

    assert summary == {"missing_count": 1, "model_identity": "test-endpoint::test-model"}


def test_embedding_coverage_summary_zero_when_fully_covered(
    db_with_vector, session_id, lag_agent_id
):
    db_with_vector.upsert_session(session_id, lag_agent_id)
    id1 = db_with_vector.mirror_message(session_id, "e1", "user", "first")
    job_id = db_with_vector.enqueue_message_embedding_job(
        lag_agent_id, session_id, id1, idempotency_key=f"key-{session_id}"
    )
    db_with_vector.complete_message_embedding_job(
        job_id, id1, db_with_vector.embedding_model_identity, [0.1, 0.2, 0.3]
    )

    summary = db_with_vector.embedding_coverage_summary(lag_agent_id)

    assert summary["missing_count"] == 0


def test_embedding_coverage_summary_none_without_a_vector_lane(db, lag_agent_id):
    assert db.embedding_coverage_summary(lag_agent_id) is None


# --- vector lane / hybrid search ---

def test_vector_search_messages_empty_without_a_provider(db, session_id):
    db.upsert_session(session_id, "main")
    db.mirror_message(session_id, "e1", "user", "hello")

    assert db._vector_search_messages("hello", "main", limit=5) == []


def test_vector_search_messages_finds_a_cached_embedding(db_with_vector, session_id):
    db_with_vector.upsert_session(session_id, "main")
    message_id = db_with_vector.mirror_message(session_id, "e1", "user", "a distinctive phrase")
    db_with_vector.complete_message_embedding_job(
        db_with_vector.enqueue_message_embedding_job(
            "main", session_id, message_id, idempotency_key=f"key-{session_id}"
        ),
        message_id, "test-endpoint::test-model", [0.1, 0.2, 0.3],
    )

    results = db_with_vector._vector_search_messages("anything", "main", limit=5)

    assert [r["id"] for r in results] == [message_id]


# ---------------------------------------------------------------------------
# Vector search relevance floor (R2-GAP-012)
# ---------------------------------------------------------------------------

def test_vector_search_messages_excludes_hits_below_min_similarity(session_id):
    from unittest.mock import Mock

    provider = Mock()
    provider.model_identity = "test-endpoint::test-model"
    provider.embed = Mock(return_value=[[1.0, 0.0, 0.0]])  # the query always embeds to this
    db = SessionDB(_DB_URL, embedding_dimensions=3, embedding_provider=provider, min_similarity=0.5)
    db.upsert_session(session_id, "main")
    message_id = db.mirror_message(session_id, "e1", "user", "unrelated content")
    job_id = db.enqueue_message_embedding_job(
        "main", session_id, message_id, idempotency_key=f"key-{session_id}"
    )
    # Orthogonal to the query vector -- cosine similarity 0, well below the 0.5 floor.
    db.complete_message_embedding_job(job_id, message_id, "test-endpoint::test-model", [0.0, 1.0, 0.0])

    results = db._vector_search_messages("query", "main", limit=5)

    assert results == []


def test_vector_search_messages_keeps_hits_at_or_above_min_similarity(session_id):
    from unittest.mock import Mock

    provider = Mock()
    provider.model_identity = "test-endpoint::test-model"
    provider.embed = Mock(return_value=[[1.0, 0.0, 0.0]])
    db = SessionDB(_DB_URL, embedding_dimensions=3, embedding_provider=provider, min_similarity=0.5)
    db.upsert_session(session_id, "main")
    message_id = db.mirror_message(session_id, "e1", "user", "closely related content")
    job_id = db.enqueue_message_embedding_job(
        "main", session_id, message_id, idempotency_key=f"key-{session_id}"
    )
    # Identical to the query vector -- cosine similarity 1, comfortably above the floor.
    db.complete_message_embedding_job(job_id, message_id, "test-endpoint::test-model", [1.0, 0.0, 0.0])

    results = db._vector_search_messages("query", "main", limit=5)

    assert [r["id"] for r in results] == [message_id]


def test_vector_search_messages_default_min_similarity_is_zero(db_with_vector):
    assert db_with_vector._min_similarity == 0.0


def test_message_embeddings_self_heals_a_dimension_change(session_id):
    # R2-GAP-015 regression: found via test-isolation work sharing one
    # schema across SessionDB instances constructed with different
    # embedding_dimensions -- also a real production scenario if
    # embeddings.dimensions ever changes in config.json. Without healing,
    # the table silently keeps its old width and a later real embed() call
    # fails with a dimension mismatch instead of picking up the new width.
    from unittest.mock import Mock

    old_provider = Mock()
    old_provider.model_identity = "old-model"
    old_provider.embed = Mock(return_value=[[0.0] * 5])
    SessionDB(_DB_URL, embedding_dimensions=5, embedding_provider=old_provider)

    new_provider = Mock()
    new_provider.model_identity = "new-model"
    db = SessionDB(_DB_URL, embedding_dimensions=3, embedding_provider=new_provider)
    db.upsert_session(session_id, "main")
    message_id = db.mirror_message(session_id, "e1", "user", "hello")
    job_id = db.enqueue_message_embedding_job(
        "main", session_id, message_id, idempotency_key=f"key-{session_id}"
    )

    db.complete_message_embedding_job(job_id, message_id, "new-model", [0.1, 0.2, 0.3])  # must not raise

    result = db._conn().execute(
        "SELECT embedding FROM message_embeddings WHERE message_id = %s", (message_id,)
    ).fetchone()
    assert result is not None


def test_vector_search_messages_is_scoped_to_the_given_agent(db_with_vector, session_id):
    db_with_vector.upsert_session(session_id, "researcher")
    message_id = db_with_vector.mirror_message(session_id, "e1", "user", "a distinctive phrase")
    db_with_vector.complete_message_embedding_job(
        db_with_vector.enqueue_message_embedding_job(
            "researcher", session_id, message_id, idempotency_key=f"key-{session_id}"
        ),
        message_id, "test-endpoint::test-model", [0.1, 0.2, 0.3],
    )

    results = db_with_vector._vector_search_messages("anything", "main", limit=5)

    assert results == []


def test_hybrid_search_messages_falls_back_to_lexical_without_a_vector_lane(db, session_id):
    db.upsert_session(session_id, "main")
    db.mirror_message(session_id, "e1", "user", "a very particular keyword")

    results = db.hybrid_search_messages("particular keyword", "main", limit=5)

    assert len(results) == 1
    assert results[0]["content"] == "a very particular keyword"


def test_hybrid_search_messages_includes_a_vector_only_match(db_with_vector, session_id):
    db_with_vector.upsert_session(session_id, "main")
    # No lexical overlap with the query at all — only the vector lane (with
    # its constant mock vector) can surface this.
    message_id = db_with_vector.mirror_message(session_id, "e1", "user", "zzz_no_overlap_zzz")
    db_with_vector.complete_message_embedding_job(
        db_with_vector.enqueue_message_embedding_job(
            "main", session_id, message_id, idempotency_key=f"key-{session_id}"
        ),
        message_id, "test-endpoint::test-model", [0.1, 0.2, 0.3],
    )

    results = db_with_vector.hybrid_search_messages("completely different wording", "main", limit=5)

    assert message_id in [r["id"] for r in results]


def test_hybrid_search_messages_result_shape_matches_search_messages(db_with_vector, session_id):
    db_with_vector.upsert_session(session_id, "main")
    message_id = db_with_vector.mirror_message(session_id, "e1", "user", "shared keyword text")
    db_with_vector.complete_message_embedding_job(
        db_with_vector.enqueue_message_embedding_job(
            "main", session_id, message_id, idempotency_key=f"key-{session_id}"
        ),
        message_id, "test-endpoint::test-model", [0.1, 0.2, 0.3],
    )

    results = db_with_vector.hybrid_search_messages("shared keyword", "main", limit=5)

    assert results
    row = results[0]
    for field in ("id", "session_id", "role", "content", "tool_name", "timestamp", "snippet", "rank"):
        assert field in row


# --- delete_session cleanup ---

def test_delete_session_removes_message_embeddings(db_with_vector, session_id):
    db_with_vector.upsert_session(session_id, "main")
    message_id = db_with_vector.mirror_message(session_id, "e1", "user", "hello")
    db_with_vector.complete_message_embedding_job(
        db_with_vector.enqueue_message_embedding_job(
            "main", session_id, message_id, idempotency_key=f"key-{session_id}"
        ),
        message_id, "test-endpoint::test-model", [0.1, 0.2, 0.3],
    )

    db_with_vector.delete_session("main", session_id)

    row = db_with_vector._conn().execute(
        "SELECT count(*) FROM message_embeddings WHERE message_id = %s", (message_id,)
    ).fetchone()
    assert row[0] == 0


# ---------------------------------------------------------------------------
# Reconciliation scan bound (R2-GAP-011)
# ---------------------------------------------------------------------------

def test_eligible_message_ids_limit_keeps_only_the_most_recent_ids(db, session_id):
    db.upsert_session(session_id, "main")
    ids = [db.mirror_message(session_id, f"e{i}", "user", f"msg {i}") for i in range(5)]

    result = db._eligible_message_ids(session_id, limit=2)

    assert result == ids[-2:]  # the two most recent, still ascending


def test_eligible_message_ids_limit_none_returns_full_history(db, session_id):
    db.upsert_session(session_id, "main")
    ids = [db.mirror_message(session_id, f"e{i}", "user", f"msg {i}") for i in range(5)]

    result = db._eligible_message_ids(session_id, limit=None)

    assert result == ids


def test_find_uncovered_capture_ranges_uses_the_bounded_scan_by_default(
    db, session_id, lag_agent_id
):
    # A very old, never-covered message must not surface once the session
    # has grown past the reconciliation scan bound -- see
    # _RECONCILIATION_SCAN_LIMIT's comment for the accepted trade-off.
    import minion_assist.session.db as db_module

    original_limit = db_module._RECONCILIATION_SCAN_LIMIT
    db_module._RECONCILIATION_SCAN_LIMIT = 2
    try:
        db.upsert_session(session_id, lag_agent_id)
        old_id1 = db.mirror_message(session_id, "e0", "user", "ancient, never covered")
        old_id2 = db.mirror_message(session_id, "e1", "assistant", "also ancient")
        recent_id1 = db.mirror_message(session_id, "e2", "user", "recent")
        recent_id2 = db.mirror_message(session_id, "e3", "assistant", "also recent")

        gaps = db.find_uncovered_capture_ranges(lag_agent_id, session_id)

        assert gaps == [(recent_id1, recent_id2)]  # the ancient pair fell outside the scan
        assert old_id1 not in [i for r in gaps for i in r]
        assert old_id2 not in [i for r in gaps for i in r]
    finally:
        db_module._RECONCILIATION_SCAN_LIMIT = original_limit


def test_find_uncovered_message_ids_for_embedding_uses_the_bounded_scan_by_default(
    db_with_vector, session_id
):
    import minion_assist.session.db as db_module

    original_limit = db_module._RECONCILIATION_SCAN_LIMIT
    db_module._RECONCILIATION_SCAN_LIMIT = 2
    try:
        db_with_vector.upsert_session(session_id, "main")
        db_with_vector.mirror_message(session_id, "e0", "user", "ancient")
        db_with_vector.mirror_message(session_id, "e1", "assistant", "also ancient")
        recent_id1 = db_with_vector.mirror_message(session_id, "e2", "user", "recent")
        recent_id2 = db_with_vector.mirror_message(session_id, "e3", "assistant", "also recent")

        result = db_with_vector.find_uncovered_message_ids_for_embedding("main", session_id)

        assert result == [recent_id1, recent_id2]
    finally:
        db_module._RECONCILIATION_SCAN_LIMIT = original_limit


# ---------------------------------------------------------------------------
# Referential integrity — foreign keys and CHECK constraints (MEM-GAP-011)
#
# These exercise the constraints directly via raw SQL (bypassing
# delete_session()'s own manual cascade, and inserting invalid values
# directly) specifically to prove the *database* now enforces these
# invariants structurally — not just that the application's own code
# already behaves correctly, which the rest of this file already covers.
# ---------------------------------------------------------------------------

def test_deleting_a_session_row_directly_cascades_to_messages(db, session_id):
    db.upsert_session(session_id, "main")
    db.add_message(session_id, "user", "hello")

    db._conn().execute("DELETE FROM sessions WHERE id = %s", (session_id,))

    row = db._conn().execute(
        "SELECT count(*) FROM messages WHERE session_id = %s", (session_id,)
    ).fetchone()
    assert row[0] == 0


def test_deleting_a_session_row_directly_cascades_to_message_mirrors(db, session_id):
    db.upsert_session(session_id, "main")
    db.mirror_message(session_id, "e1", "user", "hello")

    db._conn().execute("DELETE FROM sessions WHERE id = %s", (session_id,))

    row = db._conn().execute(
        "SELECT count(*) FROM message_mirrors WHERE session_id = %s", (session_id,)
    ).fetchone()
    assert row[0] == 0


def test_deleting_a_message_row_directly_cascades_to_message_mirrors(db, session_id):
    db.upsert_session(session_id, "main")
    message_id = db.mirror_message(session_id, "e1", "user", "hello")

    db._conn().execute("DELETE FROM messages WHERE id = %s", (message_id,))

    row = db._conn().execute(
        "SELECT count(*) FROM message_mirrors WHERE message_id = %s", (message_id,)
    ).fetchone()
    assert row[0] == 0


def test_deleting_a_session_row_directly_cascades_to_capture_jobs(db, session_id):
    db.upsert_session(session_id, "main")
    db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")

    db._conn().execute("DELETE FROM sessions WHERE id = %s", (session_id,))

    row = db._conn().execute(
        "SELECT count(*) FROM memory_capture_jobs WHERE session_id = %s", (session_id,)
    ).fetchone()
    assert row[0] == 0


def test_deleting_a_capture_job_directly_cascades_to_proposals(db, session_id):
    db.upsert_session(session_id, "main")
    db.upsert_session(session_id, "main")  # MEM-GAP-011: FK now requires a real session row
    job_id = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    db.complete_capture_job(job_id, ["a claim"])

    db._conn().execute("DELETE FROM memory_capture_jobs WHERE id = %s", (job_id,))

    row = db._conn().execute(
        "SELECT count(*) FROM memory_proposals WHERE job_id = %s", (job_id,)
    ).fetchone()
    assert row[0] == 0


def test_deleting_a_session_row_directly_cascades_to_commitment_jobs(db, session_id):
    db.upsert_session(session_id, "main")
    db.enqueue_commitment_job(
        "main", session_id, "cli", 1, 2, idempotency_key=f"key-{session_id}"
    )

    db._conn().execute("DELETE FROM sessions WHERE id = %s", (session_id,))

    row = db._conn().execute(
        "SELECT count(*) FROM memory_commitment_jobs WHERE session_id = %s", (session_id,)
    ).fetchone()
    assert row[0] == 0


def test_deleting_a_session_row_directly_cascades_to_commitments(db, session_id):
    db.upsert_session(session_id, "main")
    _create_commitment(db, session_id)

    db._conn().execute("DELETE FROM sessions WHERE id = %s", (session_id,))

    row = db._conn().execute(
        "SELECT count(*) FROM commitments WHERE session_id = %s", (session_id,)
    ).fetchone()
    assert row[0] == 0


def test_deleting_a_commitment_job_directly_detaches_but_preserves_commitments(db, session_id):
    # R2-GAP-001: fk_commitments_job is ON DELETE SET NULL, not CASCADE — a
    # commitment must survive its originating job being deleted (e.g. by
    # retention pruning), only losing the now-meaningless back-pointer.
    db.upsert_session(session_id, "main")
    commitment = _create_commitment(db, session_id)
    source_job_id = db._conn().execute(
        "SELECT source_job_id FROM commitments WHERE id = %s", (commitment["id"],)
    ).fetchone()[0]

    db._conn().execute("DELETE FROM memory_commitment_jobs WHERE id = %s", (source_job_id,))

    row = db._conn().execute(
        "SELECT source_job_id FROM commitments WHERE id = %s", (commitment["id"],)
    ).fetchone()
    assert row is not None
    assert row[0] is None


def test_deleting_a_session_row_directly_cascades_to_message_embedding_jobs(db, session_id):
    db.upsert_session(session_id, "main")
    message_id = db.mirror_message(session_id, "e1", "user", "hello")
    db.enqueue_message_embedding_job(
        "main", session_id, message_id, idempotency_key=f"emb-{session_id}"
    )

    db._conn().execute("DELETE FROM sessions WHERE id = %s", (session_id,))

    row = db._conn().execute(
        "SELECT count(*) FROM message_embedding_jobs WHERE session_id = %s", (session_id,)
    ).fetchone()
    assert row[0] == 0


def test_deleting_a_message_row_directly_cascades_to_message_embeddings(db_with_vector, session_id):
    db_with_vector.upsert_session(session_id, "main")
    message_id = db_with_vector.mirror_message(session_id, "e1", "user", "hello")
    db_with_vector.complete_message_embedding_job(
        db_with_vector.enqueue_message_embedding_job(
            "main", session_id, message_id, idempotency_key=f"key-{session_id}"
        ),
        message_id, "test-endpoint::test-model", [0.1, 0.2, 0.3],
    )

    db_with_vector._conn().execute("DELETE FROM messages WHERE id = %s", (message_id,))

    row = db_with_vector._conn().execute(
        "SELECT count(*) FROM message_embeddings WHERE message_id = %s", (message_id,)
    ).fetchone()
    assert row[0] == 0


# --- CHECK constraints on enumerated state/status columns ---

def _assert_check_violation(db, sql: str, params: tuple) -> None:
    import psycopg

    with pytest.raises(psycopg.errors.CheckViolation):
        db._conn().execute(sql, params)


def test_memory_capture_jobs_rejects_an_unrecognized_state(db, session_id):
    _assert_check_violation(
        db,
        """
        INSERT INTO memory_capture_jobs
            (agent_id, session_id, source_from_message_id, source_to_message_id,
             idempotency_key, state, attempts, run_after, created_at, updated_at)
        VALUES ('main', %s, 1, 2, %s, 'bogus', 0, 0, 0, 0)
        """,
        (session_id, f"key-{session_id}"),
    )


def test_memory_commitment_jobs_rejects_an_unrecognized_state(db, session_id):
    _assert_check_violation(
        db,
        """
        INSERT INTO memory_commitment_jobs
            (agent_id, session_id, channel, source_from_message_id, source_to_message_id,
             idempotency_key, state, attempts, run_after, created_at, updated_at)
        VALUES ('main', %s, 'cli', 1, 2, %s, 'bogus', 0, 0, 0, 0)
        """,
        (session_id, f"key-{session_id}"),
    )


def test_message_embedding_jobs_rejects_an_unrecognized_state(db, session_id):
    _assert_check_violation(
        db,
        """
        INSERT INTO message_embedding_jobs
            (agent_id, session_id, message_id, idempotency_key, state,
             attempts, run_after, created_at, updated_at)
        VALUES ('main', %s, 1, %s, 'bogus', 0, 0, 0, 0)
        """,
        (session_id, f"key-{session_id}"),
    )


def test_memory_proposals_rejects_an_unrecognized_status(db, session_id):
    db.upsert_session(session_id, "main")
    db.upsert_session(session_id, "main")  # MEM-GAP-011: FK now requires a real session row
    job_id = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    _assert_check_violation(
        db,
        "INSERT INTO memory_proposals (job_id, session_id, agent_id, claim_text, created_at, status) "
        "VALUES (%s, %s, 'main', 'x', 0, 'bogus')",
        (job_id, session_id),
    )


def test_commitments_rejects_an_unrecognized_status(db, session_id):
    _assert_check_violation(
        db,
        """
        INSERT INTO commitments
            (agent_id, session_id, channel, kind, sensitivity, source, status, reason,
             suggested_text, dedupe_key, confidence, due_earliest, due_latest,
             source_job_id, created_at, updated_at)
        VALUES ('main', %s, 'cli', 'k', 's', 'src', 'bogus', 'r', 't', 'd', 0.5, 0, 0, 1, 0, 0)
        """,
        (session_id,),
    )


def test_memory_proposals_accepts_every_documented_status(db, session_id):
    db.upsert_session(session_id, "main")
    db.upsert_session(session_id, "main")  # MEM-GAP-011: FK now requires a real session row
    job_id = db.enqueue_capture_job("main", session_id, 1, 2, idempotency_key=f"key-{session_id}")
    for status in ("pending", "promoted", "rejected", "superseded"):
        db._conn().execute(
            "INSERT INTO memory_proposals (job_id, session_id, agent_id, claim_text, created_at, status) "
            "VALUES (%s, %s, 'main', %s, 0, %s)",
            (job_id, session_id, f"claim-{status}", status),
        )  # must not raise for any of these


def test_commitments_accepts_every_documented_status(db, session_id):
    db.upsert_session(session_id, "main")
    job_id = db.enqueue_commitment_job(
        "main", session_id, "cli", 1, 2, idempotency_key=f"key-{session_id}"
    )
    for status in ("pending", "sent", "dismissed", "expired"):
        db._conn().execute(
            """
            INSERT INTO commitments
                (agent_id, session_id, channel, kind, sensitivity, source, status, reason,
                 suggested_text, dedupe_key, confidence, due_earliest, due_latest,
                 source_job_id, created_at, updated_at)
            VALUES ('main', %s, 'cli', 'k', 's', 'src', %s, 'r', 't', %s, 0.5, 0, 0, %s, 0, 0)
            """,
            (session_id, status, f"dedupe-{status}", job_id),
        )  # must not raise for any of these

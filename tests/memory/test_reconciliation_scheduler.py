"""Tests for memory/reconciliation_scheduler.py: ReconciliationScheduler (MEM-GAP-007)."""

from __future__ import annotations

import time
from unittest.mock import Mock

from minion_assist.config import MemoryReconciliationConfig
from minion_assist.memory.reconciliation_scheduler import ReconciliationScheduler
from minion_assist.worker_health import WorkerHealth


def _make_cfg(**kwargs) -> MemoryReconciliationConfig:
    defaults = dict(interval_seconds=300, quiet_seconds=60)
    defaults.update(kwargs)
    return MemoryReconciliationConfig(**defaults)


def _session_info(session_id: str, last_active: float | None) -> dict:
    return {"id": session_id, "agent_id": "main", "title": None,
            "started_at": 0.0, "last_active": last_active, "turn_count": 1}


# ---------------------------------------------------------------------------
# start / stop lifecycle
# ---------------------------------------------------------------------------

def test_start_creates_a_daemon_timer():
    scheduler = ReconciliationScheduler(_make_cfg(), Mock(), ["main"], Mock())
    scheduler.start()
    assert scheduler._timer is not None
    assert scheduler._timer.daemon is True
    scheduler.stop()


def test_stop_cancels_the_timer():
    scheduler = ReconciliationScheduler(_make_cfg(), Mock(), ["main"], Mock())
    scheduler.start()
    scheduler.stop()
    assert scheduler._stopped is True


# ---------------------------------------------------------------------------
# _run_pass — mirror reconciliation
# ---------------------------------------------------------------------------

def test_run_pass_calls_reconcile_all_sessions_with_every_configured_agent():
    db = Mock()
    db.list_session_ids_for_agent.return_value = []
    short_term = Mock()
    scheduler = ReconciliationScheduler(_make_cfg(), db, ["main", "researcher"], short_term)

    scheduler._run_pass()

    db.reconcile_all_sessions.assert_called_once_with(short_term, ["main", "researcher"])


# ---------------------------------------------------------------------------
# _run_pass — job-coverage catch-up
# ---------------------------------------------------------------------------

def test_run_pass_skips_a_session_within_the_quiet_period():
    db = Mock()
    now = time.time()
    db.list_session_ids_for_agent.return_value = ["s1"]
    db.get_sessions_by_ids.return_value = {"s1": _session_info("s1", now)}  # active right now
    scheduler = ReconciliationScheduler(_make_cfg(quiet_seconds=60), db, ["main"], Mock())

    scheduler._run_pass()

    db.find_uncovered_capture_range.assert_not_called()
    db.find_uncovered_commitment_range.assert_not_called()


def test_run_pass_reconciles_a_session_past_the_quiet_period():
    db = Mock()
    old = time.time() - 3600
    db.list_session_ids_for_agent.return_value = ["s1"]
    db.get_sessions_by_ids.return_value = {"s1": _session_info("s1", old)}
    db.find_uncovered_capture_range.return_value = None
    db.find_uncovered_commitment_range.return_value = None
    db.has_vector_lane = False
    scheduler = ReconciliationScheduler(_make_cfg(quiet_seconds=60), db, ["main"], Mock())

    scheduler._run_pass()

    db.find_uncovered_capture_range.assert_called_once_with("main", "s1")
    db.find_uncovered_commitment_range.assert_called_once_with("main", "s1")


def test_run_pass_reconciles_a_session_that_has_never_been_active():
    """last_active=None (shouldn't happen in practice, but must not crash/skip wrongly)."""
    db = Mock()
    db.list_session_ids_for_agent.return_value = ["s1"]
    db.get_sessions_by_ids.return_value = {"s1": _session_info("s1", None)}
    db.find_uncovered_capture_range.return_value = None
    db.find_uncovered_commitment_range.return_value = None
    db.has_vector_lane = False
    scheduler = ReconciliationScheduler(_make_cfg(), db, ["main"], Mock())

    scheduler._run_pass()  # must not raise

    db.find_uncovered_capture_range.assert_called_once_with("main", "s1")


def test_run_pass_skips_agents_with_no_sessions():
    db = Mock()
    db.list_session_ids_for_agent.return_value = []
    scheduler = ReconciliationScheduler(_make_cfg(), db, ["main"], Mock())

    scheduler._run_pass()  # must not raise

    db.get_sessions_by_ids.assert_not_called()


# ---------------------------------------------------------------------------
# _catch_up_capture / _catch_up_commitment
# ---------------------------------------------------------------------------

def test_catch_up_capture_enqueues_a_job_for_the_gap():
    db = Mock()
    db.find_uncovered_capture_range.return_value = (10, 12)
    scheduler = ReconciliationScheduler(_make_cfg(), db, ["main"], Mock())

    scheduler._catch_up_capture("main", "sess-1")

    db.enqueue_capture_job.assert_called_once()
    call_args = db.enqueue_capture_job.call_args.args
    assert call_args[0] == "main"
    assert call_args[1] == "sess-1"
    assert call_args[2] == 10
    assert call_args[3] == 12
    assert "reconcile" in call_args[4]  # idempotency key


def test_catch_up_capture_does_nothing_when_fully_covered():
    db = Mock()
    db.find_uncovered_capture_range.return_value = None
    scheduler = ReconciliationScheduler(_make_cfg(), db, ["main"], Mock())

    scheduler._catch_up_capture("main", "sess-1")

    db.enqueue_capture_job.assert_not_called()


def test_catch_up_commitment_enqueues_a_job_for_the_gap_with_the_right_channel():
    db = Mock()
    db.find_uncovered_commitment_range.return_value = ("!room:example.org", 10, 12)
    scheduler = ReconciliationScheduler(_make_cfg(), db, ["main"], Mock())

    scheduler._catch_up_commitment("main", "sess-1")

    db.enqueue_commitment_job.assert_called_once()
    call_args = db.enqueue_commitment_job.call_args.args
    assert call_args[0] == "main"
    assert call_args[1] == "sess-1"
    assert call_args[2] == "!room:example.org"
    assert call_args[3] == 10
    assert call_args[4] == 12
    assert "reconcile" in call_args[5]  # idempotency key


def test_catch_up_commitment_does_nothing_when_fully_covered():
    db = Mock()
    db.find_uncovered_commitment_range.return_value = None
    scheduler = ReconciliationScheduler(_make_cfg(), db, ["main"], Mock())

    scheduler._catch_up_commitment("main", "sess-1")

    db.enqueue_commitment_job.assert_not_called()


# ---------------------------------------------------------------------------
# _catch_up_message_embedding (MEM-GAP-006)
# ---------------------------------------------------------------------------

def test_catch_up_message_embedding_does_nothing_without_a_vector_lane():
    db = Mock()
    db.has_vector_lane = False
    scheduler = ReconciliationScheduler(_make_cfg(), db, ["main"], Mock())

    scheduler._catch_up_message_embedding("main", "sess-1")

    db.find_uncovered_message_ids_for_embedding.assert_not_called()
    db.enqueue_message_embedding_job.assert_not_called()


def test_catch_up_message_embedding_enqueues_one_job_per_missing_id():
    db = Mock()
    db.has_vector_lane = True
    db.embedding_model_identity = "test-endpoint::test-model"
    db.find_uncovered_message_ids_for_embedding.return_value = [10, 11]
    scheduler = ReconciliationScheduler(_make_cfg(), db, ["main"], Mock())

    scheduler._catch_up_message_embedding("main", "sess-1")

    assert db.enqueue_message_embedding_job.call_count == 2
    first_call, second_call = db.enqueue_message_embedding_job.call_args_list
    assert first_call.args == ("main", "sess-1", 10, "main:10:test-endpoint::test-model")
    assert second_call.args == ("main", "sess-1", 11, "main:11:test-endpoint::test-model")


def test_catch_up_message_embedding_does_nothing_when_fully_covered():
    db = Mock()
    db.has_vector_lane = True
    db.embedding_model_identity = "test-endpoint::test-model"
    db.find_uncovered_message_ids_for_embedding.return_value = []
    scheduler = ReconciliationScheduler(_make_cfg(), db, ["main"], Mock())

    scheduler._catch_up_message_embedding("main", "sess-1")

    db.enqueue_message_embedding_job.assert_not_called()


# ---------------------------------------------------------------------------
# _fire / WorkerHealth wiring (MEM-GAP-016)
# ---------------------------------------------------------------------------

def test_fire_records_a_poll_and_success(monkeypatch):
    health = WorkerHealth("memory_reconciliation")
    scheduler = ReconciliationScheduler(_make_cfg(), Mock(), ["main"], Mock(), health=health)
    monkeypatch.setattr(scheduler, "_run_pass", Mock())
    monkeypatch.setattr(scheduler, "start", Mock())

    scheduler._fire()

    snap = health.snapshot()
    assert snap["last_poll_at"] is not None
    assert snap["last_success_at"] is not None


def test_fire_records_failure_on_error(monkeypatch):
    health = WorkerHealth("memory_reconciliation")
    scheduler = ReconciliationScheduler(_make_cfg(), Mock(), ["main"], Mock(), health=health)
    monkeypatch.setattr(scheduler, "_run_pass", Mock(side_effect=RuntimeError("db down")))
    monkeypatch.setattr(scheduler, "start", Mock())

    scheduler._fire()  # must not raise

    snap = health.snapshot()
    assert snap["consecutive_failures"] == 1
    assert "db down" in snap["last_error"]


def test_fire_reschedules_even_after_an_error(monkeypatch):
    scheduler = ReconciliationScheduler(_make_cfg(), Mock(), ["main"], Mock())
    monkeypatch.setattr(scheduler, "_run_pass", Mock(side_effect=RuntimeError("boom")))
    restart_called = Mock()
    monkeypatch.setattr(scheduler, "start", restart_called)

    scheduler._fire()

    restart_called.assert_called_once()


def test_fire_does_nothing_once_stopped(monkeypatch):
    scheduler = ReconciliationScheduler(_make_cfg(), Mock(), ["main"], Mock())
    scheduler._stopped = True
    run_pass = Mock()
    monkeypatch.setattr(scheduler, "_run_pass", run_pass)

    scheduler._fire()

    run_pass.assert_not_called()


# ---------------------------------------------------------------------------
# _fire against a REAL SessionDB failure (MEM-GAP-018)
# ---------------------------------------------------------------------------
# The two _fire tests above only prove the wiring works when _run_pass
# itself is monkeypatched to raise — they never exercise a genuine failure
# produced by real production code (SessionDB's own connection handling).
# This test uses a real SessionDB successfully constructed against the dev
# database, then points its URL at an unreachable port so the *next* real
# query genuinely fails via psycopg's own driver — not a mocked
# side_effect — proving WorkerHealth captures an authentic production
# failure end-to-end. Skipped, not failed, without a reachable dev DB.

import pytest

_DB_URL = "postgresql://minion:minion@localhost:5433/minion_assist"

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


@_requires_live_db
def test_fire_records_a_genuine_db_connection_failure():
    from minion_assist.session.db import SessionDB

    real_db = SessionDB(_DB_URL)  # succeeds against the real dev DB
    # Force the thread-local connection closed, then point the URL at a
    # port nothing listens on — the next real query inside _run_pass()
    # (SessionDB.reconcile_all_sessions -> ... -> self._conn()) genuinely
    # fails to reconnect, raising psycopg's own OperationalError.
    real_db._conn().close()
    real_db._url = "postgresql://minion:minion@localhost:1/minion_assist"

    health = WorkerHealth("memory_reconciliation")
    scheduler = ReconciliationScheduler(
        _make_cfg(), real_db, ["main"], Mock(), health=health,
    )

    try:
        scheduler._fire()  # must not raise — _fire()'s own except must catch it
    finally:
        scheduler.stop()  # cancel the timer _fire() reschedules in its finally block

    snap = health.snapshot()
    assert snap["last_poll_at"] is not None
    assert snap["consecutive_failures"] == 1
    assert snap["last_error"]  # a genuine driver error message, not empty

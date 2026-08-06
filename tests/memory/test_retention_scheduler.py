"""Tests for memory/retention_scheduler.py: MemoryRetentionScheduler (MEM-GAP-015)."""

from __future__ import annotations

from unittest.mock import Mock

from minion_assist.config import MemoryRetentionConfig
from minion_assist.memory.retention_scheduler import MemoryRetentionScheduler
from minion_assist.worker_health import WorkerHealth


def _make_cfg(**kwargs) -> MemoryRetentionConfig:
    defaults = dict(enabled=True, hour=4, minute=45, timezone="UTC", retention_days=30)
    defaults.update(kwargs)
    return MemoryRetentionConfig(**defaults)


def _db(counts: dict | None = None) -> Mock:
    db = Mock()
    db.prune_operational_tables.return_value = counts or {
        "capture_jobs": 0, "commitment_jobs": 0, "message_embedding_jobs": 0,
    }
    return db


# ---------------------------------------------------------------------------
# start / stop lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_start_creates_a_daemon_timer(self):
        scheduler = MemoryRetentionScheduler(_make_cfg(), _db(), None)
        scheduler.start()
        assert scheduler._timer is not None
        assert scheduler._timer.daemon is True
        scheduler.stop()

    def test_stop_cancels_the_timer(self):
        scheduler = MemoryRetentionScheduler(_make_cfg(), _db(), None)
        scheduler.start()
        scheduler.stop()
        assert scheduler._stopped is True


# ---------------------------------------------------------------------------
# _run_pass
# ---------------------------------------------------------------------------

class TestRunPass:
    def test_prunes_the_db_with_the_configured_retention_days(self):
        db = _db()
        scheduler = MemoryRetentionScheduler(_make_cfg(retention_days=45), db, None)

        scheduler._run_pass()

        db.prune_operational_tables.assert_called_once_with(45)

    def test_skips_the_index_half_when_no_index_is_configured(self):
        db = _db()
        scheduler = MemoryRetentionScheduler(_make_cfg(), db, None)

        scheduler._run_pass()  # must not raise despite index=None

    def test_prunes_the_index_when_one_is_configured(self):
        db = _db()
        index = Mock()
        index.prune_operational_tables.return_value = {
            "recall_events": 0, "consolidation_previews": 0, "import_previews": 0,
        }
        scheduler = MemoryRetentionScheduler(_make_cfg(retention_days=10), db, index)

        scheduler._run_pass()

        index.prune_operational_tables.assert_called_once_with(10)

    def test_merges_db_and_index_counts(self):
        db = _db({"capture_jobs": 2, "commitment_jobs": 1, "message_embedding_jobs": 3})
        index = Mock()
        index.prune_operational_tables.return_value = {
            "recall_events": 5, "consolidation_previews": 0, "import_previews": 1,
        }
        health = WorkerHealth("memory_retention")
        scheduler = MemoryRetentionScheduler(_make_cfg(), db, index, health=health)

        scheduler._run_pass()

        assert health.snapshot()["last_success_at"] is not None


# ---------------------------------------------------------------------------
# _fire
# ---------------------------------------------------------------------------

class TestFire:
    def test_fire_reschedules_after_a_successful_pass(self, monkeypatch):
        scheduler = MemoryRetentionScheduler(_make_cfg(), _db(), None)
        monkeypatch.setattr(scheduler, "_run_pass", Mock())
        restart_called = Mock()
        monkeypatch.setattr(scheduler, "start", restart_called)

        scheduler._fire()

        restart_called.assert_called_once()

    def test_fire_reschedules_even_after_an_error(self, monkeypatch):
        scheduler = MemoryRetentionScheduler(_make_cfg(), _db(), None)
        monkeypatch.setattr(scheduler, "_run_pass", Mock(side_effect=RuntimeError("boom")))
        restart_called = Mock()
        monkeypatch.setattr(scheduler, "start", restart_called)

        scheduler._fire()  # must not raise

        restart_called.assert_called_once()

    def test_fire_does_nothing_once_stopped(self, monkeypatch):
        scheduler = MemoryRetentionScheduler(_make_cfg(), _db(), None)
        scheduler._stopped = True
        run_pass = Mock()
        monkeypatch.setattr(scheduler, "_run_pass", run_pass)

        scheduler._fire()

        run_pass.assert_not_called()


# ---------------------------------------------------------------------------
# WorkerHealth wiring (MEM-GAP-016)
# ---------------------------------------------------------------------------

class TestWorkerHealth:
    def test_fire_records_a_poll(self, monkeypatch):
        health = WorkerHealth("memory_retention")
        scheduler = MemoryRetentionScheduler(_make_cfg(), _db(), None, health=health)
        monkeypatch.setattr(scheduler, "_run_pass", Mock())
        monkeypatch.setattr(scheduler, "start", Mock())

        scheduler._fire()

        assert health.snapshot()["last_poll_at"] is not None

    def test_fire_records_failure_on_error(self, monkeypatch):
        health = WorkerHealth("memory_retention")
        scheduler = MemoryRetentionScheduler(_make_cfg(), _db(), None, health=health)
        monkeypatch.setattr(scheduler, "_run_pass", Mock(side_effect=RuntimeError("boom")))
        monkeypatch.setattr(scheduler, "start", Mock())

        scheduler._fire()

        snap = health.snapshot()
        assert snap["consecutive_failures"] == 1
        assert "boom" in snap["last_error"]

    def test_run_pass_records_success(self):
        health = WorkerHealth("memory_retention")
        scheduler = MemoryRetentionScheduler(_make_cfg(), _db(), None, health=health)

        scheduler._run_pass()

        assert health.snapshot()["last_success_at"] is not None

"""Tests for memory/consolidation_scheduler.py: MemoryConsolidationScheduler (Stage One Phase 5, slice D)."""

from __future__ import annotations

from unittest.mock import Mock

from minion_assist.config import MemoryConsolidationConfig
from minion_assist.memory.consolidation_scheduler import MemoryConsolidationScheduler
from minion_assist.worker_health import WorkerHealth


def _make_cfg(**kwargs) -> MemoryConsolidationConfig:
    defaults = dict(enabled=True, hour=4, minute=0, timezone="UTC", agent_id="main", top_n=5)
    defaults.update(kwargs)
    return MemoryConsolidationConfig(**defaults)


def _proposal(**overrides) -> dict:
    base = {"id": 1, "claim_text": "User prefers dark mode.", "score": 5}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# start / stop lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_start_creates_a_daemon_timer(self):
        scheduler = MemoryConsolidationScheduler(_make_cfg(), Mock(), Mock(), Mock())
        scheduler.start()
        assert scheduler._timer is not None
        assert scheduler._timer.daemon is True
        scheduler.stop()

    def test_stop_cancels_the_timer(self):
        scheduler = MemoryConsolidationScheduler(_make_cfg(), Mock(), Mock(), Mock())
        scheduler.start()
        scheduler.stop()
        assert scheduler._stopped is True


# ---------------------------------------------------------------------------
# _run_pass
# ---------------------------------------------------------------------------

class TestRunPass:
    def test_drafts_a_preview_for_each_never_previewed_pending_proposal(self, monkeypatch):
        import minion_assist.memory.consolidation as consolidation

        db = Mock()
        index = Mock()
        index.list_consolidation_previews.return_value = []  # never previewed
        consolidator = Mock()
        monkeypatch.setattr(
            consolidation, "rank_proposals",
            lambda db_, index_, agent_id: [_proposal(id=1), _proposal(id=2)],
        )
        scheduler = MemoryConsolidationScheduler(_make_cfg(), db, index, consolidator)

        scheduler._run_pass()

        assert consolidator.preview.call_args_list == [((1,),), ((2,),)]

    def test_skips_a_proposal_that_already_has_a_preview(self, monkeypatch):
        import minion_assist.memory.consolidation as consolidation

        db = Mock()
        index = Mock()
        index.list_consolidation_previews.side_effect = (
            lambda agent_id, proposal_id: [{"id": 99}] if proposal_id == 1 else []
        )
        consolidator = Mock()
        monkeypatch.setattr(
            consolidation, "rank_proposals",
            lambda db_, index_, agent_id: [_proposal(id=1), _proposal(id=2)],
        )
        scheduler = MemoryConsolidationScheduler(_make_cfg(), db, index, consolidator)

        scheduler._run_pass()

        consolidator.preview.assert_called_once_with(2)

    def test_stops_at_top_n_new_drafts(self, monkeypatch):
        import minion_assist.memory.consolidation as consolidation

        db = Mock()
        index = Mock()
        index.list_consolidation_previews.return_value = []
        consolidator = Mock()
        monkeypatch.setattr(
            consolidation, "rank_proposals",
            lambda db_, index_, agent_id: [_proposal(id=i) for i in range(1, 6)],
        )
        scheduler = MemoryConsolidationScheduler(_make_cfg(top_n=2), db, index, consolidator)

        scheduler._run_pass()

        assert consolidator.preview.call_count == 2

    def test_a_single_drafting_failure_does_not_abort_the_rest_of_the_pass(self, monkeypatch):
        import minion_assist.memory.consolidation as consolidation

        db = Mock()
        index = Mock()
        index.list_consolidation_previews.return_value = []
        consolidator = Mock()
        consolidator.preview.side_effect = [RuntimeError("boom"), None]
        monkeypatch.setattr(
            consolidation, "rank_proposals",
            lambda db_, index_, agent_id: [_proposal(id=1), _proposal(id=2)],
        )
        scheduler = MemoryConsolidationScheduler(_make_cfg(), db, index, consolidator)

        scheduler._run_pass()  # must not raise

        assert consolidator.preview.call_count == 2

    def test_a_failed_draft_does_not_count_toward_top_n(self, monkeypatch):
        # A failed attempt produced no new preview, so it shouldn't consume
        # the per-run drafting budget — otherwise a run full of failures
        # would silently draft nothing at all.
        import minion_assist.memory.consolidation as consolidation

        db = Mock()
        index = Mock()
        index.list_consolidation_previews.return_value = []
        consolidator = Mock()
        consolidator.preview.side_effect = [RuntimeError("boom"), None]
        monkeypatch.setattr(
            consolidation, "rank_proposals",
            lambda db_, index_, agent_id: [_proposal(id=1), _proposal(id=2)],
        )
        scheduler = MemoryConsolidationScheduler(_make_cfg(top_n=1), db, index, consolidator)

        scheduler._run_pass()

        assert consolidator.preview.call_count == 2

    def test_with_no_pending_proposals_drafts_nothing(self, monkeypatch):
        import minion_assist.memory.consolidation as consolidation

        db = Mock()
        index = Mock()
        consolidator = Mock()
        monkeypatch.setattr(consolidation, "rank_proposals", lambda db_, index_, agent_id: [])
        scheduler = MemoryConsolidationScheduler(_make_cfg(), db, index, consolidator)

        scheduler._run_pass()  # must not raise

        consolidator.preview.assert_not_called()


# ---------------------------------------------------------------------------
# _fire
# ---------------------------------------------------------------------------

class TestFire:
    def test_fire_reschedules_after_a_successful_pass(self, monkeypatch):
        scheduler = MemoryConsolidationScheduler(_make_cfg(), Mock(), Mock(), Mock())
        monkeypatch.setattr(scheduler, "_run_pass", Mock())
        restart_called = Mock()
        monkeypatch.setattr(scheduler, "start", restart_called)

        scheduler._fire()

        restart_called.assert_called_once()

    def test_fire_reschedules_even_after_an_error(self, monkeypatch):
        scheduler = MemoryConsolidationScheduler(_make_cfg(), Mock(), Mock(), Mock())
        monkeypatch.setattr(scheduler, "_run_pass", Mock(side_effect=RuntimeError("boom")))
        restart_called = Mock()
        monkeypatch.setattr(scheduler, "start", restart_called)

        scheduler._fire()  # must not raise

        restart_called.assert_called_once()

    def test_fire_does_nothing_once_stopped(self, monkeypatch):
        scheduler = MemoryConsolidationScheduler(_make_cfg(), Mock(), Mock(), Mock())
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
        health = WorkerHealth("memory_consolidation")
        scheduler = MemoryConsolidationScheduler(_make_cfg(), Mock(), Mock(), Mock(), health=health)
        monkeypatch.setattr(scheduler, "_run_pass", Mock())
        monkeypatch.setattr(scheduler, "start", Mock())

        scheduler._fire()

        assert health.snapshot()["last_poll_at"] is not None

    def test_fire_records_failure_on_a_whole_pass_error(self, monkeypatch):
        health = WorkerHealth("memory_consolidation")
        scheduler = MemoryConsolidationScheduler(_make_cfg(), Mock(), Mock(), Mock(), health=health)
        monkeypatch.setattr(scheduler, "_run_pass", Mock(side_effect=RuntimeError("db down")))
        monkeypatch.setattr(scheduler, "start", Mock())

        scheduler._fire()

        snap = health.snapshot()
        assert snap["consecutive_failures"] == 1
        assert "db down" in snap["last_error"]

    def test_run_pass_records_success_per_drafted_proposal(self, monkeypatch):
        import minion_assist.memory.consolidation as consolidation

        db = Mock()
        index = Mock()
        index.list_consolidation_previews.return_value = []
        consolidator = Mock()
        monkeypatch.setattr(
            consolidation, "rank_proposals",
            lambda db_, index_, agent_id: [_proposal(id=1), _proposal(id=2)],
        )
        health = WorkerHealth("memory_consolidation")
        scheduler = MemoryConsolidationScheduler(
            _make_cfg(top_n=5), db, index, consolidator, health=health
        )

        scheduler._run_pass()

        assert health.snapshot()["last_success_at"] is not None

    def test_run_pass_records_failure_for_a_single_proposals_error(self, monkeypatch):
        import minion_assist.memory.consolidation as consolidation

        db = Mock()
        index = Mock()
        index.list_consolidation_previews.return_value = []
        consolidator = Mock()
        consolidator.preview.side_effect = RuntimeError("malformed response")
        monkeypatch.setattr(
            consolidation, "rank_proposals",
            lambda db_, index_, agent_id: [_proposal(id=1)],
        )
        health = WorkerHealth("memory_consolidation")
        scheduler = MemoryConsolidationScheduler(
            _make_cfg(top_n=5), db, index, consolidator, health=health
        )

        scheduler._run_pass()

        snap = health.snapshot()
        assert snap["consecutive_failures"] == 1
        assert "malformed response" in snap["last_error"]

"""Tests for memory/digest_scheduler.py: KnowledgeDigestScheduler (Stage One Phase 7, slice D)."""

from __future__ import annotations

from unittest.mock import Mock

from minion_assist.config import KnowledgeDigestConfig
from minion_assist.memory.digest_scheduler import KnowledgeDigestScheduler


def _make_cfg(**kwargs) -> KnowledgeDigestConfig:
    defaults = dict(
        enabled=True, hour=4, minute=30, timezone="UTC", agent_id="main", max_chars=8000
    )
    defaults.update(kwargs)
    return KnowledgeDigestConfig(**defaults)


def _claim(**overrides) -> dict:
    base = {"id": "c-1", "rel_path": "topics/x.md", "text": "A fact.", "status": "supported"}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# start / stop lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_start_creates_a_daemon_timer(self):
        scheduler = KnowledgeDigestScheduler(_make_cfg(), Mock(), Mock())
        scheduler.start()
        assert scheduler._timer is not None
        assert scheduler._timer.daemon is True
        scheduler.stop()

    def test_stop_cancels_the_timer(self):
        scheduler = KnowledgeDigestScheduler(_make_cfg(), Mock(), Mock())
        scheduler.start()
        scheduler.stop()
        assert scheduler._stopped is True


# ---------------------------------------------------------------------------
# _run_pass
# ---------------------------------------------------------------------------

class TestRunPass:
    def test_fetches_only_supported_claims_for_the_configured_agent(self):
        index = Mock()
        index.list_claims.return_value = []
        files = Mock()
        scheduler = KnowledgeDigestScheduler(_make_cfg(agent_id="researcher"), index, files)

        scheduler._run_pass()

        index.list_claims.assert_called_once_with("researcher", status="supported")

    def test_writes_the_compiled_digest(self):
        index = Mock()
        index.list_claims.return_value = [_claim(text="Alice prefers dark mode.")]
        files = Mock()
        scheduler = KnowledgeDigestScheduler(_make_cfg(), index, files)

        scheduler._run_pass()

        files.write_digest.assert_called_once()
        (written,), _ = files.write_digest.call_args
        assert "Alice prefers dark mode." in written

    def test_writes_an_empty_digest_when_there_are_no_supported_claims(self):
        index = Mock()
        index.list_claims.return_value = []
        files = Mock()
        scheduler = KnowledgeDigestScheduler(_make_cfg(), index, files)

        scheduler._run_pass()

        files.write_digest.assert_called_once_with("")

    def test_passes_max_chars_through_to_compile_digest(self, monkeypatch):
        import minion_assist.memory.digest_scheduler as digest_scheduler

        index = Mock()
        index.list_claims.return_value = [_claim()]
        files = Mock()
        captured = {}

        def _fake_compile(claims, max_chars):
            captured["max_chars"] = max_chars
            return "digest"

        monkeypatch.setattr(digest_scheduler, "compile_digest", _fake_compile)
        scheduler = KnowledgeDigestScheduler(_make_cfg(max_chars=123), index, files)

        scheduler._run_pass()

        assert captured["max_chars"] == 123


# ---------------------------------------------------------------------------
# _fire
# ---------------------------------------------------------------------------

class TestFire:
    def test_fire_reschedules_after_a_successful_pass(self, monkeypatch):
        scheduler = KnowledgeDigestScheduler(_make_cfg(), Mock(), Mock())
        monkeypatch.setattr(scheduler, "_run_pass", Mock())
        restart_called = Mock()
        monkeypatch.setattr(scheduler, "start", restart_called)

        scheduler._fire()

        restart_called.assert_called_once()

    def test_fire_reschedules_even_after_an_error(self, monkeypatch):
        scheduler = KnowledgeDigestScheduler(_make_cfg(), Mock(), Mock())
        monkeypatch.setattr(scheduler, "_run_pass", Mock(side_effect=RuntimeError("boom")))
        restart_called = Mock()
        monkeypatch.setattr(scheduler, "start", restart_called)

        scheduler._fire()  # must not raise

        restart_called.assert_called_once()

    def test_fire_does_nothing_once_stopped(self, monkeypatch):
        scheduler = KnowledgeDigestScheduler(_make_cfg(), Mock(), Mock())
        scheduler._stopped = True
        run_pass = Mock()
        monkeypatch.setattr(scheduler, "_run_pass", run_pass)

        scheduler._fire()

        run_pass.assert_not_called()

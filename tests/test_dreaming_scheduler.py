"""Tests for DreamingScheduler and dreaming helper functions."""

from __future__ import annotations

import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from minion_assist.config import DreamingConfig
from minion_assist.dreaming import (
    DREAM_SYSTEM_PROMPT,
    DreamingScheduler,
    _build_dream_prompt,
    _read_daily_snippets,
    _read_dream_bootstrap,
    _read_recent_diary_entries,
    _seconds_until_next,
)
from minion_assist.worker_health import WorkerHealth


# ---------------------------------------------------------------------------
# _seconds_until_next
# ---------------------------------------------------------------------------


class TestSecondsUntilNext:
    def test_returns_positive_float(self) -> None:
        delay = _seconds_until_next(hour=3, minute=0, tz_name="UTC")
        assert isinstance(delay, float)
        assert delay > 0

    def test_returns_at_most_one_day(self) -> None:
        delay = _seconds_until_next(hour=3, minute=0, tz_name="UTC")
        assert delay <= 86400

    def test_target_in_future_returns_small_delay(self) -> None:
        # If we schedule 1 minute in the future from UTC midnight, delay < 60 + epsilon.
        now_utc = datetime.now(timezone.utc)
        future = now_utc + timedelta(minutes=1)
        delay = _seconds_until_next(future.hour, future.minute, "UTC")
        # Allow ±2s for clock skew between the test and implementation.
        assert delay <= 62

    def test_target_in_past_returns_next_day_delay(self) -> None:
        # Past time → delay should be around 23+ hours.
        now_utc = datetime.now(timezone.utc)
        past = now_utc - timedelta(minutes=1)
        delay = _seconds_until_next(past.hour, past.minute, "UTC")
        # Should be nearly a full day ahead.
        assert delay > 23 * 3600

    def test_invalid_timezone_falls_back_to_utc(self) -> None:
        # Should not raise; logs a warning and uses UTC fallback.
        delay = _seconds_until_next(hour=3, minute=0, tz_name="Invalid/Timezone")
        assert delay > 0


# ---------------------------------------------------------------------------
# _read_dream_bootstrap
# ---------------------------------------------------------------------------


class TestReadDreamBootstrap:
    def test_empty_string_when_files_absent(self, tmp_path: Path) -> None:
        result = _read_dream_bootstrap(tmp_path)
        assert result == ""

    def test_reads_soul_md(self, tmp_path: Path) -> None:
        (tmp_path / "SOUL.md").write_text("# SOUL\nBe good.", encoding="utf-8")
        result = _read_dream_bootstrap(tmp_path)
        assert "Be good." in result

    def test_reads_identity_md(self, tmp_path: Path) -> None:
        (tmp_path / "IDENTITY.md").write_text("# IDENTITY\nI am Ada.", encoding="utf-8")
        result = _read_dream_bootstrap(tmp_path)
        assert "I am Ada." in result

    def test_reads_both_files(self, tmp_path: Path) -> None:
        (tmp_path / "SOUL.md").write_text("Soul content.", encoding="utf-8")
        (tmp_path / "IDENTITY.md").write_text("Identity content.", encoding="utf-8")
        result = _read_dream_bootstrap(tmp_path)
        assert "Soul content." in result
        assert "Identity content." in result

    def test_skips_empty_file(self, tmp_path: Path) -> None:
        (tmp_path / "SOUL.md").write_text("", encoding="utf-8")
        (tmp_path / "IDENTITY.md").write_text("Present.", encoding="utf-8")
        result = _read_dream_bootstrap(tmp_path)
        assert "SOUL.md" not in result
        assert "Present." in result


# ---------------------------------------------------------------------------
# _read_daily_snippets
# ---------------------------------------------------------------------------


class TestReadDailySnippets:
    def test_empty_list_when_no_memory_dir(self, tmp_path: Path) -> None:
        result = _read_daily_snippets(tmp_path, lookback_days=3)
        assert result == []

    def test_reads_todays_memory_file(self, tmp_path: Path) -> None:
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        today = date.today().isoformat()
        (mem_dir / f"{today}.md").write_text(
            "# Memory\n\n- Rewrote the auth module.\n- Fixed a bug in the parser.",
            encoding="utf-8",
        )
        result = _read_daily_snippets(tmp_path, lookback_days=1)
        assert any("auth module" in s for s in result)
        assert any("parser" in s for s in result)

    def test_skips_markdown_headers(self, tmp_path: Path) -> None:
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        today = date.today().isoformat()
        (mem_dir / f"{today}.md").write_text(
            "# Daily Memory\n\n## Notes\n\nActual content here.",
            encoding="utf-8",
        )
        result = _read_daily_snippets(tmp_path, lookback_days=1)
        assert all(not s.startswith("#") for s in result)
        assert any("Actual content" in s for s in result)

    def test_reads_multiple_days(self, tmp_path: Path) -> None:
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        today = date.today()
        (mem_dir / f"{today.isoformat()}.md").write_text("Today content.", encoding="utf-8")
        yesterday = (today - timedelta(days=1)).isoformat()
        (mem_dir / f"{yesterday}.md").write_text("Yesterday content.", encoding="utf-8")
        result = _read_daily_snippets(tmp_path, lookback_days=2)
        assert any("Today content." in s for s in result)
        assert any("Yesterday content." in s for s in result)

    def test_caps_at_20_snippets(self, tmp_path: Path) -> None:
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        today = date.today().isoformat()
        lines = "\n".join(f"Line {i}." for i in range(50))
        (mem_dir / f"{today}.md").write_text(lines, encoding="utf-8")
        result = _read_daily_snippets(tmp_path, lookback_days=1)
        assert len(result) == 20

    def test_missing_file_is_skipped(self, tmp_path: Path) -> None:
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        # Only yesterday exists, not today.
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        (mem_dir / f"{yesterday}.md").write_text("Yesterday only.", encoding="utf-8")
        result = _read_daily_snippets(tmp_path, lookback_days=2)
        assert any("Yesterday only." in s for s in result)


# ---------------------------------------------------------------------------
# _read_recent_diary_entries
# ---------------------------------------------------------------------------


class TestReadRecentDiaryEntries:
    def test_empty_list_when_no_file(self, tmp_path: Path) -> None:
        result = _read_recent_diary_entries(tmp_path)
        assert result == []

    def test_empty_list_when_no_markers(self, tmp_path: Path) -> None:
        (tmp_path / "DREAMS.md").write_text("No markers here.", encoding="utf-8")
        result = _read_recent_diary_entries(tmp_path)
        assert result == []

    def test_returns_last_n_entries(self, tmp_path: Path) -> None:
        from minion_assist.dreaming import _DIARY_END, _DIARY_START

        content = (
            f"# Dream Diary\n\n{_DIARY_START}\n"
            "---\nFirst entry.\n\n"
            "---\nSecond entry.\n\n"
            "---\nThird entry.\n\n"
            f"{_DIARY_END}\n"
        )
        (tmp_path / "DREAMS.md").write_text(content, encoding="utf-8")
        result = _read_recent_diary_entries(tmp_path, limit=2)
        assert len(result) == 2
        # Most recent two should be included
        assert any("Second entry." in e or "Third entry." in e for e in result)

    def test_snippets_capped_at_250_chars(self, tmp_path: Path) -> None:
        from minion_assist.dreaming import _DIARY_END, _DIARY_START

        long_entry = "A" * 500
        content = f"# Dream Diary\n\n{_DIARY_START}\n---\n{long_entry}\n\n{_DIARY_END}\n"
        (tmp_path / "DREAMS.md").write_text(content, encoding="utf-8")
        result = _read_recent_diary_entries(tmp_path)
        assert all(len(e) <= 252 for e in result)  # 250 + "…"


# ---------------------------------------------------------------------------
# _build_dream_prompt
# ---------------------------------------------------------------------------


class TestBuildDreamPrompt:
    def test_includes_snippets(self) -> None:
        snippets = ["Fixed auth bug.", "Deployed to prod."]
        prompt = _build_dream_prompt(snippets, "2026-07-05", [])
        assert "Fixed auth bug." in prompt
        assert "Deployed to prod." in prompt

    def test_no_snippets_uses_placeholder(self) -> None:
        prompt = _build_dream_prompt([], "2026-07-05", [])
        assert "no memories" in prompt

    def test_includes_continuity_context_when_entries_present(self) -> None:
        snippets = ["Work done."]
        prompt = _build_dream_prompt(snippets, "2026-07-05", ["Old dream entry."])
        assert "continuity context" in prompt.lower()
        assert "Old dream entry." in prompt

    def test_no_continuity_when_no_entries(self) -> None:
        prompt = _build_dream_prompt(["Work."], "2026-07-05", [])
        assert "continuity context" not in prompt.lower()

    def test_date_appears_in_prompt(self) -> None:
        prompt = _build_dream_prompt(["Work."], "2026-07-05", ["Past entry."])
        assert "2026-07-05" in prompt

    def test_caps_at_12_snippets(self) -> None:
        snippets = [f"Snippet {i}." for i in range(20)]
        prompt = _build_dream_prompt(snippets, "2026-07-05", [])
        shown = [f"Snippet {i}." for i in range(12)]
        not_shown = [f"Snippet {i}." for i in range(12, 20)]
        assert all(s in prompt for s in shown)
        assert all(s not in prompt for s in not_shown)


# ---------------------------------------------------------------------------
# DreamingScheduler
# ---------------------------------------------------------------------------


class TestDreamingScheduler:
    def _make_cfg(self, **kwargs) -> DreamingConfig:
        defaults = dict(enabled=True, hour=3, minute=0, timezone="UTC", lookback_days=3, agent_id="main")
        defaults.update(kwargs)
        return DreamingConfig(**defaults)

    def test_start_creates_timer(self, tmp_path: Path) -> None:
        factory = MagicMock(return_value=MagicMock())
        scheduler = DreamingScheduler(
            cfg=self._make_cfg(),
            dream_session_factory=factory,
            workspace_dir=tmp_path,
        )
        scheduler.start()
        assert scheduler._timer is not None
        scheduler.stop()

    def test_stop_cancels_timer(self, tmp_path: Path) -> None:
        factory = MagicMock(return_value=MagicMock())
        scheduler = DreamingScheduler(
            cfg=self._make_cfg(),
            dream_session_factory=factory,
            workspace_dir=tmp_path,
        )
        scheduler.start()
        scheduler.stop()
        assert scheduler._stopped is True

    def test_timer_is_daemon(self, tmp_path: Path) -> None:
        factory = MagicMock(return_value=MagicMock())
        scheduler = DreamingScheduler(
            cfg=self._make_cfg(),
            dream_session_factory=factory,
            workspace_dir=tmp_path,
        )
        scheduler.start()
        assert scheduler._timer is not None
        assert scheduler._timer.daemon is True
        scheduler.stop()

    def test_run_dream_turn_calls_factory(self, tmp_path: Path) -> None:
        mock_session = MagicMock()
        mock_session.send.return_value = "Dream entry written."
        factory = MagicMock(return_value=mock_session)
        scheduler = DreamingScheduler(
            cfg=self._make_cfg(),
            dream_session_factory=factory,
            workspace_dir=tmp_path,
        )
        scheduler._run_dream_turn()
        factory.assert_called_once()

    def test_run_dream_turn_calls_send_with_system_suffix(self, tmp_path: Path) -> None:
        mock_session = MagicMock()
        mock_session.send.return_value = None
        factory = MagicMock(return_value=mock_session)
        scheduler = DreamingScheduler(
            cfg=self._make_cfg(),
            dream_session_factory=factory,
            workspace_dir=tmp_path,
        )
        scheduler._run_dream_turn()
        call_kwargs = mock_session.send.call_args
        assert call_kwargs is not None
        assert DREAM_SYSTEM_PROMPT in (
            call_kwargs.kwargs.get("system_suffix", "") or
            (call_kwargs.args[5] if len(call_kwargs.args) > 5 else "")
        )

    def test_run_dream_turn_injects_extra_tools(self, tmp_path: Path) -> None:
        from minion_assist.tools.read import ReadTool
        from minion_assist.tools.write_dream_entry import WriteDreamEntryTool

        mock_session = MagicMock()
        mock_session.send.return_value = None
        factory = MagicMock(return_value=mock_session)
        scheduler = DreamingScheduler(
            cfg=self._make_cfg(),
            dream_session_factory=factory,
            workspace_dir=tmp_path,
        )
        scheduler._run_dream_turn()
        call_kwargs = mock_session.send.call_args.kwargs
        extra_tools = call_kwargs.get("extra_tools", [])
        tool_types = [type(t) for t in extra_tools]
        assert ReadTool in tool_types
        assert WriteDreamEntryTool in tool_types

    def test_fire_reschedules_after_turn(self, tmp_path: Path) -> None:
        call_count = {"n": 0}

        def slow_factory():
            call_count["n"] += 1
            mock_session = MagicMock()
            mock_session.send.return_value = None
            return mock_session

        scheduler = DreamingScheduler(
            cfg=self._make_cfg(),
            dream_session_factory=slow_factory,
            workspace_dir=tmp_path,
        )

        # Patch start() to track calls without actually starting a timer.
        start_calls = []
        original_start = scheduler.start

        def patched_start():
            start_calls.append(1)

        scheduler.start = patched_start
        scheduler._fire()
        # _fire should have called start() once to reschedule.
        assert len(start_calls) == 1

    def test_fire_reschedules_after_error(self, tmp_path: Path) -> None:
        def bad_factory():
            raise RuntimeError("Provider down")

        scheduler = DreamingScheduler(
            cfg=self._make_cfg(),
            dream_session_factory=bad_factory,
            workspace_dir=tmp_path,
        )

        start_calls = []

        def patched_start():
            start_calls.append(1)

        scheduler.start = patched_start
        scheduler._fire()  # Should not raise; error is caught internally.
        assert len(start_calls) == 1

    def test_fire_does_not_reschedule_when_stopped(self, tmp_path: Path) -> None:
        mock_session = MagicMock()
        mock_session.send.return_value = None
        factory = MagicMock(return_value=mock_session)
        scheduler = DreamingScheduler(
            cfg=self._make_cfg(),
            dream_session_factory=factory,
            workspace_dir=tmp_path,
        )
        scheduler._stopped = True

        start_calls = []

        def patched_start():
            start_calls.append(1)

        scheduler.start = patched_start
        scheduler._fire()
        assert len(start_calls) == 0

    def test_snippets_passed_to_send(self, tmp_path: Path) -> None:
        # Write a memory file so snippets are non-empty.
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        today = date.today().isoformat()
        (mem_dir / f"{today}.md").write_text("Rewrote the kernel.", encoding="utf-8")

        mock_session = MagicMock()
        mock_session.send.return_value = None
        factory = MagicMock(return_value=mock_session)
        scheduler = DreamingScheduler(
            cfg=self._make_cfg(),
            dream_session_factory=factory,
            workspace_dir=tmp_path,
        )
        scheduler._run_dream_turn()
        message_arg = mock_session.send.call_args.kwargs.get("message", "") or mock_session.send.call_args.args[0]
        assert "Rewrote the kernel." in message_arg

    # -----------------------------------------------------------------
    # WorkerHealth wiring (MEM-GAP-016)
    # -----------------------------------------------------------------

    def test_fire_records_a_poll_and_success(self, tmp_path: Path) -> None:
        mock_session = MagicMock()
        mock_session.send.return_value = None
        factory = MagicMock(return_value=mock_session)
        health = WorkerHealth("dreaming")
        scheduler = DreamingScheduler(
            cfg=self._make_cfg(),
            dream_session_factory=factory,
            workspace_dir=tmp_path,
            health=health,
        )
        scheduler.start = lambda: None  # avoid actually scheduling a real timer

        scheduler._fire()

        snap = health.snapshot()
        assert snap["last_poll_at"] is not None
        assert snap["last_success_at"] is not None

    def test_fire_records_failure_on_error(self, tmp_path: Path) -> None:
        def bad_factory():
            raise RuntimeError("Provider down")

        health = WorkerHealth("dreaming")
        scheduler = DreamingScheduler(
            cfg=self._make_cfg(),
            dream_session_factory=bad_factory,
            workspace_dir=tmp_path,
            health=health,
        )
        scheduler.start = lambda: None

        scheduler._fire()

        snap = health.snapshot()
        assert snap["consecutive_failures"] == 1
        assert "Provider down" in snap["last_error"]

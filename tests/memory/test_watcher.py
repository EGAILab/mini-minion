"""Tests for memory/watcher.py: MemoryIndexWatcher (Stage One Phase 3, slice B).

Most tests exercise the debounce/reconcile logic directly (_schedule,
_due_agents, _reconcile) against a mock index and mock repositories — fast
and deterministic, no real filesystem events or threads involved. A couple
of lifecycle tests cover the real start()/stop() thread and (if the
optional ``watchdog`` package is installed) real file-event delivery.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import Mock

import pytest

from minion_assist.memory.watcher import _DEBOUNCE_SECONDS, MemoryIndexWatcher
from minion_assist.worker_health import WorkerHealth

try:
    import watchdog  # noqa: F401

    _WATCHDOG_AVAILABLE = True
except ImportError:
    _WATCHDOG_AVAILABLE = False


def _repo(root, files=None):
    repo = Mock()
    repo.root = root
    repo.list_indexable_files = Mock(return_value=files or [])
    return repo


# ---------------------------------------------------------------------------
# _schedule / _due_agents
# ---------------------------------------------------------------------------

def test_schedule_makes_an_agent_due_after_its_debounce_window(tmp_path):
    watcher = MemoryIndexWatcher(Mock(), {"main": _repo(tmp_path)})

    watcher._schedule("main")

    assert watcher._due_agents(time.monotonic()) == []  # not due yet
    future = time.monotonic() + _DEBOUNCE_SECONDS + 0.01
    assert watcher._due_agents(future) == ["main"]


def test_due_agents_pops_entries_so_they_are_not_reported_twice(tmp_path):
    watcher = MemoryIndexWatcher(Mock(), {"main": _repo(tmp_path)})
    watcher._schedule("main")

    future = time.monotonic() + _DEBOUNCE_SECONDS + 0.01
    assert watcher._due_agents(future) == ["main"]
    assert watcher._due_agents(future) == []


def test_rescheduling_an_agent_before_it_is_due_extends_the_window(tmp_path):
    watcher = MemoryIndexWatcher(Mock(), {"main": _repo(tmp_path)})

    watcher._schedule("main")
    first_due_at = watcher._pending["main"]
    watcher._schedule("main")  # a second event arrives before the first debounce elapsed
    second_due_at = watcher._pending["main"]

    assert second_due_at >= first_due_at


def test_multiple_agents_debounce_independently(tmp_path):
    watcher = MemoryIndexWatcher(
        Mock(), {"main": _repo(tmp_path), "researcher": _repo(tmp_path)}
    )

    watcher._schedule("main")

    future = time.monotonic() + _DEBOUNCE_SECONDS + 0.01
    assert watcher._due_agents(future) == ["main"]  # researcher was never scheduled


# ---------------------------------------------------------------------------
# _reconcile
# ---------------------------------------------------------------------------

def test_reconcile_calls_index_with_the_repos_current_listing(tmp_path):
    index = Mock()
    files = [("durable", "MEMORY.md", "content")]
    watcher = MemoryIndexWatcher(index, {"main": _repo(tmp_path, files)})

    watcher._reconcile("main")

    index.reconcile_agent.assert_called_once_with("main", files)


def test_reconcile_swallows_index_exceptions(tmp_path):
    index = Mock()
    index.reconcile_agent.side_effect = RuntimeError("db unavailable")
    watcher = MemoryIndexWatcher(index, {"main": _repo(tmp_path)})

    watcher._reconcile("main")  # must not raise


def test_reconcile_is_a_no_op_for_an_unknown_agent(tmp_path):
    index = Mock()
    watcher = MemoryIndexWatcher(index, {"main": _repo(tmp_path)})

    watcher._reconcile("unknown-agent")  # must not raise

    index.reconcile_agent.assert_not_called()


# ---------------------------------------------------------------------------
# WorkerHealth wiring (MEM-GAP-016)
# ---------------------------------------------------------------------------

def test_reconcile_records_success(tmp_path):
    index = Mock()
    health = WorkerHealth("memory_watcher")
    watcher = MemoryIndexWatcher(index, {"main": _repo(tmp_path)}, health=health)

    watcher._reconcile("main")

    snap = health.snapshot()
    assert snap["last_success_at"] is not None
    assert snap["consecutive_failures"] == 0


def test_reconcile_records_failure_on_index_exception(tmp_path):
    index = Mock()
    index.reconcile_agent.side_effect = RuntimeError("db unavailable")
    health = WorkerHealth("memory_watcher")
    watcher = MemoryIndexWatcher(index, {"main": _repo(tmp_path)}, health=health)

    watcher._reconcile("main")

    snap = health.snapshot()
    assert snap["consecutive_failures"] == 1
    assert "db unavailable" in snap["last_error"]


def test_debounce_loop_records_a_poll_each_iteration(tmp_path):
    index = Mock()
    health = WorkerHealth("memory_watcher")
    watcher = MemoryIndexWatcher(index, {"main": _repo(tmp_path)}, health=health)

    t = threading.Thread(target=watcher._debounce_loop, daemon=True)
    t.start()
    time.sleep(0.05)
    watcher._stop_event.set()
    t.join(timeout=2.0)

    assert health.snapshot()["last_poll_at"] is not None


# ---------------------------------------------------------------------------
# start / stop lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _WATCHDOG_AVAILABLE, reason="requires the optional watchdog package")
def test_start_and_stop_join_cleanly(tmp_path):
    watcher = MemoryIndexWatcher(Mock(), {"main": _repo(tmp_path)})

    watcher.start()
    watcher.stop(timeout=2.0)

    assert not watcher._debounce_thread.is_alive()


def test_stop_without_start_does_not_raise(tmp_path):
    watcher = MemoryIndexWatcher(Mock(), {"main": _repo(tmp_path)})
    watcher.stop(timeout=1.0)  # no observer/thread was ever started


@pytest.mark.skipif(not _WATCHDOG_AVAILABLE, reason="requires the optional watchdog package")
def test_a_real_file_edit_triggers_reconcile(tmp_path):
    index = Mock()
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    repo = _repo(tmp_path, [("daily", "memory/note.md", "hello")])
    watcher = MemoryIndexWatcher(index, {"main": repo})

    watcher.start()
    try:
        (memory_dir / "note.md").write_text("hello", encoding="utf-8")
        # Debounce window plus slack for the observer/debounce loop to notice.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not index.reconcile_agent.called:
            time.sleep(0.1)
        assert index.reconcile_agent.called
    finally:
        watcher.stop(timeout=2.0)

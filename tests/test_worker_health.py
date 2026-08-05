"""Tests for worker_health.py — WorkerHealth (MEM-GAP-016)."""

from __future__ import annotations

import threading
import time

from minion_assist.worker_health import WorkerHealth


def test_snapshot_starts_all_none_and_zero_failures():
    health = WorkerHealth("capture_worker")

    snap = health.snapshot()

    assert snap == {
        "name": "capture_worker",
        "last_poll_at": None,
        "last_success_at": None,
        "last_error": None,
        "last_error_at": None,
        "consecutive_failures": 0,
    }


def test_record_poll_sets_last_poll_at():
    health = WorkerHealth("w")
    before = time.time()

    health.record_poll()

    snap = health.snapshot()
    assert snap["last_poll_at"] is not None
    assert snap["last_poll_at"] >= before


def test_record_success_sets_last_success_at():
    health = WorkerHealth("w")

    health.record_success()

    assert health.snapshot()["last_success_at"] is not None


def test_record_success_resets_consecutive_failures():
    health = WorkerHealth("w")
    health.record_failure("boom")
    health.record_failure("boom again")
    assert health.snapshot()["consecutive_failures"] == 2

    health.record_success()

    assert health.snapshot()["consecutive_failures"] == 0


def test_record_failure_increments_consecutive_failures():
    health = WorkerHealth("w")

    health.record_failure("first")
    health.record_failure("second")

    assert health.snapshot()["consecutive_failures"] == 2


def test_record_failure_stores_the_error_message_and_timestamp():
    health = WorkerHealth("w")
    before = time.time()

    health.record_failure("connection refused")

    snap = health.snapshot()
    assert snap["last_error"] == "connection refused"
    assert snap["last_error_at"] >= before


def test_record_failure_truncates_an_overlong_error_message():
    health = WorkerHealth("w")

    health.record_failure("x" * 10_000)

    assert len(health.snapshot()["last_error"]) == 300


def test_snapshot_returns_a_copy_not_a_live_view():
    health = WorkerHealth("w")
    health.record_success()

    snap = health.snapshot()
    health.record_failure("later failure")

    # The earlier snapshot must not have mutated after the fact.
    assert snap["last_error"] is None
    assert snap["consecutive_failures"] == 0


def test_concurrent_writes_do_not_corrupt_the_failure_counter():
    # Not a proof of correctness under all interleavings, but catches a
    # regression to an unlocked read-modify-write on consecutive_failures.
    health = WorkerHealth("w")

    def _hammer():
        for _ in range(200):
            health.record_failure("boom")

    threads = [threading.Thread(target=_hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert health.snapshot()["consecutive_failures"] == 800

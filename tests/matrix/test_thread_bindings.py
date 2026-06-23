"""Tests for matrix/thread_bindings.py — MatrixThreadBindingManager."""

import asyncio
import time

import pytest

from minion_assist.matrix.config import MatrixThreadBindingsConfig
from minion_assist.matrix.thread_bindings import MatrixThreadBindingManager


def _mgr(tmp_path, idle_hours=1.0, max_age_hours=24.0):
    cfg = MatrixThreadBindingsConfig(enabled=True, idle_hours=idle_hours, max_age_hours=max_age_hours)
    return MatrixThreadBindingManager(tmp_path / "threads.db", cfg)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_creates_new_binding(tmp_path):
    async def _go():
        mgr = _mgr(tmp_path)
        await mgr.start()
        key = await mgr.get_or_create_session_key("$thread1", "!room:ex.org", "main")
        await mgr.stop()
        return key

    key = _run(_go())
    assert key.startswith("matrix-thread-")


def test_reuses_existing_binding(tmp_path):
    async def _go():
        mgr = _mgr(tmp_path)
        await mgr.start()
        key1 = await mgr.get_or_create_session_key("$thread1", "!room:ex.org", "main")
        key2 = await mgr.get_or_create_session_key("$thread1", "!room:ex.org", "main")
        await mgr.stop()
        return key1, key2

    k1, k2 = _run(_go())
    assert k1 == k2


def test_different_threads_get_different_keys(tmp_path):
    async def _go():
        mgr = _mgr(tmp_path)
        await mgr.start()
        k1 = await mgr.get_or_create_session_key("$t1", "!room:ex.org", "main")
        k2 = await mgr.get_or_create_session_key("$t2", "!room:ex.org", "main")
        await mgr.stop()
        return k1, k2

    k1, k2 = _run(_go())
    assert k1 != k2


def test_idle_eviction(tmp_path):
    import aiosqlite

    async def _go():
        mgr = _mgr(tmp_path, idle_hours=0.0001)  # ~0.36 seconds
        await mgr.start()
        key = await mgr.get_or_create_session_key("$t1", "!room:ex.org", "main")

        # Back-date the last_activity_at so it's expired
        async with aiosqlite.connect(tmp_path / "threads.db") as conn:
            await conn.execute(
                "UPDATE thread_bindings SET last_activity_at = ? WHERE thread_event_id = ?",
                (int(time.time()) - 3600, "$t1"),
            )
            await conn.commit()

        await mgr.evict_expired()
        # After eviction, creating for same thread should produce a NEW key
        new_key = await mgr.get_or_create_session_key("$t1", "!room:ex.org", "main")
        await mgr.stop()
        return key, new_key

    try:
        k1, k2 = _run(_go())
        assert k1 != k2
    except ImportError:
        pytest.skip("aiosqlite not installed")


def test_max_age_eviction(tmp_path):
    import aiosqlite

    async def _go():
        mgr = _mgr(tmp_path, max_age_hours=0.0001)
        await mgr.start()
        key = await mgr.get_or_create_session_key("$t1", "!room:ex.org", "main")

        # Back-date created_at
        async with aiosqlite.connect(tmp_path / "threads.db") as conn:
            await conn.execute(
                "UPDATE thread_bindings SET created_at = ? WHERE thread_event_id = ?",
                (int(time.time()) - 3600, "$t1"),
            )
            await conn.commit()

        await mgr.evict_expired()
        new_key = await mgr.get_or_create_session_key("$t1", "!room:ex.org", "main")
        await mgr.stop()
        return key, new_key

    try:
        k1, k2 = _run(_go())
        assert k1 != k2
    except ImportError:
        pytest.skip("aiosqlite not installed")


def test_start_required(tmp_path):
    async def _go():
        mgr = _mgr(tmp_path)
        with pytest.raises(RuntimeError, match="start"):
            await mgr.get_or_create_session_key("$t1", "!room:ex.org", "main")

    _run(_go())

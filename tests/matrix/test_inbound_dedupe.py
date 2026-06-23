"""Tests for matrix/inbound_dedupe.py — MatrixInboundDeduper."""

import asyncio
import time

import pytest

from minion_assist.matrix.inbound_dedupe import MatrixInboundDeduper, _PRUNE_AGE_SECONDS


@pytest.fixture
def dedupe(tmp_path):
    return MatrixInboundDeduper(tmp_path / "dedupe.db")


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def set_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


async def _setup_and_run(dedupe, coro):
    await dedupe.start()
    try:
        return await coro
    finally:
        await dedupe.stop()


def test_first_event_not_seen(tmp_path):
    dedupe = MatrixInboundDeduper(tmp_path / "d.db")

    async def _run():
        await dedupe.start()
        result = await dedupe.is_seen("$event1")
        await dedupe.stop()
        return result

    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(_run())
    loop.close()
    assert result is False


def test_second_event_is_seen(tmp_path):
    dedupe = MatrixInboundDeduper(tmp_path / "d.db")

    async def _run():
        await dedupe.start()
        await dedupe.is_seen("$event1")
        result = await dedupe.is_seen("$event1")
        await dedupe.stop()
        return result

    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(_run())
    loop.close()
    assert result is True


def test_different_events_not_seen(tmp_path):
    dedupe = MatrixInboundDeduper(tmp_path / "d.db")

    async def _run():
        await dedupe.start()
        await dedupe.is_seen("$event1")
        result = await dedupe.is_seen("$event2")
        await dedupe.stop()
        return result

    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(_run())
    loop.close()
    assert result is False


def test_pruning_removes_old_entries(tmp_path):
    import aiosqlite

    dedupe = MatrixInboundDeduper(tmp_path / "d.db")

    async def _run():
        await dedupe.start()
        # Manually insert an old entry
        old_ts = int(time.time()) - _PRUNE_AGE_SECONDS - 1
        async with aiosqlite.connect(tmp_path / "d.db") as conn:
            await conn.execute(
                "INSERT OR IGNORE INTO seen_events (event_id, seen_at) VALUES (?, ?)",
                ("$old_event", old_ts),
            )
            await conn.commit()

        # Re-start to trigger pruning
        await dedupe.stop()
        await dedupe.start()

        # Old entry should be gone — is_seen should return False
        result = await dedupe.is_seen("$old_event")
        await dedupe.stop()
        return result

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_run())
    except ImportError:
        pytest.skip("aiosqlite not installed")
    finally:
        loop.close()


def test_start_required_before_is_seen(tmp_path):
    dedupe = MatrixInboundDeduper(tmp_path / "d.db")

    async def _run():
        with pytest.raises(RuntimeError, match="start"):
            await dedupe.is_seen("$event1")

    loop = asyncio.new_event_loop()
    loop.run_until_complete(_run())
    loop.close()

"""Tests for matrix/room_sessions.py — MatrixRoomSessionManager (MEM-GAP-001)."""

import asyncio
import uuid

import pytest

from minion_assist.matrix.room_sessions import MatrixRoomSessionManager


def _mgr(tmp_path):
    return MatrixRoomSessionManager(tmp_path / "room_sessions.db")


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_creates_new_binding_as_a_valid_uuid(tmp_path):
    async def _go():
        mgr = _mgr(tmp_path)
        await mgr.start()
        session_id = await mgr.get_or_create_session_id("!room:ex.org", "main")
        await mgr.stop()
        return session_id

    session_id = _run(_go())
    uuid.UUID(session_id)  # must not raise — a real session_id, not an opaque key


def test_reuses_existing_binding_for_the_same_room_and_agent(tmp_path):
    async def _go():
        mgr = _mgr(tmp_path)
        await mgr.start()
        first = await mgr.get_or_create_session_id("!room:ex.org", "main")
        second = await mgr.get_or_create_session_id("!room:ex.org", "main")
        await mgr.stop()
        return first, second

    first, second = _run(_go())
    assert first == second


def test_different_rooms_get_different_session_ids(tmp_path):
    async def _go():
        mgr = _mgr(tmp_path)
        await mgr.start()
        room_a = await mgr.get_or_create_session_id("!room-a:ex.org", "main")
        room_b = await mgr.get_or_create_session_id("!room-b:ex.org", "main")
        await mgr.stop()
        return room_a, room_b

    room_a, room_b = _run(_go())
    assert room_a != room_b


def test_same_room_different_agents_get_different_session_ids(tmp_path):
    async def _go():
        mgr = _mgr(tmp_path)
        await mgr.start()
        main_session = await mgr.get_or_create_session_id("!room:ex.org", "main")
        researcher_session = await mgr.get_or_create_session_id("!room:ex.org", "researcher")
        await mgr.stop()
        return main_session, researcher_session

    main_session, researcher_session = _run(_go())
    assert main_session != researcher_session


def test_binding_survives_a_restart(tmp_path):
    async def _first_run():
        mgr = _mgr(tmp_path)
        await mgr.start()
        session_id = await mgr.get_or_create_session_id("!room:ex.org", "main")
        await mgr.stop()
        return session_id

    async def _second_run():
        mgr = _mgr(tmp_path)
        await mgr.start()
        session_id = await mgr.get_or_create_session_id("!room:ex.org", "main")
        await mgr.stop()
        return session_id

    first = _run(_first_run())
    second = _run(_second_run())
    assert first == second


def test_start_required(tmp_path):
    async def _go():
        mgr = _mgr(tmp_path)
        with pytest.raises(RuntimeError, match="start"):
            await mgr.get_or_create_session_id("!room:ex.org", "main")

    _run(_go())


# ---------------------------------------------------------------------------
# Concurrent first-message race (R2-GAP-010)
# ---------------------------------------------------------------------------

def test_concurrent_first_messages_for_the_same_room_agree_on_one_session_id(tmp_path):
    # Two Matrix events arriving together right after joining a room, both
    # racing to create the same (room_id, agent_id) binding — before the
    # fix, a plain select-then-insert could let both see no existing row
    # and both attempt an INSERT, the loser raising a uniqueness violation
    # instead of the two callers agreeing on a winner.
    async def _go():
        mgr = _mgr(tmp_path)
        await mgr.start()
        try:
            results = await asyncio.gather(
                *(mgr.get_or_create_session_id("!room:ex.org", "main") for _ in range(10))
            )
        finally:
            await mgr.stop()
        return results

    results = _run(_go())
    assert len(set(results)) == 1  # every caller agrees on the same session_id
    uuid.UUID(results[0])  # a real session_id, not an opaque key


def test_get_or_create_after_rebind_returns_the_rebound_id(tmp_path):
    async def _go():
        mgr = _mgr(tmp_path)
        await mgr.start()
        original = await mgr.get_or_create_session_id("!room:ex.org", "main")
        new_id = str(uuid.uuid4())
        await mgr.rebind("!room:ex.org", "main", new_id)
        resolved = await mgr.get_or_create_session_id("!room:ex.org", "main")
        await mgr.stop()
        return original, new_id, resolved

    original, new_id, resolved = _run(_go())
    assert resolved == new_id
    assert resolved != original

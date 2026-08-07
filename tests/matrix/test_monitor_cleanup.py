"""Tests for matrix/monitor.py's _cleanup() — shutdown-time resource disposal.

Only _cleanup() itself is exercised here, not the full monitor_matrix()
coroutine (which needs a live homeserver connection / matrix-nio client) —
_cleanup() is a small, self-contained async function, and R2-GAP-013's
wiring is entirely about what it calls and in what order.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from minion_assist.matrix.monitor import _cleanup


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _mocks():
    client = MagicMock()
    client.close = AsyncMock()
    dedupe = MagicMock()
    dedupe.stop = AsyncMock()
    room_session_mgr = MagicMock()
    room_session_mgr.stop = AsyncMock()
    msg_handler = MagicMock()
    return client, dedupe, room_session_mgr, msg_handler


def test_cleanup_closes_room_sessions():
    client, dedupe, room_session_mgr, msg_handler = _mocks()

    _run(_cleanup(client, dedupe, room_session_mgr, msg_handler))

    msg_handler.close_room_sessions.assert_called_once()


def test_cleanup_still_closes_client_and_dedupe_and_room_session_mgr():
    client, dedupe, room_session_mgr, msg_handler = _mocks()

    _run(_cleanup(client, dedupe, room_session_mgr, msg_handler))

    client.close.assert_awaited_once()
    dedupe.stop.assert_awaited_once()
    room_session_mgr.stop.assert_awaited_once()


def test_cleanup_continues_when_close_room_sessions_raises():
    client, dedupe, room_session_mgr, msg_handler = _mocks()
    msg_handler.close_room_sessions.side_effect = Exception("boom")

    _run(_cleanup(client, dedupe, room_session_mgr, msg_handler))  # must not raise

    client.close.assert_awaited_once()
    dedupe.stop.assert_awaited_once()
    room_session_mgr.stop.assert_awaited_once()


def test_cleanup_still_runs_remaining_steps_when_client_close_raises():
    client, dedupe, room_session_mgr, msg_handler = _mocks()
    client.close.side_effect = Exception("already disconnected")

    _run(_cleanup(client, dedupe, room_session_mgr, msg_handler))  # must not raise

    msg_handler.close_room_sessions.assert_called_once()
    dedupe.stop.assert_awaited_once()
    room_session_mgr.stop.assert_awaited_once()

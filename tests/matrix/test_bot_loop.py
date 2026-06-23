"""Tests for matrix/bot_loop.py — BotLoopProtection."""

import time
from unittest.mock import patch

from minion_assist.matrix.bot_loop import BotLoopProtection
from minion_assist.matrix.config import MatrixBotLoopConfig


def _config(max_events=3, window=10, cooldown=5, enabled=True):
    return MatrixBotLoopConfig(
        enabled=enabled,
        max_events_per_window=max_events,
        window_seconds=window,
        cooldown_seconds=cooldown,
    )


def test_disabled_never_suppresses():
    blp = BotLoopProtection(_config(enabled=False))
    for _ in range(100):
        assert blp.should_suppress("!room:example.org") is False


def test_under_limit_not_suppressed():
    blp = BotLoopProtection(_config(max_events=3))
    room = "!room:example.org"
    assert blp.should_suppress(room) is False
    assert blp.should_suppress(room) is False
    assert blp.should_suppress(room) is False


def test_at_limit_not_suppressed():
    # max_events=3 means the 3rd event is allowed, 4th is suppressed
    blp = BotLoopProtection(_config(max_events=3))
    room = "!room:example.org"
    blp.should_suppress(room)  # 1
    blp.should_suppress(room)  # 2
    result = blp.should_suppress(room)  # 3 — at limit, still allowed
    assert result is False


def test_over_limit_suppressed():
    blp = BotLoopProtection(_config(max_events=3, cooldown=60))
    room = "!room:example.org"
    for _ in range(3):
        blp.should_suppress(room)
    assert blp.should_suppress(room) is True  # 4th — suppressed


def test_cooldown_resets_after_period():
    blp = BotLoopProtection(_config(max_events=2, cooldown=1, window=10))
    room = "!room:example.org"
    for _ in range(3):
        blp.should_suppress(room)
    assert blp.should_suppress(room) is True  # in cooldown

    # Expire both the cooldown AND the sliding window timestamps
    state = blp._rooms[room]
    state.cooling_down_until = time.monotonic() - 1
    # Back-date all timestamps so they fall outside the window
    for i in range(len(state.timestamps)):
        state.timestamps[i] = time.monotonic() - 20  # older than window_seconds=10
    assert blp.should_suppress(room) is False


def test_different_rooms_are_independent():
    blp = BotLoopProtection(_config(max_events=2, cooldown=60))
    room_a = "!room-a:example.org"
    room_b = "!room-b:example.org"
    blp.should_suppress(room_a)
    blp.should_suppress(room_a)
    blp.should_suppress(room_a)  # room_a now suppressed
    assert blp.should_suppress(room_a) is True
    assert blp.should_suppress(room_b) is False  # room_b unaffected


def test_window_expiry_resets_count():
    blp = BotLoopProtection(_config(max_events=2, window=1, cooldown=60))
    room = "!room:example.org"
    blp.should_suppress(room)
    blp.should_suppress(room)

    # Age the timestamps beyond the window
    state = blp._rooms[room]
    for i in range(len(state.timestamps)):
        state.timestamps[i] = time.monotonic() - 2  # older than window_seconds=1

    # Now the window is clear — should not be suppressed
    assert blp.should_suppress(room) is False

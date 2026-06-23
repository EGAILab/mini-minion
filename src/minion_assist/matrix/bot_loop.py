"""Bot-loop protection for the Matrix channel.

Detects and suppresses bot-to-bot message loops by counting inbound events
per room within a sliding time window.  When a room exceeds the configured
``max_events_per_window`` in ``window_seconds``, further messages from that
room are suppressed until a ``cooldown_seconds`` period has elapsed with no
further excess events.

This mirrors openclaw's ``botLoopProtection`` config block.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from .config import MatrixBotLoopConfig


@dataclass
class _RoomState:
    # A deque (double-ended queue) is efficient for sliding window: we push to
    # the right and pop old entries from the left in O(1) time.
    timestamps: deque = field(default_factory=deque)
    # Wall-clock time (monotonic seconds) after which the cooldown expires.
    # 0.0 means not cooling down.
    cooling_down_until: float = 0.0


class BotLoopProtection:
    """Sliding-window per-room event rate limiter.

    Args:
        config: Bot-loop protection settings from ``MatrixBotLoopConfig``.
    """

    def __init__(self, config: MatrixBotLoopConfig) -> None:
        self._config = config
        # One _RoomState entry per room ID seen so far.
        self._rooms: dict[str, _RoomState] = {}

    def should_suppress(self, room_id: str) -> bool:
        """Return True if the room is over its event rate limit.

        Records the current event timestamp and returns whether the room has
        exceeded ``max_events_per_window`` events in the last ``window_seconds``.
        When the limit is exceeded the room enters a cooldown period.  During
        cooldown all events are suppressed.

        Args:
            room_id: Matrix room ID of the inbound event.

        Returns:
            True if the event should be suppressed (room is in cooldown or
            just exceeded the rate limit).
        """
        if not self._config.enabled:
            return False

        # time.monotonic() is a high-resolution clock that never goes backwards —
        # safe for measuring elapsed time even if the system clock is adjusted.
        now = time.monotonic()
        # setdefault: create a new _RoomState the first time we see a room.
        state = self._rooms.setdefault(room_id, _RoomState())

        # If the room is actively cooling down, suppress without recording a
        # new timestamp (we don't want to extend the window while cooling).
        if now < state.cooling_down_until:
            return True

        # Sliding window: remove timestamps that are older than window_seconds.
        # The deque is sorted oldest→newest so we pop from the left.
        window_start = now - self._config.window_seconds
        while state.timestamps and state.timestamps[0] < window_start:
            state.timestamps.popleft()

        # Record the current event in the window.
        state.timestamps.append(now)

        # If too many events arrived in the window, start a cooldown period.
        if len(state.timestamps) > self._config.max_events_per_window:
            state.cooling_down_until = now + self._config.cooldown_seconds
            return True

        return False

"""HeartbeatScheduler — periodic background agent turns.

The scheduler fires a ``threading.Timer`` chain that sends the heartbeat prompt
to the configured agent on a fixed interval.  On each tick:

1. Build a fresh :class:`~tools.heartbeat_respond.HeartbeatResponseCapture`.
2. Inject :class:`~tools.heartbeat_respond.HeartbeatRespondTool` as an extra_tool.
3. Call ``session.send(heartbeat_prompt, extra_tools=[...], stream=False)``.
4. If the response is ``HEARTBEAT_OK`` → suppress (nothing to report).
5. Otherwise, deliver any ``heartbeat_respond`` calls plus any residual text
   to the notification target (Matrix room or terminal).

Threading note
--------------
All agent calls go through ``AgentSession.send()``, which acquires ``_lock``
internally.  The heartbeat runs in a daemon thread so it can never prevent the
process from exiting cleanly.
"""

from __future__ import annotations

import sys
import threading
from typing import TYPE_CHECKING

from .heartbeat_token import is_heartbeat_ok, strip_heartbeat_token
from .tools.heartbeat_respond import HeartbeatResponseCapture, HeartbeatRespondTool

# TYPE_CHECKING is False at runtime (so these imports never execute and cannot cause
# circular imports), but True when a type-checker runs (so types are fully resolved).
# See tools/base.py module docstring for a full explanation of this pattern.
if TYPE_CHECKING:
    from .agents.session import AgentSession
    from .config import HeartbeatConfig


class HeartbeatScheduler:
    """Runs the heartbeat loop in a daemon thread.

    Args:
        config:    Resolved :class:`~config.HeartbeatConfig`.
        sessions:  Dict mapping agent_id → :class:`~agents.session.AgentSession`.
        matrix_outbound: Optional ``MatrixOutbound`` instance for delivering
            notifications to a Matrix room.  ``None`` → print to terminal.
        matrix_loop: The asyncio event loop that ``matrix_outbound`` runs on.
            Required when ``matrix_outbound`` is not None.
    """

    def __init__(
        self,
        config: "HeartbeatConfig",
        sessions: dict[str, "AgentSession"],
        matrix_outbound: object = None,
        matrix_loop: object = None,
    ) -> None:
        self._config = config
        self._sessions = sessions
        self._outbound = matrix_outbound
        self._loop = matrix_loop
        self._timer: threading.Timer | None = None
        self._stopped = False

    def start(self) -> None:
        """Schedule the first heartbeat tick."""
        interval = max(60, min(86400, self._config.interval_seconds))
        self._timer = threading.Timer(interval, self._tick)
        self._timer.daemon = True
        self._timer.name = "heartbeat-scheduler"
        self._timer.start()

    def stop(self) -> None:
        """Cancel the pending timer and prevent future ticks."""
        self._stopped = True
        if self._timer is not None:
            self._timer.cancel()

    def _tick(self) -> None:
        """Run one heartbeat turn then reschedule."""
        if self._stopped:
            return
        try:
            self._run_heartbeat()
        except Exception as exc:
            print(f"[heartbeat] Error during turn: {exc}", file=sys.stderr)
        finally:
            # Always reschedule (even after an error) so the loop continues.
            if not self._stopped:
                self.start()

    def _run_heartbeat(self) -> None:
        """Send the heartbeat prompt to the agent and route any notifications."""
        session = self._sessions.get(self._config.agent_id)
        if session is None:
            print(
                f"[heartbeat] Agent '{self._config.agent_id}' not found — skipping.",
                file=sys.stderr,
            )
            return

        capture = HeartbeatResponseCapture()
        respond_tool = HeartbeatRespondTool(capture)

        response = session.send(
            self._config.prompt,
            extra_tools=[respond_tool],
            stream=False,
        )

        # Deliver any explicit heartbeat_respond() calls first.
        for msg in capture.messages:
            self._deliver(msg)

        # Deliver any residual prose the agent added (after stripping HEARTBEAT_OK).
        if response and not is_heartbeat_ok(response):
            remainder = strip_heartbeat_token(response)
            if remainder:
                self._deliver(remainder)

    def _deliver(self, message: str) -> None:
        """Route a notification to the configured Matrix room or the terminal."""
        room_id = self._config.notification_room_id
        if room_id and self._outbound is not None and self._loop is not None:
            import asyncio  # noqa: PLC0415
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._outbound.send_text(room_id, message),  # type: ignore[attr-defined]
                    self._loop,  # type: ignore[arg-type]
                )
                future.result(timeout=15)
            except Exception as exc:
                print(f"[heartbeat] Matrix delivery failed: {exc}", file=sys.stderr)
        else:
            print(f"\n[heartbeat] {message}")

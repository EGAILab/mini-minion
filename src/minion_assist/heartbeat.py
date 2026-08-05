"""HeartbeatScheduler — periodic background agent turns.

The scheduler fires a ``threading.Timer`` chain that sends the heartbeat prompt
to the configured agent on a fixed interval.  On each tick:

1. Build a fresh :class:`~tools.heartbeat_respond.HeartbeatResponseCapture`.
2. Inject :class:`~tools.heartbeat_respond.HeartbeatRespondTool` as an extra_tool.
3. Fetch due commitments for the agent (Stage One Phase 6, slice C — see
   below) and, if any exist, inject
   :class:`~tools.commitment_response.RespondToCommitmentTool`/
   :class:`~tools.commitment_response.DismissCommitmentTool` too, plus
   append an untrusted-framed block listing them to the prompt.
4. Call ``session.send(heartbeat_prompt, extra_tools=[...], stream=False)``.
5. If the response is ``HEARTBEAT_OK`` → suppress (nothing to report).
6. Otherwise, deliver any ``heartbeat_respond`` calls plus any residual text
   to the notification target (Matrix room or terminal).

Multi-room-aware commitment delivery (Stage One Phase 6, slice C)
------------------------------------------------------------------
This scheduler still runs exactly one heartbeat turn per tick, for one
fixed agent — it does not (and does not need to) become a full
per-session/per-room runner to satisfy "a commitment from one Matrix room
is not delivered in another." Instead, a single turn can see due
commitments *from every room that agent has commitments in at once*
(:meth:`~minion_assist.session.db.SessionDB.list_due_commitments_for_agent`),
each one resolved to its own room *at delivery time* —
:meth:`_deliver_to_channel` sends to exactly the channel stored on that
specific commitment, never the fixed ``notification_room_id`` the base
heartbeat notification (:meth:`_deliver`) uses. A wrong-room delivery is
structurally impossible this way: the target is a direct database lookup
keyed to the commitment being responded to, not something inferred from
"what room is this turn about."

Threading note
--------------
All agent calls go through ``AgentSession.send()``, which acquires ``_lock``
internally.  The heartbeat runs in a daemon thread so it can never prevent the
process from exiting cleanly.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from typing import TYPE_CHECKING

from .heartbeat_token import is_heartbeat_ok, strip_heartbeat_token
from .tools.heartbeat_respond import HeartbeatResponseCapture, HeartbeatRespondTool

# TYPE_CHECKING is False at runtime (so these imports never execute and cannot cause
# circular imports), but True when a type-checker runs (so types are fully resolved).
# See tools/base.py module docstring for a full explanation of this pattern.
if TYPE_CHECKING:
    from .agents.session import AgentSession
    from .config import HeartbeatConfig
    from .session.db import SessionDB
    from .worker_health import WorkerHealth


class HeartbeatScheduler:
    """Runs the heartbeat loop in a daemon thread.

    Args:
        config:    Resolved :class:`~config.HeartbeatConfig`.
        sessions:  Dict mapping agent_id → :class:`~agents.session.AgentSession`.
        matrix_outbound: Optional ``MatrixOutbound`` instance for delivering
            notifications to a Matrix room.  ``None`` → print to terminal.
        matrix_loop: The asyncio event loop that ``matrix_outbound`` runs on.
            Required when ``matrix_outbound`` is not None.
        db: Optional :class:`~minion_assist.session.db.SessionDB` — when
            given, each tick also checks for due commitments (Stage One
            Phase 6, slice C). ``None`` (the default) skips commitment
            delivery entirely, same as every other database-optional path
            in this project.
        health: Optional :class:`~minion_assist.worker_health.WorkerHealth`
            (MEM-GAP-016) — see other schedulers' matching parameter.
    """

    def __init__(
        self,
        config: "HeartbeatConfig",
        sessions: dict[str, "AgentSession"],
        matrix_outbound: object = None,
        matrix_loop: object = None,
        db: "SessionDB | None" = None,
        health: "WorkerHealth | None" = None,
    ) -> None:
        self._config = config
        self._sessions = sessions
        self._outbound = matrix_outbound
        self._loop = matrix_loop
        self._db = db
        self._health = health
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
        if self._health is not None:
            self._health.record_poll()
        try:
            self._run_heartbeat()
            if self._health is not None:
                self._health.record_success()
        except Exception as exc:
            print(f"[heartbeat] Error during turn: {exc}", file=sys.stderr)
            if self._health is not None:
                self._health.record_failure(f"{type(exc).__name__}: {exc}")
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
        extra_tools: list = [HeartbeatRespondTool(capture)]
        prompt = self._config.prompt

        due_commitments = self._fetch_due_commitments()
        if due_commitments:
            from .memory.commitments import format_due_commitments_block  # noqa: PLC0415
            from .tools.commitment_response import (  # noqa: PLC0415
                DismissCommitmentTool,
                RespondToCommitmentTool,
            )
            extra_tools.append(RespondToCommitmentTool(self._db, self._deliver_to_channel))
            extra_tools.append(DismissCommitmentTool(self._db))
            prompt = f"{prompt}\n\n{format_due_commitments_block(due_commitments)}"

        response = session.send(prompt, extra_tools=extra_tools, stream=False)

        # Deliver any explicit heartbeat_respond() calls first.
        for msg in capture.messages:
            self._deliver(msg)

        # Deliver any residual prose the agent added (after stripping HEARTBEAT_OK).
        if response and not is_heartbeat_ok(response):
            remainder = strip_heartbeat_token(response)
            if remainder:
                self._deliver(remainder)

    def _fetch_due_commitments(self) -> list[dict]:
        """Look up this tick's due commitments, or ``[]`` if unconfigured/unavailable.

        Never raises — a commitment-lookup failure must not break the
        base heartbeat turn, matching every other database-optional path
        in this project.
        """
        if self._db is None:
            return []
        try:
            return self._db.list_due_commitments_for_agent(self._config.agent_id, time.time())
        except Exception as exc:
            print(f"[heartbeat] Failed to list due commitments: {exc}", file=sys.stderr)
            return []

    def _deliver(self, message: str) -> None:
        """Route a notification to the configured Matrix room or the terminal."""
        room_id = self._config.notification_room_id
        if room_id and self._outbound is not None and self._loop is not None:
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

    def _deliver_to_channel(self, message: str, channel: str) -> None:
        """Route a commitment check-in to its own channel — never the fixed notification room.

        Stage One Phase 6, slice C. The ``"cli"`` sentinel channel (a
        commitment created outside Matrix) and the case where no Matrix
        outbound is configured both fall through to a terminal print —
        there is no Matrix room to target either way.
        """
        if channel and channel != "cli" and self._outbound is not None and self._loop is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._outbound.send_text(channel, message),  # type: ignore[attr-defined]
                    self._loop,  # type: ignore[arg-type]
                )
                future.result(timeout=15)
                return
            except Exception as exc:
                print(f"[heartbeat] Commitment delivery to {channel} failed: {exc}", file=sys.stderr)
        print(f"\n[heartbeat] (commitment check-in for {channel}) {message}")

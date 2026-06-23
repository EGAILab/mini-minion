"""Matrix exec-approval handler.

When a tool call requires human approval and ``execApprovals.enabled`` is True,
this module sends a DM to the configured approver and waits for a ✅ or ❌
reaction before allowing the tool to proceed.

The reaction polling runs in the asyncio event loop (background thread) and
bridges back to the synchronous ``BashTool`` approval callback via
``asyncio.run_coroutine_threadsafe``.

Timeout: 60 seconds — after which the request is treated as denied.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from .config import MatrixExecApprovalsConfig
from .outbound import MatrixOutbound

_APPROVE_EMOJI = "✅"
_DENY_EMOJI = "❌"
_APPROVAL_TIMEOUT_SECONDS = 60
_POLL_INTERVAL_SECONDS = 2.0


class MatrixExecApprovalHandler:
    """Remote exec-approval via Matrix DM reaction.

    Args:
        client:   An authenticated matrix-nio ``AsyncClient``.
        outbound: :class:`~minion_assist.matrix.outbound.MatrixOutbound` for sending DMs.
        config:   Exec-approval settings from config.
    """

    def __init__(
        self,
        client,
        outbound: MatrixOutbound,
        config: MatrixExecApprovalsConfig,
    ) -> None:
        self._client = client
        self._outbound = outbound
        self._config = config
        # Maps pending_event_id → asyncio.Future[bool] (True=approve, False=deny).
        # When a reaction arrives, handle_reaction() looks up the future here
        # and sets its result to unblock request_approval().
        self._pending: dict[str, asyncio.Future] = {}

    async def request_approval(self, command: str) -> str:
        """Send a DM to the first approver and wait for their ✅/❌ reaction.

        Args:
            command: Shell command string that needs approval.

        Returns:
            ``"allow_once"`` if approved, ``"deny"`` if denied or timed out.
        """
        approvers = self._config.approvers
        if not approvers:
            # No approvers configured → deny by default (fail safe).
            return "deny"

        # Only the first approver is consulted.  Multi-approver support is
        # a future enhancement.
        approver_id = approvers[0]
        dm_room_id = await self._resolve_or_create_dm(approver_id)
        if not dm_room_id:
            return "deny"

        body = (
            f"**Tool approval requested**\n\n"
            f"```\n{command}\n```\n\n"
            f"React with {_APPROVE_EMOJI} to allow or {_DENY_EMOJI} to deny."
        )
        # Send the approval request and record the event ID so we can match
        # incoming reactions to this specific message.
        event_id = await self._outbound.send_text(dm_room_id, body)

        # create_future() creates an asyncio.Future: a one-shot container for a
        # result that will arrive later.  handle_reaction() will call
        # fut.set_result() when the approver reacts.
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        # Register in the pending dict so handle_reaction() can find it by event ID.
        self._pending[event_id] = fut

        try:
            deadline = time.monotonic() + _APPROVAL_TIMEOUT_SECONDS
            # Poll every 2 seconds.  The future is resolved externally by
            # handle_reaction() when a Matrix reaction event arrives during sync.
            while time.monotonic() < deadline:
                await asyncio.sleep(_POLL_INTERVAL_SECONDS)
                if fut.done():
                    break
            if not fut.done():
                # Timed out without a reaction — treat as denial.
                fut.cancel()
                return "deny"
            approved = await fut
            return "allow_once" if approved else "deny"
        finally:
            # Always clean up the pending dict entry regardless of outcome.
            self._pending.pop(event_id, None)

    def handle_reaction(self, target_event_id: str, emoji: str) -> None:
        """Called from the Matrix sync callback when a reaction event arrives.

        Resolves the pending future for ``target_event_id`` if one exists.

        Args:
            target_event_id: The event that was reacted to.
            emoji:           The reaction key string.
        """
        fut = self._pending.get(target_event_id)
        if fut is None or fut.done():
            # No pending request for this event, or it was already resolved.
            return
        if emoji == _APPROVE_EMOJI:
            fut.set_result(True)
        elif emoji == _DENY_EMOJI:
            fut.set_result(False)

    def make_sync_callback(self, loop: asyncio.AbstractEventLoop) -> Callable[[str], str]:
        """Return a synchronous approval callback suitable for ``BashTool``.

        The callback runs ``request_approval`` in the given asyncio loop and
        blocks the calling thread until a decision arrives.

        Args:
            loop: The asyncio event loop running in the Matrix background thread.

        Returns:
            A callable ``(command: str) -> str`` returning ``"allow_once"`` or ``"deny"``.
        """
        from concurrent.futures import Future as CFFuture

        def _sync_callback(command: str) -> str:
            # run_coroutine_threadsafe() submits an async coroutine to a different
            # thread's event loop and returns a concurrent.futures.Future (not an
            # asyncio.Future).  This is the correct way to call async code from a
            # synchronous context running in a different thread.
            cf: CFFuture = asyncio.run_coroutine_threadsafe(
                self.request_approval(command), loop
            )
            try:
                # .result() blocks this thread until the coroutine finishes or
                # the timeout elapses.  Add 5s slack beyond the internal timeout
                # so the coroutine always gets to return "deny" before we give up.
                result = cf.result(timeout=_APPROVAL_TIMEOUT_SECONDS + 5)
            except Exception:
                result = "deny"
            return result

        return _sync_callback

    async def _resolve_or_create_dm(self, user_id: str) -> str | None:
        """Find or open a DM room with ``user_id``.

        Returns the room ID string, or None on failure.
        """
        try:
            # Check if the client already knows about a DM room with this user
            # from a previous session (loaded during the initial sync).
            direct_rooms = getattr(self._client, "direct_rooms", {})
            if user_id in direct_rooms:
                rooms = direct_rooms[user_id]
                if rooms:
                    # Return the first known DM room for this user.
                    return list(rooms)[0]
            # No existing DM room found — create a new one via the API.
            # is_direct=True tags it as a DM in the homeserver's account data.
            resp = await self._client.room_create(
                is_direct=True,
                invite=[user_id],
            )
            return getattr(resp, "room_id", None)
        except Exception:
            return None

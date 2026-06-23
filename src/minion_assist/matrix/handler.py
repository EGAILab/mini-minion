"""Matrix inbound message handler.

``MatrixMessageHandler`` is the central routing point for inbound Matrix
room messages.  For each ``m.room.message`` event it:

1. Deduplicates (skips events already processed on a previous sync).
2. Skips the bot's own messages.
3. Applies bot-loop rate limiting.
4. Resolves the target agent from the room config or falls back to the default.
5. Checks the sender against the allowlist / group policy.
6. Posts an ack reaction if configured.
7. Dispatches the message to the appropriate ``AgentSession``.
8. Delivers the agent's response back to the room (full text, no streaming in v1).

Threading note
--------------
``handle_room_message`` is an async method called from the matrix-nio sync
callback in the background asyncio thread.  ``AgentSession.send()`` is
synchronous and runs in a thread-pool thread via ``loop.run_in_executor``
so it never blocks the asyncio event loop.
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

from .allowlist import check_allowlist, normalise_matrix_user_id
from .config import MatrixConfig

if TYPE_CHECKING:
    from .bot_loop import BotLoopProtection
    from .exec_approvals import MatrixExecApprovalHandler
    from .inbound_dedupe import MatrixInboundDeduper
    from .outbound import MatrixOutbound
    from .thread_bindings import MatrixThreadBindingManager


class MatrixMessageHandler:
    """Routes inbound Matrix room messages to AgentSession and delivers replies.

    Args:
        client:              Authenticated matrix-nio ``AsyncClient``.
        config:              Active :class:`~minion_assist.matrix.config.MatrixConfig`.
        sessions:            Dict mapping agent_id → ``AgentSession``.
        outbound:            :class:`~minion_assist.matrix.outbound.MatrixOutbound`.
        dedupe:              :class:`~minion_assist.matrix.inbound_dedupe.MatrixInboundDeduper`.
        bot_loop:            :class:`~minion_assist.matrix.bot_loop.BotLoopProtection`.
        thread_binding_mgr:  :class:`~minion_assist.matrix.thread_bindings.MatrixThreadBindingManager`.
        exec_approval_handler: Optional exec-approval handler for tool calls.
    """

    def __init__(
        self,
        client,
        config: MatrixConfig,
        sessions: dict,
        outbound: "MatrixOutbound",
        dedupe: "MatrixInboundDeduper",
        bot_loop: "BotLoopProtection",
        thread_binding_mgr: "MatrixThreadBindingManager",
        exec_approval_handler: "MatrixExecApprovalHandler | None" = None,
    ) -> None:
        self._client = client
        self._config = config
        self._sessions = sessions
        self._outbound = outbound
        self._dedupe = dedupe
        self._bot_loop = bot_loop
        self._thread_mgr = thread_binding_mgr
        self._exec_approval = exec_approval_handler

    async def handle_room_message(self, room, event) -> None:
        """Process one inbound Matrix room message event.

        Args:
            room:  matrix-nio ``MatrixRoom`` object.
            event: matrix-nio ``RoomMessageText`` (or similar) event object.
        """
        # getattr with a default safely extracts fields from nio event objects
        # even if a future SDK version renames them.
        event_id: str = getattr(event, "event_id", "") or ""
        sender: str = getattr(event, "sender", "") or ""
        room_id: str = getattr(room, "room_id", "") or ""
        body: str = getattr(event, "body", "") or ""

        # Ignore empty messages (e.g. image-only events with no text body).
        if not body.strip():
            return

        # Step 1 — Deduplication.
        # Matrix sync can replay events on reconnect; skip any we've already seen.
        if event_id and await self._dedupe.is_seen(event_id):
            return

        # Step 2 — Skip the bot's own messages.
        # Without this check the bot would respond to its own replies, creating
        # an infinite loop.
        if sender == self._client.user_id:
            return

        # Step 3 — Bot-loop suppression.
        # If another bot is posting messages rapidly, the rate limiter fires here
        # to prevent the two bots from endlessly replying to each other.
        if self._bot_loop.should_suppress(room_id):
            return

        # Step 4 — Resolve room config and pick the target agent.
        # Rooms can override which agent handles them; fall back to the default.
        room_cfg = self._config.groups.get(room_id)
        if room_cfg and not room_cfg.enabled:
            # Room is explicitly disabled in config.
            return

        agent_id = (room_cfg.agent if room_cfg else None) or self._config.default_agent_id
        # Walk fallback chain: per-room → default → first available.
        if agent_id not in self._sessions:
            agent_id = self._config.default_agent_id
        if agent_id not in self._sessions:
            agent_id = next(iter(self._sessions), None)
        if agent_id is None:
            return

        # Step 5 — Allowlist / group policy check.
        # Reject senders not permitted to trigger the agent in this room.
        normalised_sender = normalise_matrix_user_id(sender)
        if not self._is_sender_allowed(normalised_sender, room_id, room_cfg):
            return

        # Step 6 — Thread binding.
        # If this message is part of a Matrix thread, map the thread root event ID
        # to an isolated session key so threaded convos don't bleed into each other.
        thread_id = self._resolve_thread_id(event)
        session_key: str | None = None
        if thread_id and self._config.thread_bindings.enabled:
            session_key = await self._thread_mgr.get_or_create_session_key(
                thread_event_id=thread_id,
                room_id=room_id,
                agent_id=agent_id,
            )

        # Step 7 — Ack reaction.
        # Post a 👀 (or configured emoji) to acknowledge receipt before the agent
        # starts processing.  This gives the user instant visual feedback.
        if self._config.ack_reaction and event_id:
            try:
                await self._outbound.send_reaction(
                    room_id, event_id, self._config.ack_reaction
                )
            except Exception:
                pass  # reaction is best-effort; never block the agent for it

        # Step 8 — Dispatch to agent and send the reply.
        await self._dispatch_and_reply(
            room_id=room_id,
            event_id=event_id,
            text=body,
            agent_id=agent_id,
            thread_id=thread_id,
        )

    def _is_sender_allowed(self, sender: str, room_id: str, room_cfg) -> bool:
        """Return True if the sender is permitted to trigger the agent."""
        group_policy = self._config.group_policy

        # Per-room user allowlist takes precedence over the global policy.
        if room_cfg and room_cfg.users:
            return check_allowlist(sender, room_cfg.users)

        # Global group-allow-from list (applies to all rooms without a per-room list).
        if self._config.group_allow_from:
            return check_allowlist(sender, self._config.group_allow_from)

        # Fall through to the room-level policy.  "open" means anyone in the room
        # can trigger the agent; any other value defaults to deny.
        return group_policy == "open"

    def _resolve_thread_id(self, event) -> str | None:
        """Extract the thread root event ID if this event is a threaded reply."""
        # matrix-nio versions differ on whether relates_to is a dict or object,
        # so we check both forms.
        relates = getattr(event, "relates_to", None) or {}
        if isinstance(relates, dict):
            if relates.get("rel_type") == "m.thread":
                return relates.get("event_id")
        # Object form (newer matrix-nio versions).
        rel_type = getattr(relates, "rel_type", None)
        if rel_type == "m.thread":
            return getattr(relates, "event_id", None)
        return None

    async def _dispatch_and_reply(
        self,
        room_id: str,
        event_id: str,
        text: str,
        agent_id: str,
        thread_id: str | None,
    ) -> None:
        """Run the agent turn in a thread-pool thread and post the reply."""
        session = self._sessions[agent_id]
        loop = asyncio.get_running_loop()

        # Lists are used as simple mutable containers so the nested function
        # can write results back to the outer scope without `nonlocal`.
        result_holder: list[str] = []
        error_holder: list[Exception] = []

        def _run_sync():
            try:
                # session.send() is synchronous and can block for seconds while
                # the LLM generates a response.  It must NOT run on the asyncio
                # event loop thread or it would freeze all Matrix I/O.
                # on_event=None and stream=False: collect the full response
                # in one shot.  Streaming draft-preview is deferred (TODO).
                resp = session.send(text, on_event=None, stream=False)
                result_holder.append(resp or "")
            except Exception as exc:
                error_holder.append(exc)

        # Show a typing indicator while the agent is working.
        await self._outbound.send_typing(room_id, True)
        try:
            # run_in_executor() offloads _run_sync to a thread-pool thread so the
            # asyncio loop remains responsive to incoming Matrix events while the
            # LLM call blocks.  None → use the default ThreadPoolExecutor.
            await loop.run_in_executor(None, _run_sync)
        finally:
            # Always clear the typing indicator, even if the agent raised an error.
            await self._outbound.send_typing(room_id, False)

        if error_holder:
            print(
                f"[matrix] Agent '{agent_id}' error for event {event_id}: {error_holder[0]}",
                file=sys.stderr,
            )
            return

        reply_text = result_holder[0] if result_holder else ""
        if not reply_text.strip():
            return

        try:
            await self._outbound.send_text(room_id, reply_text, thread_id=thread_id)
        except Exception as exc:
            print(
                f"[matrix] Failed to send reply to room {room_id}: {exc}",
                file=sys.stderr,
            )

"""Matrix inbound message handler.

``MatrixMessageHandler`` is the central routing point for inbound Matrix
room messages.  For each ``m.room.message`` event (text or image — see
``monitor.py``'s callback registration) it:

1. Deduplicates (skips events already processed on a previous sync).
2. Skips the bot's own messages.
3. Applies bot-loop rate limiting.
4. Resolves the target agent from the room config or falls back to the default.
5. Checks the sender against the allowlist / group policy, and — for
   ``m.image`` events — downloads and stages the file as a
   ``MediaAttachment`` (``_download_image_attachment``), replying with an
   in-room error instead of silently dropping the message if that fails.
6. Posts an ack reaction if configured.
7. Dispatches the message (plus any staged image) to the appropriate
   ``AgentSession``.
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
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from .allowlist import check_allowlist, normalise_matrix_user_id
from .config import MatrixConfig
from ..heartbeat_token import is_heartbeat_ok, strip_heartbeat_token
from ..media import safe_filename, stage_attachment

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..agents.session import AgentSession
    from ..media import MediaAttachment
    from .bot_loop import BotLoopProtection
    from .exec_approvals import MatrixExecApprovalHandler
    from .inbound_dedupe import MatrixInboundDeduper
    from .outbound import MatrixOutbound
    from .room_sessions import MatrixRoomSessionManager

# Commands that make no sense outside a CLI REPL and must not be forwarded to
# the LLM or echoed back to the room as text.
_MATRIX_DISALLOWED_COMMANDS = frozenset({"/quit", "/exit", "/export"})


class MatrixMessageHandler:
    """Routes inbound Matrix room messages to AgentSession and delivers replies.

    Args:
        client:              Authenticated matrix-nio ``AsyncClient``.
        config:              Active :class:`~minion_assist.matrix.config.MatrixConfig`.
        sessions:            Dict mapping agent_id → ``AgentSession``. This is the
                              REPL's shared default session — normal Matrix traffic is
                              always dispatched to a room-scoped session instead
                              (MEM-GAP-001), built via ``session_factories``, never
                              this dict directly (R2-GAP-009: a missing factory now
                              fails the turn closed, it no longer falls back here).
                              Still the base ``CommandContext.sessions`` command
                              dispatch overrides with the room session for the calling
                              agent (R2-GAP-004) — legitimately process-global commands
                              (``/mcp-reload``, ``/diagnose``, etc.) still read every
                              other configured agent's entry from here.
        outbound:            :class:`~minion_assist.matrix.outbound.MatrixOutbound`.
        dedupe:              :class:`~minion_assist.matrix.inbound_dedupe.MatrixInboundDeduper`.
        bot_loop:            :class:`~minion_assist.matrix.bot_loop.BotLoopProtection`.
        room_session_mgr:    :class:`~minion_assist.matrix.room_sessions.MatrixRoomSessionManager`
                              — resolves the ``session_id`` bound to each ``(room_id, agent_id)``.
        session_factories:   Dict mapping agent_id → a callable that builds a fresh
                              ``AgentSession`` for that agent given a session_id
                              (built in ``minion.py``, sharing the agent's provider/
                              tools/memory). Every message's ``AgentSession`` comes
                              from here, keyed by room, instead of the one shared
                              per-agent entry in ``sessions``.
        db:                  Optional ``SessionDB`` instance — enables
                              ``/delete-session``'s cross-store cleanup
                              (MEM-GAP-003) when a database is configured.
        worker_health:       Dict mapping worker name → ``WorkerHealth``
                              (MEM-GAP-016) — enables ``/status deep`` from
                              Matrix chat.
        exec_approval_handler: Optional exec-approval handler for tool calls.
        media_dir:           Optional attachment store directory (same
                              ``{workspace}/attachments`` tree the REPL's
                              ``/attach`` command stages files into). When
                              set, inbound ``m.image`` events are downloaded
                              and staged here so they can be forwarded to
                              the LLM as multimodal attachments. When
                              ``None``, image messages are rejected with an
                              in-room explanation instead of silently
                              vanishing (the bug this parameter fixes — see
                              ``_download_image_attachment``'s docstring).
    """

    def __init__(
        self,
        client,
        config: MatrixConfig,
        sessions: dict,
        outbound: "MatrixOutbound",
        dedupe: "MatrixInboundDeduper",
        bot_loop: "BotLoopProtection",
        room_session_mgr: "MatrixRoomSessionManager",
        exec_approval_handler: "MatrixExecApprovalHandler | None" = None,
        agents_cfg: dict | None = None,
        session_store: object = None,
        mcp_manager: object = None,
        skills: dict | None = None,
        short_term: object = None,
        session_factories: "dict[str, Callable[[str], AgentSession]] | None" = None,
        db: object = None,
        worker_health: dict | None = None,
        media_dir: "Path | None" = None,
    ) -> None:
        self._client = client
        self._config = config
        self._sessions = sessions
        self._outbound = outbound
        self._dedupe = dedupe
        self._bot_loop = bot_loop
        self._room_session_mgr = room_session_mgr
        self._exec_approval = exec_approval_handler
        self._agents_cfg = agents_cfg
        self._session_store = session_store
        self._mcp_manager = mcp_manager
        self._skills = skills
        self._short_term = short_term
        self._db = db
        self._worker_health = worker_health
        self._session_factories = session_factories or {}
        self._media_dir = media_dir
        # Lazily built, then reused for the life of this handler — each
        # (agent_id, room_id) gets exactly one long-lived AgentSession
        # instance, the same way `sessions[agent_id]` is a long-lived
        # singleton for the REPL (MEM-GAP-001).
        self._room_sessions: dict[tuple[str, str], "AgentSession"] = {}

    def close_room_sessions(self) -> None:
        """Dispose every room session's provider resources (R2-GAP-013).

        Called once, during Matrix channel shutdown
        (``matrix/monitor.py``'s ``_cleanup``) — without it, each room that
        ever used a Codex-backed agent leaked its subprocess for the life
        of the bot process (each ``CodexProvider`` instance in
        ``self._room_sessions`` owns its own subprocess, never shared —
        see that class's docstring), only ever cleaned up by the OS when
        the whole process exited. A stateless provider (OpenAI-
        completions, Anthropic) has nothing to close, so this is a no-op
        for those rooms — see ``AgentSession.close()``'s docstring.
        """
        for session in self._room_sessions.values():
            try:
                session.close()
            except Exception:
                pass

    def _get_or_build_session(self, agent_id: str, room_id: str, session_id: str) -> "AgentSession | None":
        """Return this room's isolated ``AgentSession``, building it on first use.

        R2-GAP-009: returns ``None`` — fails *closed* — when no factory was
        wired for ``agent_id``, instead of the earlier fallback to the
        shared ``sessions[agent_id]`` (the REPL's default session). That
        fallback was a deliberate, accepted trade-off when room-scoping was
        first built (favoring availability — the agent still responds even
        when misconfigured), but it reintroduces the exact cross-room
        history sharing room-scoping exists to prevent, silently, for as
        long as the misconfiguration lasts. For a single-user deployment,
        refusing to respond until it's fixed is the safer default: a
        missing factory becomes immediately visible (the room gets an
        explicit error, not silence) rather than a quiet, ongoing isolation
        leak. Still logged loudly either way.
        """
        cache_key = (agent_id, room_id)
        cached = self._room_sessions.get(cache_key)
        if cached is not None:
            return cached
        factory = self._session_factories.get(agent_id)
        if factory is None:
            print(
                f"[matrix] ERROR: no room-session factory for agent '{agent_id}' — "
                f"refusing to process room {room_id}'s message rather than share history "
                "with every other room routed to this agent. Wire a session_factory for "
                "this agent in minion.py's Matrix construction.",
                file=sys.stderr,
            )
            return None
        session = factory(session_id)
        self._room_sessions[cache_key] = session
        return session

    async def handle_room_message(self, room, event) -> None:
        """Process one inbound Matrix room message event.

        Args:
            room:  matrix-nio ``MatrixRoom`` object.
            event: matrix-nio ``RoomMessageText`` or ``RoomMessageImage``
                event object (``monitor.py`` registers this callback for
                both types).
        """
        # getattr with a default safely extracts fields from nio event objects
        # even if a future SDK version renames them.
        event_id: str = getattr(event, "event_id", "") or ""
        sender: str = getattr(event, "sender", "") or ""
        room_id: str = getattr(room, "room_id", "") or ""
        body: str = getattr(event, "body", "") or ""

        # RoomMessageImage is a distinct nio event class from
        # RoomMessageText — isinstance, not msgtype string comparison,
        # because a bare (unspec'd) test MagicMock always evaluates False
        # here, leaving every existing text-path test unaffected.
        from nio import RoomMessageImage  # noqa: PLC0415 — lazy import

        is_image = isinstance(event, RoomMessageImage)

        # Ignore empty non-image messages (e.g. edits, redactions, or other
        # non-text events with no body). Image events proceed even with an
        # empty body — a caption-less image is still a real image to answer
        # questions about.
        if not is_image and not body.strip():
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

        # Step 5b — Mention gate.
        # When require_mention=True for this room, only respond when the bot's
        # userId appears in the message body.  This prevents the agent from
        # replying to every message in busy group chats.
        if room_cfg and room_cfg.require_mention:
            if not self._is_mentioned(body):
                return

        # Step 5c — Image download and staging.
        # Previously the bot never even saw m.image events (see class
        # docstring's "Threading note" and monitor.py's callback
        # registration), so a sent image silently vanished with no
        # feedback — the LLM never learned an image existed at all,
        # image-with-no-caption or not. Download the file from the
        # homeserver's media repo now and stage it through the same
        # media.stage_attachment() the REPL's /attach command uses, so the
        # rest of the turn (below) is identical to a text message except
        # for the extra `attachment` argument.
        attachment: "MediaAttachment | None" = None
        if is_image:
            attachment, error = await self._download_image_attachment(event)
            if attachment is None:
                thread_id = self._resolve_thread_id(event)
                await self._outbound.send_text(
                    room_id,
                    f"Couldn't process that image: {error}",
                    thread_id=thread_id,
                )
                return
            if not body.strip():
                # No caption — give the LLM a minimal prompt so the turn
                # still proceeds instead of looking like an empty message.
                body = "[Image attached]"

        # Step 6 — Room session resolution (MEM-GAP-001).
        # A room is this deployment's unit of conversation isolation (see
        # docs/adr/0006-room-scoped-matrix-sessions.md) — every room gets its
        # own session_id, resolved unconditionally, not just when a message
        # happens to be posted inside a Matrix thread.
        session_id = await self._room_session_mgr.get_or_create_session_id(room_id, agent_id)
        # thread_id is unrelated to session isolation: it's only used below to
        # reply inside the same Matrix thread UI-wise, if the incoming message
        # happened to be posted in one.
        thread_id = self._resolve_thread_id(event)

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
            session_id=session_id,
            thread_id=thread_id,
            room_cfg=room_cfg,
            attachment=attachment,
        )

    async def _download_image_attachment(
        self, event
    ) -> "tuple[MediaAttachment | None, str | None]":
        """Download an m.image event's file and stage it as a MediaAttachment.

        Returns a ``(attachment, error)`` pair where exactly one side is
        ``None`` — mirrors the shape callers need to either proceed or show
        an in-room error, without a separate exception type for what is
        routine, expected failure (bad upload, oversized file, no media_dir
        configured).

        Reuses ``media.stage_attachment()`` — the exact same validation
        (MIME sniffing, 15 MB size cap) and content-addressed storage the
        REPL's ``/attach`` command already goes through, so a Matrix image
        and a ``/attach``-ed file behave identically once staged.
        """
        if self._media_dir is None:
            return None, "image attachments are not configured on this bot"

        mxc = getattr(event, "url", None)
        if not mxc:
            return None, "message had no image URL"

        try:
            response = await self._client.download(mxc=mxc)
        except Exception as exc:
            return None, f"download failed ({exc})"

        from nio.responses import DownloadError  # noqa: PLC0415 — lazy import

        if isinstance(response, DownloadError):
            return None, response.message or "download failed"

        file_bytes = getattr(response, "body", None)
        if not file_bytes:
            return None, "downloaded file was empty"

        # event.body is whatever the sender's client put there — often an
        # actual filename, but for clients that support captions it's the
        # free-text caption instead (e.g. "what is this?", or the "图里三
        # 条线各是什么" caption from the original bug report this feature
        # fixes), which can contain characters illegal in a Windows path
        # (`?`, `:`, ...) or path-traversal segments. It is never used as a
        # path directly: Path(...).name first drops any directory
        # components, then safe_filename() (media.py's sanitizer, also
        # used for the final staged filename) strips everything but
        # alphanumerics/dot/dash/underscore, so the write below is always
        # confined to tmp_dir regardless of what body contains.
        # stage_attachment() sniffs the real MIME type from bytes, so an
        # imprecise or missing extension here doesn't affect validation —
        # it only affects the display name shown back to the user.
        raw_filename = getattr(event, "body", None) or "matrix-image"
        filename = safe_filename(Path(raw_filename).name) or "matrix-image"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / filename
            tmp_path.write_bytes(file_bytes)
            try:
                attachment = stage_attachment(tmp_path, self._media_dir)
            except (ValueError, FileNotFoundError) as exc:
                return None, str(exc)

        return attachment, None

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

    def _is_mentioned(self, body: str) -> bool:
        """Return True when the bot's userId appears in the message body."""
        user_id = self._client.user_id or ""
        # Match full user ID (e.g. @ada:example.org) or bare localpart (e.g. ada).
        localpart = user_id.split(":")[0].lstrip("@") if ":" in user_id else user_id.lstrip("@")
        body_lower = body.lower()
        return (
            user_id.lower() in body_lower
            or (bool(localpart) and localpart.lower() in body_lower)
        )

    async def _dispatch_and_reply(
        self,
        room_id: str,
        event_id: str,
        text: str,
        agent_id: str,
        session_id: str,
        thread_id: str | None,
        room_cfg=None,
        attachment: "MediaAttachment | None" = None,
    ) -> None:
        """Run the agent turn in a thread-pool thread and post the reply.

        attachment: An image staged by ``_download_image_attachment``, if
            this turn came from an ``m.image`` event. Forwarded to
            ``session.send()``'s existing multimodal ``attachments`` param
            unchanged — the same param the REPL's ``/attach`` command
            already exercises, so no new downstream handling is needed.
        """
        # R2-GAP-004: resolved *before* command interception below, not
        # after — a command handler (/new, /compact, /session, etc.) must
        # see this room's own isolated session, not the shared REPL
        # default. Building it unconditionally here (rather than only on
        # the non-command path, as before) is what fixes that.
        session = self._get_or_build_session(agent_id, room_id, session_id)
        if session is None:
            # R2-GAP-009: fail closed — no session_factory is wired for
            # this agent, so there is no isolated session to use and no
            # safe fallback. Refuse the turn (command or not) rather than
            # silently share history via the old shared-session fallback.
            await self._outbound.send_text(
                room_id,
                "This agent has no room-scoped session configured — refusing to respond "
                "rather than risk mixing this room's history with another's. "
                "Contact the operator.",
                thread_id=thread_id,
            )
            return

        # Slash command interception — only active when agents_cfg is provided.
        # Supports both / and ! prefix.  Element Web blocks unknown /commands
        # client-side (showing "Unrecognised command"); ! is the conventional
        # Matrix bot prefix that always passes through to the room unmodified.
        # We normalise !cmd → /cmd so parse_command can handle both uniformly.
        if self._agents_cfg is not None:
            from ..commands import CommandContext, dispatch_command, parse_command  # noqa: PLC0415
            _cmd_text = (
                "/" + text[1:]
                if text.startswith("!") and len(text) > 1 and not text[1].isspace()
                else text
            )
            parsed = parse_command(_cmd_text)
            if parsed is not None:
                cmd, args = parsed
                if cmd in _MATRIX_DISALLOWED_COMMANDS:
                    await self._outbound.send_text(
                        room_id,
                        f"`{cmd}` is not available in Matrix.",
                        thread_id=thread_id,
                    )
                    return
                # R2-GAP-004: every single-target command handler in
                # commands.py resolves "the session to act on" via
                # ctx.sessions.get(ctx.target_agent_id) — target_agent_id is
                # always this room's own agent_id here, so overriding just
                # that one entry (leaving every other agent's shared REPL
                # session untouched) redirects every such lookup to this
                # room's real session, with no per-command special-casing
                # and no change to the CLI REPL's own dispatch path
                # (self._sessions is never mutated, only a fresh dict is
                # built for this one call).
                _matrix_sessions = dict(self._sessions)
                _matrix_sessions[agent_id] = session
                ctx = CommandContext(
                    raw=text,
                    command=cmd,
                    args=args,
                    target_agent_id=agent_id,
                    sessions=_matrix_sessions,
                    agents_cfg=self._agents_cfg,
                    session_store=self._session_store,
                    mcp_manager=self._mcp_manager,
                    skills=self._skills,
                    short_term=self._short_term,
                    db=self._db,
                    worker_health=self._worker_health,
                )
                result = dispatch_command(ctx)
                if result.handled:
                    # R2-GAP-004: /session <arg> switches this room's live
                    # AgentSession to a different session_id in place
                    # (commands.py's switch_session() call) but has no way
                    # to persist that — only this handler has a reference
                    # to room_session_mgr. Without this, the switch would
                    # silently revert to the old binding on the next bot
                    # restart.
                    if result.new_session_id is not None:
                        await self._room_session_mgr.rebind(room_id, agent_id, result.new_session_id)
                    if result.message:
                        await self._outbound.send_text(room_id, result.message, thread_id=thread_id)
                    return  # consumed — skip LLM

        loop = asyncio.get_running_loop()

        # Build per-turn extra tools (injected for this turn only).
        extra_tools = []

        # ReactToMessageTool: available in rooms where reaction_level != "off".
        reaction_level = getattr(room_cfg, "reaction_level", "off") if room_cfg else "off"
        if reaction_level != "off" and event_id:
            from ..tools.react_to_message import ReactToMessageTool  # noqa: PLC0415
            extra_tools.append(
                ReactToMessageTool(
                    outbound=self._outbound,
                    room_id=room_id,
                    event_id=event_id,
                    loop=loop,
                )
            )

        # Per-turn system suffix from the room's custom system_prompt.
        system_suffix = getattr(room_cfg, "system_prompt", None) if room_cfg else None

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
                resp = session.send(
                    text,
                    attachments=[attachment] if attachment else None,
                    on_event=None,
                    stream=False,
                    extra_tools=extra_tools or None,
                    system_suffix=system_suffix,
                    channel=room_id,
                )
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

        # Suppress silent heartbeat acknowledgements — they are not user-facing.
        if is_heartbeat_ok(reply_text):
            return
        reply_text = strip_heartbeat_token(reply_text)

        if not reply_text.strip():
            return

        try:
            await self._outbound.send_text(room_id, reply_text, thread_id=thread_id)
        except Exception as exc:
            print(
                f"[matrix] Failed to send reply to room {room_id}: {exc}",
                file=sys.stderr,
            )

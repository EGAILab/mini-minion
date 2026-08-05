"""Matrix monitor — the long-running async coroutine that drives the channel.

``monitor_matrix`` connects to the homeserver, registers event callbacks, and
runs ``client.sync_forever()`` until the ``stop_event`` is set.

Lifecycle
---------
1. Authenticate via :func:`~minion_assist.matrix.auth.resolve_matrix_auth`.
2. Open the inbound-dedupe and room-session databases.
3. Construct the outbound adapter, bot-loop protection, exec-approval handler,
   and room-message handler.
4. Register matrix-nio callbacks for ``RoomMessageText``, ``InviteEvent``, and
   ``ReactionEvent`` (used by exec approvals).
5. Start ``client.sync_forever()`` in a cancellation-aware task.
6. On ``stop_event`` set: cancel the sync task and clean up.

This coroutine runs in a background daemon thread with its own asyncio event
loop.  The main thread's REPL continues unaffected.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from .auth import resolve_matrix_auth
from .auto_join import handle_invite
from .bot_loop import BotLoopProtection
from .config import MatrixConfig
from .exec_approvals import MatrixExecApprovalHandler
from .handler import MatrixMessageHandler
from .inbound_dedupe import MatrixInboundDeduper
from .outbound import MatrixOutbound
from .room_sessions import MatrixRoomSessionManager


async def monitor_matrix(
    config: MatrixConfig,
    sessions: dict,
    stop_event: asyncio.Event,
    workspace: Path,
    agents_cfg: dict | None = None,
    session_store: object = None,
    mcp_manager: object = None,
    skills: dict | None = None,
    short_term: object = None,
    session_factories: dict | None = None,
    db: object = None,
) -> None:
    """Run the Matrix sync loop until ``stop_event`` is set.

    Args:
        config:       Active :class:`~minion_assist.matrix.config.MatrixConfig`.
        sessions:     Dict mapping agent_id → ``AgentSession`` (shared with REPL;
                      used by the Matrix handler only as a last-resort fallback —
                      see :meth:`~minion_assist.matrix.handler.MatrixMessageHandler._get_or_build_session`).
        stop_event:   asyncio.Event; set by :class:`~minion_assist.matrix.channel.MatrixChannel`
                      to initiate a clean shutdown.
        workspace:    Root workspace path for database files.
        agents_cfg:   Agent model config dict — enables slash command dispatch when provided.
        session_store: SessionStore instance (for /agents command).
        mcp_manager:  McpClientManager instance (for /mcp-* commands).
        skills:       Loaded skill map (for /skills command).
        short_term:   ShortTermMemory instance (for /session and /rename commands).
        session_factories: Dict mapping agent_id → a callable that builds a
                      fresh ``AgentSession`` for that agent given a session_id
                      (MEM-GAP-001) — see ``minion.py``'s
                      ``matrix_session_factories``.
        db:           Optional ``SessionDB`` instance — enables
                      ``/delete-session``'s cross-store cleanup (MEM-GAP-003).
    """
    # All database files live under workspace/matrix/ so they're easy to find
    # and back up alongside other workspace data.
    matrix_dir = workspace / "matrix"
    dedupe_db = matrix_dir / "inbound_dedupe.db"
    room_sessions_db = matrix_dir / "room_sessions.db"

    # Step 1: authenticate.
    client = await resolve_matrix_auth(config)

    # Step 2: open supporting databases.
    dedupe = MatrixInboundDeduper(dedupe_db)
    await dedupe.start()

    room_session_mgr = MatrixRoomSessionManager(room_sessions_db)
    await room_session_mgr.start()

    # Step 3: build the outbound adapter and optional helpers.
    outbound = MatrixOutbound(client, config)
    bot_loop = BotLoopProtection(config.bot_loop)

    # Exec-approval handler is optional; only built when enabled in config.
    exec_approval: MatrixExecApprovalHandler | None = None
    if config.exec_approvals.enabled:
        exec_approval = MatrixExecApprovalHandler(outbound, config.exec_approvals)

    # Step 4: build the message handler that ties everything together.
    msg_handler = MatrixMessageHandler(
        client=client,
        config=config,
        sessions=sessions,
        outbound=outbound,
        dedupe=dedupe,
        bot_loop=bot_loop,
        room_session_mgr=room_session_mgr,
        exec_approval_handler=exec_approval,
        agents_cfg=agents_cfg,
        session_store=session_store,
        mcp_manager=mcp_manager,
        skills=skills,
        short_term=short_term,
        session_factories=session_factories,
        db=db,
    )

    try:
        # Import matrix-nio event types for use as callback type filters.
        from nio import InviteEvent, ReactionEvent, RoomMessageText  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "matrix-nio is not installed. "
            "Run: uv add --optional matrix matrix-nio aiosqlite"
        ) from exc

    # Step 5: register callbacks.  matrix-nio calls these for each matching event
    # during the sync loop.
    async def _on_room_message(room, event):
        await msg_handler.handle_room_message(room, event)

    async def _on_invite(room, event):
        await handle_invite(client, room.room_id, event.sender, config)

    client.add_event_callback(_on_room_message, RoomMessageText)
    client.add_event_callback(_on_invite, InviteEvent)

    # Reactions are exec approvals' confirm/deny mechanism.
    if exec_approval:
        async def _on_reaction(room, event):
            exec_approval.handle_reaction(event.reacts_to, event.key)

        client.add_event_callback(_on_reaction, ReactionEvent)

    print(f"[matrix] Connected as {config.user_id} on {config.homeserver}")

    # Step 6: start the sync loop in the background.
    # create_task() schedules the coroutine on the running event loop WITHOUT
    # blocking here — it runs concurrently while we await the stop signal.
    # timeout=30_000ms keeps long-polling efficient; full_state=True loads room
    # membership on the first sync so we can resolve room IDs immediately.
    sync_task = asyncio.create_task(
        client.sync_forever(timeout=30_000, full_state=True)
    )

    # Step 7: wait for either a fatal sync error OR a stop signal.
    # asyncio.wait() returns (done_set, pending_set) as soon as the first task
    # completes.  We use FIRST_COMPLETED so a stop() call exits immediately
    # rather than waiting for the sync loop to time out on its own.
    stop_waiter = asyncio.create_task(stop_event.wait())
    done, pending = await asyncio.wait(
        [sync_task, stop_waiter],
        return_when=asyncio.FIRST_COMPLETED,
    )
    # Cancel whichever task didn't finish first and wait for it to clean up.
    for task in pending:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    if sync_task in done and not sync_task.cancelled():
        exc = sync_task.exception()
        if exc:
            print(f"[matrix] Sync loop exited with error: {exc}", file=sys.stderr)

    await _cleanup(client, dedupe, room_session_mgr)
    print("[matrix] Disconnected.")


async def _cleanup(
    client, dedupe: MatrixInboundDeduper, room_session_mgr: MatrixRoomSessionManager
) -> None:
    # Each step is wrapped individually: one failure shouldn't prevent the others
    # from running during shutdown.
    try:
        await client.close()
    except Exception:
        pass
    try:
        await dedupe.stop()
    except Exception:
        pass
    try:
        await room_session_mgr.stop()
    except Exception:
        pass

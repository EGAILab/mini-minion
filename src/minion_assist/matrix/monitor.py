"""Matrix monitor — the long-running async coroutine that drives the channel.

``monitor_matrix`` connects to the homeserver, registers event callbacks, and
runs ``client.sync_forever()`` until the ``stop_event`` is set.

Lifecycle
---------
1. Authenticate via :func:`~minion_assist.matrix.auth.resolve_matrix_auth`.
2. Set up E2E crypto if ``config.encryption`` is True.
3. Open the inbound-dedupe and thread-binding databases.
4. Construct the outbound adapter, bot-loop protection, exec-approval handler,
   and room-message handler.
5. Register matrix-nio callbacks for ``RoomMessageText`` and ``InviteEvent``.
6. Start ``client.sync_forever()`` in a cancellation-aware task.
7. On ``stop_event`` set: cancel the sync task and clean up.

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
from .crypto import setup_crypto
from .exec_approvals import MatrixExecApprovalHandler
from .handler import MatrixMessageHandler
from .inbound_dedupe import MatrixInboundDeduper
from .outbound import MatrixOutbound
from .thread_bindings import MatrixThreadBindingManager


async def monitor_matrix(
    config: MatrixConfig,
    sessions: dict,
    stop_event: asyncio.Event,
    workspace: Path,
) -> None:
    """Run the Matrix sync loop until ``stop_event`` is set.

    Args:
        config:     Active :class:`~minion_assist.matrix.config.MatrixConfig`.
        sessions:   Dict mapping agent_id → ``AgentSession`` (shared with REPL).
        stop_event: asyncio.Event; set by :class:`~minion_assist.matrix.channel.MatrixChannel`
                    to initiate a clean shutdown.
        workspace:  Root workspace path for database files.
    """
    # All database files live under workspace/matrix/ so they're easy to find
    # and back up alongside other workspace data.
    matrix_dir = workspace / "matrix"
    dedupe_db = matrix_dir / "inbound_dedupe.db"
    thread_db = matrix_dir / "thread_bindings.db"
    crypto_db = matrix_dir / "crypto.db"

    # Step 1: authenticate.  Returns a fully configured AsyncClient.
    client = await resolve_matrix_auth(config)

    # Step 2: optional E2E encryption.  Skipped if encryption=false in config.
    if config.encryption:
        await setup_crypto(client, crypto_db)

    # Step 3: open supporting databases.
    dedupe = MatrixInboundDeduper(dedupe_db)
    await dedupe.start()

    thread_mgr = MatrixThreadBindingManager(thread_db, config.thread_bindings)
    await thread_mgr.start()

    # Step 4: build the outbound adapter and optional helpers.
    outbound = MatrixOutbound(client, config)
    bot_loop = BotLoopProtection(config.bot_loop)

    # Exec-approval handler is optional; only built when enabled in config.
    exec_approval: MatrixExecApprovalHandler | None = None
    if config.exec_approvals.enabled:
        exec_approval = MatrixExecApprovalHandler(client, outbound, config.exec_approvals)

    # Step 5: build the message handler that ties everything together.
    msg_handler = MatrixMessageHandler(
        client=client,
        config=config,
        sessions=sessions,
        outbound=outbound,
        dedupe=dedupe,
        bot_loop=bot_loop,
        thread_binding_mgr=thread_mgr,
        exec_approval_handler=exec_approval,
    )

    try:
        # Import matrix-nio event types for use as callback type filters.
        from nio import InviteEvent, RoomMessageText  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "matrix-nio is not installed. "
            "Run: uv add --optional matrix 'matrix-nio[e2e]' aiosqlite"
        ) from exc

    # Step 6: register callbacks.  matrix-nio calls these for each matching event
    # during the sync loop.
    async def _on_room_message(room, event):
        await msg_handler.handle_room_message(room, event)

    async def _on_invite(room, event):
        await handle_invite(client, room.room_id, event.sender, config)

    client.add_event_callback(_on_room_message, RoomMessageText)
    client.add_event_callback(_on_invite, InviteEvent)

    print(f"[matrix] Connected as {config.user_id} on {config.homeserver}")

    # Step 7: start the sync loop in the background.
    # create_task() schedules the coroutine on the running event loop WITHOUT
    # blocking here — it runs concurrently while we await the stop signal.
    # timeout=30_000ms keeps long-polling efficient; full_state=True loads room
    # membership on the first sync so we can resolve room IDs immediately.
    sync_task = asyncio.create_task(
        client.sync_forever(timeout=30_000, full_state=True)
    )

    # Step 8: wait for either a fatal sync error OR a stop signal.
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

    await _cleanup(client, dedupe, thread_mgr)
    print("[matrix] Disconnected.")


async def _cleanup(client, dedupe: MatrixInboundDeduper, thread_mgr: MatrixThreadBindingManager) -> None:
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
        await thread_mgr.stop()
    except Exception:
        pass

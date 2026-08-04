"""Matrix monitor — the long-running async coroutine that drives the channel.

``monitor_matrix`` connects to the homeserver, registers event callbacks, and
runs ``client.sync_forever()`` until the ``stop_event`` is set.

Lifecycle
---------
1. Authenticate via :func:`~minion_assist.matrix.auth.resolve_matrix_auth`,
   which also wires up E2E crypto if ``config.encryption`` is True.
2. Upload device keys if encryption activated (:func:`~minion_assist.matrix.crypto.setup_crypto`).
3. Open the inbound-dedupe and thread-binding databases.
4. Construct the outbound adapter, bot-loop protection, exec-approval handler,
   verification handler, and room-message handler.
5. Register matrix-nio callbacks for ``RoomMessageText``, ``InviteEvent``,
   ``ReactionEvent`` (shared by exec approvals and verification), and — when
   ``config.verification.enabled`` — the ``KeyVerification*`` to-device events.
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
from .verification import MatrixVerificationHandler


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
) -> None:
    """Run the Matrix sync loop until ``stop_event`` is set.

    Args:
        config:       Active :class:`~minion_assist.matrix.config.MatrixConfig`.
        sessions:     Dict mapping agent_id → ``AgentSession`` (shared with REPL).
        stop_event:   asyncio.Event; set by :class:`~minion_assist.matrix.channel.MatrixChannel`
                      to initiate a clean shutdown.
        workspace:    Root workspace path for database files.
        agents_cfg:   Agent model config dict — enables slash command dispatch when provided.
        session_store: SessionStore instance (for /agents command).
        mcp_manager:  McpClientManager instance (for /mcp-* commands).
        skills:       Loaded skill map (for /skills command).
        short_term:   ShortTermMemory instance (for /session and /rename commands).
    """
    # All database files live under workspace/matrix/ so they're easy to find
    # and back up alongside other workspace data.
    matrix_dir = workspace / "matrix"
    dedupe_db = matrix_dir / "inbound_dedupe.db"
    thread_db = matrix_dir / "thread_bindings.db"
    # A directory, not a single file: matrix-nio's SqliteStore derives its own
    # filename (f"{user_id}_{device_id}.db") inside whatever path it's given.
    crypto_dir = matrix_dir / "crypto"

    # Step 1: authenticate.  Also wires up E2E encryption when config.encryption
    # is True — matrix-nio requires store_path to be known at AsyncClient
    # construction time, so this can't happen as a separate step afterwards.
    client = await resolve_matrix_auth(config, crypto_store_dir=crypto_dir)

    # Step 2: upload device keys if encryption actually activated above.
    if config.encryption:
        await setup_crypto(client)

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
        exec_approval = MatrixExecApprovalHandler(outbound, config.exec_approvals)

    # Verification handler is optional too. It reuses exec_approvals.approvers
    # as its allowlist rather than a second "people Ada trusts" list.
    verification: MatrixVerificationHandler | None = None
    if config.verification.enabled:
        verification = MatrixVerificationHandler(
            client, outbound, config.exec_approvals.approvers
        )

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
        agents_cfg=agents_cfg,
        session_store=session_store,
        mcp_manager=mcp_manager,
        skills=skills,
        short_term=short_term,
    )

    try:
        # Import matrix-nio event types for use as callback type filters.
        from nio import InviteEvent, ReactionEvent, RoomMessageText  # noqa: PLC0415
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

    # Reactions are the shared confirm/deny mechanism for both exec approvals
    # and device verification — one callback, dispatched to whichever handler
    # (if any) is waiting on that specific event ID.
    if exec_approval or verification:
        async def _on_reaction(room, event):
            if exec_approval:
                exec_approval.handle_reaction(event.reacts_to, event.key)
            if verification:
                await verification.handle_reaction(event.reacts_to, event.key)

        client.add_event_callback(_on_reaction, ReactionEvent)

    # Device verification runs entirely over to-device events — matrix-nio
    # handles the crypto protocol internally, these callbacks just decide
    # whether to accept and show the human the emoji to confirm.
    if verification:
        try:
            from nio import (  # noqa: PLC0415
                KeyVerificationCancel,
                KeyVerificationKey,
                KeyVerificationMac,
                KeyVerificationStart,
            )
        except ImportError:
            pass
        else:
            async def _on_verification_start(event):
                await verification.handle_verification_start(event)

            async def _on_verification_key(event):
                await verification.handle_verification_key(event)

            async def _on_verification_mac(event):
                await verification.handle_verification_mac(event)

            async def _on_verification_cancel(event):
                await verification.handle_verification_cancel(event)

            client.add_to_device_callback(_on_verification_start, KeyVerificationStart)
            client.add_to_device_callback(_on_verification_key, KeyVerificationKey)
            client.add_to_device_callback(_on_verification_mac, KeyVerificationMac)
            client.add_to_device_callback(_on_verification_cancel, KeyVerificationCancel)

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

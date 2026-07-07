"""MatrixChannel — lifecycle manager for the Matrix background listener.

``MatrixChannel`` spawns a daemon thread that runs its own asyncio event loop.
The loop runs :func:`~minion_assist.matrix.monitor.monitor_matrix` until
``stop()`` is called, at which point a threading.Event signals the asyncio
``stop_event`` and the thread is joined with a 5-second timeout.

The main thread's REPL loop is unaffected: both run concurrently and share
the same ``AgentSession`` dict passed in via ``start(sessions)``.

Usage::

    channel = MatrixChannel(config, workspace)
    channel.start(sessions)    # non-blocking; spawns daemon thread
    # ... REPL runs here ...
    channel.stop()             # blocks until listener shuts down (≤5s)
"""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

from .config import MatrixConfig


class MatrixChannel:
    """Manages the Matrix listener background thread.

    Args:
        config:    Validated :class:`~minion_assist.matrix.config.MatrixConfig`.
        workspace: Workspace root path for database files (passed to monitor).
    """

    def __init__(self, config: MatrixConfig, workspace: Path) -> None:
        self._config = config
        self._workspace = workspace
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._started = False

    def start(self, sessions: dict) -> None:
        """Start the Matrix listener in a background daemon thread.

        Args:
            sessions: Dict mapping agent_id → ``AgentSession``.  Shared with REPL.
        """
        if self._started:
            return
        self._started = True
        # daemon=True means the thread is killed automatically when the main
        # process exits — no need to call stop() on a clean REPL exit.
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(sessions,),
            daemon=True,
            name="matrix-listener",  # visible in debuggers and stack traces
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the listener to stop and wait up to 5 seconds for it to exit."""
        if not self._started or self._thread is None:
            return
        if self._loop is not None and self._stop_event is not None:
            # call_soon_threadsafe is the correct way to schedule a callback on
            # an asyncio event loop from a *different* thread.  Direct calls to
            # asyncio primitives from the wrong thread cause undefined behaviour.
            self._loop.call_soon_threadsafe(self._stop_event.set)
        # join() blocks the calling thread until the listener finishes or times out.
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            print("[matrix] Warning: listener thread did not stop cleanly.", file=sys.stderr)

    def _run_loop(self, sessions: dict) -> None:
        """Entry point for the background daemon thread."""
        # Each thread needs its own asyncio event loop — loops are not thread-safe.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        # asyncio.Event is the async equivalent of threading.Event.
        # It lives inside this thread's event loop.
        self._stop_event = asyncio.Event()
        try:
            loop.run_until_complete(self._run_monitor(sessions))
        except Exception as exc:
            print(f"[matrix] Listener crashed: {exc}", file=sys.stderr)
        finally:
            loop.close()
            self._loop = None

    async def _run_monitor(self, sessions: dict) -> None:
        """Launch the Matrix monitor coroutine, stopping when ``_stop_event`` is set."""
        # Lazy import: importing monitor pulls in matrix-nio which loads libolm.
        # We defer this until the background thread actually starts so the CLI
        # can import minion_assist quickly even if matrix-nio is not installed.
        from .monitor import monitor_matrix  # noqa: PLC0415 — lazy import avoids nio at startup

        await monitor_matrix(
            config=self._config,
            sessions=sessions,
            stop_event=self._stop_event,
            workspace=self._workspace,
        )

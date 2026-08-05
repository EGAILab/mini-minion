"""``MemoryIndexWatcher`` — live debounced filesystem sync for the lexical index.

Stage One Phase 3, slice B.

Why this exists, given write-path sync already updates the index
--------------------------------------------------------------------
``MemoryService``'s write methods (``remember``, ``remember_import``,
``append_daily``, ``delete``) already update the index synchronously for
everything *the app itself* writes. That leaves exactly one gap: an edit
made entirely outside the app — e.g. a user hand-editing ``MEMORY.md`` in a
text editor while minion-assist is running. This watcher exists only to
close that gap. It is not the primary sync mechanism and is not required
for correctness — startup reconciliation (``minion.py``) already catches
any out-of-band edit made while the process *wasn't* running, the same way
``session/db.py``'s ``reconcile_all_sessions`` catches a crash mid-mirror.

Why debounce?
-------------
A single save in most editors produces several filesystem events in quick
succession (e.g. a temp-file write followed by a rename). Reconciling on
every individual event would mean redundant reindex work for one logical
edit. This collects events per agent and waits for a short quiet period
before reconciling, so a burst of edits to the same agent's files triggers
one reconcile, not several.

Why ``reconcile_agent``, not ``rebuild_agent``?
--------------------------------------------------
``reconcile_agent`` only touches files whose content hash actually
changed — the right cost for "something changed somewhere in this agent's
memory directory," which is all a filesystem event tells us (it doesn't
reliably say *which* file or what changed, especially once several events
are collapsed by debouncing).

Talks to
--------
- ``memory/postgres_index.py`` — :meth:`PostgresMemoryIndex.reconcile_agent`,
  called once per debounced burst.
- ``memory/files.py`` — each watched agent's :class:`MemoryFileRepository`
  supplies both the directory to watch (:attr:`MemoryFileRepository.root`)
  and the fresh file listing to reconcile against
  (:meth:`MemoryFileRepository.list_indexable_files`).
- ``minion.py`` — constructs one :class:`MemoryIndexWatcher` at startup
  (only when a database is configured) covering every configured agent, and
  calls :meth:`start`/:meth:`stop`, the same lifecycle shape as
  ``memory/capture_worker.py``'s ``CaptureWorker``.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..worker_health import WorkerHealth
    from .files import MemoryFileRepository
    from .postgres_index import PostgresMemoryIndex

_log = logging.getLogger("minion_assist.memory_watcher")

# How long to wait, after the most recent filesystem event for an agent,
# before reconciling that agent's index. Long enough to collapse a typical
# editor's multi-event save into one reconcile; short enough that an
# external edit is reflected in search results well within the same
# conversation.
_DEBOUNCE_SECONDS = 1.0

# How often the debounce loop checks whether any pending agent's quiet
# period has elapsed. Independent of _DEBOUNCE_SECONDS — this just bounds
# how promptly a due reconcile actually fires once its debounce expires.
_POLL_INTERVAL_SECONDS = 0.25


class MemoryIndexWatcher:
    """Watches every configured agent's workspace root for on-disk memory edits.

    Args:
        index: The shared lexical index to reconcile into.
        agents: ``{agent_id: MemoryFileRepository}`` — every agent whose
            workspace root should be watched. One ``watchdog`` observer
            covers all of them; each agent's own repository supplies both
            the directory to watch and the listing to reconcile against.
        health: Optional :class:`~minion_assist.worker_health.WorkerHealth`
            (MEM-GAP-016) — see ``CaptureWorker``'s matching parameter.
            A "poll" is recorded once per debounce-loop iteration (whether
            or not any agent was actually due); "success"/"failure" are
            recorded per agent reconcile.
    """

    def __init__(
        self,
        index: PostgresMemoryIndex,
        agents: dict[str, MemoryFileRepository],
        health: "WorkerHealth | None" = None,
    ) -> None:
        self._index = index
        self._agents = agents
        self._health = health
        self._observer = None
        self._stop_event = threading.Event()
        self._debounce_thread: threading.Thread | None = None
        # agent_id -> monotonic time.monotonic() timestamp when this agent's
        # debounce window is due to elapse. Protected by _lock since
        # watchdog event callbacks run on the observer's own thread.
        self._pending: dict[str, float] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start watching every configured agent's workspace root.

        Requires the optional ``watchdog`` package (installed via the
        ``postgres`` extra — see ``pyproject.toml``). Imported lazily so
        this module can be imported (and its debounce logic unit-tested)
        without ``watchdog`` installed, matching ``session/db.py``'s lazy
        ``psycopg`` import for the same reason.
        """
        from watchdog.events import FileSystemEventHandler  # noqa: PLC0415
        from watchdog.observers import Observer  # noqa: PLC0415

        class _Handler(FileSystemEventHandler):
            def __init__(handler_self, agent_id: str) -> None:
                handler_self._agent_id = agent_id

            def on_any_event(handler_self, event) -> None:
                if event.is_directory or not str(event.src_path).endswith(".md"):
                    return
                self._schedule(handler_self._agent_id)

        self._observer = Observer()
        for agent_id, repo in self._agents.items():
            self._observer.schedule(_Handler(agent_id), str(repo.root), recursive=True)
        self._observer.start()

        self._debounce_thread = threading.Thread(
            target=self._debounce_loop, daemon=True, name="memory-index-watcher"
        )
        self._debounce_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop watching and wait for both the observer and debounce loop to drain."""
        self._stop_event.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=timeout)
        if self._debounce_thread is not None:
            self._debounce_thread.join(timeout=timeout)

    def _schedule(self, agent_id: str) -> None:
        """Record a filesystem event for ``agent_id`` just now, (re)starting its debounce window."""
        with self._lock:
            self._pending[agent_id] = time.monotonic() + _DEBOUNCE_SECONDS

    def _due_agents(self, now: float) -> list[str]:
        """Pop and return every agent whose debounce window has elapsed by ``now``."""
        due = []
        with self._lock:
            for agent_id, due_at in list(self._pending.items()):
                if now >= due_at:
                    due.append(agent_id)
                    del self._pending[agent_id]
        return due

    def _debounce_loop(self) -> None:
        """Poll for agents whose debounce window has elapsed and reconcile them."""
        while not self._stop_event.is_set():
            if self._health is not None:
                self._health.record_poll()
            for agent_id in self._due_agents(time.monotonic()):
                self._reconcile(agent_id)
            self._stop_event.wait(_POLL_INTERVAL_SECONDS)

    def _reconcile(self, agent_id: str) -> None:
        """Reconcile one agent's index against its current on-disk files.

        Never raises — a failed reconcile here must not kill the watcher
        thread; the next filesystem event (or the next process restart's
        startup reconciliation) will simply try again.
        """
        repo = self._agents.get(agent_id)
        if repo is None:
            return
        try:
            self._index.reconcile_agent(agent_id, repo.list_indexable_files())
            if self._health is not None:
                self._health.record_success()
        except Exception as exc:
            _log.debug(
                "Watcher reconcile failed for agent %s: %s: %s", agent_id, type(exc).__name__, exc
            )
            if self._health is not None:
                self._health.record_failure(f"{type(exc).__name__}: {exc}")

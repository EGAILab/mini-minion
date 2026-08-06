"""``MemoryRetentionScheduler`` — bounded cleanup of operational/telemetry tables (MEM-GAP-015).

Structurally mirrors ``memory/digest_scheduler.py``'s
``KnowledgeDigestScheduler``: the same daily wall-clock shape (reuses
:func:`~minion_assist.dreaming._seconds_until_next` rather than
re-deriving timezone/DST-sensitive scheduling logic), but each pass calls
:meth:`~minion_assist.session.db.SessionDB.prune_operational_tables` and
:meth:`~minion_assist.memory.postgres_index.PostgresMemoryIndex.prune_operational_tables`
instead of compiling a digest.

Scope, deliberately narrow
---------------------------
This only ever deletes rows from tables that are pure operational
bookkeeping — completed/failed job-queue rows, recall telemetry, and
review-preview drafts — never messages, memory files, knowledge-graph
claims, or ``memory_topic_revisions`` (the rollback undo-stack). See each
``prune_operational_tables`` method's own docstring for the exact table
list and why. Indefinite retention remains the default for everything
this scheduler doesn't touch, matching the gap analysis's own framing:
"indefinite retention matches the requested default... archive/retention/
cleanup [should be] configurable... distinct from deletion."

Talks to
--------
- ``session/db.py`` — :meth:`~minion_assist.session.db.SessionDB.prune_operational_tables`.
- ``memory/postgres_index.py`` — :meth:`~minion_assist.memory.postgres_index.PostgresMemoryIndex.prune_operational_tables`.
- ``dreaming.py`` — :func:`~minion_assist.dreaming._seconds_until_next`
  (shared scheduling primitive, not duplicated).
- ``minion.py`` — constructs and starts one instance per process, only
  when ``memory_retention.enabled`` and a database is configured. The
  lexical index (``PostgresMemoryIndex``) is optional — its half of the
  cleanup is simply skipped when no index is configured, same as every
  other optional-index code path in this codebase.
"""

from __future__ import annotations

import sys
import threading
from typing import TYPE_CHECKING

from ..dreaming import _seconds_until_next

if TYPE_CHECKING:
    from ..config import MemoryRetentionConfig
    from ..session.db import SessionDB
    from ..worker_health import WorkerHealth
    from .postgres_index import PostgresMemoryIndex


class MemoryRetentionScheduler:
    """Runs a daily operational-table cleanup pass in a daemon thread.

    Args:
        cfg: Resolved :class:`~minion_assist.config.MemoryRetentionConfig`.
        db: The :class:`~minion_assist.session.db.SessionDB` whose
            completed job-queue tables get pruned.
        index: The lexical index (if configured) whose recall-event and
            preview tables get pruned. ``None`` skips that half of the
            pass — a file-only-index deployment still gets job-table
            cleanup from ``db`` alone.
        health: Optional :class:`~minion_assist.worker_health.WorkerHealth`
            (MEM-GAP-016) — see ``KnowledgeDigestScheduler``'s matching
            parameter.
    """

    def __init__(
        self,
        cfg: MemoryRetentionConfig,
        db: SessionDB,
        index: "PostgresMemoryIndex | None",
        health: "WorkerHealth | None" = None,
    ) -> None:
        self._cfg = cfg
        self._db = db
        self._index = index
        self._health = health
        self._timer: threading.Timer | None = None
        self._stopped = False

    def start(self) -> None:
        """Schedule the first pass at the next configured wall-clock time."""
        delay = _seconds_until_next(self._cfg.hour, self._cfg.minute, self._cfg.timezone)
        self._timer = threading.Timer(delay, self._fire)
        self._timer.daemon = True
        self._timer.name = "memory-retention-scheduler"
        self._timer.start()

    def stop(self) -> None:
        """Cancel the pending timer and prevent future firings."""
        self._stopped = True
        if self._timer is not None:
            self._timer.cancel()

    def _fire(self) -> None:
        """Timer callback: run one pass then reschedule for tomorrow."""
        if self._stopped:
            return
        if self._health is not None:
            self._health.record_poll()
        try:
            self._run_pass()
        except Exception as exc:
            print(f"[memory-retention] Error during pass: {exc}", file=sys.stderr)
            if self._health is not None:
                self._health.record_failure(f"{type(exc).__name__}: {exc}")
        finally:
            # Always reschedule so the loop continues even after errors.
            if not self._stopped:
                self.start()

    def _run_pass(self) -> None:
        """Prune terminal-state job rows, then telemetry/preview rows if an index is configured."""
        counts = self._db.prune_operational_tables(self._cfg.retention_days)
        if self._index is not None:
            counts.update(self._index.prune_operational_tables(self._cfg.retention_days))
        total = sum(counts.values())
        if total:
            print(f"[memory-retention] Pruned {total} operational row(s): {counts}")
        if self._health is not None:
            self._health.record_success()

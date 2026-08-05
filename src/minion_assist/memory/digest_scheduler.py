"""``KnowledgeDigestScheduler`` — compiles and writes ``KNOWLEDGE_DIGEST.md`` on a timer.

Stage One Phase 7, slice D. Structurally mirrors
``memory/consolidation_scheduler.py``'s ``MemoryConsolidationScheduler``:
the same daily wall-clock shape (reuses
:func:`~minion_assist.dreaming._seconds_until_next` rather than
re-deriving timezone/DST-sensitive scheduling logic), but each pass
fetches the configured agent's ``status="supported"`` claims, renders them
with :func:`~minion_assist.memory.knowledge.compile_digest`, and
overwrites ``KNOWLEDGE_DIGEST.md`` via
:meth:`~minion_assist.memory.files.MemoryFileRepository.write_digest`
instead of drafting a consolidation preview.

Never reads or edits any topic note — this scheduler only ever reads
already-synced claims out of ``kb_claims`` (Postgres, the derived cache)
and writes the one file it owns outright. Nothing here can conflict with
a human hand-editing a topic note, the way ``MemoryConsolidator``'s
preview/apply flow has to guard against.

Talks to
--------
- ``memory/knowledge.py`` — :func:`~minion_assist.memory.knowledge.compile_digest`.
- ``memory/postgres_index.py`` — :meth:`PostgresMemoryIndex.list_claims`.
- ``memory/files.py`` — :meth:`~minion_assist.memory.files.MemoryFileRepository.write_digest`.
- ``dreaming.py`` — :func:`~minion_assist.dreaming._seconds_until_next`
  (shared scheduling primitive, not duplicated).
- ``minion.py`` — constructs and starts one instance per process, only
  when ``knowledge_digest.enabled`` and a database + lexical index are
  both configured.
"""

from __future__ import annotations

import sys
import threading
from typing import TYPE_CHECKING

from ..dreaming import _seconds_until_next
from .knowledge import compile_digest

if TYPE_CHECKING:
    from ..config import KnowledgeDigestConfig
    from ..worker_health import WorkerHealth
    from .files import MemoryFileRepository
    from .postgres_index import PostgresMemoryIndex


class KnowledgeDigestScheduler:
    """Runs a daily digest-compilation pass in a daemon thread.

    Args:
        cfg: Resolved :class:`~minion_assist.config.KnowledgeDigestConfig`.
        index: The lexical index ``kb_claims`` lives in, used to fetch
            ``status="supported"`` claims.
        files: The :class:`~minion_assist.memory.files.MemoryFileRepository`
            for ``cfg.agent_id`` — already bound to that agent's workspace
            root, so :meth:`~minion_assist.memory.files.MemoryFileRepository.write_digest`
            writes to the right place.
        health: Optional :class:`~minion_assist.worker_health.WorkerHealth`
            (MEM-GAP-016) — see ``MemoryConsolidationScheduler``'s matching
            parameter.
    """

    def __init__(
        self,
        cfg: KnowledgeDigestConfig,
        index: PostgresMemoryIndex,
        files: MemoryFileRepository,
        health: "WorkerHealth | None" = None,
    ) -> None:
        self._cfg = cfg
        self._index = index
        self._files = files
        self._health = health
        self._timer: threading.Timer | None = None
        self._stopped = False

    def start(self) -> None:
        """Schedule the first pass at the next configured wall-clock time."""
        delay = _seconds_until_next(self._cfg.hour, self._cfg.minute, self._cfg.timezone)
        self._timer = threading.Timer(delay, self._fire)
        self._timer.daemon = True
        self._timer.name = "knowledge-digest-scheduler"
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
            print(f"[knowledge-digest] Error during pass: {exc}", file=sys.stderr)
            if self._health is not None:
                self._health.record_failure(f"{type(exc).__name__}: {exc}")
        finally:
            # Always reschedule so the loop continues even after errors.
            if not self._stopped:
                self.start()

    def _run_pass(self) -> None:
        """Compile the configured agent's supported claims and overwrite the digest file."""
        claims = self._index.list_claims(self._cfg.agent_id, status="supported")
        digest = compile_digest(claims, max_chars=self._cfg.max_chars)
        self._files.write_digest(digest)
        if self._health is not None:
            self._health.record_success()

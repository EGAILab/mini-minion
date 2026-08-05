"""``MemoryConsolidationScheduler`` — automated preview drafting on a timer.

Stage One Phase 5, slice D, Task 9: "keep the poetic ``DreamingScheduler``
independently configurable; rename the consolidation schedule in code/UI
to avoid ambiguity." This is that separately-configured schedule — the
same daily wall-clock shape as ``dreaming.py``'s ``DreamingScheduler``
(reuses its :func:`~minion_assist.dreaming._seconds_until_next` rather
than re-deriving timezone/DST-sensitive scheduling logic), but calls
:meth:`~minion_assist.memory.consolidation.MemoryConsolidator.preview`
for an agent's top-ranked pending proposals instead of writing a diary
entry.

Never applies, promotes, or rejects anything — only drafts previews. This
matches the 100% human-gated design Phase 5 settled on throughout: this
scheduler exists purely to keep a human's review queue
(``minion-assist memory consolidate list``) populated with fresh drafts,
so reviewing doesn't require a manual ``preview`` call per proposal
first.

Skips proposals that already have a preview
---------------------------------------------
A proposal sitting ``"pending"`` for many days (waiting on a human to get
around to reviewing it) would otherwise get *redrafted every single day*
this scheduler fires — wasted LLM calls for no new information. Each pass
only drafts a proposal that has zero existing
``memory_consolidation_previews`` rows yet, so ``top_n`` really means "at
most this many *new* drafts per run," not "reconsider the top N every
time." A human who wants a fresh draft after editing something can always
run ``memory consolidate preview`` manually.

Talks to
--------
- ``memory/consolidation.py`` — :func:`~minion_assist.memory.consolidation.rank_proposals`,
  :class:`~minion_assist.memory.consolidation.MemoryConsolidator`.
- ``dreaming.py`` — :func:`~minion_assist.dreaming._seconds_until_next`
  (shared scheduling primitive, not duplicated).
- ``minion.py`` — constructs and starts/stops one instance per process,
  only when ``memory_consolidation.enabled`` and a database + lexical
  index are both configured.
"""

from __future__ import annotations

import sys
import threading
from typing import TYPE_CHECKING

from ..dreaming import _seconds_until_next

if TYPE_CHECKING:
    from ..config import MemoryConsolidationConfig
    from ..session.db import SessionDB
    from ..worker_health import WorkerHealth
    from .consolidation import MemoryConsolidator
    from .postgres_index import PostgresMemoryIndex


class MemoryConsolidationScheduler:
    """Runs a daily consolidation-preview pass in a daemon thread.

    Args:
        cfg: Resolved :class:`~minion_assist.config.MemoryConsolidationConfig`.
        db: The ``SessionDB`` proposals live in.
        index: The lexical index used for ranking and preview lookups.
        consolidator: The agent-scoped
            :class:`~minion_assist.memory.consolidation.MemoryConsolidator`
            to draft with — already bound to ``cfg.agent_id`` (see that
            class's docstring for why one instance is single-agent).
        health: Optional :class:`~minion_assist.worker_health.WorkerHealth`
            (MEM-GAP-016) — a "poll" is recorded each time the timer fires;
            "success"/"failure" are recorded per proposal drafted (or once
            for a whole-pass failure, e.g. a database error).
    """

    def __init__(
        self,
        cfg: MemoryConsolidationConfig,
        db: SessionDB,
        index: PostgresMemoryIndex,
        consolidator: MemoryConsolidator,
        health: "WorkerHealth | None" = None,
    ) -> None:
        self._cfg = cfg
        self._db = db
        self._index = index
        self._consolidator = consolidator
        self._health = health
        self._timer: threading.Timer | None = None
        self._stopped = False

    def start(self) -> None:
        """Schedule the first pass at the next configured wall-clock time."""
        delay = _seconds_until_next(self._cfg.hour, self._cfg.minute, self._cfg.timezone)
        self._timer = threading.Timer(delay, self._fire)
        self._timer.daemon = True
        self._timer.name = "memory-consolidation-scheduler"
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
            print(f"[memory-consolidation] Error during pass: {exc}", file=sys.stderr)
            if self._health is not None:
                self._health.record_failure(f"{type(exc).__name__}: {exc}")
        finally:
            # Always reschedule so the loop continues even after errors.
            if not self._stopped:
                self.start()

    def _run_pass(self) -> None:
        """Draft a preview for up to ``top_n`` never-previewed pending proposals.

        A single proposal's drafting failure (e.g. a malformed provider
        response) is logged and skipped — it must never abort the whole
        pass, since one bad response shouldn't prevent every other
        proposal in the run from getting a preview.
        """
        from .consolidation import rank_proposals  # noqa: PLC0415

        ranked = rank_proposals(self._db, self._index, self._cfg.agent_id)
        drafted = 0
        for proposal in ranked:
            if drafted >= self._cfg.top_n:
                break
            already_previewed = self._index.list_consolidation_previews(
                self._cfg.agent_id, proposal["id"]
            )
            if already_previewed:
                continue
            try:
                self._consolidator.preview(proposal["id"])
                drafted += 1
                if self._health is not None:
                    self._health.record_success()
            except Exception as exc:
                print(
                    f"[memory-consolidation] Failed to draft preview for proposal "
                    f"#{proposal['id']}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                if self._health is not None:
                    self._health.record_failure(f"{type(exc).__name__}: {exc}")

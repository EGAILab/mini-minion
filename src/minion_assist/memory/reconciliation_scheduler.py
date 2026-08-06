"""``ReconciliationScheduler`` — periodic self-healing for transient write failures (MEM-GAP-007).

``AgentSession._send_locked()`` mirrors every turn's messages to
PostgreSQL and enqueues capture/commitment extraction jobs, but a
transient failure (e.g. a momentary connection drop) at any of those
points is caught and swallowed — the turn must never fail just because an
optional durability step did. Before this scheduler existed, a swallowed
failure meant that turn's mirror or job simply never existed; nothing
ever retried it, and the only way to catch up was a full process restart
(``SessionDB.reconcile_all_sessions()`` only ever ran once, at startup).

This scheduler runs that same reconciliation — plus a second, analogous
pass for capture/commitment job *coverage* — on a timer, so a gap heals
itself within one interval instead of requiring a restart.

Two kinds of gap, two mechanisms
------------------------------------
1. **Mirror gaps** — reuses :meth:`~minion_assist.session.db.SessionDB.reconcile_all_sessions`
   unchanged: it diffs every agent's JSONL session files (the always-durable
   source of truth) against ``message_mirrors`` and mirrors whatever's
   missing. Cheap and exact — no new logic needed here.
2. **Job-coverage gaps** — capture/commitment jobs have no equivalent
   "diff against a source of truth" (an enqueue intent isn't itself
   persisted anywhere durable before the attempt). Instead,
   :meth:`~minion_assist.session.db.SessionDB.find_uncovered_capture_range`/
   :meth:`~minion_assist.session.db.SessionDB.find_uncovered_commitment_range`
   look for mirrored messages newer than a session's last enqueued job and
   enqueue one coarse catch-up job for the whole gap — see those methods'
   docstrings for why "coarse catch-up" rather than exact per-turn
   reconstruction.

Quiet-period guard
-------------------
A session whose ``last_active`` is more recent than
``cfg.quiet_seconds`` is skipped for job-coverage catch-up (not for mirror
reconciliation, which is always safe and idempotent) — its turn may still
be in progress and about to enqueue its own job normally; racing that
with a catch-up job for the same range would just be redundant, not
harmful (both use idempotency keys), but skipping avoids the redundancy
entirely in the common case.

Talks to
--------
- ``session/db.py`` — :meth:`SessionDB.reconcile_all_sessions`,
  :meth:`SessionDB.find_uncovered_capture_range`,
  :meth:`SessionDB.find_uncovered_commitment_range`,
  :meth:`SessionDB.list_session_ids_for_agent`, :meth:`SessionDB.get_sessions_by_ids`,
  :meth:`SessionDB.enqueue_capture_job`, :meth:`SessionDB.enqueue_commitment_job`.
- ``memory/extractor.py`` / ``memory/commitments.py`` — the prompt-version
  constants embedded in a catch-up job's idempotency key.
- ``minion.py`` — constructs and starts one instance per process, only
  when a database is configured, passing every configured agent id.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import MemoryReconciliationConfig
    from ..session.db import SessionDB
    from ..worker_health import WorkerHealth


class ReconciliationScheduler:
    """Runs a periodic mirror + job-coverage reconciliation pass in a daemon thread.

    Args:
        cfg: Resolved :class:`~minion_assist.config.MemoryReconciliationConfig`.
        db: The shared ``SessionDB`` instance.
        agent_ids: Every configured agent id to reconcile.
        short_term: The shared ``ShortTermMemory`` instance — duck-typed,
            only ``list_sessions``/``load``/``save`` are used (passed
            straight through to ``reconcile_all_sessions``).
        health: Optional :class:`~minion_assist.worker_health.WorkerHealth`
            (MEM-GAP-016) — a "poll" is recorded each time the timer
            fires; "success"/"failure" are recorded for the whole pass.
    """

    def __init__(
        self,
        cfg: MemoryReconciliationConfig,
        db: SessionDB,
        agent_ids: list[str],
        short_term: object,
        health: "WorkerHealth | None" = None,
    ) -> None:
        self._cfg = cfg
        self._db = db
        self._agent_ids = agent_ids
        self._short_term = short_term
        self._health = health
        self._timer: threading.Timer | None = None
        self._stopped = False

    def start(self) -> None:
        """Schedule the first pass. Safe to call once per process."""
        interval = max(60, min(3600, self._cfg.interval_seconds))
        self._timer = threading.Timer(interval, self._fire)
        self._timer.daemon = True
        self._timer.name = "memory-reconciliation-scheduler"
        self._timer.start()

    def stop(self) -> None:
        """Cancel the pending timer and prevent future firings."""
        self._stopped = True
        if self._timer is not None:
            self._timer.cancel()

    def _fire(self) -> None:
        """Timer callback: run one pass then reschedule."""
        if self._stopped:
            return
        if self._health is not None:
            self._health.record_poll()
        try:
            self._run_pass()
            if self._health is not None:
                self._health.record_success()
        except Exception as exc:
            print(f"[memory-reconciliation] Error during pass: {exc}", file=sys.stderr)
            if self._health is not None:
                self._health.record_failure(f"{type(exc).__name__}: {exc}")
        finally:
            # Always reschedule so the loop continues even after errors.
            if not self._stopped:
                self.start()

    def _run_pass(self) -> None:
        """Heal mirror gaps, then job-coverage gaps, for every configured agent."""
        self._db.reconcile_all_sessions(self._short_term, self._agent_ids)

        quiet_cutoff = time.time() - self._cfg.quiet_seconds
        for agent_id in self._agent_ids:
            session_ids = self._db.list_session_ids_for_agent(agent_id)
            if not session_ids:
                continue
            sessions = self._db.get_sessions_by_ids(session_ids, agent_id)
            for session_id, info in sessions.items():
                last_active = info.get("last_active")
                if last_active is not None and last_active > quiet_cutoff:
                    continue  # possibly still mid-turn -- don't race a live enqueue
                self._catch_up_capture(agent_id, session_id)
                self._catch_up_commitment(agent_id, session_id)

    def _catch_up_capture(self, agent_id: str, session_id: str) -> None:
        gap = self._db.find_uncovered_capture_range(agent_id, session_id)
        if gap is None:
            return
        from_id, to_id = gap
        from .extractor import _EXTRACTION_PROMPT_VERSION  # noqa: PLC0415

        idem_key = f"{agent_id}:{session_id}:{from_id}-{to_id}:{_EXTRACTION_PROMPT_VERSION}:reconcile"
        self._db.enqueue_capture_job(agent_id, session_id, from_id, to_id, idem_key)

    def _catch_up_commitment(self, agent_id: str, session_id: str) -> None:
        gap = self._db.find_uncovered_commitment_range(agent_id, session_id)
        if gap is None:
            return
        channel, from_id, to_id = gap
        from .commitments import _COMMITMENT_PROMPT_VERSION  # noqa: PLC0415

        idem_key = (
            f"{agent_id}:{session_id}:{channel}:{from_id}-{to_id}:"
            f"{_COMMITMENT_PROMPT_VERSION}:reconcile"
        )
        self._db.enqueue_commitment_job(
            agent_id, session_id, channel, from_id, to_id, idem_key
        )

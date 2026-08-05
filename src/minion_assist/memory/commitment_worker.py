"""``CommitmentWorker`` — the durable commitment-extraction worker (Stage One Phase 6, slice B).

Structurally identical to ``memory/capture_worker.py``'s ``CaptureWorker``
— one long-running background thread, started once at process startup
(not per turn), polls ``memory_commitment_jobs`` for due work, extracts
via the provider, and records results as ``commitments`` rows. A separate
table/worker from the capture-job pipeline entirely — see
``session/db.py``'s module docstring for why (incompatible output
schemas: structured kind/sensitivity/due-window candidates here, plain
claim strings there).

Why one worker, not one thread per job?
------------------------------------------
Same reasoning ``CaptureWorker``'s own docstring already gives: a
continuously-running worker thread means jobs enqueued while the process
wasn't actively handling a turn (after a crash-and-restart, or between
turns) still get processed, unlike a fire-and-forget per-turn thread.

Talks to
--------
- ``session/db.py`` — :meth:`SessionDB.claim_next_commitment_job`,
  :meth:`SessionDB.complete_commitment_job`,
  :meth:`SessionDB.fail_commitment_job`,
  :meth:`SessionDB.list_pending_commitments_for_scope`,
  :meth:`SessionDB.get_messages_in_range`.
- ``memory/commitments.py`` — :func:`~minion_assist.memory.commitments.extract_commitments`,
  the shared prompt/parsing/validation primitive (this worker relies on
  it *raising* on provider failure, so its own retry/backoff loop can
  catch and reschedule the job — same contract
  ``memory/extractor.py``'s ``extract_facts`` has with ``CaptureWorker``).
- ``minion.py`` — constructs one :class:`CommitmentWorker` at startup
  (only when a database is configured and commitments are enabled) and
  calls :meth:`start`/:meth:`stop`.
- ``agents/session.py`` — enqueues jobs via
  :meth:`SessionDB.enqueue_commitment_job` after each turn; does not
  touch this worker directly.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from .commitments import extract_commitments

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..providers.base import LLMProvider
    from ..session.db import SessionDB
    from ..worker_health import WorkerHealth

_log = logging.getLogger("minion_assist.commitment_worker")

# Same polling/backoff/retry constants as capture_worker.py — no
# independent tuning rationale for these being different, so they stay
# identical rather than diverging without a reason.
_POLL_INTERVAL_SECONDS = 5.0
_BACKOFF_BASE_SECONDS = 2.0
_MAX_ATTEMPTS = 5


class CommitmentWorker:
    """Single background thread that processes ``memory_commitment_jobs``.

    Args:
        db: The shared :class:`~minion_assist.session.db.SessionDB` instance.
        provider_for_agent: Callable that returns the configured
            :class:`~minion_assist.providers.base.LLMProvider` for a given
            agent ID — same reasoning as ``CaptureWorker``'s own
            ``provider_for_agent``.
        min_due_seconds: Minimum seconds a validated commitment's earliest
            due time must be pushed out from "now" — see
            ``memory/commitments.py``'s module docstring for why this
            should typically be the configured heartbeat interval.
        health: Optional :class:`~minion_assist.worker_health.WorkerHealth`
            (MEM-GAP-016) — see ``CaptureWorker``'s matching parameter.
    """

    def __init__(
        self,
        db: SessionDB,
        provider_for_agent: Callable[[str], LLMProvider],
        min_due_seconds: float = 1800.0,
        health: "WorkerHealth | None" = None,
    ) -> None:
        self._db = db
        self._provider_for_agent = provider_for_agent
        self._min_due_seconds = min_due_seconds
        self._health = health
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background polling thread. Safe to call once per process."""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="commitment-worker"
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the worker to stop and wait for it to drain.

        Args:
            timeout: Maximum seconds to wait for the current job (if any)
                to finish before giving up on a clean join.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        """Poll loop — runs until :meth:`stop` is called."""
        while not self._stop_event.is_set():
            processed = self._process_one()
            if not processed:
                self._stop_event.wait(_POLL_INTERVAL_SECONDS)

    def _process_one(self) -> bool:
        """Claim and process one due job, if any.

        Returns:
            bool: ``True`` if a job was claimed (whether it succeeded or
                failed), ``False`` if the queue had nothing due.
        """
        if self._health is not None:
            self._health.record_poll()
        job = self._db.claim_next_commitment_job()
        if job is None:
            return False

        try:
            messages = self._db.get_messages_in_range(
                job["session_id"], job["source_from_message_id"], job["source_to_message_id"]
            )
            # A commitment job's range is one exchange (the same 2-message
            # range a capture job would cover for the same turn) — the
            # *last* user/assistant message in range, not every same-role
            # message concatenated, matches that "one exchange" semantics.
            user_msgs = [m["content"] for m in messages if m.get("role") == "user" and m.get("content")]
            assistant_msgs = [
                m["content"] for m in messages if m.get("role") == "assistant" and m.get("content")
            ]
            user_text = user_msgs[-1] if user_msgs else ""
            assistant_text = assistant_msgs[-1] if assistant_msgs else ""

            existing_pending = self._db.list_pending_commitments_for_scope(
                job["agent_id"], job["channel"]
            )
            provider = self._provider_for_agent(job["agent_id"])
            candidates = extract_commitments(
                provider, user_text, assistant_text, existing_pending,
                time.time(), self._min_due_seconds,
            )
            self._db.complete_commitment_job(job["id"], candidates)
            if self._health is not None:
                self._health.record_success()
        except Exception as exc:
            _log.debug("Commitment job %s failed: %s: %s", job["id"], type(exc).__name__, exc)
            backoff = _BACKOFF_BASE_SECONDS * (2 ** job["attempts"])
            self._db.fail_commitment_job(
                job["id"], f"{type(exc).__name__}: {exc}", backoff, max_attempts=_MAX_ATTEMPTS
            )
            if self._health is not None:
                self._health.record_failure(f"{type(exc).__name__}: {exc}")
        return True

"""``MessageEmbeddingWorker`` — the durable message-embedding job worker (MEM-GAP-006).

One long-running background thread — started once at process startup, not
per turn — polls ``message_embedding_jobs`` for due work, embeds the
message's content via the configured
:class:`~minion_assist.providers.embeddings.EmbeddingProvider`, and stores
the resulting vector in ``message_embeddings``. Structurally identical to
:class:`~minion_assist.memory.capture_worker.CaptureWorker` (same poll loop,
same ``FOR UPDATE SKIP LOCKED`` claim, same exponential-backoff retry), but
processes one message per job instead of extracting from a range — see
``session/db.py``'s "Durable message-embedding jobs" module docstring
section for why a worker (not an inline call in ``agents/session.py``) does
this: embedding is a network round trip, and the whole point of a durable
queue instead of a per-turn synchronous call is to never block a turn on
that.

Talks to
--------
- ``session/db.py`` — :meth:`SessionDB.claim_next_message_embedding_job`,
  :meth:`SessionDB.complete_message_embedding_job`,
  :meth:`SessionDB.fail_message_embedding_job`,
  :meth:`SessionDB.get_messages_in_range` (reused to fetch one message's
  content by id — a single-id range is just that one row).
- ``providers/embeddings.py`` — :class:`EmbeddingProvider`, this worker's
  only external dependency for turning content into a vector.
- ``minion.py`` — constructs one :class:`MessageEmbeddingWorker` at startup
  (only when a database and an embedding provider are both configured) and
  calls :meth:`start`/:meth:`stop`.
- ``agents/session.py`` — enqueues jobs via
  :meth:`SessionDB.enqueue_message_embedding_job` after each turn; does not
  touch this worker directly.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..providers.embeddings import EmbeddingProvider
    from ..session.db import SessionDB
    from ..worker_health import WorkerHealth

_log = logging.getLogger("minion_assist.message_embedding_worker")

# Same polling/backoff constants as CaptureWorker — no tuning knobs without
# evaluation data to justify them yet (see the project's simplicity rule).
_POLL_INTERVAL_SECONDS = 5.0
_BACKOFF_BASE_SECONDS = 2.0
_MAX_ATTEMPTS = 5


class MessageEmbeddingWorker:
    """Single background thread that processes ``message_embedding_jobs``.

    Args:
        db: The shared :class:`SessionDB` instance.
        embedding_provider: The :class:`EmbeddingProvider` to call for each
            job's message content.
        health: Optional :class:`~minion_assist.worker_health.WorkerHealth`
            (MEM-GAP-016) — if given, every poll/success/failure is
            recorded on it so a same-process caller (e.g. the REPL's
            ``/status deep``) can tell whether this worker is actually
            alive and draining the queue, not just constructed.
    """

    def __init__(
        self,
        db: SessionDB,
        embedding_provider: EmbeddingProvider,
        health: "WorkerHealth | None" = None,
    ) -> None:
        self._db = db
        self._embedding_provider = embedding_provider
        self._health = health
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background polling thread. Safe to call once per process."""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="message-embedding-worker"
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
                # Queue was empty — wait before polling again. wait() returns
                # early if stop() sets the event, so shutdown isn't delayed
                # by a full poll interval.
                self._stop_event.wait(_POLL_INTERVAL_SECONDS)

    def _process_one(self) -> bool:
        """Claim and process one due job, if any.

        Returns:
            bool: ``True`` if a job was claimed (whether it succeeded or
                failed), ``False`` if the queue had nothing due.
        """
        if self._health is not None:
            self._health.record_poll()
        job = self._db.claim_next_message_embedding_job()
        if job is None:
            return False

        try:
            messages = self._db.get_messages_in_range(
                job["session_id"], job["message_id"], job["message_id"]
            )
            content = messages[0]["content"] if messages else None
            if not content:
                # Message was deleted, or its content is empty — nothing to
                # embed and nothing that will ever change on retry.
                # max_attempts=1 marks it terminally 'failed' immediately
                # rather than retrying a condition that can't resolve
                # itself, while still recording why in last_error.
                self._db.fail_message_embedding_job(
                    job["id"], "message has no content", 0.0, max_attempts=1
                )
                return True
            [vector] = self._embedding_provider.embed([content])
            self._db.complete_message_embedding_job(
                job["id"], job["message_id"], self._embedding_provider.model_identity, vector
            )
            if self._health is not None:
                self._health.record_success()
        except Exception as exc:
            _log.debug(
                "Message-embedding job %s failed: %s: %s", job["id"], type(exc).__name__, exc
            )
            backoff = _BACKOFF_BASE_SECONDS * (2 ** job["attempts"])
            self._db.fail_message_embedding_job(
                job["id"], f"{type(exc).__name__}: {exc}", backoff, max_attempts=_MAX_ATTEMPTS
            )
            if self._health is not None:
                self._health.record_failure(f"{type(exc).__name__}: {exc}")
        return True

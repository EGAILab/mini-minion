"""``CaptureWorker`` — the durable capture-job worker (Stage One Phase 2, slice C).

Replaces the per-turn daemon-thread extractor (``memory/extractor.py``) when
a database is configured. One long-running background thread — started
once at process startup, not per turn — polls ``memory_capture_jobs`` for
due work, extracts facts via the provider, and records them as
``memory_proposals``: structured, unreviewed claims, deliberately *not*
written into any note file directly. Stage One Phase 5 (consolidation)
decides how/whether proposals get promoted into curated memory; until a
human approves one, it stays a proposal.

Searchable, but gated (Stage One Phase 5, slice B)
-----------------------------------------------------
Each new proposal is also indexed into :class:`~minion_assist.memory.postgres_index.PostgresMemoryIndex`
right after :meth:`~minion_assist.session.db.SessionDB.complete_capture_job`
returns (via the optional ``index_proposal`` callable below), so Phase 5
slice C's consolidation ranking can use recall telemetry
(``hash_query``/``recall_stats``) on proposals the same way it does on
notes. This does *not* mean proposals show up in normal conversation:
``PostgresMemoryIndex.hybrid_search``'s corpus-agnostic default explicitly
excludes ``source_kind = "proposal"`` chunks — they're reachable only via
an explicit ``corpus="proposal"`` query (what Phase 5 slice C/D's
consolidation review does), never through per-turn ``<relevant_memories>``
injection or a normal ``search_memory`` call.

Why one worker, not one thread per job?
------------------------------------------
The old extractor fired a brand-new daemon thread every turn — fine for a
fire-and-forget best-effort note, but wrong for a durable queue: nothing
would poll for jobs that were enqueued while the process wasn't actively
handling a turn (e.g. after a crash-and-restart, or before the next turn
happens to fire another thread). One worker thread, running continuously
from startup to shutdown, means jobs get processed even if the agent that
enqueued them isn't actively in a turn right now.

Talks to
--------
- ``session/db.py`` — :meth:`SessionDB.claim_next_capture_job`,
  :meth:`SessionDB.complete_capture_job`, :meth:`SessionDB.fail_capture_job`,
  :meth:`SessionDB.get_messages_in_range`.
- ``memory/extractor.py`` — :func:`extract_facts`, the shared prompt/parsing
  primitive (this worker relies on it *raising* on provider failure, so its
  own retry/backoff loop can catch and reschedule the job).
- ``minion.py`` — constructs one :class:`CaptureWorker` at startup (only
  when a database is configured) and calls :meth:`start`/:meth:`stop`;
  passes ``index_proposal=_memory_index.reindex_proposal`` when a lexical
  index is also configured.
- ``memory/postgres_index.py`` — :meth:`PostgresMemoryIndex.reindex_proposal`,
  the typical ``index_proposal`` callable (Stage One Phase 5, slice B).
- ``agents/session.py`` — enqueues jobs via
  :meth:`SessionDB.enqueue_capture_job` after each turn; does not touch this
  worker directly.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from .extractor import extract_facts

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..providers.base import LLMProvider
    from ..session.db import SessionDB
    from ..worker_health import WorkerHealth

_log = logging.getLogger("minion_assist.capture_worker")

# How long to sleep between polls when the queue is empty. Not configurable
# (yet) — Phase 2's scope is correctness, not tuning knobs without
# evaluation data to justify them (see the project's simplicity rule).
_POLL_INTERVAL_SECONDS = 5.0

# Exponential backoff base for retrying a failed job: attempt 1 waits
# _BACKOFF_BASE_SECONDS, attempt 2 waits 2x that, etc.
_BACKOFF_BASE_SECONDS = 2.0

# After this many failed attempts, a job is marked 'failed' and stops being
# retried automatically (still inspectable in memory_capture_jobs).
_MAX_ATTEMPTS = 5


class CaptureWorker:
    """Single background thread that processes ``memory_capture_jobs``.

    Args:
        db: The shared :class:`SessionDB` instance.
        provider_for_agent: Callable that returns the configured
            :class:`LLMProvider` for a given agent ID — the worker isn't
            tied to one agent, so it looks up the right provider per job.
        index_proposal: Optional ``(agent_id, proposal_id, claim_text) ->
            None`` callable — typically
            :meth:`~minion_assist.memory.postgres_index.PostgresMemoryIndex.reindex_proposal`
            (Stage One Phase 5, slice B). ``None`` (the default) when no
            lexical index is configured: proposals are still recorded in
            ``memory_proposals``, just not searchable until an index
            exists. Passed in rather than imported directly, matching
            ``provider_for_agent``'s own reasoning — this module stays
            free of a direct dependency on ``postgres_index.py``.
        health: Optional :class:`~minion_assist.worker_health.WorkerHealth`
            (MEM-GAP-016) — if given, every poll/success/failure is
            recorded on it so a same-process caller (e.g. the REPL's
            ``/status deep``) can tell whether this worker is actually
            alive and draining the queue, not just constructed.
    """

    def __init__(
        self,
        db: SessionDB,
        provider_for_agent: Callable[[str], LLMProvider],
        index_proposal: Callable[[str, int, str], None] | None = None,
        health: "WorkerHealth | None" = None,
    ) -> None:
        self._db = db
        self._provider_for_agent = provider_for_agent
        self._index_proposal = index_proposal
        self._health = health
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background polling thread. Safe to call once per process."""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="memory-capture-worker"
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
        job = self._db.claim_next_capture_job()
        if job is None:
            return False

        try:
            messages = self._db.get_messages_in_range(
                job["session_id"], job["source_from_message_id"], job["source_to_message_id"]
            )
            exchange = [
                {"role": m["role"], "content": m["content"]}
                for m in messages
                if m.get("role") in ("user", "assistant") and m.get("content")
            ]
            provider = self._provider_for_agent(job["agent_id"])
            facts = extract_facts(provider, exchange)
            new_proposals = self._db.complete_capture_job(job["id"], facts)
            if self._index_proposal is not None:
                for proposal in new_proposals:
                    try:
                        self._index_proposal(
                            proposal["agent_id"], proposal["id"], proposal["claim_text"]
                        )
                    except Exception as exc:
                        # Best-effort: an indexing failure must never turn an
                        # already-successful capture job into a failed one —
                        # the proposal itself is already safely recorded.
                        _log.debug(
                            "Indexing proposal %s failed: %s: %s",
                            proposal["id"], type(exc).__name__, exc,
                        )
            if self._health is not None:
                self._health.record_success()
        except Exception as exc:
            _log.debug("Capture job %s failed: %s: %s", job["id"], type(exc).__name__, exc)
            backoff = _BACKOFF_BASE_SECONDS * (2 ** job["attempts"])
            self._db.fail_capture_job(
                job["id"], f"{type(exc).__name__}: {exc}", backoff, max_attempts=_MAX_ATTEMPTS
            )
            if self._health is not None:
                self._health.record_failure(f"{type(exc).__name__}: {exc}")
        return True

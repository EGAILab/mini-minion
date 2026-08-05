"""``WorkerHealth`` — thread-safe live status for one background worker/scheduler.

MEM-GAP-016 ("Health and diagnostics do not describe end-to-end freshness"
— see ``minion-assist-docs/improve/openclaw-memory-gap-analysis.md``):
before this module, none of minion-assist's seven background workers
(``CaptureWorker``, ``CommitmentWorker``, ``MemoryIndexWatcher``,
``MemoryConsolidationScheduler``, ``KnowledgeDigestScheduler``,
``DreamingScheduler``, ``HeartbeatScheduler``) exposed any live status at
all — a runtime failure was only ever a swallowed exception and a
``print()`` to stderr. An operator had no way to ask "is the capture queue
actually being drained?" without reading logs.

Same-process only, by design
-----------------------------
This tracks *in-process* liveness — it answers "is the worker thread in
*this* running process still polling/succeeding," which only the process
that owns the thread can know. It is deliberately not persisted anywhere
(no new DB table, no status file): a separate CLI invocation (e.g.
``minion-assist memory status --deep``) is a different process and cannot
see this regardless of where it's stored in-memory. That command already
correctly limits itself to DB-observable facts (index summary, queue lag —
see ``session/db.py``'s ``queue_lag_summary``); this module is what a
same-process caller (the REPL's ``/status deep``, or Matrix's equivalent)
reads instead.

Precedent
---------
Generalizes the same "mutable object written by a worker, read back by
another part of the same process" shape ``tools/heartbeat_respond.py``'s
``HeartbeatResponseCapture`` already uses — but with a lock, since these
workers write from their own poll thread while a REPL/Matrix command reads
from the main thread concurrently (``HeartbeatResponseCapture`` gets away
without one only because it's read strictly after a synchronous call
returns on the same thread).

Never logs memory content
--------------------------
:meth:`record_failure` stores ``str(exception)`` truncated to
:data:`_MAX_ERROR_LENGTH` characters — exception messages are almost always
programming/connectivity errors (a bad connection string, a SQL error), not
memory content, but the cap exists so a pathological case (e.g. a SQL error
echoing back an oversized parameter) can't turn a health snapshot into a
content leak.
"""

from __future__ import annotations

import threading
import time

# Exception messages are truncated to this length before being stored, so a
# health snapshot can never become a way to exfiltrate memory content
# through an unusually chatty exception message.
_MAX_ERROR_LENGTH = 300


class WorkerHealth:
    """Thread-safe, in-memory liveness record for one background worker.

    Args:
        name: A short, stable identifier for the worker (e.g.
            ``"capture_worker"``) — shown as-is in ``/status deep`` output.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._lock = threading.Lock()
        self._last_poll_at: float | None = None
        self._last_success_at: float | None = None
        self._last_error: str | None = None
        self._last_error_at: float | None = None
        self._consecutive_failures: int = 0

    def record_poll(self) -> None:
        """Record that the worker is alive and just started a poll/pass.

        Called unconditionally at the top of each loop iteration/timer
        fire, regardless of whether there was any work to do — this is
        what proves the thread hasn't silently died, separately from
        whether it's finding (or successfully processing) any work.
        """
        with self._lock:
            self._last_poll_at = time.time()

    def record_success(self) -> None:
        """Record a successfully completed unit of work, resetting the failure streak."""
        with self._lock:
            self._last_success_at = time.time()
            self._consecutive_failures = 0

    def record_failure(self, error: str) -> None:
        """Record a failed unit of work.

        Args:
            error: A short description of what failed, typically
                ``str(exception)`` — truncated to :data:`_MAX_ERROR_LENGTH`
                characters before being stored.
        """
        with self._lock:
            self._last_error = error[:_MAX_ERROR_LENGTH]
            self._last_error_at = time.time()
            self._consecutive_failures += 1

    def snapshot(self) -> dict:
        """Return a consistent, point-in-time copy of this worker's status.

        Returns:
            dict: ``{"name", "last_poll_at", "last_success_at",
                "last_error", "last_error_at", "consecutive_failures"}`` —
                every timestamp is epoch seconds or ``None`` if it never
                happened yet.
        """
        with self._lock:
            return {
                "name": self.name,
                "last_poll_at": self._last_poll_at,
                "last_success_at": self._last_success_at,
                "last_error": self._last_error,
                "last_error_at": self._last_error_at,
                "consecutive_failures": self._consecutive_failures,
            }

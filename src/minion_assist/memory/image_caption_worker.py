"""``ImageCaptionWorker`` — the durable image-captioning job worker (R3-GAP-002).

One long-running background thread — started once at process startup, not
per turn — polls ``image_captions`` for due work, asks the owning agent's
own provider to describe the attached image, and folds the result into the
originating message's searchable text.

Why a worker instead of an inline call in ``agents/session.py``?
------------------------------------------------------------------
Same reasoning as :class:`~minion_assist.memory.message_embedding_worker.MessageEmbeddingWorker`:
captioning is a network round trip to an LLM, and the whole point of a
durable queue instead of a per-turn synchronous call is to never block a
turn on that. ``agents/session.py`` only enqueues (see its module's R3-GAP-002
comment at the user-message mirror site) — this worker does the actual work,
asynchronously, off the turn's critical path.

Structurally identical to ``MessageEmbeddingWorker`` (same poll loop, same
``FOR UPDATE SKIP LOCKED`` claim via :meth:`SessionDB.claim_next_image_caption_job`,
same exponential-backoff retry), plus one extra step neither
``MessageEmbeddingWorker`` nor ``CaptureWorker`` needs: checking whether the
owning agent's configured model can even see images before attempting a
call — ``agents_cfg[agent_id].input_modalities`` doesn't change between
retries, so an agent with a text-only model is marked ``'unsupported'``
immediately (a real, inspectable diagnostic — R3-GAP-002's acceptance
criteria) rather than silently exhausting retries as if it were a transient
failure.

Talks to
--------
- ``session/db.py`` — :meth:`SessionDB.claim_next_image_caption_job`,
  :meth:`SessionDB.complete_image_caption_job`,
  :meth:`SessionDB.fail_image_caption_job`,
  :meth:`SessionDB.mark_image_caption_unsupported`.
- ``providers/*.py`` — via the injected ``provider_for_agent`` callable
  (same injection pattern ``CaptureWorker`` already uses); the actual
  multimodal request-building (base64 materialization from the staged
  file's path) happens inside the provider's own ``chat()``, exactly the
  same code path a normal agent turn with an image attachment already
  goes through — nothing here reimplements that.
- ``minion.py`` — constructs one :class:`ImageCaptionWorker` at startup
  (only when a database is configured) and calls :meth:`start`/:meth:`stop`.
- ``agents/session.py`` — enqueues jobs via
  :meth:`SessionDB.enqueue_image_caption`; does not touch this worker
  directly.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..config import AgentModelConfig
    from ..providers.base import LLMProvider
    from ..session.db import SessionDB
    from ..worker_health import WorkerHealth

_log = logging.getLogger("minion_assist.image_caption_worker")

# Same polling/backoff constants as MessageEmbeddingWorker/CaptureWorker —
# no tuning knobs without evaluation data to justify them yet (see the
# project's simplicity rule).
_POLL_INTERVAL_SECONDS = 5.0
_BACKOFF_BASE_SECONDS = 2.0
_MAX_ATTEMPTS = 5

# Deliberately generic and factual — this text becomes searchable database
# content (session FTS/vector, capture extraction), not a user-facing
# response, so it should read like a caption/alt-text, not a chat reply.
_CAPTION_SYSTEM_PROMPT = (
    "Describe this image factually and concisely, in one or two sentences. "
    "Only describe what is actually visible in the image — do not "
    "speculate, infer intent, or add commentary. This description will be "
    "stored as searchable text alongside the conversation, not shown "
    "directly to the user."
)


class ImageCaptionWorker:
    """Single background thread that processes ``image_captions`` jobs.

    Args:
        db: The shared :class:`SessionDB` instance.
        agents_cfg: ``dict[str, AgentModelConfig]`` — read only for each
            job's ``input_modalities``, to decide up front whether the
            owning agent's model can see images at all.
        provider_for_agent: Callable that returns the configured
            :class:`~minion_assist.providers.base.LLMProvider` for a given
            agent id — same injection shape
            :class:`~minion_assist.memory.capture_worker.CaptureWorker`
            already uses, for the same reason (one already-constructed
            provider per agent, reused, not built fresh per job).
        health: Optional :class:`~minion_assist.worker_health.WorkerHealth`
            — if given, every poll/success/failure is recorded on it so a
            same-process caller (e.g. the REPL's ``/status deep``) can tell
            whether this worker is actually alive and draining the queue.
    """

    def __init__(
        self,
        db: SessionDB,
        agents_cfg: dict[str, AgentModelConfig],
        provider_for_agent: Callable[[str], LLMProvider],
        health: WorkerHealth | None = None,
    ) -> None:
        self._db = db
        self._agents_cfg = agents_cfg
        self._provider_for_agent = provider_for_agent
        self._health = health
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background polling thread. Safe to call once per process."""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="image-caption-worker"
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
            bool: ``True`` if a job was claimed (whether it succeeded,
                failed, or was marked unsupported), ``False`` if the queue
                had nothing due.
        """
        if self._health is not None:
            self._health.record_poll()
        job = self._db.claim_next_image_caption_job()
        if job is None:
            return False

        agent_id = job["agent_id"]
        cfg = self._agents_cfg.get(agent_id)
        # getattr, not direct attribute access: agents_cfg's values are
        # AgentModelConfig in production, but nothing here guarantees every
        # caller passes a fully-populated one (a stray/incomplete object
        # must be treated the same as "no image modality," not crash this
        # background thread's poll loop — that's the whole point of
        # 'unsupported' being a graceful, inspectable terminal state).
        modalities = getattr(cfg, "input_modalities", ()) if cfg is not None else ()
        if "image" not in modalities:
            # Terminal, not retried — see this class's own docstring for
            # why 'unsupported' is a distinct state from 'failed' here.
            self._db.mark_image_caption_unsupported(
                job["id"],
                f"agent '{agent_id}' has no configured image input modality",
            )
            return True

        try:
            provider = self._provider_for_agent(agent_id)
            response = provider.chat(
                system=_CAPTION_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image."},
                        {
                            "type": "image",
                            "media_type": job["media_type"],
                            "path": job["path"],
                            "size_bytes": 0,
                            "source_name": job["source_name"],
                        },
                    ],
                }],
                tools=[],
                max_tokens=150,
            )
            caption = (response.text or "").strip()
            if not caption:
                # Nothing about retrying would produce a different empty
                # response — terminal, same reasoning as
                # MessageEmbeddingWorker's "message has no content" case.
                self._db.fail_image_caption_job(
                    job["id"], "provider returned an empty caption", 0.0, max_attempts=1
                )
                return True
            self._db.complete_image_caption_job(
                job["id"], job["message_id"], job["source_name"], caption
            )
            if self._health is not None:
                self._health.record_success()
        except FileNotFoundError as exc:
            # The staged file is gone (e.g. manual cleanup, or a future
            # retention policy that prunes attachment_store contents) —
            # terminal, not retryable; retrying can't make a deleted file
            # reappear.
            self._db.fail_image_caption_job(
                job["id"], f"{type(exc).__name__}: {exc}", 0.0, max_attempts=1
            )
        except Exception as exc:
            _log.debug(
                "Image-caption job %s failed: %s: %s", job["id"], type(exc).__name__, exc
            )
            backoff = _BACKOFF_BASE_SECONDS * (2 ** job["attempts"])
            self._db.fail_image_caption_job(
                job["id"], f"{type(exc).__name__}: {exc}", backoff, max_attempts=_MAX_ATTEMPTS
            )
            if self._health is not None:
                self._health.record_failure(f"{type(exc).__name__}: {exc}")
        return True

"""Memory fact extraction — the degraded (no-database) daemon-thread path,
plus the shared prompt/parsing logic the durable capture worker also uses.

The note is quarantined (``memory/imports/_auto_extracted.md``, via
:meth:`MemoryService.remember_import`), not a curated topic page — nobody
has reviewed these facts, so per
``docs/adr/0003-per-agent-memory-scope.md`` they stay searchable but are
never auto-promoted.

Two callers, two failure-handling contracts (Stage One Phase 2, slice C)
--------------------------------------------------------------------------
:func:`extract_facts` is the shared primitive — one prompt, one parsing
implementation — used by both:

- :func:`extract_and_save_async`/:func:`_worker` below: the original
  per-turn daemon-thread path, used when no database is configured. Fires
  a new thread every turn and fails silently — there is no retry mechanism
  here, so a swallowed exception is the only sane behavior.
- ``memory/capture_worker.py``'s :class:`CaptureWorker`: the durable,
  database-backed replacement, used when a database *is* configured. One
  long-running worker (not one thread per turn) processes
  ``memory_capture_jobs`` and needs failures to propagate so its own
  retry/backoff loop can catch and reschedule them.

:func:`extract_facts` therefore does NOT catch provider exceptions itself —
each caller wraps it according to its own contract.

Design (this module's degraded path only)
------------------------------------------
- Runs in a daemon thread so it never blocks the REPL.
- Extracts from the last exchange only (not the full history) to stay
  incremental and cheap.
- Caps at 3 facts per turn and 50 total entries in the rolling file.
- Fails silently — extraction is best-effort; a failure here must never
  affect the conversation in any way.

Talks to
--------
- ``memory/service.py`` — appends extracted facts via
  :meth:`MemoryService.remember_import`/:meth:`MemoryService.load_import`.
- ``memory/capture_worker.py`` — the durable-job replacement; imports
  :func:`extract_facts` and :data:`_EXTRACTION_PROMPT_VERSION`.
- ``providers/base.py``   — calls ``provider.chat()`` for the extraction LLM call.
- ``agents/session.py``   — calls :func:`extract_and_save_async` when no
                            database is configured, or enqueues a durable
                            capture job (via ``session/db.py``) otherwise.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..memory.service import MemoryService
    from ..providers.base import LLMProvider

_log = logging.getLogger("minion_assist.extractor")

_EXTRACT_SYSTEM = """\
Extract 0 to 3 short facts worth remembering from this conversation exchange.
Only extract facts useful in a future conversation:
- User preferences, background, goals, or relationships
- Decisions made and their rationale
- Key research findings with context

If nothing is worth remembering, respond with exactly: NOTHING

Otherwise write one fact per line, each under 100 characters.
No bullet points, no numbering — bare facts only."""

_MAX_FACTS_PER_TURN = 3
_MAX_ROLLING_ENTRIES = 50

# Bumped whenever _EXTRACT_SYSTEM's wording changes meaningfully. Included in
# the durable capture worker's idempotency key (session/db.py's
# enqueue_capture_job) so a prompt change causes previously-processed message
# ranges to be re-extracted under a new key, rather than silently reusing
# results produced under the old prompt.
_EXTRACTION_PROMPT_VERSION = "v1"


def extract_facts(provider: LLMProvider, exchange: list[dict]) -> list[str]:
    """Call the provider to extract 0-3 short facts from a message exchange.

    Shared by the degraded daemon-thread path (:func:`_worker`) and the
    durable capture-job worker (``memory/capture_worker.py``) — see this
    module's docstring for why failure handling is split between the two
    callers instead of living here.

    Args:
        provider: The LLM provider to call.
        exchange: The messages to extract from — plain ``{"role", "content"}``
            dicts. Callers are responsible for building clean dicts (no
            extra keys) before calling this, the same way
            ``providers/openai_compatible.py`` must strip ``EVENT_ID_KEY``
            before any message list reaches a provider.

    Returns:
        list[str]: 0-3 extracted facts, or ``[]`` if the provider explicitly
            said there was nothing worth remembering.

    Raises:
        Exception: Whatever ``provider.chat()`` raises — not caught here.
    """
    response = provider.chat(
        system=_EXTRACT_SYSTEM,
        messages=exchange,
        tools=[],
        max_tokens=150,
    )
    text = (response.text or "").strip()
    if not text or text.upper() == "NOTHING":
        return []
    return [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ][:_MAX_FACTS_PER_TURN]


def extract_and_save_async(
    memory: MemoryService,
    provider: LLMProvider,
    last_exchange: list[dict],
) -> None:
    """Trigger background memory extraction. Returns immediately.

    Degraded-mode path only — used when no database is configured (see
    ``agents/session.py``). When a database is configured, a durable
    ``memory_capture_jobs`` row is enqueued instead (Stage One Phase 2,
    slice C); this daemon-thread path is not used in that case.

    Args:
        memory:        The agent's :class:`MemoryService` instance.
        provider:      The agent's LLM provider (same one used for the turn).
        last_exchange: The last 1–2 messages (ideally last user + last assistant).
                       Extraction is skipped if fewer than 2 messages are provided.
    """
    if len(last_exchange) < 2:
        return
    exchange = last_exchange[-2:]
    threading.Thread(
        target=_worker,
        args=(memory, provider, exchange),
        daemon=True,
        name="memory-extractor",
    ).start()


def _worker(
    memory: MemoryService,
    provider: LLMProvider,
    exchange: list[dict],
) -> None:
    """Worker — runs in a background thread.  Never raises."""
    try:
        facts = extract_facts(provider, exchange)
        if facts:
            _append(memory, facts)
    except Exception as exc:
        _log.debug("Memory extraction failed: %s: %s", type(exc).__name__, exc)


def _append(memory: MemoryService, facts: list[str]) -> None:
    """Append new facts to the rolling, quarantined ``_auto_extracted`` note."""
    existing = memory.load_import("_auto_extracted") or ""
    lines = [ln for ln in existing.splitlines() if ln.strip()]
    # Trim oldest entries to stay within the cap.
    lines = lines[-(_MAX_ROLLING_ENTRIES - len(facts)):]
    lines.extend(facts)
    memory.remember_import("_auto_extracted", "\n".join(lines))

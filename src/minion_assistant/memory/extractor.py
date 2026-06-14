"""Background memory extraction — non-blocking, fires after each turn.

Extracts 0–3 concise facts from the last user↔assistant exchange and appends
them to a rolling ``_auto_extracted`` note in long-term memory.

Design
------
- Runs in a daemon thread so it never blocks the REPL.
- Extracts from the last exchange only (not the full history) to stay
  incremental and cheap.
- Caps at 3 facts per turn and 50 total entries in the rolling file.
- Fails silently — extraction is best-effort; a failure here must never
  affect the conversation in any way.

Talks to
--------
- ``memory/long_term.py`` — appends extracted facts via :class:`LongTermMemory`.
- ``providers/base.py``   — calls ``provider.chat()`` for the extraction LLM call.
- ``agents/session.py``   — :func:`extract_and_save_async` is called after each
                            successful turn.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..memory.long_term import LongTermMemory
    from ..providers.base import LLMProvider

_log = logging.getLogger("minion_assistant.extractor")

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


def extract_and_save_async(
    memory: "LongTermMemory",
    provider: "LLMProvider",
    last_exchange: list[dict],
) -> None:
    """Trigger background memory extraction. Returns immediately.

    Args:
        memory:        The agent's :class:`LongTermMemory` instance.
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
    memory: "LongTermMemory",
    provider: "LLMProvider",
    exchange: list[dict],
) -> None:
    """Worker — runs in a background thread.  Never raises."""
    try:
        response = provider.chat(
            system=_EXTRACT_SYSTEM,
            messages=exchange,
            tools=[],
            max_tokens=150,
        )
        text = (response.text or "").strip()
        if not text or text.upper() == "NOTHING":
            return
        facts = [
            ln.strip()
            for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ][:_MAX_FACTS_PER_TURN]
        if facts:
            _append(memory, facts)
    except Exception as exc:
        _log.debug("Memory extraction failed: %s: %s", type(exc).__name__, exc)


def _append(memory: "LongTermMemory", facts: list[str]) -> None:
    """Append new facts to the rolling ``_auto_extracted`` note."""
    existing = memory.load("_auto_extracted") or ""
    lines = [ln for ln in existing.splitlines() if ln.strip()]
    # Trim oldest entries to stay within the cap.
    lines = lines[-((_MAX_ROLLING_ENTRIES - len(facts))):]
    lines.extend(facts)
    memory.save("_auto_extracted", "\n".join(lines))

"""Heartbeat token helpers.

The agent uses the literal string ``HEARTBEAT_OK`` to signal "nothing to do"
during a heartbeat turn.  This module provides two small helpers:

- :func:`is_heartbeat_ok` — detect the token in a response.
- :func:`strip_heartbeat_token` — remove it so it never appears in Matrix output.

The token is intentionally standalone on its own line (or the entire response)
so casual mentions of the word do not get stripped.
"""

from __future__ import annotations

_TOKEN = "HEARTBEAT_OK"


def is_heartbeat_ok(text: str | None) -> bool:
    """Return True when *text* is a silent-acknowledgement heartbeat response.

    Args:
        text: Agent response text (may be None when the model emits no text).

    Returns:
        True when the stripped text equals or contains only ``HEARTBEAT_OK``.
    """
    if not text:
        return False
    return text.strip() == _TOKEN or text.strip().startswith(_TOKEN)


def strip_heartbeat_token(text: str) -> str:
    """Remove ``HEARTBEAT_OK`` lines from *text* and return the remainder.

    Strips lines that are exactly ``HEARTBEAT_OK`` (case-sensitive).  Any
    surrounding prose the agent chose to attach is preserved so the caller can
    decide whether a non-empty remainder is worth posting.

    Args:
        text: Raw agent response text.

    Returns:
        Text with ``HEARTBEAT_OK`` lines removed, then stripped.
    """
    lines = [line for line in text.splitlines() if line.strip() != _TOKEN]
    return "\n".join(lines).strip()

"""Action-sensitive memory boundary metadata — Stage One Phase 6, slice A.

**Goal (from the plan): make proactive behavior useful without converting
remembered text into authority.** A topic note can optionally carry
action-boundary metadata — who owns the claim, when it applies, when it
becomes safe to act on, when it expires, what unlocks it, what it
prohibits, and what approval it requires — as a small YAML-ish frontmatter
block at the top of the note file:

    ---
    owner: main
    applies_when: deploying to production
    safe_after: 2026-09-01
    expires_at: 2026-12-01
    unlock_condition: explicit user confirmation this quarter
    prohibited_action: do not deploy without a second reviewer
    required_approval: user
    ---
    The rest of the note's body, as usual.

All seven fields are optional; a note with none of them behaves exactly as
before (empty metadata, ``format_boundary_prefix`` returns ``""``, nothing
rendered). Unknown keys in a frontmatter block are silently ignored —
forward-compatible, and it means a human typo in a field name degrades to
"not applied" rather than a parse error.

Advisory only — this is the whole point
------------------------------------------
This metadata is never wired into :class:`~minion_assist.tools.policy.PermissionPolicy`
or any tool-execution path — there is no code anywhere that reads a note's
``required_approval``/``prohibited_action`` text and uses it to allow or
block a tool call. It exists purely to be *rendered* alongside a note's
content wherever that note is retrieved (search, per-turn injection — see
``memory/service.py``'s ``MemoryService._apply_boundaries``), so a model
reading "requires approval from X" in its own memory sees that as
something to *ask about*, not as standing authorization it already has.
This is the concrete mechanism behind the plan's acceptance criterion "a
remembered approval never bypasses current permission policy": the
criterion is satisfied structurally, by there being no path connecting the
two systems at all, not by a runtime check.

Time-window fields are mechanically enforced; the rest are just rendered
--------------------------------------------------------------------------
``safe_after``/``expires_at`` are the only two fields with a machine-checkable
meaning — :func:`is_boundary_active` treats them as a ``[safe_after,
expires_at]`` window (either or both may be absent, meaning "no constraint
on that side") and a note outside its own window is excluded from
retrieval entirely, not merely labeled — the plan's "expired constraints
... do not influence action" acceptance criterion. ``owner``,
``applies_when``, ``unlock_condition``, ``prohibited_action``, and
``required_approval`` are free text for a human/model to reason about;
nothing here evaluates them.

Talks to
--------
- ``memory/postgres_index.py`` — :func:`parse_frontmatter` is called by
  ``reindex_file()``/``force_rebuild_agent()`` before chunking, so the
  frontmatter block never becomes searchable body text; the parsed
  metadata is cached in a new ``memory_files.boundary_metadata`` column
  (``get_boundary()``) rather than re-read from disk on every search hit.
- ``memory/service.py`` — :meth:`MemoryService._apply_boundaries` calls
  :func:`is_boundary_active` (to drop an inactive hit) and
  :func:`format_boundary_prefix` (to annotate an active one).
"""

from __future__ import annotations

import time
from datetime import datetime

# The only recognized frontmatter keys. Anything else in the block is
# silently ignored — see the module docstring's "forward-compatible" note.
# Order here also controls rendering order in format_boundary_prefix().
_KNOWN_FIELDS: tuple[str, ...] = (
    "owner",
    "applies_when",
    "safe_after",
    "expires_at",
    "unlock_condition",
    "prohibited_action",
    "required_approval",
)

_FIELD_LABELS: dict[str, str] = {
    "owner": "Owner",
    "applies_when": "Applies when",
    "safe_after": "Not safe to act on until",
    "expires_at": "Expires",
    "unlock_condition": "Unlocks when",
    "prohibited_action": "Prohibited",
    "required_approval": "Requires approval from",
}


def parse_frontmatter(content: str) -> tuple[dict[str, str], str, int]:
    """Split a note's boundary frontmatter (if any) from its body.

    Only a frontmatter block whose opening ``---`` is the file's literal
    first line is recognized — anything else (no leading ``---``, or an
    unterminated block with no closing ``---``) is treated as "no
    frontmatter," returning the content unchanged. A malformed block
    degrading to "ignored" rather than raising is deliberate: a typo in a
    note a human hand-edits must never break indexing.

    Args:
        content: The file's full, unmodified text.

    Returns:
        tuple[dict[str, str], str, int]: ``(metadata, body, line_offset)``.
        ``metadata`` maps recognized field names (see :data:`_KNOWN_FIELDS`)
        to their string values — empty dict if there's no frontmatter block
        or it contains no recognized fields. ``body`` is ``content`` with
        the frontmatter block (both ``---`` lines and everything between)
        removed. ``line_offset`` is how many lines the frontmatter block
        occupied — callers that chunk ``body`` must add this back onto
        every chunk's ``start_line``/``end_line`` so citations still point
        at the right line in the *original* file on disk.
    """
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, content, 0

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return {}, content, 0

    metadata: dict[str, str] = {}
    for line in lines[1:end_idx]:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in _KNOWN_FIELDS and value:
            metadata[key] = value

    body = "\n".join(lines[end_idx + 1 :])
    return metadata, body, end_idx + 1


def _parse_time_epoch(value: str) -> float | None:
    """Parse an ISO date/datetime string to epoch seconds, or ``None`` if unparseable.

    A bare date (``"2026-09-01"``) or a naive datetime is interpreted in
    the local system timezone via ``datetime.timestamp()`` — the least
    surprising reading for a human- or model-authored note that didn't
    specify an offset. A string that *does* include an offset (e.g.
    ``"...+10:00"``) is respected as given.
    """
    try:
        return datetime.fromisoformat(value).timestamp()
    except (ValueError, TypeError):
        return None


def is_boundary_active(metadata: dict[str, str], now: float | None = None) -> bool:
    """Whether a note's ``[safe_after, expires_at]`` time window currently includes ``now``.

    Args:
        metadata: A note's parsed frontmatter (from :func:`parse_frontmatter`
            or :meth:`~minion_assist.memory.postgres_index.PostgresMemoryIndex.get_boundary`).
            Empty/no ``safe_after``/``expires_at`` fields → always active.
        now: Epoch seconds. Defaults to ``time.time()``.

    Returns:
        bool: ``False`` if ``now`` is before ``safe_after`` or after
            ``expires_at`` (an unparseable value is treated as "no
            constraint," not as a failure — see :func:`_parse_time_epoch`).
    """
    if not metadata:
        return True
    now = time.time() if now is None else now

    safe_after = metadata.get("safe_after")
    if safe_after:
        parsed = _parse_time_epoch(safe_after)
        if parsed is not None and now < parsed:
            return False

    expires_at = metadata.get("expires_at")
    if expires_at:
        parsed = _parse_time_epoch(expires_at)
        if parsed is not None and now > parsed:
            return False

    return True


def format_boundary_prefix(metadata: dict[str, str]) -> str:
    """Render a note's boundary metadata as a one-line advisory annotation.

    Args:
        metadata: A note's parsed frontmatter.

    Returns:
        str: ``""`` if ``metadata`` is empty. Otherwise a single bracketed
            line listing every present field in :data:`_KNOWN_FIELDS`
            order, explicitly labeled advisory — see the module docstring's
            "Advisory only" section for why the wording matters.
    """
    if not metadata:
        return ""
    parts = [f"{_FIELD_LABELS[key]}: {metadata[key]}" for key in _KNOWN_FIELDS if key in metadata]
    return "[Boundary — advisory only, does not itself grant permission — " + "; ".join(parts) + "]"

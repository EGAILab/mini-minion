"""Claim markers — Stage One Phase 7, slice A: knowledge schema and claim sync.

**Goal (from the plan): add wiki-like belief maintenance only after capture
and retrieval are reliable.** This is the plan's only phase with no
analogous feature in OpenClaw's actual source to ground design decisions
against (verified by searching — nothing resembling a claim/entity/
evidence graph exists there); the design below is original to this
project, built to be consistent with everything Stage One already
established rather than borrowed from a reference implementation.

A "page" is not a new concept
--------------------------------
Task 1 asks for a "stable page identifier." A topic note's
``(agent_id, rel_path)`` — already the stable identifier every other part
of this project uses (``memory_files``, ``memory_pins``,
``memory_consolidation_previews``, ...) — already *is* that. No separate
``kb_pages`` table is added here; it would just duplicate what
``memory_files`` already tracks.

Claim markers: the file is still the source of truth
-----------------------------------------------------------
Consistent with every phase before this one ("files are the source of
truth, Postgres is a derived cache"), a claim's stable id and structured
fields live *in the canonical Markdown page itself*, as an inline HTML
comment attached to the text it annotates::

    - User's dog is named Biscuit.
      <!-- claim:c-a1b2c3d4 status=supported confidence=0.9
           observed=2026-06-01 evidence=proposal:42 -->

:func:`parse_claims` is read-only relative to file content — it never
invents a claim from unmarked prose. Plain, unmarked sentences in a topic
note stay exactly what they've always been: prose outside the tracked
system. A sentence only enters ``kb_claims`` once something — a human
hand-writing the marker, or :class:`~minion_assist.memory.consolidation.MemoryConsolidator`
in a later slice — explicitly attaches one. ``PostgresMemoryIndex``
(``reindex_file()``/``force_rebuild_agent()``) calls :func:`parse_claims`
and syncs the result into Postgres the same way it already syncs chunks
and boundary metadata — Postgres never invents or edits a marker itself.

Recognized fields
------------------
- ``status`` — one of ``supported``/``contested``/``superseded``/``unknown``
  (Task 2). Defaults to ``"unknown"`` when absent or unrecognized — a
  claim nobody has classified yet is exactly that, not an error.
- ``confidence`` — a float; ``None`` if absent or unparseable.
- ``observed`` — ISO date/datetime the claim was learned (Task 2's
  "observed time"). ``None`` if absent — the sync step then uses the
  moment of syncing as a fallback (see ``PostgresMemoryIndex``'s
  ``_sync_claims``).
- ``valid_from`` / ``valid_to`` — ISO date/datetime bounds on when the
  claim is/was true in the world (Task 2's "valid time" — distinct from
  *observed* time, the bi-temporal distinction the plan explicitly asks
  for). Either or both may be absent, meaning no bound on that side.
- ``privacy`` — free text (Task 2's "privacy tier"); not validated
  against an enum, the same way ``memory/boundaries.py``'s fields are
  free text.
- ``entity`` — an optional entity name this claim is about. Resolved by
  exact, case-insensitive name match within the agent's scope (see
  ``PostgresMemoryIndex.get_or_create_entity``) — deliberately no fuzzy
  entity resolution/merging in this slice, a genuinely hard NLP problem
  with no evaluation data yet to justify building, the same "collect
  data before choosing thresholds" posture the rest of Stage One has
  taken throughout.
- ``evidence`` — comma-separated ``kind:ref`` pairs (Task 1's "evidence
  identifiers"), e.g. ``evidence=proposal:42,message:1189``. A claim
  marker with no ``evidence`` field is still parsed and synced — hand-
  authored content can't be retroactively forced to cite a source — but
  shows up in the provenance-gap dashboard (a later slice).

``freshness`` (also Task 2) is deliberately *not* a marker field — it's
derived at query time from ``observed_at`` (a decay function, mirroring
``postgres_index.py``'s existing ``_decay_factor``), not something a
human or model would hand-author.

Talks to
--------
- ``memory/postgres_index.py`` — :meth:`PostgresMemoryIndex._sync_claims`
  calls :func:`parse_claims` from ``reindex_file()``/``force_rebuild_agent()``
  (``source_kind == "durable"`` only) and writes the result into
  ``kb_entities``/``kb_claims``/``kb_evidence``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")
_CLAIM_MARKER_RE = re.compile(r"<!--\s*claim:(\S+)(.*?)-->", re.DOTALL)
_FIELD_RE = re.compile(r"(\w+)=(\S+)")

_KNOWN_STATUSES = frozenset({"supported", "contested", "superseded", "unknown"})


@dataclass
class ParsedClaim:
    """One claim marker parsed from a page's raw content.

    Attributes:
        id: The claim's stable id, from the marker (e.g. ``"c-a1b2c3d4"``)
            — whatever string follows ``claim:``, verbatim.
        text: The claim's stated text — the content of the block (list
            item or paragraph) the marker is attached to, with the
            marker's own comment text removed and any leading list
            bullet stripped.
        status: One of :data:`_KNOWN_STATUSES`; ``"unknown"`` if absent
            or unrecognized.
        confidence: Parsed float, or ``None`` if absent/unparseable.
        observed: Raw ``observed=`` value (still a string — parsed to
            epoch seconds by the caller, mirroring
            ``memory/boundaries.py``'s own time handling), or ``None``.
        valid_from: Raw ``valid_from=`` value, or ``None``.
        valid_to: Raw ``valid_to=`` value, or ``None``.
        privacy: Raw ``privacy=`` value, or ``""`` if absent.
        entity: Raw ``entity=`` value, or ``None`` if absent.
        evidence: ``(source_kind, source_ref)`` pairs parsed from
            ``evidence=kind:ref,kind:ref``. Empty if the field is absent
            or empty.
        line: 1-indexed line number the marker itself starts on, for
            citations/debugging.
    """

    id: str
    text: str
    status: str = "unknown"
    confidence: float | None = None
    observed: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    privacy: str = ""
    entity: str | None = None
    evidence: list[tuple[str, str]] = field(default_factory=list)
    line: int = 0


def _extract_claim_text(raw_span: str) -> str:
    """From the text between the previous marker (or file start) and this one,
    take just the last paragraph/list-item block — the specific text this
    marker is attached to, not everything since the previous claim.
    """
    lines = raw_span.splitlines()
    # Drop a trailing whitespace-only fragment: this is the indentation
    # before the marker's own "<!--" on its own line (raw_span is cut off
    # mid-line right at the marker's start), not a genuine blank-line
    # paragraph break in the source document — treating it as one would
    # incorrectly reset the block and discard the claim text just found.
    while lines and not lines[-1].strip():
        lines.pop()
    block: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            block = []
            continue
        if _LIST_ITEM_RE.match(line):
            block = [line]
        else:
            block.append(line)
    text = " ".join(ln.strip() for ln in block).strip()
    return _LIST_ITEM_RE.sub("", text, count=1).strip()


def _parse_confidence(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_evidence(raw: str | None) -> list[tuple[str, str]]:
    if not raw:
        return []
    pairs = []
    for item in raw.split(","):
        if ":" not in item:
            continue
        kind, _, ref = item.partition(":")
        if kind and ref:
            pairs.append((kind, ref))
    return pairs


def parse_claims(content: str) -> list[ParsedClaim]:
    """Parse every claim marker in a page's raw content.

    Args:
        content: The page's full, unmodified text.

    Returns:
        list[ParsedClaim]: In document order. Empty if the page has no
            claim markers — the overwhelmingly common case for any note
            nobody has annotated yet.
    """
    claims = []
    last_end = 0
    for match in _CLAIM_MARKER_RE.finditer(content):
        claim_id = match.group(1)
        fields = dict(_FIELD_RE.findall(match.group(2)))
        text = _extract_claim_text(content[last_end : match.start()])
        line_number = content.count("\n", 0, match.start()) + 1

        status = fields.get("status", "unknown")
        if status not in _KNOWN_STATUSES:
            status = "unknown"

        claims.append(
            ParsedClaim(
                id=claim_id,
                text=text,
                status=status,
                confidence=_parse_confidence(fields.get("confidence")),
                observed=fields.get("observed"),
                valid_from=fields.get("valid_from"),
                valid_to=fields.get("valid_to"),
                privacy=fields.get("privacy", ""),
                entity=fields.get("entity"),
                evidence=_parse_evidence(fields.get("evidence")),
                line=line_number,
            )
        )
        last_end = match.end()
    return claims


def parse_time_epoch(value: str | None) -> float | None:
    """Parse an ISO date/datetime string to epoch seconds, or ``None``.

    Same approach as ``memory/boundaries.py``'s own ``_parse_time_epoch``
    and ``memory/commitments.py``'s own copy — duplicated rather than
    imported, the established precedent for this exact tiny helper
    across this project's memory modules.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except (ValueError, TypeError):
        return None

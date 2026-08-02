"""``forget_source`` — cascade forgetting one evidence source to its derivative claims.

Stage One Phase 7, slice F — the plan's last Phase 7 acceptance criterion:
"forgetting a source identifies and removes or re-evaluates all
derivatives." A "source" here is an evidence citation
(``source_kind``/``source_ref`` — e.g. ``("proposal", "42")`` or
``("import", "_auto_extracted")``, the same pairs a claim marker's
``evidence=`` field encodes, see ``memory/knowledge.py``'s module
docstring). Forgetting one finds every claim citing it
(``PostgresMemoryIndex.list_claims_citing_evidence``) and edits each
affected page's claim marker directly
(``memory/knowledge.py``'s ``remove_evidence_from_content``) — never just
the derived Postgres cache, which would be silently overwritten back to
the stale citation the next time that page is reindexed.

Human-triggered, not automatic
----------------------------------
Like every other mutating action Stage One has added (``approve()``,
``reject()`` in both ``consolidation.py`` and ``import_review.py``),
:func:`forget_source` only runs when a human explicitly invokes it (the
CLI's ``memory knowledge forget`` command) — nothing in this project ever
decides on its own that a source should be forgotten. And like
``reject()`` in both sibling pipelines, there is no preview step and no
rollback: forgetting is a deliberate, explicitly-named cleanup action a
human types directly, not a draft awaiting review.

Claims left with no evidence at all are re-flagged ``status=unknown``
(never silently deleted) — the same "surface gaps, don't hide them"
posture the rest of Phase 7 already takes (dangling contradictions,
provenance gaps). See :func:`~minion_assist.memory.knowledge.remove_evidence_from_content`
for exactly how a marker is edited.

``MEMORY.md`` is skipped, not edited
----------------------------------------
A claim can technically live in ``MEMORY.md`` too (it is
``source_kind == "durable"``, so ``_sync_claims`` parses it same as any
topic note). But nothing in this project ever programmatically writes
``MEMORY.md`` — it is human/bootstrap-owned everywhere else
(``memory/files.py``'s module docstring). Rather than break that
invariant, a claim living in a page :func:`_topic_key_from_rel_path`
doesn't recognize as a topic note is left untouched and reported back in
``"skipped_manual_review"`` for a human to handle by hand.

Talks to
--------
- ``memory/postgres_index.py`` — :meth:`~minion_assist.memory.postgres_index.PostgresMemoryIndex.list_claims_citing_evidence`
  (the read-only lookup), :meth:`~minion_assist.memory.postgres_index.PostgresMemoryIndex.reindex_file`
  (re-syncs a page after its marker is edited).
- ``memory/knowledge.py`` — :func:`~minion_assist.memory.knowledge.remove_evidence_from_content`.
- ``memory/files.py`` — :meth:`~minion_assist.memory.files.MemoryFileRepository.load`/``remember``
  (topic notes only — see above).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .knowledge import remove_evidence_from_content

if TYPE_CHECKING:
    from .files import MemoryFileRepository
    from .postgres_index import PostgresMemoryIndex

# memory/files.py's MemoryFileRepository always stores topic notes under
# this prefix (see its module docstring) — used to recognize a claim's
# rel_path as an editable topic note, and to recover its key.
#
# Duplicated from consolidation.py's _topic_key_from_rel_path rather than
# imported: it's a tiny (two-line), purely syntactic helper with no real
# logic to share — the same "tiny helper, safe to keep local to each
# module" precedent consolidation.py's own _hash_text docstring already
# sets, as opposed to find_merge_target/parse_draft_response (real,
# nontrivial logic), which were promoted to shared public functions.
_TOPICS_PREFIX = "memory/topics/"


def _topic_key_from_rel_path(rel_path: str) -> str | None:
    """Recover a topic note's key from its indexed ``rel_path``, or ``None`` if it isn't one."""
    if not rel_path.startswith(_TOPICS_PREFIX) or not rel_path.endswith(".md"):
        return None
    return rel_path[len(_TOPICS_PREFIX) : -len(".md")]


def forget_source(
    index: PostgresMemoryIndex,
    files: MemoryFileRepository,
    agent_id: str,
    source_kind: str,
    source_ref: str,
) -> dict:
    """Cascade forgetting one evidence source to every claim citing it.

    Args:
        index: The lexical index ``kb_claims``/``kb_evidence`` live in.
        files: The agent's ``MemoryFileRepository`` — reads/writes the
            affected topic notes.
        agent_id: Which agent's claims to search.
        source_kind: The evidence kind to forget, e.g. ``"proposal"``.
        source_ref: The evidence reference to forget, e.g. a proposal id
            (as a string — evidence refs are always stored as text).

    Returns:
        dict: ``{"source_kind", "source_ref", "reevaluated",
            "still_grounded", "skipped_manual_review"}``.

            - ``reevaluated``: claim ids that lost their last evidence
              and were re-flagged ``status=unknown``.
            - ``still_grounded``: claim ids that cited this source but
              still have at least one other citation — status untouched.
            - ``skipped_manual_review``: ``{"claim_id", "rel_path"}``
              dicts for claims living in a page that can't be
              auto-edited (i.e. ``MEMORY.md`` — see the module
              docstring), left entirely untouched.

            A source cited by nothing returns all-empty lists — a valid,
            harmless no-op, not an error.
    """
    affected = index.list_claims_citing_evidence(agent_id, source_kind, source_ref)

    by_rel_path: dict[str, list[dict]] = {}
    for claim in affected:
        by_rel_path.setdefault(claim["rel_path"], []).append(claim)

    reevaluated: list[str] = []
    still_grounded: list[str] = []
    skipped_manual_review: list[dict] = []

    for rel_path, claims in by_rel_path.items():
        target_key = _topic_key_from_rel_path(rel_path)
        if target_key is None:
            skipped_manual_review.extend(
                {"claim_id": c["id"], "rel_path": rel_path} for c in claims
            )
            continue

        content = files.load(target_key) or ""
        for claim in claims:
            content, has_remaining = remove_evidence_from_content(
                content, claim["id"], source_kind, source_ref
            )
            (still_grounded if has_remaining else reevaluated).append(claim["id"])

        files.remember(target_key, content)
        index.reindex_file(agent_id, rel_path, "durable", content)

    return {
        "source_kind": source_kind,
        "source_ref": source_ref,
        "reevaluated": reevaluated,
        "still_grounded": still_grounded,
        "skipped_manual_review": skipped_manual_review,
    }

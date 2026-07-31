"""Proposal ranking and preview drafting — Stage One Phase 5, slice C.

Nothing in this module ever writes to disk or changes a proposal's status.
It only ranks unreviewed ``memory_proposals`` (Phase 2, slice C) into a
review queue and, for a chosen proposal, drafts what a topic-note update
*would* look like — storing the draft as a
``memory_consolidation_previews`` row (``postgres_index.py``) for a human
to look at. A later slice adds apply/reject/rollback on top of these rows;
until then a preview is inert, exactly like a proposal.

Ranking signals: what's used, and what's deliberately skipped
------------------------------------------------------------------
The plan's Task 3 asks for confidence, source authority, recall count,
query diversity, recall days, recency, contradiction status, and user
pinning. This slice's :func:`rank_proposals` only uses **recall count**,
**query diversity** (``unique_queries``), and **injected count** — all
real signals already recorded by Phase 5 slice A/B's telemetry. The rest
are skipped for a concrete reason, not an oversight:

- **Confidence**: ``memory/extractor.py``'s ``extract_facts`` returns bare
  claim strings, no per-claim confidence score. Adding one means changing
  the extraction prompt/parser — a separate change, not this slice's job.
- **Source authority**: every proposal comes from the same capture
  pipeline (``CaptureWorker``) — there is no second source to
  differentiate from yet.
- **Contradiction status**: detecting this needs a semantic comparison
  against existing notes, which does not exist as a precomputed signal.
  Instead, :class:`MemoryConsolidator`'s drafting prompt tells the model
  to surface a contradiction *in the drafted text* (never silently
  resolve it) rather than pre-scoring it — see "Contradictions" below.
- **User pinning**: pinning (Phase 4, slice B) is scoped to topic notes
  only; proposals were never made pinnable.

The ranking score is therefore a deliberately simple, documented
placeholder — this directly matches the plan's own Task 4 language,
"begin in preview-only mode; collect data before choosing thresholds."
Nothing is gated on this score; it only orders a human's review queue.

Evidence provenance (Task 8)
-----------------------------
"Never use ``DREAMS.md`` or generated consolidation prose as evidence for
a new promotion. Evidence must resolve to user messages, trusted source
notes, or reviewed imports." This module never reads ``DREAMS.md`` or any
``dreaming.py``/``DreamingScheduler`` state — a proposal's claim text is
the only "new evidence" a draft is built from, and it is only ever
produced by ``extract_facts()`` reading real captured message exchanges
(``memory_capture_jobs`` → ``session/db.py`` messages), never diary text.
The *existing* topic-note content a draft merges into is the **merge
target**, not "evidence" for the new claim — the claim's evidence chain
runs through its own ``job_id``, unaffected by what it's being merged
into. See ``tests/memory/test_consolidation.py`` for an explicit
regression test of this boundary.

Contradictions
--------------
The drafting prompt (:data:`_DRAFT_SYSTEM`) explicitly instructs the model
to keep both statements and flag the disagreement in the drafted text
rather than silently pick one, whenever a new claim contradicts the
target note's existing content — this is what the plan's acceptance
criterion "contradictory preferences remain contested until resolved and
are never merged into a false synthesis" requires of *this* slice (the
resolution itself is a human's job, in whatever review flow a later slice
adds).

Talks to
--------
- ``session/db.py`` — :meth:`SessionDB.list_pending_proposals`,
  :meth:`SessionDB.get_proposal`.
- ``memory/postgres_index.py`` — :meth:`PostgresMemoryIndex.recall_stats`
  (ranking), :meth:`PostgresMemoryIndex.hybrid_search` (finding a merge
  target, restricted to ``corpus="durable"``), and
  :meth:`PostgresMemoryIndex.record_consolidation_preview`.
- ``memory/files.py`` — :meth:`MemoryFileRepository.load` reads a merge
  target's current content; nothing here ever calls ``remember()`` — no
  write happens in this slice.
- ``providers/base.py`` — :class:`LLMProvider`, for the drafting call.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..providers.base import LLMProvider
    from ..session.db import SessionDB
    from .files import MemoryFileRepository
    from .postgres_index import PostgresMemoryIndex

# memory/files.py's MemoryFileRepository always stores topic notes under
# this prefix (see its module docstring) — used to recognize a
# hybrid_search hit as a candidate merge target, and to recover the note's
# key from its rel_path.
_TOPICS_PREFIX = "memory/topics/"

_DRAFT_SYSTEM = """\
You are drafting a personal-memory note update for a human to review. \
Nothing you write is saved automatically.

You are given a new claim extracted from a real conversation, and either \
the current content of an existing note it likely belongs in, or a note \
that no existing note matches (in which case propose a new one).

Rules:
- If revising an existing note, preserve everything in it that's still \
true. Integrate the new claim naturally — don't just append an \
unintegrated bullet point.
- If the new claim CONTRADICTS the existing content, do NOT silently \
resolve it. Keep both statements and clearly flag them as contested/
unresolved in the drafted text, so a human decides.
- If there's no existing note, propose a short kebab-case topic key (e.g. \
"coding-preferences") and draft a new note from scratch.
- Keep the note short, factual, and free of filler — matching the style \
of a personal memory note, not an essay.

Respond in exactly this format, nothing before or after it:
KEY: <kebab-case-key>
RATIONALE: <one sentence: why this draft, or what changed>
---
<the full note content>"""


def _hash_text(text: str) -> str:
    """SHA256 hex digest of a file's content.

    Mirrors ``postgres_index.py``'s own ``_hash_text`` (same algorithm) —
    duplicated rather than imported, the same precedent that module's own
    docstring already sets against ``memory/migration.py``'s
    ``_hash_bytes``: a tiny, stable hash helper, safe to keep local to
    each module rather than centralizing.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _topic_key_from_rel_path(rel_path: str) -> str | None:
    """Recover a topic note's key from its indexed ``rel_path``, or ``None`` if it isn't one.

    ``MemoryFileRepository.topic_path()`` always writes to
    ``memory/topics/{sanitized_key}.md`` — this is the exact inverse.
    """
    if not rel_path.startswith(_TOPICS_PREFIX) or not rel_path.endswith(".md"):
        return None
    return rel_path[len(_TOPICS_PREFIX):-len(".md")]


def rank_proposals(db: SessionDB, index: PostgresMemoryIndex, agent_id: str) -> list[dict]:
    """Rank one agent's pending proposals into a human review queue.

    See the module docstring for exactly which signals feed the score and
    why several plan-listed ones are deliberately not used yet. This only
    orders the queue — nothing here gates or auto-applies anything.

    Score: ``5 * injected_count + 2 * recall_count + unique_queries``.
    Weighted so a proposal that was actually selected for per-turn
    injection (the strongest real signal that it mattered to a
    conversation) ranks above one that was merely returned by a search,
    with query diversity as a light tiebreak. A proposal never recalled
    still appears (score 0) rather than being hidden — a human should be
    able to see brand-new proposals too, not only ones that happened to
    already be searched.

    Args:
        db: The ``SessionDB`` proposals live in.
        index: The lexical index recall telemetry (Phase 5, slice A) and
            proposal chunks (slice B) live in.
        agent_id: Which agent's proposals to rank.

    Returns:
        list[dict]: Each pending proposal's fields (``id``, ``job_id``,
            ``agent_id``, ``claim_text``, ``created_at``) plus
            ``recall_count``, ``unique_queries``, ``injected_count``,
            ``last_recalled_at``, and ``score`` — highest score first,
            ties broken by proposal id (oldest first) for deterministic
            ordering.
    """
    pending = db.list_pending_proposals(agent_id)
    ranked = []
    for proposal in pending:
        stats = index.recall_stats(agent_id, f"proposals/{proposal['id']}")
        score = 5 * stats["injected_count"] + 2 * stats["recall_count"] + stats["unique_queries"]
        ranked.append({**proposal, **stats, "score": score})
    ranked.sort(key=lambda p: (-p["score"], p["id"]))
    return ranked


def _parse_draft_response(text: str) -> tuple[str, str, str]:
    """Parse :data:`_DRAFT_SYSTEM`'s fixed ``KEY:``/``RATIONALE:``/``---`` response format.

    Returns:
        tuple[str, str, str]: ``(key, rationale, content)``.

    Raises:
        ValueError: If the response doesn't contain the ``---`` content
            separator — a malformed drafting response is a real failure,
            not something to silently paper over, since a preview with
            garbage content would be shown to a human as if it were real.
    """
    lines = text.splitlines()
    key = ""
    rationale = ""
    content_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.upper().startswith("KEY:"):
            key = stripped.split(":", 1)[1].strip()
        elif stripped.upper().startswith("RATIONALE:"):
            rationale = stripped.split(":", 1)[1].strip()
        elif stripped == "---":
            content_start = i + 1
            break
    if content_start is None:
        raise ValueError(
            f"Consolidator drafting response missing '---' content separator: {text[:200]!r}"
        )
    content = "\n".join(lines[content_start:]).strip()
    return key, rationale, content


def format_preview_report(preview: dict) -> str:
    """Human-readable review report for one preview (Task 7).

    "Decision" always reads "pending review" — nothing decides anything in
    this slice; a later slice's approve/reject flow is what fills that in.

    Args:
        preview: A preview dict, as returned by
            :meth:`MemoryConsolidator.preview` or one row from
            :meth:`~minion_assist.memory.postgres_index.PostgresMemoryIndex.list_consolidation_previews`.

    Returns:
        str: Multi-line plain text — candidate, rationale, decision
            placeholder, and the full drafted content.
    """
    action = "Create new topic" if preview["target_kind"] == "new_topic" else "Revise topic"
    lines = [
        f"Proposal #{preview['proposal_id']}",
        f"{action}: {preview['target_key']}",
        f"Rationale: {preview['rationale']}",
        "Decision: pending review",
        "",
        "--- Drafted content ---",
        preview["drafted_content"],
    ]
    return "\n".join(lines)


class MemoryConsolidator:
    """Drafts (never applies) topic-note updates from pending proposals.

    Args:
        db: The ``SessionDB`` proposals live in.
        index: The lexical index used to find a merge target and to store
            the resulting preview.
        files: The agent's ``MemoryFileRepository`` — read-only here
            (``load()`` only); nothing is ever written to disk by this
            class.
        provider: The LLM provider used to draft the revised note text.
    """

    def __init__(
        self,
        db: SessionDB,
        index: PostgresMemoryIndex,
        files: MemoryFileRepository,
        provider: LLMProvider,
    ) -> None:
        self._db = db
        self._index = index
        self._files = files
        self._provider = provider

    def preview(self, agent_id: str, proposal_id: int) -> dict:
        """Draft a preview for one pending proposal — never writes to disk.

        Args:
            agent_id: The agent the proposal belongs to.
            proposal_id: Which ``memory_proposals`` row to draft from.

        Returns:
            dict: ``{"id", "agent_id", "proposal_id", "target_kind",
                "target_key", "based_on_content_hash", "drafted_content",
                "rationale"}`` — the same shape
                :meth:`~minion_assist.memory.postgres_index.PostgresMemoryIndex.list_consolidation_previews`
                returns (minus ``created_at``, added once persisted).

        Raises:
            ValueError: If ``proposal_id`` doesn't exist, or if the
                provider's drafting response is malformed (see
                :func:`_parse_draft_response`).
        """
        proposal = self._db.get_proposal(proposal_id)
        if proposal is None:
            raise ValueError(f"No proposal with id {proposal_id!r}")
        claim_text = proposal["claim_text"]

        target_key = self._find_merge_target(agent_id, claim_text)
        if target_key is not None:
            target_kind = "revise_topic"
            existing_content = self._files.load(target_key) or ""
        else:
            target_kind = "new_topic"
            existing_content = ""

        based_on_content_hash = _hash_text(existing_content)
        drafted_key, rationale, drafted_content = self._draft(claim_text, existing_content)
        final_key = target_key if target_kind == "revise_topic" else drafted_key

        preview_id = self._index.record_consolidation_preview(
            agent_id, proposal_id, target_kind, final_key,
            based_on_content_hash, drafted_content, rationale,
        )
        return {
            "id": preview_id,
            "agent_id": agent_id,
            "proposal_id": proposal_id,
            "target_kind": target_kind,
            "target_key": final_key,
            "based_on_content_hash": based_on_content_hash,
            "drafted_content": drafted_content,
            "rationale": rationale,
        }

    def _find_merge_target(self, agent_id: str, claim_text: str) -> str | None:
        """Find an existing topic note this claim likely belongs in, or ``None``.

        Restricted to ``corpus="durable"`` and, within that, to
        ``memory/topics/`` hits specifically — never ``MEMORY.md`` itself
        (out of scope for this slice — see the plan discussion this slice
        was scoped from), never daily notes (ephemeral) or imports
        (unreviewed/quarantined per ``docs/adr/0003-per-agent-memory-scope.md``,
        the same reasoning Phase 4 slice B's pinning scope already used).

        No score threshold is applied: the first topic-note hit (if any)
        is treated as the merge target. Without real evaluation data an
        absolute cutoff would be arbitrary (see the module docstring's
        Task 4 note) — and a wrong guess here only produces a preview a
        human can reject, never an applied change.
        """
        hits = self._index.hybrid_search(agent_id, claim_text, corpus="durable", max_results=5)
        for hit in hits:
            key = _topic_key_from_rel_path(hit["rel_path"])
            if key is not None:
                return key
        return None

    def _draft(self, claim_text: str, existing_content: str) -> tuple[str, str, str]:
        """Call the provider to draft the revised/new note text.

        Only ``claim_text`` (traceable to real captured messages) and the
        merge target's own existing content are ever included in the
        prompt — see the module docstring's "Evidence provenance" section.
        """
        existing_block = existing_content or "(no existing note — propose a new one)"
        user_message = f"New claim:\n{claim_text}\n\nExisting note content:\n{existing_block}"
        response = self._provider.chat(
            system=_DRAFT_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
            tools=[],
            max_tokens=800,
        )
        return _parse_draft_response((response.text or "").strip())

"""Proposal ranking, preview drafting, and apply/reject/rollback — Stage One Phase 5, slices C-D.

:func:`rank_proposals` ranks unreviewed ``memory_proposals`` (Phase 2,
slice C) into a review queue. :class:`MemoryConsolidator`'s ``preview()``
drafts what a topic-note update *would* look like, storing the draft as a
``memory_consolidation_previews`` row (``postgres_index.py``) — nothing is
written to disk yet at that point. ``approve()``/``reject()``/
``rollback()`` (slice D) are what actually change something: ``approve()``
writes a preview's drafted content to disk and reindexes it, ``reject()``
only updates the proposal's status, and ``rollback()`` restores a topic
note's pre-apply content. Every one of these three is still a deliberate,
human-triggered action — nothing in this module decides *when* to apply
anything on its own (a later slice's ``MemoryConsolidationScheduler`` only
ever calls ``preview()`` on a timer, never ``approve()``).

One :class:`MemoryConsolidator` instance is scoped to a single agent (its
``agent_id`` is fixed at construction, matching ``MemoryService``'s own
per-agent design) — unlike ``rank_proposals()``, which stays a free
function taking ``agent_id`` per call, since it never touches an agent's
on-disk files (only ``db``/``index``, both already agent-partitioned
internally).

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
regression test of this boundary. Stage One Phase 7, slice B adds the
target's *existing claim markers* (id/status/text — see
``memory/knowledge.py``) to the drafting prompt too, but this is the
same existing-content data already covered by this boundary, just
restructured for the model to reference by id — not a new evidence
source.

Contradictions and claim markers (Stage One Phase 7, slice B)
------------------------------------------------------------------
The drafting prompt (:data:`_DRAFT_SYSTEM`) explicitly instructs the model
to keep both statements and flag the disagreement in the drafted text
rather than silently pick one, whenever a new claim contradicts the
target note's existing content — this is what the plan's acceptance
criterion "contradictory preferences remain contested until resolved and
are never merged into a false synthesis" requires of *this* slice (the
resolution itself is a human's job, in whatever review flow a later slice
adds).

Since Phase 7, slice B, this isn't just prose anymore: ``preview()``
generates a fresh claim id (code-generated, never left to the model to
invent — the same reasoning ``PostgresMemoryIndex.get_or_create_entity``
already applies to entity ids) and gives it to the model along with the
target's existing claim markers (id/status/text). The prompt instructs
the model to attach a real ``<!-- claim:ID ... -->`` marker to the new
claim, and — when it recognizes a conflict — to set ``status=contested``
and ``contradicts=EXISTING_ID`` on it *and* flip the existing marker's
own ``status`` field to ``contested`` too (leaving everything else about
that existing marker untouched). Nothing about this requires new sync
code here: once a human calls ``approve()``, the existing
``reindex_file()`` call already parses and syncs whatever markers ended
up in the drafted content — see ``memory/knowledge.py``'s module
docstring for that whole mechanism. A malformed or missing marker in the
model's response is not fatal — it just means this particular claim
never enters ``kb_claims``, the same as any other hand-authored note
that has no markers at all.

Staleness and rollback (Task 6, slice D)
------------------------------------------
A preview records ``based_on_content_hash`` — the merge target's content
hash *at draft time*. ``approve()`` re-hashes the target's *current*
content and refuses to apply (raising :class:`StaleProposalError`) if it
no longer matches — a human's manual edit made between preview and
approval always wins over stale consolidator output, never gets silently
clobbered. ``rollback()`` restores a topic note's exact pre-apply content
from ``memory_topic_revisions`` (a snapshot ``approve()`` records right
before writing), consuming that revision row so repeated rollbacks step
back through history one apply at a time, and puts the proposal itself
back to ``"pending"`` — undoing an approval undoes the review decision
too, not just the file content.

Historical backfill (Task 10)
--------------------------------
:func:`backfill_agent` finds message ranges in an agent's session history
that no ``memory_capture_jobs`` row has ever covered (comparing every
session's actual message ids against every capture job's recorded
range — true gap-filling, not just "skip sessions with any coverage"),
chunks each gap into bounded windows (:data:`_BACKFILL_WINDOW_MESSAGES`),
and enqueues one capture job per window via the existing
``SessionDB.enqueue_capture_job``. It does not run extraction itself —
the already-running ``CaptureWorker`` picks these jobs up and processes
them exactly like a live turn's job, so no second extraction path exists.

Talks to
--------
- ``session/db.py`` — :meth:`SessionDB.list_pending_proposals`,
  :meth:`SessionDB.get_proposal`, :meth:`SessionDB.set_proposal_status`,
  :meth:`SessionDB.list_session_ids_for_agent`,
  :meth:`SessionDB.list_message_ids`,
  :meth:`SessionDB.list_capture_job_ranges`,
  :meth:`SessionDB.enqueue_capture_job`.
- ``memory/postgres_index.py`` — :meth:`PostgresMemoryIndex.recall_stats`
  (ranking), :meth:`PostgresMemoryIndex.hybrid_search` (finding a merge
  target, restricted to ``corpus="durable"``),
  :meth:`PostgresMemoryIndex.record_consolidation_preview`/
  :meth:`~PostgresMemoryIndex.get_consolidation_preview`, and
  :meth:`~PostgresMemoryIndex.record_topic_revision`/
  :meth:`~PostgresMemoryIndex.latest_topic_revision`/
  :meth:`~PostgresMemoryIndex.delete_topic_revision`, and (Stage One
  Phase 7, slice B) :meth:`~PostgresMemoryIndex.list_claims` (the
  target's existing claims, shown to the drafting prompt) —
  :meth:`~PostgresMemoryIndex.reindex_file`, already called by
  ``approve()``, is what actually syncs any claim marker the model
  drafted; this module never writes to ``kb_claims`` directly.
- ``memory/files.py`` — :meth:`MemoryFileRepository.load` reads a merge
  target's current content; ``approve()``/``rollback()`` are the only
  methods in this module that call ``remember()``/``delete()``.
- ``memory/extractor.py`` — :data:`_EXTRACTION_PROMPT_VERSION`, reused by
  :func:`backfill_agent` so a backfilled job's idempotency key has the
  exact same shape a live per-turn job's key would.
- ``providers/base.py`` — :class:`LLMProvider`, for the drafting call.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import date
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

Claim markers:
- Attach exactly one claim marker to the specific line stating the new \
claim, using exactly this id (never invent your own — it's given to you \
below): <!-- claim:NEW_CLAIM_ID status=STATUS confidence=N \
observed=DATE evidence=proposal:PROPOSAL_ID -->
- status is "supported" by default.
- If the new claim contradicts one of the "Existing claims" listed \
below, set status=contested and add contradicts=EXISTING_ID (that \
claim's id) to the new marker. Also find that existing claim's own \
marker, already present in "Existing note content", and change only \
its status field to contested — leave its id, confidence, evidence, \
and everything else about it exactly as given. Never silently pick a \
side.
- confidence is your own 0.0-1.0 estimate of how reliable the new \
claim is.
- observed is today's date (given below), in ISO format (YYYY-MM-DD).
- Do not add or modify any claim marker besides these.

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


def find_merge_target(index: PostgresMemoryIndex, agent_id: str, claim_text: str) -> str | None:
    """Find an existing topic note a piece of new text likely belongs in, or ``None``.

    A free function (not a method) so both :class:`MemoryConsolidator` and
    ``memory/import_review.py``'s ``ImportReviewer`` can share it without
    duplicating the ``hybrid_search`` call — the same merge-target-finding
    logic applies whether the "new claim" came from a capture-job proposal
    or a quarantined import.

    Restricted to ``corpus="durable"`` and, within that, to
    ``memory/topics/`` hits specifically — never ``MEMORY.md`` itself (out
    of scope for this slice — see the plan discussion this slice was
    scoped from), never daily notes (ephemeral) or imports (unreviewed/
    quarantined per ``docs/adr/0003-per-agent-memory-scope.md``, the same
    reasoning Phase 4 slice B's pinning scope already used).

    No score threshold is applied: the first topic-note hit (if any) is
    treated as the merge target. Without real evaluation data an absolute
    cutoff would be arbitrary (see the module docstring's Task 4 note) —
    and a wrong guess here only produces a preview a human can reject,
    never an applied change.
    """
    hits = index.hybrid_search(agent_id, claim_text, corpus="durable", max_results=5)
    for hit in hits:
        key = _topic_key_from_rel_path(hit["rel_path"])
        if key is not None:
            return key
    return None


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


def parse_draft_response(text: str) -> tuple[str, str, str]:
    """Parse the fixed ``KEY:``/``RATIONALE:``/``---`` drafting-response format.

    A public free function (Stage One Phase 7, slice E) — not just
    :data:`_DRAFT_SYSTEM`'s format, ``memory/import_review.py``'s
    ``_IMPORT_DRAFT_SYSTEM`` prompt uses the exact same shape, so both
    ``MemoryConsolidator._draft`` and ``ImportReviewer._draft`` share this
    one parser rather than each keeping their own copy.

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

    "Decision" always reads "pending review" — this renders the *draft*
    itself, not the proposal's current review status (which may since
    have become ``"promoted"``/``"rejected"`` via
    :meth:`MemoryConsolidator.approve`/``reject``). A caller that wants
    the current status should check it separately, e.g. via
    ``SessionDB.get_proposal()``.

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


def is_preview_stale(files: MemoryFileRepository, preview: dict) -> bool:
    """Whether a preview's merge target has changed since it was drafted.

    Used by the CLI's ``explain``/``approve`` commands to warn a human
    before a stale apply is attempted — the read-only check behind
    :meth:`MemoryConsolidator.approve`'s own (enforced) staleness guard.

    Args:
        files: The agent's ``MemoryFileRepository`` (must be the same
            agent the preview belongs to).
        preview: A preview dict — needs ``target_key`` and
            ``based_on_content_hash``.

    Returns:
        bool: ``True`` if the target's current content hash no longer
            matches ``based_on_content_hash``.
    """
    current_content = files.load(preview["target_key"]) or ""
    return _hash_text(current_content) != preview["based_on_content_hash"]


# Fixed message-count window for one backfilled capture job (Stage One
# Phase 5, slice D). Bounds how much history a single extract_facts() call
# ever receives — a live per-turn job only ever covers one exchange (2
# messages), so an unbounded backfill window could dwarf that by orders of
# magnitude and risk exceeding the extraction model's context. Not
# configurable (yet) — same "no knob without evaluation data" reasoning as
# capture_worker.py's own hardcoded constants.
_BACKFILL_WINDOW_MESSAGES = 20


def _chunk_run(run: list[int]) -> list[tuple[int, int]]:
    """Split one contiguous run of uncovered message ids into bounded (from, to) windows."""
    return [
        (chunk[0], chunk[-1])
        for chunk in (
            run[i : i + _BACKFILL_WINDOW_MESSAGES]
            for i in range(0, len(run), _BACKFILL_WINDOW_MESSAGES)
        )
    ]


def _compute_backfill_windows(
    message_ids: list[int], covered_ranges: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Find bounded (from_id, to_id) windows covering every uncaptured message in a session.

    "Contiguous" here means adjacent *by position in message_ids* (this
    session's own message order), not adjacent integers — ``messages`` is
    one shared table with a single id sequence across every session, so
    two of one session's own messages are almost never numerically
    consecutive (other sessions' messages get ids in between). Walking
    ``message_ids`` in order and tracking runs of "not covered" sidesteps
    that entirely.

    Args:
        message_ids: Every message id in one session, ascending (as
            :meth:`~minion_assist.session.db.SessionDB.list_message_ids`
            returns).
        covered_ranges: Every ``(from_id, to_id)`` pair any capture job
            (of any state) has ever been enqueued for, in this session
            (as :meth:`~minion_assist.session.db.SessionDB.list_capture_job_ranges`
            returns).

    Returns:
        list[tuple[int, int]]: Windows to enqueue, each at most
            :data:`_BACKFILL_WINDOW_MESSAGES` messages wide, covering
            exactly the gaps — messages already covered by an existing
            job are never included in any window.
    """
    covered: set[int] = set()
    for from_id, to_id in covered_ranges:
        covered.update(range(from_id, to_id + 1))

    windows: list[tuple[int, int]] = []
    current_run: list[int] = []
    for mid in message_ids:
        if mid in covered:
            if current_run:
                windows.extend(_chunk_run(current_run))
                current_run = []
        else:
            current_run.append(mid)
    if current_run:
        windows.extend(_chunk_run(current_run))
    return windows


def backfill_agent(db: SessionDB, agent_id: str, model_id: str) -> int:
    """Enqueue capture jobs for every historical message range never captured (Task 10).

    Does not run extraction itself — enqueues into the same
    ``memory_capture_jobs`` queue a live turn uses, so the already-running
    ``CaptureWorker`` processes backfilled ranges exactly like a live
    turn's job (same retry/backoff, same proposal-indexing wiring). Safe
    to re-run: a gap already backfilled produces the same idempotency key
    the second time, so ``enqueue_capture_job`` no-ops it.

    Args:
        db: The ``SessionDB`` sessions/messages/capture jobs live in.
        agent_id: Which agent's history to backfill.
        model_id: The model id to embed in each enqueued job's
            idempotency key — should match whatever model this agent is
            actually configured to use, the same key shape a live turn's
            enqueue already uses (``agents/session.py``'s ``send()``).

    Returns:
        int: How many *new* capture jobs were actually enqueued (0 means
            every session was already fully covered).
    """
    from .extractor import _EXTRACTION_PROMPT_VERSION  # noqa: PLC0415

    enqueued = 0
    for session_id in db.list_session_ids_for_agent(agent_id):
        message_ids = db.list_message_ids(session_id)
        if not message_ids:
            continue
        covered = db.list_capture_job_ranges(session_id)
        for from_id, to_id in _compute_backfill_windows(message_ids, covered):
            idempotency_key = (
                f"{agent_id}:{session_id}:{from_id}-{to_id}:"
                f"{_EXTRACTION_PROMPT_VERSION}:{model_id}"
            )
            job_id = db.enqueue_capture_job(agent_id, session_id, from_id, to_id, idempotency_key)
            if job_id is not None:
                enqueued += 1
    return enqueued


class StaleProposalError(Exception):
    """Raised by :meth:`MemoryConsolidator.approve` when the merge target changed since preview.

    The concrete mechanism behind the plan's "user edits win over stale
    consolidator output" acceptance criterion: ``approve()`` never
    overwrites a topic note whose content no longer matches what the
    preview was drafted against.
    """


class MemoryConsolidator:
    """Drafts, applies, rejects, and rolls back topic-note updates for one agent's proposals.

    Scoped to a single agent (unlike ``rank_proposals()``, a free
    function) because ``files``/``index`` writes are inherently
    agent-rooted — see the module docstring.

    Args:
        db: The ``SessionDB`` proposals live in.
        index: The lexical index used to find a merge target, store
            previews/revisions, and reindex after a write.
        files: The agent's ``MemoryFileRepository`` — ``load()`` only in
            ``preview()``; ``approve()``/``rollback()`` are the only
            methods here that call ``remember()``/``delete()``.
        provider: The LLM provider used to draft the revised note text.
            Only ``preview()`` calls it — ``None`` is fine for a caller
            that only ever needs ``approve()``/``reject()``/``rollback()``
            (e.g. the CLI's ``reject``/``rollback`` commands, which have
            no reason to require a live model/API key).
        agent_id: Which agent this instance drafts/applies for. Must match
            the agent ``files`` is rooted at and ``index``'s rows are
            partitioned under.
    """

    def __init__(
        self,
        db: SessionDB,
        index: PostgresMemoryIndex,
        files: MemoryFileRepository,
        provider: LLMProvider | None,
        agent_id: str,
    ) -> None:
        self._db = db
        self._index = index
        self._files = files
        self._provider = provider
        self._agent_id = agent_id

    def preview(self, proposal_id: int) -> dict:
        """Draft a preview for one pending proposal — never writes to disk.

        Args:
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
                :func:`parse_draft_response`).
        """
        proposal = self._db.get_proposal(proposal_id)
        if proposal is None:
            raise ValueError(f"No proposal with id {proposal_id!r}")
        claim_text = proposal["claim_text"]

        target_key = self._find_merge_target(claim_text)
        if target_key is not None:
            target_kind = "revise_topic"
            existing_content = self._files.load(target_key) or ""
            target_rel_path = (
                self._files.topic_path(target_key).relative_to(self._files.root).as_posix()
            )
            existing_claims = self._index.list_claims(self._agent_id, rel_path=target_rel_path)
        else:
            target_kind = "new_topic"
            existing_content = ""
            existing_claims = []

        based_on_content_hash = _hash_text(existing_content)
        # Code-generated, never left to the model to invent -- the same
        # reasoning PostgresMemoryIndex.get_or_create_entity already
        # applies to entity ids (Stage One Phase 7, slice B).
        new_claim_id = f"c-{uuid.uuid4().hex[:8]}"
        drafted_key, rationale, drafted_content = self._draft(
            claim_text, existing_content, existing_claims, new_claim_id, proposal_id
        )
        final_key = target_key if target_kind == "revise_topic" else drafted_key

        preview_id = self._index.record_consolidation_preview(
            self._agent_id, proposal_id, target_kind, final_key,
            based_on_content_hash, drafted_content, rationale,
        )
        return {
            "id": preview_id,
            "agent_id": self._agent_id,
            "proposal_id": proposal_id,
            "target_kind": target_kind,
            "target_key": final_key,
            "based_on_content_hash": based_on_content_hash,
            "drafted_content": drafted_content,
            "rationale": rationale,
        }

    def approve(self, preview_id: int) -> dict:
        """Apply a preview: write its drafted content to disk and reindex it.

        Refuses to apply (see :class:`StaleProposalError`) if the target's
        current content no longer matches ``based_on_content_hash`` —
        someone edited it since this preview was drafted, and that edit
        wins.

        Args:
            preview_id: Which ``memory_consolidation_previews`` row to
                apply.

        Returns:
            dict: ``{"proposal_id", "target_key", "rel_path"}``.

        Raises:
            ValueError: If ``preview_id`` doesn't exist.
            StaleProposalError: If the target changed since this preview
                was drafted.
        """
        preview = self._index.get_consolidation_preview(preview_id)
        if preview is None:
            raise ValueError(f"No consolidation preview with id {preview_id!r}")

        current_content = self._files.load(preview["target_key"]) or ""
        current_hash = _hash_text(current_content)
        if current_hash != preview["based_on_content_hash"]:
            raise StaleProposalError(
                f"Topic {preview['target_key']!r} changed since this preview was drafted "
                f"(based on hash {preview['based_on_content_hash'][:8]}, "
                f"now {current_hash[:8]}) — re-run preview before approving."
            )

        # Snapshot BEFORE writing, so rollback() can restore it.
        self._index.record_topic_revision(
            self._agent_id, preview["target_key"], preview["proposal_id"], current_content
        )
        path = self._files.remember(preview["target_key"], preview["drafted_content"])
        rel_path = path.relative_to(self._files.root).as_posix()
        self._index.reindex_file(self._agent_id, rel_path, "durable", preview["drafted_content"])
        # The raw proposal claim is now redundant — its content lives in the
        # real note. Reject keeps the chunk (still useful to audit what was
        # rejected); promote does not.
        self._index.remove_proposal(self._agent_id, preview["proposal_id"])
        self._db.set_proposal_status(preview["proposal_id"], "promoted")

        return {
            "proposal_id": preview["proposal_id"],
            "target_key": preview["target_key"],
            "rel_path": rel_path,
        }

    def reject(self, proposal_id: int, reason: str = "") -> None:
        """Mark a proposal reviewed and rejected — never promoted, no file touched.

        Args:
            proposal_id: Which proposal to reject.
            reason: Optional human-readable reason, stored on the
                proposal row (``memory_proposals.rejected_reason``) for
                later audit/explain purposes.
        """
        self._db.set_proposal_status(proposal_id, "rejected", reason=reason)

    def rollback(self, target_key: str) -> dict:
        """Undo the most recent ``approve()`` for one topic note.

        Restores the note's exact pre-apply content (or deletes the file
        entirely if it didn't exist before that apply — i.e. the apply
        created a brand new topic), reindexes accordingly, restores the
        associated proposal to ``"pending"`` and re-indexes its raw claim
        chunk (removed by ``approve()``), and consumes the revision row —
        a second rollback steps back one apply further, not the same one
        again.

        Args:
            target_key: Which topic note to roll back.

        Returns:
            dict: ``{"target_key", "proposal_id", "restored_content"}``.

        Raises:
            ValueError: If this topic has no revision history to roll
                back (never applied, or already fully rolled back).
        """
        revision = self._index.latest_topic_revision(self._agent_id, target_key)
        if revision is None:
            raise ValueError(f"No revision history for topic {target_key!r} to roll back")

        rel_path = self._files.topic_path(target_key).relative_to(self._files.root).as_posix()
        if revision["prior_content"] == "":
            # "" means the apply this undoes created the note from
            # scratch — there is nothing to restore it to, so the file
            # goes away entirely rather than becoming an empty note.
            self._files.delete(target_key)
            self._index.remove_file(self._agent_id, rel_path)
        else:
            self._files.remember(target_key, revision["prior_content"])
            self._index.reindex_file(self._agent_id, rel_path, "durable", revision["prior_content"])

        self._db.set_proposal_status(revision["proposal_id"], "pending")
        proposal = self._db.get_proposal(revision["proposal_id"])
        if proposal is not None:
            self._index.reindex_proposal(
                self._agent_id, revision["proposal_id"], proposal["claim_text"]
            )
        self._index.delete_topic_revision(revision["id"])

        return {
            "target_key": target_key,
            "proposal_id": revision["proposal_id"],
            "restored_content": revision["prior_content"],
        }

    def _find_merge_target(self, claim_text: str) -> str | None:
        """Find an existing topic note this claim likely belongs in, or ``None``.

        Thin wrapper around the free function :func:`find_merge_target`
        (Stage One Phase 7, slice E) — kept as a method so existing
        callers/tests are unaffected; see that function's docstring for
        the actual logic and reasoning.
        """
        return find_merge_target(self._index, self._agent_id, claim_text)

    def _draft(
        self,
        claim_text: str,
        existing_content: str,
        existing_claims: list[dict],
        new_claim_id: str,
        proposal_id: int,
    ) -> tuple[str, str, str]:
        """Call the provider to draft the revised/new note text.

        Only ``claim_text`` (traceable to real captured messages) and the
        merge target's own existing content — ``existing_claims`` is that
        same content, restructured — are ever included in the prompt; see
        the module docstring's "Evidence provenance" section.
        ``new_claim_id``/``proposal_id`` are code-generated/already-known
        values, not new evidence.
        """
        existing_block = existing_content or "(no existing note — propose a new one)"
        claims_block = "(none)"
        if existing_claims:
            claims_block = "\n".join(
                f"- {c['id']} ({c['status']}): {c['text']}" for c in existing_claims
            )
        user_message = (
            f"New claim:\n{claim_text}\n\n"
            f"Existing note content:\n{existing_block}\n\n"
            f"Existing claims in this note:\n{claims_block}\n\n"
            f"New claim id to use: {new_claim_id}\n"
            f"Proposal id (for evidence=proposal:...): {proposal_id}\n"
            f"Today's date: {date.today().isoformat()}"
        )
        response = self._provider.chat(
            system=_DRAFT_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
            tools=[],
            max_tokens=1000,
        )
        return parse_draft_response((response.text or "").strip())

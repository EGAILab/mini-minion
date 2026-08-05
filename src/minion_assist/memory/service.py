"""``MemoryService`` — the one object agents/tools depend on for memory.

Stage One Phase 1, slice 2.

Why a facade over ``MemoryFileRepository``?
--------------------------------------------
Before this existed, ``AgentSession`` and every memory tool
(``SaveMemoryTool``, ``NoteTool``, ``SearchMemoryTool``,
``WriteDailyMemoryTool``) each held their own reference to a raw
:class:`~minion_assist.memory.long_term.LongTermMemory` instance and called
its methods directly. That meant every consumer had to know the storage
layer's internals (how a key becomes a filename, where daily notes live)
and there was no single place to add cross-cutting behavior (status
reporting, future scope checks) without touching every call site.
``NoteTool`` was later retired (Phase 1, slice 4) once
``WriteDailyMemoryTool`` absorbed its responsibility.

``MemoryService`` is that single place. It wraps one
:class:`~minion_assist.memory.files.MemoryFileRepository` per agent and
exposes the operations tools/session code actually need, translating
``files.py``'s lower-level primitives (path resolution, locators) into
simpler calls a tool can use directly — e.g. :meth:`get` takes a plain path
string rather than requiring the caller to build a
:class:`~minion_assist.memory.models.MemoryLocator` itself.

Scope, deliberately not enforced here
--------------------------------------
Stage One's target design names several memory scopes (agent-private,
user-shared, workspace, session-lineage, channel, import-quarantine — see
``docs/adr/0003-per-agent-memory-scope.md``). This service only enforces
``agent-private``, and it does so *structurally*: one ``MemoryService`` per
agent, each backed by its own repository rooted at that agent's own
workspace directory. There is no runtime scope-check method here, because
nothing in the codebase yet produces a request tagged with any other scope
— adding that plumbing now would be exactly the kind of speculative,
no-current-caller abstraction the project's simplicity rule warns against.
Add real scope enforcement when a real cross-agent/channel use case exists.

Talks to
--------
- ``memory/files.py`` — :class:`MemoryFileRepository`, the actual storage
  backend this service wraps.
- ``memory/models.py`` — the typed request/result objects passed through.
- ``context.py`` — :meth:`Compactor.peek_compaction_head` supplies the
  messages :meth:`flush_head` writes out before they're summarized away
  (Stage One Phase 2, slice B).
- ``memory/postgres_index.py`` — :class:`PostgresMemoryIndex`, the optional
  lexical index this service keeps in sync on every write (Stage One
  Phase 3, slice B). Only imported under ``TYPE_CHECKING``: constructing
  one requires ``psycopg``, which this module must not require just to be
  imported (matching ``agents/session.py``'s and ``minion.py``'s existing
  "optional database" pattern).
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from ..messages import format_message_excerpt
from .boundaries import format_boundary_prefix, is_boundary_active
from .files import MemoryFileRepository
from .models import (
    FLUSH_STATUS_EMPTY,
    FLUSH_STATUS_FAILED,
    FLUSH_STATUS_FLUSHED,
    FlushOutcome,
    MemoryExcerpt,
    MemoryHit,
    MemoryLocator,
    MemoryStatus,
)

if TYPE_CHECKING:
    from .postgres_index import PostgresMemoryIndex

_log = logging.getLogger("minion_assist.memory_service")

# Mirrors MemoryFileRepository's / LongTermMemory's default search cap.
_SEARCH_MAX_RESULTS = 20

# Maps a lexical-index corpus name to the linear scan's equivalent `source`
# tag, so search()'s corpus filter still narrows results in degraded mode
# (no index configured, or the index search just failed). "durable" is the
# only one that actually differs — the linear scan calls topic notes
# "topic", never "durable" (and never returns MEMORY.md at all).
_CORPUS_TO_LEGACY_SOURCE = {"durable": "topic", "daily": "daily", "import": "import"}


class MemoryService:
    """Thin orchestration facade over :class:`MemoryFileRepository`.

    Args:
        files: The repository backing this service. Construct one
            ``MemoryService`` per agent, each wrapping a repository rooted
            at that agent's own workspace directory.
        index: The optional lexical index to keep in sync on every write
            (Stage One Phase 3, slice B). ``None`` (the default, and every
            existing call site before this slice) means no database is
            configured — writes behave exactly as before, index sync is
            simply skipped.
        agent_id: Required whenever ``index`` is given — the partition key
            :class:`PostgresMemoryIndex` uses to keep agents' chunks apart.
            Ignored if ``index`` is ``None``.
    """

    def __init__(
        self,
        files: MemoryFileRepository,
        *,
        index: PostgresMemoryIndex | None = None,
        agent_id: str | None = None,
    ) -> None:
        self._files = files
        self._index = index
        self._agent_id = agent_id

    def _sync_index(self, source_kind: str, path: Path, content: str | None) -> None:
        """Update the lexical index for one file, if one is configured.

        ``content=None`` means the file was deleted (remove it from the
        index); otherwise ``content`` is the file's new full text.

        Never raises — an index-update failure must not block the memory
        write that already succeeded on disk. This is a best-effort,
        immediate sync; the periodic/startup reconciliation pass (also
        Phase 3 slice B) heals anything missed here, the same self-healing
        split responsibility as message mirroring (Phase 2 slice A).
        """
        if self._index is None or self._agent_id is None:
            return
        rel_path = path.relative_to(self._files.root).as_posix()
        try:
            if content is None:
                self._index.remove_file(self._agent_id, rel_path)
            else:
                self._index.reindex_file(self._agent_id, rel_path, source_kind, content)
        except Exception as exc:
            _log.debug(
                "Memory index sync failed for %s (%s): %s: %s",
                rel_path, self._agent_id, type(exc).__name__, exc,
            )

    # -----------------------------------------------------------------
    # Explicit notes
    # -----------------------------------------------------------------

    def remember(self, key: str, content: str) -> None:
        """Save an explicit note under ``key``, overwriting any existing note.

        Args:
            key: Note identifier, e.g. ``"project-goals"``.
            content: Markdown text to store.
        """
        path = self._files.remember(key, content)
        self._sync_index("durable", path, content)

    def load(self, key: str) -> str | None:
        """Load an explicit note's content by key, or ``None`` if it doesn't exist."""
        return self._files.load(key)

    def delete(self, key: str) -> bool:
        """Delete an explicit note. Returns ``True`` if a file was removed."""
        removed = self._files.delete(key)
        if removed:
            self._sync_index("durable", self._files.topic_path(key), None)
        return removed

    def list_keys(self) -> list[str]:
        """Return every explicit note's key, sorted alphabetically."""
        return self._files.list_keys()

    # -----------------------------------------------------------------
    # Quarantined notes — unreviewed, never auto-promoted (see files.py)
    # -----------------------------------------------------------------

    def remember_import(self, key: str, content: str) -> None:
        """Save quarantined, unreviewed content under ``key``.

        Used by the background extractor and (until retired) the ``note``
        tool — content nobody has reviewed yet, searchable but never
        auto-promoted. See ``docs/adr/0003-per-agent-memory-scope.md``.
        """
        path = self._files.remember_import(key, content)
        self._sync_index("import", path, content)

    def load_import(self, key: str) -> str | None:
        """Load quarantined content by key, or ``None`` if it doesn't exist."""
        return self._files.load_import(key)

    def list_import_keys(self) -> list[str]:
        """Return every quarantined note's key, sorted alphabetically."""
        return self._files.list_import_keys()

    # -----------------------------------------------------------------
    # Session-derived cleanup (MEM-GAP-003)
    # -----------------------------------------------------------------

    def forget_proposals(self, proposal_ids: list[int]) -> dict:
        """Remove every indexed trace of deleted ``memory_proposals`` rows.

        Called by ``commands.py``'s ``/delete-session`` after
        :meth:`~minion_assist.session.db.SessionDB.delete_session` returns
        the ids of the proposal rows it just deleted — ``SessionDB`` owns
        those rows but has no access to :class:`PostgresMemoryIndex`
        (separate class/connection by design), so this is where the two
        stores' cleanup meets.

        For each proposal id:

        1. **Forgets it as a knowledge-graph evidence source**
           (:func:`~minion_assist.memory.forgetting.forget_source`) — edits
           the actual claim markers in any topic note citing
           ``("proposal", str(proposal_id))``, not just the derived
           ``kb_evidence`` cache (which a later reindex would otherwise
           silently restore). A claim left with no other evidence is
           re-flagged ``status=unknown``, never silently deleted.
        2. Removes its indexed chunk (:meth:`PostgresMemoryIndex.remove_proposal`).
        3. Removes any draft preview referencing it
           (:meth:`PostgresMemoryIndex.remove_consolidation_previews_for_proposal`).

        Deliberately does **not** touch ``memory_topic_revisions`` or any
        durable note a *promoted* proposal's content was merged into — that
        note is independent, reviewed memory now, and survives deletion of
        the session it originally came from.

        A no-op (returns empty results) if no index is configured, since
        none of the rows this cleans up can exist without one.

        Args:
            proposal_ids: Ids of ``memory_proposals`` rows that were just
                deleted from ``SessionDB``.

        Returns:
            dict: ``{"proposal_ids": [...], "forget_results": [...]}`` —
                ``forget_results`` is one
                :func:`~minion_assist.memory.forgetting.forget_source`
                result dict per proposal id, in order, for callers that want
                to report exactly what was re-evaluated.
        """
        if self._index is None or self._agent_id is None or not proposal_ids:
            return {"proposal_ids": list(proposal_ids), "forget_results": []}

        from .forgetting import forget_source  # noqa: PLC0415

        forget_results = []
        for proposal_id in proposal_ids:
            forget_results.append(
                forget_source(
                    self._index, self._files, self._agent_id, "proposal", str(proposal_id)
                )
            )
            self._index.remove_proposal(self._agent_id, proposal_id)
            self._index.remove_consolidation_previews_for_proposal(self._agent_id, proposal_id)

        return {"proposal_ids": list(proposal_ids), "forget_results": forget_results}

    # -----------------------------------------------------------------
    # Search and recall
    # -----------------------------------------------------------------

    def search(
        self,
        query: str,
        max_results: int = _SEARCH_MAX_RESULTS,
        *,
        corpus: str | None = None,
    ) -> list[MemoryHit]:
        """Search memory for ``query``, using hybrid retrieval when an index is configured.

        With a configured index (Stage One Phase 3/4), this fuses path,
        lexical, vector, pinned, and recent lanes
        (``PostgresMemoryIndex.hybrid_search`` — see its docstring for how)
        across ``MEMORY.md``, topic notes, daily notes, and imports — a
        strictly larger corpus than the Phase 1 linear scan, which never
        covers root ``MEMORY.md`` at all (see ``memory/files.py``'s
        ``list_indexable_files``). Without a configured index, or if an
        index search fails (e.g. a transient connection drop), this falls
        back to the Phase 1 linear scan — a turn's memory injection must
        never break over a database hiccup. The fallback is not silent: a
        failed index search is logged at WARNING, and ``deep_status()``
        surfaces ongoing index health so a persistently broken index isn't
        invisible.

        Returns raw, structured hits — formatting them into a prompt block
        or tool-result string is the caller's job (``session.py`` for
        per-turn injection, ``SearchMemoryTool`` for the explicit
        ``search_memory`` tool call). See the module docstring's "Scope,
        deliberately not enforced here" note — this is also where a future
        recall-telemetry hook (Stage One Phase 5) would attach, once one
        exists.

        Args:
            query: One or more keywords, space-separated.
            max_results: Maximum notes/chunks to return.
            corpus: Restrict to one corpus (``"durable"``, ``"daily"``,
                ``"import"``, or ``"proposal"`` — Stage One Phase 5, slice
                B; unreviewed capture-job proposals, only ever present
                with a configured index), or ``None`` to search
                everything. Only affects the indexed path in practice —
                the deliberate degraded-mode scope reduction. See
                :data:`_CORPUS_TO_LEGACY_SOURCE`.

        Returns:
            list[MemoryHit]: Best matches first, tagged by source. Hits from
                the lexical index also carry ``rel_path``/``start_line``/
                ``end_line``/``score``; linear-scan hits leave those unset.
        """
        if self._index is not None and self._agent_id is not None:
            try:
                rows = self._index.hybrid_search(
                    self._agent_id, query, corpus=corpus, max_results=max_results
                )
                hits = [
                    MemoryHit(
                        key=Path(r["rel_path"]).stem,
                        content=r["content"],
                        source=r["source_kind"],
                        rel_path=r["rel_path"],
                        start_line=r["start_line"],
                        end_line=r["end_line"],
                        score=r["score"],
                    )
                    for r in rows
                ]
                return self._apply_boundaries(hits)
            except Exception as exc:
                _log.warning(
                    "Hybrid index search failed for agent %s, falling back to linear scan: "
                    "%s: %s",
                    self._agent_id, type(exc).__name__, exc,
                )

        # MEM-GAP-004: quarantined imports must not enter a corpus-agnostic
        # search's results (the automatic per-turn injection path, and a
        # plain search_memory call) — only an explicit corpus="import"
        # request may see them, matching the indexed path's policy above.
        # Excluding them from the candidate pool (rather than filtering
        # after the fact) also means an excluded import can never crowd out
        # an eligible note within max_results — see files.py's docstring.
        exclude = frozenset() if corpus == "import" else frozenset({"import"})
        hits = self._files.search(query, max_results=max_results, exclude_sources=exclude)
        if corpus is not None:
            legacy_source = _CORPUS_TO_LEGACY_SOURCE.get(corpus, corpus)
            hits = [h for h in hits if h.source == legacy_source]
        return hits

    def _apply_boundaries(self, hits: list[MemoryHit]) -> list[MemoryHit]:
        """Attach an advisory boundary annotation, or drop an inactive one (Stage One Phase 6, slice A).

        Only applies to indexed-path hits (every hit here has ``rel_path``
        set, since this is only ever called from the branch of
        :meth:`search` that used the lexical index) — degraded-mode linear
        scan hits never carry boundary metadata, since it lives in Postgres
        (see ``memory/boundaries.py``'s module docstring for why the file
        itself is still the source of truth; this is a cached derivative).

        A hit whose note has boundary metadata but is currently outside its
        ``[safe_after, expires_at]`` window is *excluded* from the returned
        list entirely, not merely labeled — see
        ``memory/boundaries.py``'s ``is_boundary_active`` docstring. This
        can occasionally return fewer than ``max_results`` hits on a turn
        where an inactive boundary-bearing note would otherwise have
        ranked in the top results; a documented, deliberate trade-off in
        favor of correctness over exact recall-count preservation.

        Never raises: a lookup failure for one file is logged at DEBUG and
        treated as "no boundary," the same best-effort posture every other
        index-backed enrichment in this class already has.
        """
        if self._index is None or self._agent_id is None:
            return hits
        now = time.time()
        cache: dict[str, dict[str, str] | None] = {}
        kept: list[MemoryHit] = []
        for hit in hits:
            if hit.rel_path is None:
                kept.append(hit)
                continue
            if hit.rel_path not in cache:
                try:
                    cache[hit.rel_path] = self._index.get_boundary(self._agent_id, hit.rel_path)
                except Exception as exc:
                    _log.debug(
                        "Boundary lookup failed for %s (%s): %s: %s",
                        hit.rel_path, self._agent_id, type(exc).__name__, exc,
                    )
                    cache[hit.rel_path] = None
            metadata = cache[hit.rel_path]
            if not metadata:
                kept.append(hit)
                continue
            if not is_boundary_active(metadata, now):
                continue
            kept.append(replace(hit, boundary=format_boundary_prefix(metadata)))
        return kept

    def mark_injected(self, rel_paths: list[str], query: str) -> None:
        """Record that these files (from a prior :meth:`search` call) were actually injected.

        Stage One Phase 5, slice A. Called by ``agents/session.py``'s
        ``send()`` right after ``build_prompt_section()`` decides which of
        this turn's search results fit its token budget — the "was
        actually surfaced" vs. "was actually injected" distinction Task 1
        asks for. Correlates with the index's own recall recording (done
        inside ``hybrid_search()`` for every result it returns) via
        ``hash_query(query)``, so both calls must be given the exact same
        query text.

        A no-op with no index configured, an empty ``rel_paths`` list, or
        (like every telemetry write in this module) if the update itself
        fails — a turn must never break over a telemetry hiccup.
        """
        if self._index is None or self._agent_id is None or not rel_paths:
            return
        try:
            from .postgres_index import hash_query  # noqa: PLC0415

            self._index.mark_injected(self._agent_id, rel_paths, hash_query(query))
        except Exception as exc:
            _log.debug(
                "Recording injection telemetry failed for agent %s: %s: %s",
                self._agent_id, type(exc).__name__, exc,
            )

    # -----------------------------------------------------------------
    # Daily notes
    # -----------------------------------------------------------------

    def append_daily(self, text: str, *, when: date | None = None) -> Path:
        """Append a timestamped entry to today's (or ``when``'s) daily note.

        Args:
            text: The note content to append.
            when: Date to file this entry under. Defaults to today.

        Returns:
            Path: The daily note file that was written to.
        """
        path = self._files.append_daily(text, when=when)
        if self._index is not None and self._agent_id is not None:
            # append_daily() only returns the file it touched, not its full
            # new content (it appends rather than replacing) — the index
            # needs the whole file re-chunked, so read it back once here.
            # A daily note append isn't a hot path, so the extra read is a
            # non-issue.
            self._sync_index("daily", path, path.read_text(encoding="utf-8"))
        return path

    # -----------------------------------------------------------------
    # Exact bounded reads
    # -----------------------------------------------------------------

    def get(
        self,
        path: str,
        *,
        from_line: int | None = None,
        lines: int | None = None,
    ) -> MemoryExcerpt:
        """Read an exact, bounded slice of a memory file by path string.

        Wraps :meth:`MemoryFileRepository.resolve_path` and
        :class:`MemoryLocator` construction so callers never need to know
        about the locator type or perform their own containment check —
        this is what a future ``memory_get`` tool (Phase 1, slice 5) will
        call directly with the path string an LLM provides.

        Args:
            path: A relative (to the agent's workspace root) or absolute
                path. Resolved and validated as inside the memory root.
            from_line: 1-indexed starting line. ``None`` means from the
                start of the file.
            lines: Maximum number of lines to return. ``None`` means to the
                end of the file.

        Returns:
            MemoryExcerpt: The requested slice, plus the file's total line
                count.

        Raises:
            ValueError: ``path`` resolves outside the memory root.
            FileNotFoundError: ``path`` does not exist.
        """
        resolved = self._files.resolve_path(path)
        locator = MemoryLocator(path=resolved, from_line=from_line, lines=lines)
        return self._files.get(locator)

    # -----------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------

    def status(self) -> MemoryStatus:
        """Return a snapshot of this agent's memory store — note counts by source.

        Returns:
            MemoryStatus: Topic/import/daily note counts and the store's root
                directory. See :class:`MemoryStatus` for why this is
                intentionally minimal in Phase 1.
        """
        counts = self._files.count_notes()
        return MemoryStatus(
            root=self._files.root,
            topic_count=counts["topic"],
            import_count=counts["import"],
            daily_count=counts["daily"],
        )

    def deep_status(self) -> dict | None:
        """Lexical-index health for this agent — the primitive behind ``memory status --deep``.

        Deliberately returns ``None`` (not a zeroed-out dict) when no index
        is configured, so a caller can tell "index configured but empty"
        apart from "no index at all" — see
        ``docs/adr/0004-degraded-operation.md``. Everything reported comes
        from tables a one-shot CLI process can actually read; it does not
        (and cannot) claim to know whether a live ``CaptureWorker``/
        ``MemoryIndexWatcher`` thread is currently running in some other
        process.

        Returns:
            dict | None: ``PostgresMemoryIndex.index_summary()``'s result
                (``total_chunks``, ``file_count``, ``by_corpus``,
                ``last_indexed_at``), or ``None`` if no index is configured.
        """
        if self._index is None or self._agent_id is None:
            return None
        return self._index.index_summary(self._agent_id)

    def force_reindex(self) -> int:
        """Crash-safely rebuild this agent's entire lexical index from scratch.

        Delegates to :meth:`PostgresMemoryIndex.force_rebuild_agent` — see
        its docstring for the shadow-table swap that makes this safe to
        interrupt. This is a deliberate, manual maintenance operation (the
        ``memory reindex --force`` CLI command); nothing in the running app
        calls this automatically.

        Returns:
            int: Total chunk count in the rebuilt index.

        Raises:
            RuntimeError: No index is configured for this agent. A caller
                asking to force-reindex with no database configured is
                almost certainly a mistake worth surfacing, not a silent
                no-op.
        """
        if self._index is None or self._agent_id is None:
            raise RuntimeError("No lexical index configured for this agent — nothing to reindex.")
        return self._index.force_rebuild_agent(self._agent_id, self._files.list_indexable_files())

    def reconcile_index(self) -> int:
        """Hash-diff reconcile this agent's lexical index against its current on-disk files.

        Delegates to :meth:`PostgresMemoryIndex.reconcile_agent` — the
        cheaper counterpart to :meth:`force_reindex`: only touches files
        whose content actually changed since the last reconciliation,
        rather than rebuilding everything. This is what ``minion.py`` calls
        at startup and what ``minion-assist memory reindex`` (without
        ``--force``) calls on demand.

        Returns:
            int: How many files were reindexed or removed (0 means the
                index was already fully up to date).

        Raises:
            RuntimeError: No index is configured for this agent.
        """
        if self._index is None or self._agent_id is None:
            raise RuntimeError("No lexical index configured for this agent — nothing to reconcile.")
        return self._index.reconcile_agent(self._agent_id, self._files.list_indexable_files())

    # -----------------------------------------------------------------
    # Pinning (Stage One Phase 4, slice B)
    # -----------------------------------------------------------------
    #
    # Scoped to explicit topic notes only (memory/topics/{key}.md) — not
    # MEMORY.md (already unconditionally injected into every turn via
    # bootstrap.py, a separate mechanism entirely), not daily notes
    # (ephemeral by nature), and not imports (unreviewed/quarantined —
    # pinning one would contradict that status). "Pin whatever you
    # explicitly saved" mirrors how remember()/delete() already address
    # notes by key, not by raw path.

    def _pin_rel_path(self, key: str) -> str:
        """The lexical index's rel_path for a topic note key."""
        return self._files.topic_path(key).relative_to(self._files.root).as_posix()

    def pin(self, key: str) -> None:
        """Pin a topic note so the pinned fusion lane always surfaces it, regardless of query match.

        Args:
            key: The note identifier, same as :meth:`remember`/:meth:`delete`.

        Raises:
            RuntimeError: No lexical index is configured for this agent.
            FileNotFoundError: No note exists under this key — pinning
                something that doesn't exist would just create an orphaned
                pin, so this is rejected rather than silently accepted.
        """
        if self._index is None or self._agent_id is None:
            raise RuntimeError("No lexical index configured for this agent — nothing to pin.")
        if self._files.load(key) is None:
            raise FileNotFoundError(f"No note found for key {key!r} — nothing to pin.")
        self._index.pin_file(self._agent_id, self._pin_rel_path(key))

    def unpin(self, key: str) -> None:
        """Unpin a topic note. A no-op if it wasn't pinned (or doesn't exist).

        Deliberately does not require the note to still exist, unlike
        :meth:`pin` — this must always be able to clear a pin, including a
        stale one left behind by a note deleted outside :meth:`delete`.

        Raises:
            RuntimeError: No lexical index is configured for this agent.
        """
        if self._index is None or self._agent_id is None:
            raise RuntimeError("No lexical index configured for this agent — nothing to unpin.")
        self._index.unpin_file(self._agent_id, self._pin_rel_path(key))

    def is_pinned(self, key: str) -> bool:
        """Whether a topic note is pinned. ``False`` (not an error) with no index configured."""
        if self._index is None or self._agent_id is None:
            return False
        return self._index.is_pinned(self._agent_id, self._pin_rel_path(key))

    def list_pinned(self) -> list[str]:
        """Every pinned note's key, most recently pinned first. Empty with no index configured."""
        if self._index is None or self._agent_id is None:
            return []
        return [Path(p).stem for p in self._index.pinned_files(self._agent_id)]

    # -----------------------------------------------------------------
    # Pre-compaction flush (Stage One Phase 2, slice B)
    # -----------------------------------------------------------------

    def flush_head(self, head: list[dict]) -> FlushOutcome:
        """Append a deterministic transcript excerpt of ``head`` to today's daily note.

        Called by ``agents/session.py`` right before
        :meth:`~minion_assist.context.Compactor.compact` summarizes and
        discards these same messages — so their raw content survives even if
        summarization itself fails immediately afterward. Deliberately makes
        no LLM call: :func:`~minion_assist.messages.format_message_excerpt`
        is pure text rendering, so this can't fail the way an LLM call can,
        and adds no latency to the turn.

        Never raises — a failure to write is reported via the returned
        :class:`FlushOutcome`, not an exception, so it can never block the
        turn that triggered it.

        Args:
            head: The messages about to be summarized away (typically
                ``Compactor.peek_compaction_head()``'s result).

        Returns:
            FlushOutcome: ``FLUSH_STATUS_EMPTY`` if ``head`` is empty or
                renders to blank text, ``FLUSH_STATUS_FLUSHED`` on a
                successful write, or ``FLUSH_STATUS_FAILED`` with the
                exception description if the write itself raised.
        """
        if not head:
            return FlushOutcome(status=FLUSH_STATUS_EMPTY)

        text = format_message_excerpt(head)
        if not text.strip():
            return FlushOutcome(status=FLUSH_STATUS_EMPTY)

        try:
            self._files.append_daily(f"[Pre-compaction checkpoint]\n{text}")
        except Exception as exc:
            return FlushOutcome(status=FLUSH_STATUS_FAILED, detail=f"{type(exc).__name__}: {exc}")

        return FlushOutcome(status=FLUSH_STATUS_FLUSHED)

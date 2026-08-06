"""``MemoryFileRepository`` — the canonical on-disk memory layout for one agent (Stage One Phase 1).

This replaces :class:`~minion_assist.memory.long_term.LongTermMemory` as the
thing that actually touches disk. Where ``LongTermMemory`` stored every note
as a flat ``{base_dir}/{key}.md`` file (mixing explicit notes, daily logs,
and the extractor's rolling note in one directory), this repository targets
the merged per-agent layout Stage One Phase 0 migrates existing data into
(see ``docs/adr/0003-per-agent-memory-scope.md``)::

    workspaces/{agent_id}/
      USER.md, MEMORY.md, DREAMS.md   # untouched here — bootstrap.py/dreaming.py own these
      KNOWLEDGE_DIGEST.md             # write_digest() — fully machine-owned, no human-edit concern
      memory/
        YYYY-MM-DD.md                 # daily notes — append_daily()
        topics/{key}.md               # explicit notes — remember()/load()/delete()
        imports/{key}.md              # quarantined, unreviewed — search()/get() only, no write API

Why a separate repository from ``MemoryService``?
--------------------------------------------------
This class owns *only* disk I/O and path safety — atomic writes, key
sanitization, containment checks, line-bounded reads. ``memory/service.py``
(added in the next Phase 1 slice) will own orchestration (scope enforcement,
formatting for tools). Keeping them separate means the path-safety logic can
be tested in isolation, the same way ``PermissionPolicy`` is tested
independently of the tools that use it.

Search behavior is intentionally unchanged from ``LongTermMemory.search()``
in this slice — same term-frequency-with-recency-tiebreak scoring, same
``_SEARCH_MAX_RESULTS`` cap. Phase 1's goal is one canonical service with
existing behavior, not better retrieval (that's Phase 3+).

Talks to
--------
- ``memory/models.py`` — :class:`MemoryHit`, :class:`MemoryLocator`,
  :class:`MemoryExcerpt` are this module's return types.
- ``memory/migration.py`` — Phase 0's migration populates the directories
  this class reads from.

Concurrency (MEM-GAP-007/009)
-------------------------------
Every writer that can touch this agent's files (an interactive turn,
``CaptureWorker``, ``MemoryConsolidationScheduler``, ``KnowledgeDigestScheduler``,
...) is a thread inside the *same* process, not a separate process — this
deployment doesn't run separate worker processes. ``_atomic_write_text``'s
own temp-file-then-rename swap already makes a *reader* never see a
half-written file; what it doesn't protect against on its own is two
*writers* targeting the same path concurrently (the read-modify-write in
:meth:`~MemoryFileRepository.append_daily` could lose an update, and two
overwrites could interleave unpredictably). Each write method acquires
this path's lock (:meth:`~MemoryFileRepository._lock_for`) for its full
read-modify-write sequence, which fully closes that race for this
deployment's actual (single-process, multi-thread) concurrency model — see
``session/db.py``'s module docstring for the same reasoning applied to
PostgreSQL writes. A per-*instance* lock dict, not a global one: two
different agents' repositories never touch the same file (separate
workspace roots), so they never need to share locks.
"""

from __future__ import annotations

import os
import threading
from datetime import date, datetime
from pathlib import Path

from .models import MemoryExcerpt, MemoryHit, MemoryLocator

# Mirrors LongTermMemory._SEARCH_MAX_RESULTS — same cap, same reasoning:
# prevents context flooding when many notes match a broad query.
_SEARCH_MAX_RESULTS = 20


def _sanitize_key(key: str) -> str:
    """Make a note key filesystem-safe.

    Mirrors ``LongTermMemory._path()``'s behavior exactly, so migrated keys
    round-trip to the same filenames: forward/back slashes are replaced with
    underscores to keep a flat structure within each subdirectory and avoid
    accidentally creating nested directories from a key like ``"api/notes"``.
    """
    return key.replace("/", "_").replace("\\", "_")


def _atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` via a temp-file-then-rename swap.

    ``LongTermMemory.save()`` uses a plain ``write_text()``, which can leave
    a half-written (truncated) file if the process crashes mid-write. This
    writes to a sibling temp file first, then uses ``os.replace()`` — atomic
    on both POSIX and Windows — so a reader never observes a partial file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


class MemoryFileRepository:
    """Disk-backed memory store for one agent, under the merged workspace layout.

    Args:
        root: The agent's workspace root (i.e. what
            ``workspace.agent_workspace_root()`` returns) — e.g.
            ``~/.minion-assist/workspaces/main``. The ``memory/``,
            ``memory/topics/``, and ``memory/imports/`` subdirectories are
            created automatically if missing.
    """

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._memory_dir = self._root / "memory"
        self._topics_dir = self._memory_dir / "topics"
        self._imports_dir = self._memory_dir / "imports"
        for d in (self._memory_dir, self._topics_dir, self._imports_dir):
            d.mkdir(parents=True, exist_ok=True)
        # One lock per target file path, created lazily on first use — see
        # the module docstring's "Concurrency" section for why this (not a
        # global lock, not OS-level file locking) is the right scope for
        # this deployment's actual concurrency model.
        self._write_locks: dict[Path, threading.Lock] = {}
        self._write_locks_guard = threading.Lock()

    def _lock_for(self, path: Path) -> threading.Lock:
        """Return the write lock for ``path``, creating it on first use.

        Every writer targeting the same ``path`` waits on the exact same
        ``Lock`` object, serializing their read-modify-write sequences —
        see the module docstring. ``_write_locks_guard`` only protects the
        lazy-creation step itself (a classic double-checked-locking
        pattern), not the caller's actual file operation.
        """
        with self._write_locks_guard:
            lock = self._write_locks.get(path)
            if lock is None:
                lock = threading.Lock()
                self._write_locks[path] = lock
            return lock

    @property
    def root(self) -> Path:
        """The agent's workspace root this repository reads/writes under."""
        return self._root

    # -----------------------------------------------------------------
    # Explicit notes (memory/topics/) — replaces LongTermMemory.save/load/delete
    # -----------------------------------------------------------------

    def topic_path(self, key: str) -> Path:
        """Resolve the on-disk path for a topic note key (doesn't require the file to exist).

        Exposed publicly (not just used internally) so callers like
        ``MemoryService`` can compute a topic's path — e.g. to tell
        :class:`~minion_assist.memory.postgres_index.PostgresMemoryIndex`
        which file was just deleted — without duplicating
        :func:`_sanitize_key`'s filename rules.
        """
        return self._topics_dir / f"{_sanitize_key(key)}.md"

    def remember(self, key: str, content: str) -> Path:
        """Save a note under ``memory/topics/{key}.md``, overwriting any existing note.

        Args:
            key: Note identifier, e.g. ``"project-goals"``. Sanitized via
                :func:`_sanitize_key` before use as a filename.
            content: Markdown text to store.

        Returns:
            Path: The file that was written — lets a caller (e.g.
                ``MemoryService``, for Phase 3 slice B's write-path index
                sync) compute a path relative to :attr:`root` without
                re-deriving the sanitized filename itself.
        """
        path = self.topic_path(key)
        with self._lock_for(path):
            _atomic_write_text(path, content)
        return path

    def load(self, key: str) -> str | None:
        """Load a topic note's content by key.

        Args:
            key: The note identifier.

        Returns:
            str | None: The note's text, or ``None`` if no such note exists.
        """
        path = self.topic_path(key)
        return path.read_text(encoding="utf-8") if path.exists() else None

    def delete(self, key: str) -> bool:
        """Delete a topic note.

        Args:
            key: The note identifier to delete.

        Returns:
            bool: ``True`` if a file was deleted, ``False`` if it didn't exist.
        """
        path = self.topic_path(key)
        with self._lock_for(path):
            if path.exists():
                path.unlink()
                return True
            return False

    def list_keys(self) -> list[str]:
        """Return every topic note's key, sorted alphabetically."""
        return [p.stem for p in sorted(self._topics_dir.glob("*.md"))]

    def count_notes(self) -> dict[str, int]:
        """Count notes by source — the primitive behind :class:`MemoryService`'s ``status()``.

        Returns:
            dict[str, int]: ``{"topic": N, "import": N, "daily": N}``.
        """
        return {
            "topic": sum(1 for _ in self._topics_dir.glob("*.md")),
            "import": sum(1 for _ in self._imports_dir.glob("*.md")),
            "daily": sum(1 for _ in self._memory_dir.glob("*.md")),
        }

    # -----------------------------------------------------------------
    # Knowledge digest (KNOWLEDGE_DIGEST.md) — Stage One Phase 7, slice D
    # -----------------------------------------------------------------
    #
    # Unlike MEMORY.md/USER.md/DREAMS.md (root-level files this repository
    # deliberately never writes — see the module docstring), this file IS
    # written here: it's a fully machine-owned, regenerated-on-a-schedule
    # artifact, not something a human ever hand-edits, so there is no
    # "don't clobber human edits" concern the way there would be for those.

    def write_digest(self, content: str) -> Path:
        """Overwrite ``KNOWLEDGE_DIGEST.md`` at the workspace root with ``content``.

        Lives directly under the workspace root (next to ``MEMORY.md``),
        not under ``memory/topics/`` — so ``bootstrap.py``'s existing
        file-injection machinery picks it up on every turn once
        ``"KNOWLEDGE_DIGEST.md"`` is added to its ``_BOOTSTRAP_FILES``
        tuple, with no other changes needed there.

        Args:
            content: The compiled digest text, typically from
                :func:`~minion_assist.memory.knowledge.compile_digest`.
                Written verbatim, including an empty string (which simply
                leaves an empty file — harmless, since ``bootstrap.py``
                already skips empty files when building the prompt).

        Returns:
            Path: The file that was written.
        """
        path = self._root / "KNOWLEDGE_DIGEST.md"
        with self._lock_for(path):
            _atomic_write_text(path, content)
        return path

    # -----------------------------------------------------------------
    # Quarantined notes (memory/imports/) — unreviewed, never auto-promoted
    # -----------------------------------------------------------------
    #
    # Used by the background extractor's rolling "_auto_extracted" note and
    # (until it is retired in a later Phase 1 slice) the "note" tool's daily
    # quick-log — both write content nobody has reviewed yet, which per
    # docs/adr/0003-per-agent-memory-scope.md must stay searchable but must
    # never be auto-promoted into curated memory/topics/ pages.

    def import_path(self, key: str) -> Path:
        """Resolve the on-disk path for an import key (doesn't require the file to exist).

        Mirrors :meth:`topic_path` exactly, just scoped to
        ``memory/imports/`` — exposed publicly for the same reason: so a
        caller (``memory/import_review.py``'s ``ImportReviewer``) can
        compute an import's ``rel_path`` (e.g. to tell
        :class:`~minion_assist.memory.postgres_index.PostgresMemoryIndex`
        which file to remove after a review) without duplicating
        :func:`_sanitize_key`'s filename rules.
        """
        return self._imports_dir / f"{_sanitize_key(key)}.md"

    def remember_import(self, key: str, content: str) -> Path:
        """Save quarantined, unreviewed content under ``memory/imports/{key}.md``.

        Args:
            key: Note identifier, e.g. ``"_auto_extracted"``.
            content: Markdown text to store.

        Returns:
            Path: The file that was written — see :meth:`remember`'s return
                value for why.
        """
        path = self.import_path(key)
        with self._lock_for(path):
            _atomic_write_text(path, content)
        return path

    def load_import(self, key: str) -> str | None:
        """Load quarantined content by key, or ``None`` if it doesn't exist."""
        path = self.import_path(key)
        return path.read_text(encoding="utf-8") if path.exists() else None

    def list_import_keys(self) -> list[str]:
        """Return every quarantined note's key, sorted alphabetically."""
        return [p.stem for p in sorted(self._imports_dir.glob("*.md"))]

    def delete_import(self, key: str) -> bool:
        """Delete a quarantined import note (Stage One Phase 7, slice E).

        Called by ``memory/import_review.py``'s ``ImportReviewer`` once an
        import has been reviewed (whether approved or rejected) — the
        reviewed snapshot is retired either way, so it never lingers to be
        offered again in a future ``preview()`` call. Mirrors :meth:`delete`
        exactly, just scoped to ``memory/imports/`` instead of
        ``memory/topics/``.

        Args:
            key: The import identifier to delete.

        Returns:
            bool: ``True`` if a file was deleted, ``False`` if it didn't exist.
        """
        path = self.import_path(key)
        with self._lock_for(path):
            if path.exists():
                path.unlink()
                return True
            return False

    # -----------------------------------------------------------------
    # Daily notes (memory/YYYY-MM-DD.md) — the one merged daily-note path
    # -----------------------------------------------------------------

    def append_daily(self, text: str, *, when: date | None = None) -> Path:
        """Append a timestamped entry to today's (or ``when``'s) daily note file.

        The file lives at ``memory/{date}.md`` — the same path
        ``dreaming.py`` already reads as source material and
        ``bootstrap.py``/``write_daily_memory.py`` already agree is the
        canonical daily-note location (see Finding 2 in the Phase 1 plan:
        this repository is what the merged ``note``/``write_daily_memory``
        tools will call once the tool layer is consolidated in a later
        slice). The first entry of a new day writes a ``## {date}`` header;
        every entry (including the first) is a timestamped bullet, so
        multiple notes on the same day read as a simple running log.

        Args:
            text: The note content to append.
            when: Date to file this entry under. Defaults to today. Exposed
                as a parameter (rather than always using ``date.today()``)
                so tests are deterministic.

        Returns:
            Path: The daily note file that was written to.
        """
        day = when or date.today()
        path = self._memory_dir / f"{day.isoformat()}.md"

        # The whole read-modify-write sequence must hold this path's lock,
        # not just the final write — this is the one method in this class
        # where the *read* result feeds the write, so two concurrent
        # appends (e.g. an interactive turn and MemoryConsolidationScheduler
        # both writing today's note) could otherwise both read the same
        # "before" content and one append would silently overwrite the
        # other's (MEM-GAP-009).
        with self._lock_for(path):
            existing = path.read_text(encoding="utf-8") if path.exists() else ""
            stamp = datetime.now().strftime("%H:%M")
            entry = f"- {stamp}: {text}"

            if existing.strip():
                new_content = existing.rstrip("\n") + "\n" + entry + "\n"
            else:
                new_content = f"## {day.isoformat()}\n\n{entry}\n"

            _atomic_write_text(path, new_content)
        return path

    # -----------------------------------------------------------------
    # Search — same scoring as LongTermMemory.search(), now spanning three sources
    # -----------------------------------------------------------------

    def search(
        self,
        query: str,
        max_results: int = _SEARCH_MAX_RESULTS,
        *,
        exclude_sources: frozenset[str] = frozenset(),
    ) -> list[MemoryHit]:
        """Search topic notes, imported notes, and daily notes, ranked by term frequency.

        Scoring is unchanged from ``LongTermMemory.search()``: notes matching
        more distinct query terms (3+ characters, case-insensitive) rank
        higher, with a small recency tie-breaker among equal-scoring notes.
        This slice intentionally keeps the same recall characteristics as
        today — see ``tests/fixtures/memory_corpus/README.md`` for the
        recorded baseline this will eventually be compared against.

        Args:
            query: One or more keywords, space-separated.
            max_results: Maximum notes to return.
            exclude_sources: Source tags to leave out of the candidate pool
                entirely (MEM-GAP-004) — applied *before* scoring/truncation,
                so an excluded source can never crowd out an eligible one
                within ``max_results``. ``MemoryService.search()`` passes
                ``{"import"}`` here for a corpus-agnostic call, so quarantined
                notes never reach automatic per-turn recall in the
                degraded/local-fallback path (they're still reachable via an
                explicit ``corpus="import"`` search — see its docstring).

        Returns:
            list[MemoryHit]: Best matches first, each tagged with its source
                ("topic", "import", or "daily").
        """
        terms = [t.lower() for t in query.split() if len(t) >= 3]
        if not terms:
            return []

        candidates: list[tuple[str, Path]] = []
        if "topic" not in exclude_sources:
            candidates.extend(("topic", p) for p in self._topics_dir.glob("*.md"))
        if "import" not in exclude_sources:
            candidates.extend(("import", p) for p in self._imports_dir.glob("*.md"))
        # Non-recursive glob on memory_dir itself only matches YYYY-MM-DD.md
        # files directly inside it — topics/ and imports/ are subdirectories
        # and are not matched again here.
        if "daily" not in exclude_sources:
            candidates.extend(("daily", p) for p in self._memory_dir.glob("*.md"))

        scored: list[tuple[float, str, str, str]] = []  # (score, key, content, source)
        for source, p in candidates:
            try:
                content = p.read_text(encoding="utf-8")
            except OSError:
                continue

            content_lower = content.lower()
            stem_lower = p.stem.lower()
            match_count = sum(1 for t in terms if t in content_lower or t in stem_lower)
            if match_count == 0:
                continue

            try:
                recency = p.stat().st_mtime / 1e10
            except OSError:
                recency = 0.0

            scored.append((match_count + recency, p.stem, content, source))

        scored.sort(key=lambda x: (-x[0], x[1]))
        return [
            MemoryHit(key=key, content=content, source=source)
            for _, key, content, source in scored[:max_results]
        ]

    # -----------------------------------------------------------------
    # Indexable file enumeration — Stage One Phase 3 (memory/postgres_index.py)
    # -----------------------------------------------------------------

    def list_indexable_files(self) -> list[tuple[str, str, str]]:
        """Enumerate every file the Phase 3 lexical index should cover.

        Per the plan's Phase 3 Task 4: index ``MEMORY.md``, dated notes,
        topic pages, and imports; exclude ``DREAMS.md``, ``USER.md``, and
        any other root-level file. There is no "reviewed" flag on imports
        yet (that distinction is Phase 5's job), so every import is indexed
        for now — still tagged ``"import"`` so a caller can filter it out.

        Returns:
            list[tuple[str, str, str]]: ``(source_kind, rel_path, content)``
                triples. ``source_kind`` is one of ``"durable"`` (
                ``MEMORY.md`` and topic notes), ``"daily"`` (
                ``memory/YYYY-MM-DD.md``), or ``"import"`` (
                ``memory/imports/*.md``). ``rel_path`` is relative to this
                repository's root (:attr:`root`), stable across runs so it
                can be used as an indexing key. Unreadable files (e.g. a
                permissions error) are silently skipped, matching
                :meth:`search`'s existing tolerance for a bad file.
        """
        results: list[tuple[str, str, str]] = []

        memory_md = self._root / "MEMORY.md"
        if memory_md.exists():
            try:
                results.append(("durable", "MEMORY.md", memory_md.read_text(encoding="utf-8")))
            except OSError:
                pass

        for p in sorted(self._topics_dir.glob("*.md")):
            try:
                content = p.read_text(encoding="utf-8")
            except OSError:
                continue
            results.append(("durable", f"memory/topics/{p.name}", content))

        for p in sorted(self._imports_dir.glob("*.md")):
            try:
                content = p.read_text(encoding="utf-8")
            except OSError:
                continue
            results.append(("import", f"memory/imports/{p.name}", content))

        # Non-recursive glob on memory_dir itself only matches YYYY-MM-DD.md
        # files directly inside it — see search()'s identical comment.
        for p in sorted(self._memory_dir.glob("*.md")):
            try:
                content = p.read_text(encoding="utf-8")
            except OSError:
                continue
            results.append(("daily", f"memory/{p.name}", content))

        return results

    # -----------------------------------------------------------------
    # Bounded exact reads — new in Phase 1 (the plan's memory_get)
    # -----------------------------------------------------------------

    def resolve_path(self, path_str: str) -> Path:
        """Resolve a path string to an absolute path inside this memory root.

        Relative paths are resolved against the memory root itself
        (``workspaces/{agent_id}/``, not just ``memory/``), so ``"MEMORY.md"``
        or ``"memory/topics/foo.md"`` both work. Absolute paths are accepted
        only if they already fall inside the root.

        Args:
            path_str: A relative or absolute path.

        Returns:
            Path: The resolved, validated absolute path.

        Raises:
            ValueError: If the resolved path falls outside the memory root
                (path traversal, symlink escape, or an absolute path
                elsewhere on disk).
        """
        candidate = Path(path_str)
        candidate = candidate if candidate.is_absolute() else self._root / candidate
        try:
            resolved = candidate.resolve()
            resolved.relative_to(self._root)
        except (ValueError, OSError) as exc:
            raise ValueError(f"'{path_str}' is outside the memory root {self._root}") from exc
        return resolved

    def get(self, locator: MemoryLocator) -> MemoryExcerpt:
        """Read an exact, bounded slice of a file — for citing specific lines, not searching.

        Args:
            locator: Identifies the file and (optionally) the line range.
                Build one via :meth:`resolve_path` first to ensure the path
                has already been validated as inside this memory root.

        Returns:
            MemoryExcerpt: The requested slice, plus the file's total line
                count so a caller can request a continuation page.

        Raises:
            FileNotFoundError: If ``locator.path`` does not exist.
        """
        all_lines = locator.path.read_text(encoding="utf-8").splitlines()
        total = len(all_lines)

        if total == 0:
            return MemoryExcerpt(
                path=locator.path, start_line=0, end_line=0, total_lines=0, text=""
            )

        start = max(1, locator.from_line or 1)
        start = min(start, total)  # clamp a too-large from_line to the last line, not empty
        end = total if locator.lines is None else min(total, start + locator.lines - 1)

        text = "\n".join(all_lines[start - 1:end])
        return MemoryExcerpt(
            path=locator.path, start_line=start, end_line=end, total_lines=total, text=text
        )

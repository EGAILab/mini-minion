"""``MemoryFileRepository`` — the canonical on-disk memory layout for one agent (Stage One Phase 1).

This replaces :class:`~minion_assist.memory.long_term.LongTermMemory` as the
thing that actually touches disk. Where ``LongTermMemory`` stored every note
as a flat ``{base_dir}/{key}.md`` file (mixing explicit notes, daily logs,
and the extractor's rolling note in one directory), this repository targets
the merged per-agent layout Stage One Phase 0 migrates existing data into
(see ``docs/adr/0003-per-agent-memory-scope.md``)::

    workspaces/{agent_id}/
      USER.md, MEMORY.md, DREAMS.md   # untouched here — bootstrap.py/dreaming.py own these
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
"""

from __future__ import annotations

import os
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

    # -----------------------------------------------------------------
    # Explicit notes (memory/topics/) — replaces LongTermMemory.save/load/delete
    # -----------------------------------------------------------------

    def remember(self, key: str, content: str) -> None:
        """Save a note under ``memory/topics/{key}.md``, overwriting any existing note.

        Args:
            key: Note identifier, e.g. ``"project-goals"``. Sanitized via
                :func:`_sanitize_key` before use as a filename.
            content: Markdown text to store.
        """
        path = self._topics_dir / f"{_sanitize_key(key)}.md"
        _atomic_write_text(path, content)

    def load(self, key: str) -> str | None:
        """Load a topic note's content by key.

        Args:
            key: The note identifier.

        Returns:
            str | None: The note's text, or ``None`` if no such note exists.
        """
        path = self._topics_dir / f"{_sanitize_key(key)}.md"
        return path.read_text(encoding="utf-8") if path.exists() else None

    def delete(self, key: str) -> bool:
        """Delete a topic note.

        Args:
            key: The note identifier to delete.

        Returns:
            bool: ``True`` if a file was deleted, ``False`` if it didn't exist.
        """
        path = self._topics_dir / f"{_sanitize_key(key)}.md"
        if path.exists():
            path.unlink()
            return True
        return False

    def list_keys(self) -> list[str]:
        """Return every topic note's key, sorted alphabetically."""
        return [p.stem for p in sorted(self._topics_dir.glob("*.md"))]

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

    def search(self, query: str, max_results: int = _SEARCH_MAX_RESULTS) -> list[MemoryHit]:
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

        Returns:
            list[MemoryHit]: Best matches first, each tagged with its source
                ("topic", "import", or "daily").
        """
        terms = [t.lower() for t in query.split() if len(t) >= 3]
        if not terms:
            return []

        candidates: list[tuple[str, Path]] = []
        candidates.extend(("topic", p) for p in self._topics_dir.glob("*.md"))
        candidates.extend(("import", p) for p in self._imports_dir.glob("*.md"))
        # Non-recursive glob on memory_dir itself only matches YYYY-MM-DD.md
        # files directly inside it — topics/ and imports/ are subdirectories
        # and are not matched again here.
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

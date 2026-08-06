"""Typed data contracts for the memory subsystem (Stage One Phase 1).

Why typed dataclasses instead of raw tuples/str?
-------------------------------------------------
:class:`LongTermMemory` (the legacy store) returns ``list[tuple[str, str]]``
from ``search()`` — a ``(key, content)`` pair — and callers format that tuple
into a string by hand wherever they need it. That works when there is one
consumer, but Phase 1 introduces a second one: :class:`MemoryService`'s
``recall()`` (used for per-turn ``<relevant_memories>`` injection) needs the
same underlying data shaped differently — truncated snippets and a "where
did this come from" label, not full content. Passing a typed object through
``files.py`` -> ``service.py`` -> tools lets each layer read the fields it
needs without renegotiating a tuple shape every time a new consumer appears.

Talks to
--------
- ``memory/files.py`` — :class:`MemoryFileRepository` methods return these
  types.
- ``memory/service.py`` — :class:`MemoryService` methods pass these types
  through to tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemoryHit:
    """One search result — a whole note (Phase 1 scan) or one chunk (Phase 3 index).

    Attributes:
        key: The note's identifier (filename stem), e.g. ``"api-research"``.
        content: The matched content — a whole note's text (linear scan) or
            one chunk's text (lexical index; see ``memory/chunking.py``).
        source: Which part of the memory root the content came from. The
            linear scan tags ``"topic"``/``"import"``/``"daily"``; the
            lexical index (Stage One Phase 3, slice C) tags
            ``"durable"``/``"daily"``/``"import"`` — ``"durable"`` covers
            both topic notes *and* root ``MEMORY.md``, and so does the
            linear scan's ``"topic"`` tag (MEM-GAP-008: root ``MEMORY.md``
            is now also a linear-scan candidate, tagged ``"topic"`` to
            match this same corpus grouping — see ``memory/files.py``'s
            ``MemoryFileRepository.search()``). The tag *string* still
            differs between the two paths; only the corpus each one covers
            is now the same.
        rel_path: Path relative to the agent's workspace root, e.g.
            ``"memory/topics/project-goals.md"``. ``None`` for a linear-scan
            hit (Phase 1 never tracked this).
        start_line: 1-indexed first line of this chunk in its source file.
            ``None`` for a linear-scan hit (whole-file, no chunk range).
        end_line: 1-indexed last line of this chunk (inclusive). ``None``
            for a linear-scan hit.
        score: The lexical index's ``ts_rank`` relevance score. ``None`` for
            a linear-scan hit (that scoring isn't rank-comparable to
            ``ts_rank``, so it's left unset rather than forced into this
            field).
        boundary: Stage One Phase 6, slice A — a formatted, advisory
            ``[Boundary: ...]`` annotation (see ``memory/boundaries.py``'s
            ``format_boundary_prefix``) when this hit's source note
            carries action-boundary frontmatter, else ``None``. Set on an
            indexed-path hit by ``MemoryService._apply_boundaries``, and
            on a linear-scan hit directly by
            ``MemoryFileRepository.search()`` (MEM-GAP-008) — both paths
            parse the same frontmatter, so a note outside its boundary
            window is excluded identically either way.
    """

    key: str
    content: str
    source: str
    rel_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    score: float | None = None
    boundary: str | None = None


@dataclass(frozen=True)
class MemoryLocator:
    """Identifies an exact memory file (and optionally a line range) to read.

    Used by :meth:`MemoryFileRepository.get` for bounded, cited reads — as
    opposed to :meth:`MemoryFileRepository.search`, which returns whole-note
    content ranked by relevance. A locator names *which* file and *which
    lines*, with no ranking involved.

    Attributes:
        path: Absolute path to the file. Must already be resolved and
            validated as inside the memory root — obtain one via
            :meth:`MemoryFileRepository.resolve_path` rather than
            constructing this from raw, unchecked user input.
        from_line: 1-indexed starting line (inclusive). ``None`` means "from
            the start of the file".
        lines: Maximum number of lines to return starting at ``from_line``.
            ``None`` means "to the end of the file".
    """

    path: Path
    from_line: int | None = None
    lines: int | None = None


@dataclass(frozen=True)
class MemoryExcerpt:
    """The result of a bounded :meth:`MemoryFileRepository.get` read.

    Always reports the *actual* line range returned (``start_line``/
    ``end_line``) and the file's total line count, so a caller can tell
    whether they've reached the end of the file or need to request another
    page — this is what makes the read "bounded" rather than "read whole
    file, hope it's small."

    Attributes:
        path: The file that was read.
        start_line: 1-indexed first line included in ``text``.
        end_line: 1-indexed last line included in ``text`` (inclusive).
        total_lines: Total number of lines in the file, for continuation.
        text: The requested slice of the file's content.
    """

    path: Path
    start_line: int
    end_line: int
    total_lines: int
    text: str


@dataclass(frozen=True)
class MemoryStatus:
    """A snapshot of one agent's memory store — counts only, for now.

    Returned by :meth:`MemoryService.status`. Deliberately minimal in Phase 1
    (no jobs, no index health, no degraded-mode reporting — there is no
    database or background worker yet). Later phases extend this rather than
    replace it, once there is durable job/index state worth reporting on
    (see ``docs/adr/0004-degraded-operation.md``).

    Attributes:
        root: The agent's workspace root this store reads from.
        topic_count: Number of explicit notes under ``memory/topics/``.
        import_count: Number of quarantined notes under ``memory/imports/``.
        daily_count: Number of dated ``memory/YYYY-MM-DD.md`` files.
    """

    root: Path
    topic_count: int
    import_count: int
    daily_count: int


# Status values for FlushOutcome. Kept as plain strings (not an enum) to
# match the rest of this module's style (see MemoryHit.source) and because
# the plan's own language ("succeeded, found nothing, failed transiently, or
# was skipped") maps directly onto these four cases.
FLUSH_STATUS_FLUSHED = "flushed"
FLUSH_STATUS_EMPTY = "empty"
FLUSH_STATUS_FAILED = "failed"
FLUSH_STATUS_SKIPPED = "skipped"


@dataclass(frozen=True)
class FlushOutcome:
    """The result of one :meth:`MemoryService.flush_head` call.

    Stage One Phase 2, slice B: before :class:`~minion_assist.context.Compactor`
    summarizes and discards old history, its content is flushed to a durable
    daily note first — so a failed or lossy summarization can never be the
    only place that context existed. This is always computed (never silently
    skipped) whenever a memory backend is configured and compaction is about
    to run; see ``agents/session.py``'s pre-compaction flush integration.

    Attributes:
        status: One of :data:`FLUSH_STATUS_FLUSHED` (content was written),
            :data:`FLUSH_STATUS_EMPTY` (nothing worth writing — the head
            rendered to blank text), :data:`FLUSH_STATUS_FAILED` (writing
            raised an exception — reported, never lets the exception
            propagate and block the turn), or :data:`FLUSH_STATUS_SKIPPED`
            (no memory backend configured; nothing to flush to).
        detail: Human-readable detail — empty for "flushed"/"empty"/
            "skipped", the exception description for "failed".
    """

    status: str
    detail: str = ""

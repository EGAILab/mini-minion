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
    """One search result: a note found by :meth:`MemoryFileRepository.search`.

    Attributes:
        key: The note's identifier (filename stem), e.g. ``"api-research"``.
        content: The note's full Markdown content.
        source: Which part of the memory root the note came from —
            ``"topic"`` (an explicit ``save_memory`` note under
            ``memory/topics/``), ``"import"`` (quarantined, unreviewed
            extractor/daily-log output under ``memory/imports/`` — see
            ``docs/adr/0003-per-agent-memory-scope.md``), or ``"daily"`` (a
            dated ``memory/YYYY-MM-DD.md`` log file).
    """

    key: str
    content: str
    source: str


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

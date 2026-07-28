"""Heading-aware Markdown chunker for the Stage One Phase 3 lexical index.

Why chunk at all?
-----------------
Phase 1's ``MemoryFileRepository.search()`` scores whole files. That is fine
while every note is short, but a citation like "your preference is in
MEMORY.md" stops being useful once ``MEMORY.md`` grows into a long,
multi-section file over months of use — a caller needs the specific
section, not "somewhere in this 2 000-line file." This module splits a
file's Markdown text into overlapping, heading-aware, token-bounded pieces
so :mod:`memory.postgres_index` can index and cite at that granularity.

Why token-bounded *and* heading-aware, not just one or the other?
------------------------------------------------------------------
A pure token-count splitter can cut a chunk in the middle of a heading's
section, which loses context (the heading, and thus the *topic*, ends up in
the previous chunk while the content it describes lands in the next one). A
pure heading splitter can produce a wildly oversized chunk if one section is
very long. This chunker grows a chunk by token count but, once it is large
enough, prefers to end it at the most recent heading boundary rather than
mid-section — the best of both without needing a full document outline
pass.

Why overlap?
------------
Without overlap, a sentence describing one fact could be split exactly at
the chunk boundary, with half its context in each chunk and neither chunk
scoring well for a query about it. Carrying the last ``overlap_tokens``
worth of lines from the end of one chunk into the start of the next means
that content near a boundary is still fully present in at least one chunk.

Talks to
--------
- ``context.py`` — this module's token counter mirrors
  ``_make_token_estimator()``'s tiktoken-or-char/4-heuristic fallback
  exactly, but counts plain chunk text rather than a message dict, so it is
  its own small function here rather than an import.
- ``memory/postgres_index.py`` — the only caller; turns each :class:`Chunk`
  into one ``memory_chunks`` row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Matches a Markdown ATX heading ("#" through "######") and captures its
# level (by the number of "#" characters) and text.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

# ~400 tokens keeps a chunk small enough to inject into a prompt as a single
# citation without dominating the token budget, while still covering a
# typical short section in one piece.
DEFAULT_TARGET_TOKENS = 400

# 80 tokens (~20% of the target) is enough to carry a full sentence or two
# of context across a boundary without duplicating so much that the index
# balloons in size.
DEFAULT_OVERLAP_TOKENS = 80


def _count_tokens(text: str) -> int:
    """Approximate the token count of plain text.

    Tries tiktoken's cl100k_base encoding (exact for the GPT-4 family, a
    good proxy for other models) and falls back to a 4-chars-per-token
    heuristic when tiktoken is not installed — the same fallback shape as
    ``context.py``'s per-message estimator, applied to plain chunk text
    instead of a message dict.
    """
    try:
        import tiktoken  # noqa: PLC0415

        enc = tiktoken.get_encoding("cl100k_base")
        return max(1, len(enc.encode(text)))
    except ImportError:
        return max(1, len(text) // 4)


@dataclass(frozen=True)
class Chunk:
    """One indexable slice of a Markdown file.

    Attributes:
        content: The chunk's raw text (a contiguous, unmodified slice of
            the source file's lines — never a line fragment).
        start_line: 1-indexed, inclusive first line of this chunk in the
            source file.
        end_line: 1-indexed, inclusive last line of this chunk in the
            source file.
        heading_path: The Markdown heading hierarchy in effect at
            ``start_line``, root to leaf (e.g. ``("Projects", "Minion
            Assist")``). Empty for a file (or chunk) with no headings above
            it yet.
        token_count: This chunk's own approximate token count (see
            :func:`_count_tokens`) — stored so ``postgres_index.py`` doesn't
            need to recompute it for logging/diagnostics.
    """

    content: str
    start_line: int
    end_line: int
    heading_path: tuple[str, ...]
    token_count: int


def chunk_markdown(
    text: str,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Split Markdown text into heading-aware, token-bounded, overlapping chunks.

    Args:
        text: The file's full Markdown content.
        target_tokens: Approximate token budget per chunk. A chunk may end
            up smaller than this if a heading boundary is found first, or
            larger if a single line alone exceeds the budget (a chunk is
            never split mid-line).
        overlap_tokens: Approximate token count to carry back from the end
            of one chunk into the start of the next, so content near a
            boundary is never confined to just one side of it.

    Returns:
        list[Chunk]: Chunks in file order. Empty for empty/whitespace-only
            text.
    """
    if not text.strip():
        return []
    lines = text.splitlines()
    n = len(lines)

    # heading_path_after[i] = the heading stack in effect immediately after
    # line i has been read (so if line i is itself a heading, it is already
    # included — a chunk starting on a heading line should show that
    # heading as part of its own path, not just its ancestors).
    heading_path_after: list[tuple[tuple[int, str], ...]] = []
    stack: tuple[tuple[int, str], ...] = ()
    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            stack = tuple(h for h in stack if h[0] < level) + ((level, heading_text),)
        heading_path_after.append(stack)

    chunks: list[Chunk] = []
    start = 0  # 0-indexed line, inclusive start of the chunk being built

    while start < n:
        idx = start
        token_count = 0
        last_heading_boundary: int | None = None  # 0-indexed line of a candidate cut point
        hit_token_limit = False

        while idx < n:
            # A heading strictly after the chunk's first line is a candidate
            # place to end the chunk instead of cutting mid-section. (A
            # heading exactly on the first line just starts this chunk.)
            if idx > start and _HEADING_RE.match(lines[idx]):
                last_heading_boundary = idx

            line_tokens = _count_tokens(lines[idx])
            # Always include at least one line, even if it alone exceeds
            # the budget — never produce an empty chunk.
            if idx > start and token_count + line_tokens > target_tokens:
                hit_token_limit = True
                break
            token_count += line_tokens
            idx += 1

        end = idx  # 0-indexed, exclusive

        # Only prefer a heading boundary over the natural end when the
        # token budget actually forced an early stop. If the remaining
        # file simply fit within budget (loop ran to EOF on its own), there
        # is no mid-section cut to avoid in the first place — cutting back
        # to a heading found along the way would fragment a file that
        # never needed splitting at all.
        _has_earlier_heading = (
            last_heading_boundary is not None and start < last_heading_boundary < end
        )
        if hit_token_limit and _has_earlier_heading:
            end = last_heading_boundary

        chunk_lines = lines[start:end]
        chunk_text = "\n".join(chunk_lines)
        chunks.append(
            Chunk(
                content=chunk_text,
                start_line=start + 1,
                end_line=end,
                # The heading context as of the chunk's *last* line — the
                # most specific section this chunk's content ultimately
                # belongs to (a heading appearing partway through simply
                # becomes the active context for the rest of the chunk).
                heading_path=tuple(h[1] for h in heading_path_after[end - 1]),
                token_count=_count_tokens(chunk_text),
            )
        )

        if end >= n:
            break

        # Walk backwards from the end of this chunk to find where the next
        # chunk should start, carrying ~overlap_tokens worth of trailing
        # lines forward as shared context.
        overlap_start = end
        acc = 0
        while overlap_start > start and acc < overlap_tokens:
            overlap_start -= 1
            acc += _count_tokens(lines[overlap_start])

        # Guarantee forward progress even if the overlap walk reached all
        # the way back to this chunk's own start (e.g. overlap_tokens is
        # larger than the chunk itself) — otherwise the loop would spin on
        # the same `start` forever.
        start = max(overlap_start, start + 1)

    return chunks

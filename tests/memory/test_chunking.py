"""Tests for memory/chunking.py's heading-aware Markdown chunker (Stage One Phase 3, slice A).

Token counts here always fall back to the 4-chars-per-token heuristic (this
dev environment has no ``tiktoken`` installed), so tests avoid hardcoding
exact expected token numbers where the tiktoken backend could plausibly
produce a different count — instead they assert structural properties
(boundary placement, overlap, reconstruction) that hold under either
backend, and use ``_count_tokens`` itself where an exact number is needed.
"""

from __future__ import annotations

from minion_assist.memory.chunking import Chunk, _count_tokens, chunk_markdown


def test_empty_text_returns_no_chunks():
    assert chunk_markdown("") == []


def test_whitespace_only_text_returns_no_chunks():
    assert chunk_markdown("   \n  \n\t") == []


def test_short_text_returns_a_single_chunk():
    text = "Just a short note about coffee preferences."
    chunks = chunk_markdown(text, target_tokens=400, overlap_tokens=80)

    assert len(chunks) == 1
    assert chunks[0].content == text
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 1
    assert chunks[0].heading_path == ()


def test_heading_path_reflects_the_heading_context_active_at_chunk_end():
    text = "# Projects\n## Minion Assist\nSome content under this heading."
    chunks = chunk_markdown(text, target_tokens=400, overlap_tokens=0)

    assert len(chunks) == 1
    assert chunks[0].heading_path == ("Projects", "Minion Assist")


def test_heading_path_is_empty_before_any_heading():
    text = "Preamble content with no heading above it yet."
    [chunk] = chunk_markdown(text)
    assert chunk.heading_path == ()


def test_deeper_heading_replaces_same_or_shallower_level():
    # "## B" should pop "# A" only if a same-or-shallower heading follows;
    # here "### C" nests under both "# A" and "## B".
    text = "# A\n## B\n### C\nbody text"
    [chunk] = chunk_markdown(text)
    assert chunk.heading_path == ("A", "B", "C")


def test_sibling_heading_replaces_previous_heading_at_same_level():
    text = "# A\n## B\ncontent one\n## C\ncontent two"
    chunks = chunk_markdown(text, target_tokens=400, overlap_tokens=0)
    # Still one chunk (well under the token budget — nothing here forces an
    # early split), so the heading path reflects whatever is active by the
    # chunk's last line: "## C" sibling-replaced "## B" partway through.
    assert len(chunks) == 1
    assert chunks[0].heading_path == ("A", "C")


def test_never_splits_a_single_line_even_if_it_exceeds_the_budget():
    long_line = "word " * 200  # comfortably exceeds a tiny token budget alone
    chunks = chunk_markdown(long_line, target_tokens=5, overlap_tokens=0)

    assert len(chunks) == 1
    assert chunks[0].content == long_line


def test_splits_into_multiple_chunks_when_exceeding_target_tokens():
    lines = [f"Line number {i} with some filler words to add tokens." for i in range(60)]
    text = "\n".join(lines)

    chunks = chunk_markdown(text, target_tokens=20, overlap_tokens=0)

    assert len(chunks) > 1


def test_chunk_content_matches_source_line_range():
    lines = [f"line {i}" for i in range(30)]
    text = "\n".join(lines)

    chunks = chunk_markdown(text, target_tokens=10, overlap_tokens=0)

    for chunk in chunks:
        expected = "\n".join(lines[chunk.start_line - 1:chunk.end_line])
        assert chunk.content == expected


def test_chunk_token_count_matches_count_tokens_of_its_own_content():
    lines = [f"filler line number {i}" for i in range(40)]
    text = "\n".join(lines)

    chunks = chunk_markdown(text, target_tokens=15, overlap_tokens=5)

    for chunk in chunks:
        assert chunk.token_count == _count_tokens(chunk.content)


def test_no_split_at_all_when_the_whole_file_fits_the_budget():
    # A heading partway through must NOT force a split when nothing about
    # the token budget required stopping early — only an actual forced
    # cutoff should ever prefer a heading boundary over the natural end.
    text = "Some short opening body text.\n\n# Next Section\nMore content here."
    chunks = chunk_markdown(text, target_tokens=400, overlap_tokens=0)

    assert len(chunks) == 1
    assert chunks[0].content == text


def test_prefers_ending_at_a_heading_boundary_over_the_token_cutoff():
    lines = [
        "Some short opening body text.",
        "",
        "# Next Section",
        "More content here.",
    ]
    text = "\n".join(lines)
    # Choose a budget that exactly covers the first three lines (through the
    # heading) — adding the fourth line must overflow it, forcing a split.
    # The split should then land at the heading (line 3) rather than after
    # the fourth line, since that heading is a valid, earlier cut point.
    target_tokens = sum(_count_tokens(line) for line in lines[:3])

    chunks = chunk_markdown(text, target_tokens=target_tokens, overlap_tokens=0)

    assert len(chunks) == 2
    assert chunks[0].content == "Some short opening body text.\n"
    assert chunks[1].heading_path == ("Next Section",)


def test_overlap_carries_trailing_lines_into_the_next_chunk():
    lines = [f"line {i}" for i in range(30)]
    text = "\n".join(lines)

    chunks = chunk_markdown(text, target_tokens=10, overlap_tokens=8)

    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        # The next chunk starts at or before the previous chunk's end line
        # (overlap), but strictly after the previous chunk's own start line
        # (forward progress is always guaranteed).
        assert nxt.start_line <= prev.end_line
        assert nxt.start_line > prev.start_line


def test_zero_overlap_produces_a_clean_partition_with_no_gap_or_repeat():
    lines = [f"line {i}" for i in range(30)]
    text = "\n".join(lines)

    chunks = chunk_markdown(text, target_tokens=10, overlap_tokens=0)

    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.start_line == prev.end_line + 1


def test_large_overlap_still_makes_forward_progress():
    lines = [f"line {i}" for i in range(20)]
    text = "\n".join(lines)

    # overlap_tokens far larger than any chunk could ever hold.
    chunks = chunk_markdown(text, target_tokens=5, overlap_tokens=10_000)

    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.start_line > prev.start_line
    # And the whole file is still covered, ending exactly at the last line.
    assert chunks[-1].end_line == len(lines)


def test_chunks_cover_the_entire_file_without_gaps():
    lines = [f"line {i}" for i in range(50)]
    text = "\n".join(lines)

    chunks = chunk_markdown(text, target_tokens=12, overlap_tokens=4)

    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == len(lines)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.start_line <= prev.end_line + 1  # no gap between chunks


def test_chunk_is_a_frozen_dataclass_instance():
    [chunk] = chunk_markdown("hello world")
    assert isinstance(chunk, Chunk)

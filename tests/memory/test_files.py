"""Tests for memory/files.py: MemoryFileRepository (Stage One Phase 1, slice 1).

All tests use tmp_path as a fake agent workspace root — no real filesystem
side effects outside the test's own temp directory.
"""

from __future__ import annotations

import threading
from datetime import date

import pytest

from minion_assist.memory.files import MemoryFileRepository, _sanitize_key
from minion_assist.memory.models import MemoryLocator

# ---------------------------------------------------------------------------
# __init__ — directory creation
# ---------------------------------------------------------------------------

def test_init_creates_memory_subdirectories(tmp_path):
    MemoryFileRepository(tmp_path)
    assert (tmp_path / "memory").is_dir()
    assert (tmp_path / "memory" / "topics").is_dir()
    assert (tmp_path / "memory" / "imports").is_dir()


def test_init_is_idempotent(tmp_path):
    MemoryFileRepository(tmp_path)
    MemoryFileRepository(tmp_path)  # must not raise on second call
    assert (tmp_path / "memory" / "topics").is_dir()


def test_root_property_returns_resolved_root(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    assert repo.root == tmp_path.resolve()


# ---------------------------------------------------------------------------
# _sanitize_key
# ---------------------------------------------------------------------------

def test_sanitize_key_replaces_forward_slash():
    assert _sanitize_key("api/notes") == "api_notes"


def test_sanitize_key_replaces_backslash():
    assert _sanitize_key("api\\notes") == "api_notes"


def test_sanitize_key_leaves_plain_key_unchanged():
    assert _sanitize_key("project-goals") == "project-goals"


# ---------------------------------------------------------------------------
# remember / load / delete / list_keys (memory/topics/)
# ---------------------------------------------------------------------------

def test_remember_writes_to_topics_dir(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember("project-goals", "# Goals\nShip Phase 1.")
    path = tmp_path / "memory" / "topics" / "project-goals.md"
    assert path.read_text(encoding="utf-8") == "# Goals\nShip Phase 1."


def test_remember_returns_the_path_it_wrote(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    returned = repo.remember("project-goals", "content")
    assert returned == tmp_path / "memory" / "topics" / "project-goals.md"


def test_topic_path_matches_remembers_actual_write_location(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    assert repo.topic_path("api/notes") == tmp_path / "memory" / "topics" / "api_notes.md"


def test_remember_sanitizes_key(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember("api/rest-notes", "content")
    assert (tmp_path / "memory" / "topics" / "api_rest-notes.md").exists()


def test_remember_overwrites_existing_note(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember("note", "first")
    repo.remember("note", "second")
    assert repo.load("note") == "second"


def test_remember_leaves_no_leftover_temp_file(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember("note", "content")
    leftovers = list((tmp_path / "memory" / "topics").glob(".*.tmp"))
    assert leftovers == []


def test_load_returns_none_for_missing_note(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    assert repo.load("does-not-exist") is None


def test_delete_removes_existing_note(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember("note", "content")
    assert repo.delete("note") is True
    assert repo.load("note") is None


def test_delete_returns_false_for_missing_note(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    assert repo.delete("does-not-exist") is False


def test_list_keys_returns_sorted_topic_keys(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember("zeta", "z")
    repo.remember("alpha", "a")
    assert repo.list_keys() == ["alpha", "zeta"]


def test_list_keys_excludes_imports_and_daily_notes(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember("topic-note", "content")
    (tmp_path / "memory" / "imports" / "imported.md").write_text("x", encoding="utf-8")
    repo.append_daily("daily entry")
    assert repo.list_keys() == ["topic-note"]


def test_count_notes_reports_all_three_sources(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember("topic-note", "content")
    (tmp_path / "memory" / "imports" / "imported.md").write_text("x", encoding="utf-8")
    repo.append_daily("daily entry")
    assert repo.count_notes() == {"topic": 1, "import": 1, "daily": 1}


def test_count_notes_all_zero_for_empty_store(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    assert repo.count_notes() == {"topic": 0, "import": 0, "daily": 0}


# ---------------------------------------------------------------------------
# write_digest (KNOWLEDGE_DIGEST.md, Stage One Phase 7, slice D)
# ---------------------------------------------------------------------------

def test_write_digest_writes_to_the_workspace_root(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    path = repo.write_digest("# Knowledge Digest\n\n- A fact.")

    assert path == tmp_path / "KNOWLEDGE_DIGEST.md"
    assert path.read_text(encoding="utf-8") == "# Knowledge Digest\n\n- A fact."


def test_write_digest_is_not_nested_under_memory(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.write_digest("content")

    assert not (tmp_path / "memory" / "KNOWLEDGE_DIGEST.md").exists()


def test_write_digest_overwrites_an_existing_digest(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.write_digest("first version")
    repo.write_digest("second version")

    assert (tmp_path / "KNOWLEDGE_DIGEST.md").read_text(encoding="utf-8") == "second version"


def test_write_digest_accepts_an_empty_string(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    path = repo.write_digest("")

    assert path.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# remember_import / load_import / list_import_keys (memory/imports/)
# ---------------------------------------------------------------------------

def test_remember_import_writes_to_imports_dir(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember_import("_auto_extracted", "fact one\nfact two")
    path = tmp_path / "memory" / "imports" / "_auto_extracted.md"
    assert path.read_text(encoding="utf-8") == "fact one\nfact two"


def test_remember_import_returns_the_path_it_wrote(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    returned = repo.remember_import("_auto_extracted", "content")
    assert returned == tmp_path / "memory" / "imports" / "_auto_extracted.md"


def test_remember_import_overwrites_existing(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember_import("_auto_extracted", "first")
    repo.remember_import("_auto_extracted", "second")
    assert repo.load_import("_auto_extracted") == "second"


def test_load_import_returns_none_for_missing_note(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    assert repo.load_import("does-not-exist") is None


def test_list_import_keys_returns_sorted_import_keys(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember_import("zeta", "z")
    repo.remember_import("alpha", "a")
    assert repo.list_import_keys() == ["alpha", "zeta"]


def test_list_import_keys_excludes_topics_and_daily_notes(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember("topic-note", "content")
    repo.remember_import("import-note", "content")
    repo.append_daily("daily entry")
    assert repo.list_import_keys() == ["import-note"]


# ---------------------------------------------------------------------------
# append_daily
# ---------------------------------------------------------------------------

def test_append_daily_creates_file_with_header(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    when = date(2026, 7, 20)
    path = repo.append_daily("did a thing", when=when)

    assert path == tmp_path / "memory" / "2026-07-20.md"
    content = path.read_text(encoding="utf-8")
    assert content.startswith("## 2026-07-20")
    assert "did a thing" in content


def test_append_daily_second_call_same_day_does_not_repeat_header(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    when = date(2026, 7, 20)
    repo.append_daily("first entry", when=when)
    path = repo.append_daily("second entry", when=when)

    content = path.read_text(encoding="utf-8")
    assert content.count("## 2026-07-20") == 1
    assert "first entry" in content
    assert "second entry" in content


def test_append_daily_entries_are_timestamped_bullets(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    path = repo.append_daily("some text", when=date(2026, 7, 20))
    content = path.read_text(encoding="utf-8")
    # "- HH:MM: some text" — just check the bullet/colon shape, not the exact clock time.
    assert "- " in content
    assert ": some text" in content


def test_append_daily_defaults_to_today(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    path = repo.append_daily("entry")
    assert path.name == f"{date.today().isoformat()}.md"


def test_append_daily_different_days_use_different_files(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    p1 = repo.append_daily("entry one", when=date(2026, 7, 20))
    p2 = repo.append_daily("entry two", when=date(2026, 7, 21))
    assert p1 != p2
    assert "entry one" not in p2.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def test_search_finds_topic_note_and_tags_source(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember("api-notes", "REST API best practices")
    [hit] = repo.search("REST")
    assert hit.key == "api-notes"
    assert hit.source == "topic"


def test_search_finds_import_note_and_tags_source(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    (tmp_path / "memory" / "imports" / "wiki-note.md").write_text(
        "rate limit is 100 requests per minute", encoding="utf-8"
    )
    [hit] = repo.search("rate limit")
    assert hit.key == "wiki-note"
    assert hit.source == "import"


def test_search_finds_daily_note_and_tags_source(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.append_daily("debugging the payment webhook", when=date(2026, 7, 20))
    [hit] = repo.search("webhook")
    assert hit.key == "2026-07-20"
    assert hit.source == "daily"


def test_search_ignores_terms_shorter_than_three_chars(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember("note", "a very short note")
    # "is", "a" are both < 3 chars and should be filtered as stop words —
    # nothing else in the query means no match at all.
    assert repo.search("is a") == []


def test_search_returns_empty_list_when_nothing_matches(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember("note", "unrelated content")
    assert repo.search("xyzzy") == []


def test_search_ranks_more_matching_terms_higher(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember("weak-match", "python")
    repo.remember("strong-match", "python django rest")
    results = repo.search("python django rest")
    assert [hit.key for hit in results][0] == "strong-match"


def test_search_respects_max_results(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    for i in range(5):
        repo.remember(f"note-{i}", "matching keyword")
    results = repo.search("matching", max_results=2)
    assert len(results) == 2


def test_search_exclude_sources_omits_import_notes(tmp_path):
    # MEM-GAP-004: exclude_sources lets a caller keep unreviewed imports out
    # of a corpus-agnostic search entirely.
    repo = MemoryFileRepository(tmp_path)
    (tmp_path / "memory" / "imports" / "wiki-note.md").write_text(
        "rate limit is 100 requests per minute", encoding="utf-8"
    )
    assert repo.search("rate limit", exclude_sources=frozenset({"import"})) == []


def test_search_exclude_sources_does_not_affect_other_sources(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember("api-notes", "rate limit best practices")
    [hit] = repo.search("rate limit", exclude_sources=frozenset({"import"}))
    assert hit.key == "api-notes"


def test_search_exclude_sources_does_not_let_an_excluded_hit_crowd_out_max_results(tmp_path):
    # An excluded import must not consume one of max_results' slots even
    # though it would otherwise rank first (exact match vs. partial).
    repo = MemoryFileRepository(tmp_path)
    (tmp_path / "memory" / "imports" / "best-match.md").write_text(
        "python django rest python django rest", encoding="utf-8"
    )
    repo.remember("other-match", "python")
    results = repo.search("python django rest", max_results=1, exclude_sources=frozenset({"import"}))
    assert [hit.key for hit in results] == ["other-match"]


# ---------------------------------------------------------------------------
# search — MEM-GAP-008: root MEMORY.md and boundary parity with the indexed path
# ---------------------------------------------------------------------------

def test_search_finds_root_memory_md_and_tags_it_topic(tmp_path):
    (tmp_path / "MEMORY.md").write_text("The user prefers dark mode.", encoding="utf-8")
    repo = MemoryFileRepository(tmp_path)

    [hit] = repo.search("dark mode")

    assert hit.key == "MEMORY"
    assert hit.source == "topic"


def test_search_exclude_sources_topic_also_omits_root_memory_md(tmp_path):
    (tmp_path / "MEMORY.md").write_text("The user prefers dark mode.", encoding="utf-8")
    repo = MemoryFileRepository(tmp_path)

    assert repo.search("dark mode", exclude_sources=frozenset({"topic"})) == []


def test_search_root_memory_md_absent_is_not_an_error(tmp_path):
    # No MEMORY.md file at all — must not raise, just contribute no candidate.
    repo = MemoryFileRepository(tmp_path)
    repo.remember("note", "matching keyword")

    [hit] = repo.search("matching")

    assert hit.key == "note"


def test_search_excludes_a_note_outside_its_boundary_window(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember(
        "future-plan",
        "---\nsafe_after: 2999-01-01\n---\nSome content mentioning goals.",
    )

    assert repo.search("goals") == []


def test_search_excludes_an_expired_note(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember(
        "old-plan",
        "---\nexpires_at: 2000-01-01\n---\nSome content mentioning goals.",
    )

    assert repo.search("goals") == []


def test_search_includes_an_active_boundary_note_with_annotation(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember(
        "active-plan",
        "---\nowner: main\n---\nSome content mentioning goals.",
    )

    [hit] = repo.search("goals")

    assert hit.boundary is not None
    assert "Owner: main" in hit.boundary


def test_search_strips_frontmatter_from_returned_content(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember(
        "active-plan",
        "---\nowner: main\n---\nSome content mentioning goals.",
    )

    [hit] = repo.search("goals")

    assert "owner: main" not in hit.content
    assert "---" not in hit.content
    assert "Some content mentioning goals." in hit.content


def test_search_does_not_match_against_frontmatter_text(tmp_path):
    # A term that only appears inside the frontmatter block (not the body)
    # must not produce a match — matching is against the stripped body.
    repo = MemoryFileRepository(tmp_path)
    repo.remember(
        "some-note",
        "---\napplies_when: deploying to production\n---\nUnrelated body text.",
    )

    assert repo.search("deploying") == []


def test_search_a_note_without_frontmatter_has_no_boundary(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember("plain-note", "Some content mentioning goals.")

    [hit] = repo.search("goals")

    assert hit.boundary is None


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------

def test_resolve_path_accepts_relative_path_inside_root(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    (tmp_path / "MEMORY.md").write_text("x", encoding="utf-8")
    resolved = repo.resolve_path("MEMORY.md")
    assert resolved == (tmp_path / "MEMORY.md").resolve()


def test_resolve_path_accepts_nested_relative_path(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember("note", "content")
    resolved = repo.resolve_path("memory/topics/note.md")
    assert resolved == (tmp_path / "memory" / "topics" / "note.md").resolve()


def test_resolve_path_accepts_absolute_path_inside_root(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    abs_path = str((tmp_path / "MEMORY.md").resolve())
    resolved = repo.resolve_path(abs_path)
    assert resolved == (tmp_path / "MEMORY.md").resolve()


def test_resolve_path_rejects_traversal_outside_root(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    with pytest.raises(ValueError, match="outside the memory root"):
        repo.resolve_path("../../etc/passwd")


def test_resolve_path_rejects_absolute_path_outside_root(tmp_path):
    other_dir = tmp_path.parent / "unrelated-dir"
    other_dir.mkdir(exist_ok=True)
    repo = MemoryFileRepository(tmp_path)
    with pytest.raises(ValueError, match="outside the memory root"):
        repo.resolve_path(str(other_dir / "secret.md"))


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

def _make_locator(repo: MemoryFileRepository, rel_path: str, **kwargs) -> MemoryLocator:
    return MemoryLocator(path=repo.resolve_path(rel_path), **kwargs)


def test_get_returns_whole_file_by_default(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember("note", "line1\nline2\nline3")
    excerpt = repo.get(_make_locator(repo, "memory/topics/note.md"))
    assert excerpt.text == "line1\nline2\nline3"
    assert excerpt.start_line == 1
    assert excerpt.end_line == 3
    assert excerpt.total_lines == 3


def test_get_respects_from_line(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember("note", "line1\nline2\nline3")
    excerpt = repo.get(_make_locator(repo, "memory/topics/note.md", from_line=2))
    assert excerpt.text == "line2\nline3"
    assert excerpt.start_line == 2
    assert excerpt.end_line == 3


def test_get_respects_lines_cap(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember("note", "line1\nline2\nline3\nline4")
    excerpt = repo.get(_make_locator(repo, "memory/topics/note.md", from_line=2, lines=2))
    assert excerpt.text == "line2\nline3"
    assert excerpt.start_line == 2
    assert excerpt.end_line == 3
    assert excerpt.total_lines == 4


def test_get_clamps_from_line_beyond_end_of_file(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember("note", "line1\nline2")
    excerpt = repo.get(_make_locator(repo, "memory/topics/note.md", from_line=100))
    assert excerpt.start_line == 2  # clamped to the last line, not empty
    assert excerpt.text == "line2"


def test_get_handles_empty_file(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    repo.remember("note", "")
    excerpt = repo.get(_make_locator(repo, "memory/topics/note.md"))
    assert excerpt == excerpt  # sanity — construct without error
    assert excerpt.total_lines == 0
    assert excerpt.start_line == 0
    assert excerpt.end_line == 0
    assert excerpt.text == ""


def test_get_raises_for_missing_file(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    locator = MemoryLocator(path=tmp_path / "memory" / "topics" / "missing.md")
    with pytest.raises(FileNotFoundError):
        repo.get(locator)


# ---------------------------------------------------------------------------
# list_indexable_files — Stage One Phase 3, slice A
# ---------------------------------------------------------------------------

def test_list_indexable_files_covers_every_source(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    (tmp_path / "MEMORY.md").write_text("durable root memory", encoding="utf-8")
    repo.remember("project-goals", "topic content")
    repo.remember_import("_auto_extracted", "import content")
    repo.append_daily("daily entry", when=date(2026, 7, 20))

    files = repo.list_indexable_files()

    by_path = {rel: (kind, content) for kind, rel, content in files}
    assert by_path["MEMORY.md"] == ("durable", "durable root memory")
    assert by_path["memory/topics/project-goals.md"] == ("durable", "topic content")
    assert by_path["memory/imports/_auto_extracted.md"] == ("import", "import content")
    assert by_path["memory/2026-07-20.md"][0] == "daily"


def test_list_indexable_files_excludes_dreams_and_user_and_missing_memory_md(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    (tmp_path / "DREAMS.md").write_text("dream content", encoding="utf-8")
    (tmp_path / "USER.md").write_text("user content", encoding="utf-8")

    files = repo.list_indexable_files()

    assert files == []  # no MEMORY.md, no topics/imports/daily notes yet


def test_list_indexable_files_returns_empty_list_for_a_fresh_repository(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    assert repo.list_indexable_files() == []


# ---------------------------------------------------------------------------
# Concurrency (MEM-GAP-009)
# ---------------------------------------------------------------------------

def test_lock_for_returns_the_same_lock_object_for_the_same_path(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    path = repo.topic_path("project-goals")

    first = repo._lock_for(path)
    second = repo._lock_for(path)

    assert first is second


def test_lock_for_returns_different_locks_for_different_paths(tmp_path):
    repo = MemoryFileRepository(tmp_path)

    a = repo._lock_for(repo.topic_path("a"))
    b = repo._lock_for(repo.topic_path("b"))

    assert a is not b


def test_concurrent_append_daily_loses_no_entries(tmp_path):
    """20 threads each append one entry to the same daily note concurrently.

    Before the per-path lock, append_daily()'s read-modify-write was not
    atomic with respect to other writers to the same file — two threads
    could both read the same "before" content and one append would
    silently overwrite the other's. With the lock, every entry must
    survive regardless of interleaving.
    """
    repo = MemoryFileRepository(tmp_path)
    day = date(2026, 7, 20)
    thread_count = 20
    barrier = threading.Barrier(thread_count)

    def _append(i: int) -> None:
        barrier.wait()  # maximize actual concurrent overlap
        repo.append_daily(f"entry-{i}", when=day)

    threads = [threading.Thread(target=_append, args=(i,)) for i in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    content = repo._memory_dir / "2026-07-20.md"
    text = content.read_text(encoding="utf-8")
    for i in range(thread_count):
        assert f"entry-{i}" in text
    # One bullet line per entry, no lost/merged/corrupted lines.
    bullet_lines = [ln for ln in text.splitlines() if ln.startswith("- ")]
    assert len(bullet_lines) == thread_count


def test_concurrent_remember_to_different_keys_does_not_interfere(tmp_path):
    repo = MemoryFileRepository(tmp_path)
    thread_count = 10
    barrier = threading.Barrier(thread_count)

    def _remember(i: int) -> None:
        barrier.wait()
        repo.remember(f"topic-{i}", f"content-{i}")

    threads = [threading.Thread(target=_remember, args=(i,)) for i in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i in range(thread_count):
        assert repo.load(f"topic-{i}") == f"content-{i}"

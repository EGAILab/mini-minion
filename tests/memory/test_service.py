"""Tests for memory/service.py: MemoryService (Stage One Phase 1, slice 2).

MemoryService is a thin facade over MemoryFileRepository — these tests
mostly verify delegation is correct and the get()/status() conveniences
work, rather than re-testing MemoryFileRepository's own behavior in detail
(that's tests/memory/test_files.py's job).
"""

from __future__ import annotations

from datetime import date
from unittest.mock import Mock

import pytest

from minion_assist.memory.files import MemoryFileRepository
from minion_assist.memory.service import MemoryService


@pytest.fixture
def service(tmp_path):
    return MemoryService(MemoryFileRepository(tmp_path))


@pytest.fixture
def indexed_service(tmp_path):
    """A MemoryService wired to a mock index — Stage One Phase 3, slice B."""
    mock_index = Mock()
    svc = MemoryService(MemoryFileRepository(tmp_path), index=mock_index, agent_id="main")
    return svc, mock_index


# ---------------------------------------------------------------------------
# Explicit notes
# ---------------------------------------------------------------------------

def test_remember_and_load_round_trip(service):
    service.remember("project-goals", "# Goals\nShip Phase 1.")
    assert service.load("project-goals") == "# Goals\nShip Phase 1."


def test_load_returns_none_for_missing_note(service):
    assert service.load("does-not-exist") is None


def test_delete_removes_existing_note(service):
    service.remember("note", "content")
    assert service.delete("note") is True
    assert service.load("note") is None


def test_delete_returns_false_for_missing_note(service):
    assert service.delete("does-not-exist") is False


def test_list_keys_returns_sorted_keys(service):
    service.remember("zeta", "z")
    service.remember("alpha", "a")
    assert service.list_keys() == ["alpha", "zeta"]


# ---------------------------------------------------------------------------
# Quarantined notes (remember_import / load_import / list_import_keys)
# ---------------------------------------------------------------------------

def test_remember_import_and_load_import_round_trip(service):
    service.remember_import("_auto_extracted", "fact one\nfact two")
    assert service.load_import("_auto_extracted") == "fact one\nfact two"


def test_load_import_returns_none_for_missing_note(service):
    assert service.load_import("does-not-exist") is None


def test_list_import_keys_returns_sorted_keys(service):
    service.remember_import("zeta", "z")
    service.remember_import("alpha", "a")
    assert service.list_import_keys() == ["alpha", "zeta"]


def test_import_notes_are_separate_from_topic_notes(service):
    service.remember("topic-note", "content")
    service.remember_import("import-note", "content")
    assert service.list_keys() == ["topic-note"]
    assert service.list_import_keys() == ["import-note"]


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def test_search_returns_memory_hits(service):
    service.remember("api-notes", "REST API best practices")
    [hit] = service.search("REST")
    assert hit.key == "api-notes"
    assert hit.source == "topic"


def test_search_respects_max_results(service):
    for i in range(5):
        service.remember(f"note-{i}", "matching keyword")
    assert len(service.search("matching", max_results=2)) == 2


def test_search_returns_empty_list_when_nothing_matches(service):
    service.remember("note", "unrelated content")
    assert service.search("xyzzy") == []


def test_search_with_corpus_filters_linear_scan_results_by_legacy_source(service):
    service.remember("topic-note", "shared keyword")
    service.remember_import("import-note", "shared keyword")

    durable_only = service.search("shared keyword", corpus="durable")

    assert len(durable_only) == 1
    assert durable_only[0].key == "topic-note"


# ---------------------------------------------------------------------------
# search — with a configured lexical index (Stage One Phase 3, slice C)
# ---------------------------------------------------------------------------

def _index_row(**overrides):
    row = {
        "rel_path": "MEMORY.md", "source_kind": "durable", "chunk_index": 0,
        "heading_path": "", "content": "matched content", "start_line": 1,
        "end_line": 3, "score": 0.5,
    }
    row.update(overrides)
    return row


def test_search_uses_the_index_when_configured(indexed_service):
    svc, mock_index = indexed_service
    mock_index.search.return_value = [_index_row()]

    [hit] = svc.search("query")

    mock_index.search.assert_called_once_with("main", "query", corpus=None, max_results=20)
    assert hit.key == "MEMORY"
    assert hit.content == "matched content"
    assert hit.source == "durable"
    assert hit.rel_path == "MEMORY.md"
    assert hit.start_line == 1
    assert hit.end_line == 3
    assert hit.score == 0.5


def test_search_passes_corpus_through_to_the_index(indexed_service):
    svc, mock_index = indexed_service
    mock_index.search.return_value = []

    svc.search("query", corpus="daily")

    mock_index.search.assert_called_once_with("main", "query", corpus="daily", max_results=20)


def test_search_falls_back_to_linear_scan_when_index_search_raises(indexed_service, tmp_path):
    svc, mock_index = indexed_service
    mock_index.search.side_effect = RuntimeError("connection lost")
    svc.remember("project-goals", "fallback content")

    [hit] = svc.search("fallback")

    assert hit.key == "project-goals"
    assert hit.rel_path is None  # a plain linear-scan MemoryHit, not an index one


# ---------------------------------------------------------------------------
# append_daily
# ---------------------------------------------------------------------------

def test_append_daily_writes_to_dated_file(service, tmp_path):
    path = service.append_daily("did a thing", when=date(2026, 7, 20))
    assert path == tmp_path / "memory" / "2026-07-20.md"
    assert "did a thing" in path.read_text(encoding="utf-8")


def test_append_daily_defaults_to_today(service):
    path = service.append_daily("entry")
    assert path.name == f"{date.today().isoformat()}.md"


# ---------------------------------------------------------------------------
# get — path-string convenience over MemoryFileRepository.get/resolve_path
# ---------------------------------------------------------------------------

def test_get_reads_whole_file_by_default(service):
    service.remember("note", "line1\nline2\nline3")
    excerpt = service.get("memory/topics/note.md")
    assert excerpt.text == "line1\nline2\nline3"
    assert excerpt.total_lines == 3


def test_get_respects_from_line_and_lines(service):
    service.remember("note", "line1\nline2\nline3\nline4")
    excerpt = service.get("memory/topics/note.md", from_line=2, lines=2)
    assert excerpt.text == "line2\nline3"
    assert excerpt.start_line == 2
    assert excerpt.end_line == 3


def test_get_raises_value_error_for_path_outside_root(service):
    with pytest.raises(ValueError, match="outside the memory root"):
        service.get("../../etc/passwd")


def test_get_raises_file_not_found_for_missing_file(service):
    with pytest.raises(FileNotFoundError):
        service.get("memory/topics/missing.md")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def test_status_reports_zero_counts_for_empty_store(service, tmp_path):
    status = service.status()
    assert status.root == tmp_path.resolve()
    assert status.topic_count == 0
    assert status.import_count == 0
    assert status.daily_count == 0


def test_status_reports_counts_across_all_sources(service, tmp_path):
    service.remember("topic-note", "content")
    (tmp_path / "memory" / "imports" / "imported.md").write_text("x", encoding="utf-8")
    service.append_daily("daily entry")

    status = service.status()
    assert status.topic_count == 1
    assert status.import_count == 1
    assert status.daily_count == 1


# ---------------------------------------------------------------------------
# deep_status / force_reindex (Stage One Phase 3, slice C)
# ---------------------------------------------------------------------------

def test_deep_status_returns_none_without_an_index(service):
    assert service.deep_status() is None


def test_deep_status_delegates_to_the_index(indexed_service):
    svc, mock_index = indexed_service
    mock_index.index_summary.return_value = {
        "total_chunks": 3, "file_count": 2, "by_corpus": {"durable": 3}, "last_indexed_at": 1.0
    }

    result = svc.deep_status()

    mock_index.index_summary.assert_called_once_with("main")
    assert result["total_chunks"] == 3


def test_force_reindex_raises_without_an_index(service):
    with pytest.raises(RuntimeError, match="No lexical index configured"):
        service.force_reindex()


def test_force_reindex_delegates_to_the_index_with_the_current_file_listing(indexed_service):
    svc, mock_index = indexed_service
    svc.remember("project-goals", "content")  # so list_indexable_files() has something
    mock_index.force_rebuild_agent.return_value = 5

    result = svc.force_reindex()

    assert result == 5
    mock_index.force_rebuild_agent.assert_called_once()
    call_args = mock_index.force_rebuild_agent.call_args.args
    assert call_args[0] == "main"
    assert ("durable", "memory/topics/project-goals.md", "content") in call_args[1]


# ---------------------------------------------------------------------------
# flush_head (Stage One Phase 2, slice B)
# ---------------------------------------------------------------------------

def test_flush_head_empty_list_returns_empty_status(service):
    outcome = service.flush_head([])
    assert outcome.status == "empty"


def test_flush_head_writes_to_daily_note(service, tmp_path):
    outcome = service.flush_head([{"role": "user", "content": "important context"}])

    assert outcome.status == "flushed"
    today = date.today().isoformat()
    content = (tmp_path / "memory" / f"{today}.md").read_text(encoding="utf-8")
    assert "important context" in content
    assert "[Pre-compaction checkpoint]" in content


def test_flush_head_blank_content_returns_empty_status(service):
    """A message with no renderable content (e.g. blank) counts as nothing to flush."""
    outcome = service.flush_head([{"role": "user", "content": ""}])
    assert outcome.status == "empty"


def test_flush_head_never_raises_on_write_failure(service, monkeypatch):
    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(service._files, "append_daily", _boom)

    outcome = service.flush_head([{"role": "user", "content": "hi"}])

    assert outcome.status == "failed"
    assert "disk full" in outcome.detail


def test_flush_head_multiple_messages_all_included(service, tmp_path):
    service.flush_head([
        {"role": "user", "content": "first message"},
        {"role": "assistant", "content": "second message"},
    ])

    today = date.today().isoformat()
    content = (tmp_path / "memory" / f"{today}.md").read_text(encoding="utf-8")
    assert "first message" in content
    assert "second message" in content


# ---------------------------------------------------------------------------
# Write-path index sync (Stage One Phase 3, slice B)
# ---------------------------------------------------------------------------

def test_remember_without_index_never_touches_index(service):
    # No index configured — must behave exactly like before this slice
    # (nothing to assert on an index that doesn't exist; this just
    # documents that remember() still works with no index/agent_id).
    service.remember("project-goals", "content")
    assert service.load("project-goals") == "content"


def test_remember_reindexes_the_written_file(indexed_service):
    svc, mock_index = indexed_service
    svc.remember("project-goals", "# Goals\nShip it.")

    mock_index.reindex_file.assert_called_once_with(
        "main", "memory/topics/project-goals.md", "durable", "# Goals\nShip it."
    )


def test_remember_import_reindexes_as_the_import_corpus(indexed_service):
    svc, mock_index = indexed_service
    svc.remember_import("_auto_extracted", "fact one")

    mock_index.reindex_file.assert_called_once_with(
        "main", "memory/imports/_auto_extracted.md", "import", "fact one"
    )


def test_delete_removes_from_index_only_when_a_file_was_actually_deleted(indexed_service):
    svc, mock_index = indexed_service
    svc.remember("project-goals", "content")
    mock_index.reset_mock()

    deleted = svc.delete("project-goals")

    assert deleted is True
    mock_index.remove_file.assert_called_once_with("main", "memory/topics/project-goals.md")


def test_delete_of_nonexistent_key_does_not_call_the_index(indexed_service):
    svc, mock_index = indexed_service
    deleted = svc.delete("never-existed")

    assert deleted is False
    mock_index.remove_file.assert_not_called()


def test_append_daily_reindexes_with_the_files_full_current_content(indexed_service):
    svc, mock_index = indexed_service
    svc.append_daily("first entry", when=date(2026, 7, 20))
    mock_index.reset_mock()

    svc.append_daily("second entry", when=date(2026, 7, 20))

    args = mock_index.reindex_file.call_args.args
    assert args[0] == "main"
    assert args[1] == "memory/2026-07-20.md"
    assert args[2] == "daily"
    assert "first entry" in args[3]  # full file content, not just the new entry
    assert "second entry" in args[3]


def test_index_sync_failure_never_raises_out_of_remember(indexed_service):
    svc, mock_index = indexed_service
    mock_index.reindex_file.side_effect = RuntimeError("db unavailable")

    svc.remember("project-goals", "content")  # must not raise

    assert svc.load("project-goals") == "content"  # the actual write still succeeded


def test_index_ignored_when_agent_id_is_missing(tmp_path):
    mock_index = Mock()
    svc = MemoryService(MemoryFileRepository(tmp_path), index=mock_index)  # no agent_id

    svc.remember("project-goals", "content")

    mock_index.reindex_file.assert_not_called()

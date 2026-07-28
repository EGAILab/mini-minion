"""Tests for memory/service.py: MemoryService (Stage One Phase 1, slice 2).

MemoryService is a thin facade over MemoryFileRepository — these tests
mostly verify delegation is correct and the get()/status() conveniences
work, rather than re-testing MemoryFileRepository's own behavior in detail
(that's tests/memory/test_files.py's job).
"""

from __future__ import annotations

from datetime import date

import pytest

from minion_assist.memory.files import MemoryFileRepository
from minion_assist.memory.service import MemoryService


@pytest.fixture
def service(tmp_path):
    return MemoryService(MemoryFileRepository(tmp_path))


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

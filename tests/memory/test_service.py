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
    """A MemoryService wired to a mock index — Stage One Phase 3, slice B.

    ``get_boundary`` defaults to ``None`` (Stage One Phase 6, slice A) —
    "no boundary metadata" — so every existing test here that doesn't care
    about boundaries isn't tripped up by an unconfigured Mock call
    returning a truthy-but-meaningless Mock object.
    """
    mock_index = Mock()
    mock_index.get_boundary.return_value = None
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


# ---------------------------------------------------------------------------
# root — exposes the underlying repository's workspace root
# ---------------------------------------------------------------------------

def test_root_returns_repository_root(tmp_path, service):
    assert service.root == tmp_path.resolve()


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


def test_search_excludes_imports_by_default_in_the_linear_scan(service):
    # MEM-GAP-004: a corpus-agnostic search (the default — used for
    # automatic per-turn injection) must not surface unreviewed imports,
    # even in the no-index local/basic fallback path.
    service.remember("topic-note", "shared keyword")
    service.remember_import("import-note", "shared keyword")

    results = service.search("shared keyword")

    assert [hit.key for hit in results] == ["topic-note"]


def test_search_with_corpus_import_still_finds_import_notes_in_the_linear_scan(service):
    # Explicit review access to quarantined content must still work.
    service.remember_import("import-note", "shared keyword")

    results = service.search("shared keyword", corpus="import")

    assert [hit.key for hit in results] == ["import-note"]


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
    mock_index.hybrid_search.return_value = [_index_row()]

    [hit] = svc.search("query")

    mock_index.hybrid_search.assert_called_once_with("main", "query", corpus=None, max_results=20)
    assert hit.key == "MEMORY"
    assert hit.content == "matched content"
    assert hit.source == "durable"
    assert hit.rel_path == "MEMORY.md"
    assert hit.start_line == 1
    assert hit.end_line == 3
    assert hit.score == 0.5


# ---------------------------------------------------------------------------
# Action-boundary annotation and filtering (Stage One Phase 6, slice A)
# ---------------------------------------------------------------------------

def _iso(epoch: float) -> str:
    import datetime as _dt

    return _dt.datetime.fromtimestamp(epoch).isoformat()


def test_search_attaches_boundary_annotation_when_the_note_has_active_metadata(indexed_service):
    svc, mock_index = indexed_service
    mock_index.hybrid_search.return_value = [_index_row()]
    mock_index.get_boundary.return_value = {"owner": "main"}

    [hit] = svc.search("query")

    assert hit.boundary is not None
    assert "Owner: main" in hit.boundary


def test_search_leaves_boundary_none_when_the_note_has_no_metadata(indexed_service):
    svc, mock_index = indexed_service
    mock_index.hybrid_search.return_value = [_index_row()]
    mock_index.get_boundary.return_value = None

    [hit] = svc.search("query")

    assert hit.boundary is None


def test_search_excludes_a_hit_whose_boundary_has_expired(indexed_service):
    import time as _time

    svc, mock_index = indexed_service
    mock_index.hybrid_search.return_value = [_index_row()]
    mock_index.get_boundary.return_value = {"expires_at": _iso(_time.time() - 3600)}

    results = svc.search("query")

    assert results == []


def test_search_excludes_a_hit_that_is_not_yet_safe(indexed_service):
    import time as _time

    svc, mock_index = indexed_service
    mock_index.hybrid_search.return_value = [_index_row()]
    mock_index.get_boundary.return_value = {"safe_after": _iso(_time.time() + 3600)}

    results = svc.search("query")

    assert results == []


def test_search_includes_a_hit_within_its_active_boundary_window(indexed_service):
    import time as _time

    svc, mock_index = indexed_service
    mock_index.hybrid_search.return_value = [_index_row()]
    mock_index.get_boundary.return_value = {"expires_at": _iso(_time.time() + 3600)}

    results = svc.search("query")

    assert len(results) == 1
    assert results[0].boundary is not None


def test_search_looks_up_boundary_at_most_once_per_unique_rel_path(indexed_service):
    svc, mock_index = indexed_service
    mock_index.hybrid_search.return_value = [
        _index_row(chunk_index=0), _index_row(chunk_index=1),
    ]
    mock_index.get_boundary.return_value = None

    svc.search("query")

    mock_index.get_boundary.assert_called_once_with("main", "MEMORY.md")


def test_search_treats_a_boundary_lookup_failure_as_no_boundary(indexed_service):
    svc, mock_index = indexed_service
    mock_index.hybrid_search.return_value = [_index_row()]
    mock_index.get_boundary.side_effect = RuntimeError("db unavailable")

    [hit] = svc.search("query")  # must not raise

    assert hit.boundary is None


def test_search_has_no_boundary_for_a_note_without_frontmatter_in_degraded_mode(service):
    # `service` (no index configured) exercises the linear-scan fallback.
    # A note with no boundary frontmatter at all naturally has none to find.
    service.remember("project-goals", "Some content mentioning goals.")

    results = service.search("goals")

    assert all(h.boundary is None for h in results)


def test_search_applies_boundaries_in_degraded_mode(service):
    # MEM-GAP-008: boundary frontmatter is parsed locally by
    # MemoryFileRepository.search() itself (memory/boundaries.py has no
    # PostgreSQL dependency), so degraded/no-index mode enforces the same
    # boundary window as the indexed path — not a gap anymore.
    service.remember(
        "future-plan",
        "---\nsafe_after: 2999-01-01\n---\nSome content mentioning goals.",
    )

    results = service.search("goals")

    assert results == []  # not yet safe to surface


def test_search_annotates_an_active_boundary_note_in_degraded_mode(service):
    service.remember(
        "active-plan",
        "---\nowner: main\n---\nSome content mentioning goals.",
    )

    results = service.search("goals")

    assert len(results) == 1
    assert "Owner: main" in results[0].boundary
    # The raw frontmatter block must never leak into displayed content.
    assert "owner: main" not in results[0].content
    assert "---" not in results[0].content


def test_search_passes_corpus_through_to_the_index(indexed_service):
    svc, mock_index = indexed_service
    mock_index.hybrid_search.return_value = []

    svc.search("query", corpus="daily")

    mock_index.hybrid_search.assert_called_once_with(
        "main", "query", corpus="daily", max_results=20
    )


def test_search_falls_back_to_linear_scan_when_index_search_raises(indexed_service, tmp_path):
    svc, mock_index = indexed_service
    mock_index.hybrid_search.side_effect = RuntimeError("connection lost")
    svc.remember("project-goals", "fallback content")

    [hit] = svc.search("fallback")

    assert hit.key == "project-goals"
    assert hit.rel_path is None  # a plain linear-scan MemoryHit, not an index one


# ---------------------------------------------------------------------------
# search — degraded-mode visibility via WorkerHealth (MEM-GAP-008)
# ---------------------------------------------------------------------------

def test_search_records_health_success_on_a_working_index(tmp_path):
    from minion_assist.worker_health import WorkerHealth

    mock_index = Mock()
    mock_index.hybrid_search.return_value = []
    mock_index.get_boundary.return_value = None
    health = WorkerHealth("memory_search:main")
    svc = MemoryService(
        MemoryFileRepository(tmp_path), index=mock_index, agent_id="main", health=health,
    )

    svc.search("query")

    snap = health.snapshot()
    assert snap["last_poll_at"] is not None
    assert snap["last_success_at"] is not None
    assert snap["consecutive_failures"] == 0


def test_search_records_health_failure_when_index_search_raises(tmp_path):
    from minion_assist.worker_health import WorkerHealth

    mock_index = Mock()
    mock_index.hybrid_search.side_effect = RuntimeError("connection lost")
    health = WorkerHealth("memory_search:main")
    svc = MemoryService(
        MemoryFileRepository(tmp_path), index=mock_index, agent_id="main", health=health,
    )

    svc.search("query")  # falls back to linear scan, must not raise

    snap = health.snapshot()
    assert snap["consecutive_failures"] == 1
    assert "connection lost" in snap["last_error"]
    assert snap["last_success_at"] is None


def test_search_without_health_configured_does_not_raise(indexed_service):
    svc, mock_index = indexed_service
    mock_index.hybrid_search.return_value = []

    svc.search("query")  # health=None is the default; must not raise


def test_search_never_touches_health_without_an_index(tmp_path):
    # No index configured at all — nothing to poll/succeed/fail against.
    health = Mock()
    svc = MemoryService(MemoryFileRepository(tmp_path), health=health)

    svc.search("query")

    health.record_poll.assert_not_called()
    health.record_success.assert_not_called()
    health.record_failure.assert_not_called()


# ---------------------------------------------------------------------------
# mark_injected (Stage One Phase 5, slice A)
# ---------------------------------------------------------------------------

def test_mark_injected_is_a_no_op_without_an_index(service):
    service.mark_injected(["MEMORY.md"], "some query")  # must not raise


def test_mark_injected_is_a_no_op_for_an_empty_list(indexed_service):
    svc, mock_index = indexed_service
    svc.mark_injected([], "some query")
    mock_index.mark_injected.assert_not_called()


def test_mark_injected_delegates_to_the_index_with_a_hashed_query(indexed_service):
    from minion_assist.memory.postgres_index import hash_query

    svc, mock_index = indexed_service
    svc.mark_injected(["MEMORY.md", "memory/topics/goal.md"], "some query")

    mock_index.mark_injected.assert_called_once_with(
        "main", ["MEMORY.md", "memory/topics/goal.md"], hash_query("some query")
    )


def test_mark_injected_never_raises_when_the_index_call_fails(indexed_service):
    svc, mock_index = indexed_service
    mock_index.mark_injected.side_effect = RuntimeError("db unavailable")

    svc.mark_injected(["MEMORY.md"], "some query")  # must not raise


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
# pin / unpin / is_pinned / list_pinned (Stage One Phase 4, slice B)
# ---------------------------------------------------------------------------

def test_pin_raises_without_an_index(service):
    service.remember("project-goals", "content")
    with pytest.raises(RuntimeError, match="No lexical index configured"):
        service.pin("project-goals")


def test_pin_raises_for_a_key_with_no_note(indexed_service):
    svc, _mock_index = indexed_service
    with pytest.raises(FileNotFoundError, match="No note found"):
        svc.pin("never-saved")


def test_pin_delegates_to_the_index_with_the_topic_rel_path(indexed_service):
    svc, mock_index = indexed_service
    svc.remember("project-goals", "content")
    mock_index.reset_mock()

    svc.pin("project-goals")

    mock_index.pin_file.assert_called_once_with("main", "memory/topics/project-goals.md")


def test_unpin_raises_without_an_index(service):
    with pytest.raises(RuntimeError, match="No lexical index configured"):
        service.unpin("project-goals")


def test_unpin_delegates_to_the_index_even_for_a_nonexistent_note(indexed_service):
    svc, mock_index = indexed_service

    svc.unpin("never-saved")  # must not raise — clears a possibly-stale pin

    mock_index.unpin_file.assert_called_once_with("main", "memory/topics/never-saved.md")


def test_is_pinned_returns_false_without_an_index(service):
    assert service.is_pinned("project-goals") is False


def test_is_pinned_delegates_to_the_index(indexed_service):
    svc, mock_index = indexed_service
    mock_index.is_pinned.return_value = True

    assert svc.is_pinned("project-goals") is True
    mock_index.is_pinned.assert_called_once_with("main", "memory/topics/project-goals.md")


def test_list_pinned_returns_empty_list_without_an_index(service):
    assert service.list_pinned() == []


def test_list_pinned_maps_rel_paths_back_to_keys(indexed_service):
    svc, mock_index = indexed_service
    mock_index.pinned_files.return_value = [
        "memory/topics/b.md", "memory/topics/a.md", "MEMORY.md",
    ]

    assert svc.list_pinned() == ["b", "a", "MEMORY"]


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


# ---------------------------------------------------------------------------
# Revision history for explicit writes (MEM-GAP-020)
# ---------------------------------------------------------------------------

def test_remember_records_a_revision_of_the_prior_content(indexed_service):
    svc, mock_index = indexed_service
    svc.remember("project-goals", "Ship it this quarter.")
    mock_index.reset_mock()

    svc.remember("project-goals", "Delayed to next quarter.")

    mock_index.record_topic_revision.assert_called_once_with(
        "main", "project-goals", None, "Ship it this quarter."
    )


def test_remember_records_empty_prior_content_for_a_brand_new_note(indexed_service):
    svc, mock_index = indexed_service

    svc.remember("brand-new", "First content.")

    mock_index.record_topic_revision.assert_called_once_with("main", "brand-new", None, "")


def test_delete_records_a_revision_of_the_deleted_content(indexed_service):
    svc, mock_index = indexed_service
    svc.remember("project-goals", "Ship it this quarter.")
    mock_index.reset_mock()

    svc.delete("project-goals")

    mock_index.record_topic_revision.assert_called_once_with(
        "main", "project-goals", None, "Ship it this quarter."
    )


def test_delete_of_nonexistent_key_does_not_record_a_revision(indexed_service):
    svc, mock_index = indexed_service

    svc.delete("never-existed")

    mock_index.record_topic_revision.assert_not_called()


def test_remember_without_an_index_never_records_a_revision(service):
    # No index configured — nothing to record into, and it must not raise.
    service.remember("project-goals", "content")  # must not raise


def test_revision_recording_failure_never_raises_out_of_remember(indexed_service):
    svc, mock_index = indexed_service
    mock_index.record_topic_revision.side_effect = RuntimeError("db unavailable")

    svc.remember("project-goals", "content")  # must not raise

    assert svc.load("project-goals") == "content"  # the actual write still succeeded


def test_revision_recording_failure_never_raises_out_of_delete(indexed_service):
    svc, mock_index = indexed_service
    svc.remember("project-goals", "content")
    mock_index.record_topic_revision.side_effect = RuntimeError("db unavailable")

    deleted = svc.delete("project-goals")  # must not raise

    assert deleted is True  # the actual deletion still succeeded


def test_remember_import_reindexes_as_the_import_corpus(indexed_service):
    svc, mock_index = indexed_service
    svc.remember_import("_auto_extracted", "fact one")

    mock_index.reindex_file.assert_called_once_with(
        "main", "memory/imports/_auto_extracted.md", "import", "fact one"
    )


def test_forget_proposals_without_an_index_is_a_no_op(service):
    result = service.forget_proposals([1, 2])

    assert result == {"proposal_ids": [1, 2], "forget_results": []}


def test_forget_proposals_with_an_empty_list_never_touches_the_index(indexed_service):
    svc, mock_index = indexed_service

    result = svc.forget_proposals([])

    mock_index.remove_proposal.assert_not_called()
    assert result == {"proposal_ids": [], "forget_results": []}


def test_forget_proposals_removes_each_proposals_chunk_and_previews(indexed_service):
    svc, mock_index = indexed_service
    mock_index.list_claims_citing_evidence.return_value = []  # forget_source no-op

    svc.forget_proposals([10, 11])

    assert mock_index.remove_proposal.call_args_list == [
        (("main", 10),), (("main", 11),)
    ]
    assert mock_index.remove_consolidation_previews_for_proposal.call_args_list == [
        (("main", 10),), (("main", 11),)
    ]


def test_forget_proposals_forgets_each_id_as_an_evidence_source(indexed_service):
    svc, mock_index = indexed_service
    mock_index.list_claims_citing_evidence.return_value = []

    result = svc.forget_proposals([10, 11])

    assert [r["source_kind"] for r in result["forget_results"]] == ["proposal", "proposal"]
    assert [r["source_ref"] for r in result["forget_results"]] == ["10", "11"]
    # list_claims_citing_evidence is forget_source's actual lookup call —
    # this proves the real cascade ran, not just a stub.
    assert mock_index.list_claims_citing_evidence.call_args_list == [
        (("main", "proposal", "10"),), (("main", "proposal", "11"),)
    ]


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

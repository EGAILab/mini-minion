"""Tests for memory/import_review.py: ImportReviewer preview/approve/reject
(Stage One Phase 7, slice E).

Same testing shape as tests/memory/test_consolidation.py: a fake
PostgresMemoryIndex (no live database needed) and a real, tmp_path-backed
MemoryFileRepository so "never writes to disk before approve" and "writes
exactly the expected content" can be verified for real.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from minion_assist.memory.files import MemoryFileRepository
from minion_assist.memory.import_review import (
    ImportReviewer,
    StaleImportError,
    _hash_text,
    format_import_preview_report,
    is_import_preview_stale,
)
from minion_assist.providers.base import LLMResponse


class _FakeIndex:
    """Fakes just the PostgresMemoryIndex methods ImportReviewer calls."""

    def __init__(
        self,
        hits: list[dict] | None = None,
        claims_by_rel_path: dict[str, list[dict]] | None = None,
    ):
        self._hits = hits or []
        self._claims_by_rel_path = claims_by_rel_path or {}
        self.recorded_previews: list[dict] = []
        self.reindexed_files: list[tuple] = []
        self.removed_files: list[tuple] = []

    def hybrid_search(self, agent_id: str, query: str, *, corpus: str | None = None,
                       max_results: int = 20) -> list[dict]:
        return self._hits

    def list_claims(self, agent_id: str, rel_path: str | None = None, status=None) -> list[dict]:
        return self._claims_by_rel_path.get(rel_path, [])

    def record_import_preview(
        self, agent_id, import_key, target_kind, target_key,
        based_on_content_hash, drafted_content, rationale,
    ) -> int:
        preview_id = len(self.recorded_previews) + 1
        self.recorded_previews.append({
            "id": preview_id, "agent_id": agent_id, "import_key": import_key,
            "target_kind": target_kind, "target_key": target_key,
            "based_on_content_hash": based_on_content_hash,
            "drafted_content": drafted_content, "rationale": rationale,
        })
        return preview_id

    def get_import_preview(self, preview_id: int) -> dict | None:
        for p in self.recorded_previews:
            if p["id"] == preview_id:
                return dict(p)
        return None

    def reindex_file(self, agent_id: str, rel_path: str, source_kind: str, content: str) -> int:
        self.reindexed_files.append((agent_id, rel_path, source_kind, content))
        return 1

    def remove_file(self, agent_id: str, rel_path: str) -> None:
        self.removed_files.append((agent_id, rel_path))


def _draft_response(key: str = "some-key", rationale: str = "Because reasons.",
                     content: str = "Drafted content.") -> Mock:
    provider = Mock()
    provider.chat = Mock(
        return_value=LLMResponse(
            text=f"KEY: {key}\nRATIONALE: {rationale}\n---\n{content}", finish_reason="stop"
        )
    )
    return provider


@pytest.fixture
def files(tmp_path: Path) -> MemoryFileRepository:
    return MemoryFileRepository(tmp_path)


# ---------------------------------------------------------------------------
# ImportReviewer.list_pending_imports
# ---------------------------------------------------------------------------

def test_list_pending_imports_returns_every_import_key(files):
    files.remember_import("_auto_extracted", "Some scratch content.")
    files.remember_import("scraped-doc", "Other content.")
    reviewer = ImportReviewer(_FakeIndex(), files, Mock(), agent_id="main")

    assert reviewer.list_pending_imports() == ["_auto_extracted", "scraped-doc"]


def test_list_pending_imports_is_empty_with_no_imports(files):
    reviewer = ImportReviewer(_FakeIndex(), files, Mock(), agent_id="main")

    assert reviewer.list_pending_imports() == []


# ---------------------------------------------------------------------------
# ImportReviewer.preview
# ---------------------------------------------------------------------------

def test_preview_creates_a_new_topic_when_no_existing_note_matches(files):
    files.remember_import("_auto_extracted", "User prefers dark mode.")
    index = _FakeIndex(hits=[])
    provider = _draft_response(key="dark-mode", rationale="New preference.",
                                content="User prefers dark mode.")
    reviewer = ImportReviewer(index, files, provider, agent_id="main")

    preview = reviewer.preview("_auto_extracted")

    assert preview["target_kind"] == "new_topic"
    assert preview["target_key"] == "dark-mode"
    assert preview["based_on_content_hash"] == _hash_text("")


def test_preview_revises_an_existing_topic_when_a_topic_hit_matches(files):
    files.remember("project-goals", "Existing goal: ship v1.")
    files.remember_import("_auto_extracted", "Goal deadline moved to March.")
    index = _FakeIndex(hits=[{"rel_path": "memory/topics/project-goals.md", "score": 1.0}])
    provider = _draft_response(key="ignored-by-reviewer", content="Merged content.")
    reviewer = ImportReviewer(index, files, provider, agent_id="main")

    preview = reviewer.preview("_auto_extracted")

    assert preview["target_kind"] == "revise_topic"
    assert preview["target_key"] == "project-goals"
    assert preview["based_on_content_hash"] == _hash_text("Existing goal: ship v1.")


def test_preview_ignores_a_memory_md_hit_and_falls_back_to_new_topic(files):
    files.remember_import("_auto_extracted", "Some content.")
    index = _FakeIndex(hits=[{"rel_path": "MEMORY.md", "score": 5.0}])
    provider = _draft_response(key="new-key")
    reviewer = ImportReviewer(index, files, provider, agent_id="main")

    preview = reviewer.preview("_auto_extracted")

    assert preview["target_kind"] == "new_topic"
    assert preview["target_key"] == "new-key"


def test_preview_never_writes_to_disk(files):
    files.remember_import("_auto_extracted", "Some content.")
    index = _FakeIndex(hits=[])
    provider = _draft_response(key="dark-mode", content="Drafted content.")
    reviewer = ImportReviewer(index, files, provider, agent_id="main")

    reviewer.preview("_auto_extracted")

    assert files.load("dark-mode") is None  # drafted, never saved
    assert files.load_import("_auto_extracted") == "Some content."  # import untouched


def test_preview_leaves_an_existing_target_notes_content_unchanged(files):
    files.remember("project-goals", "Existing goal: ship v1.")
    files.remember_import("_auto_extracted", "Some content.")
    index = _FakeIndex(hits=[{"rel_path": "memory/topics/project-goals.md", "score": 1.0}])
    provider = _draft_response(content="Some drafted merge nobody approved yet.")
    reviewer = ImportReviewer(index, files, provider, agent_id="main")

    reviewer.preview("_auto_extracted")

    assert files.load("project-goals") == "Existing goal: ship v1."


def test_preview_records_the_draft_via_the_index(files):
    files.remember_import("_auto_extracted", "Some content.")
    index = _FakeIndex(hits=[])
    provider = _draft_response(key="dark-mode", rationale="New preference.", content="Drafted.")
    reviewer = ImportReviewer(index, files, provider, agent_id="main")

    reviewer.preview("_auto_extracted")

    assert len(index.recorded_previews) == 1
    recorded = index.recorded_previews[0]
    assert recorded["import_key"] == "_auto_extracted"
    assert recorded["target_key"] == "dark-mode"
    assert recorded["rationale"] == "New preference."
    assert recorded["drafted_content"] == "Drafted."


def test_preview_raises_for_an_unknown_import_key(files):
    reviewer = ImportReviewer(_FakeIndex(), files, Mock(), agent_id="main")

    with pytest.raises(ValueError, match="unknown-key"):
        reviewer.preview("unknown-key")


def test_preview_raises_for_an_import_that_is_only_whitespace(files):
    files.remember_import("_auto_extracted", "   \n  ")
    reviewer = ImportReviewer(_FakeIndex(), files, Mock(), agent_id="main")

    with pytest.raises(ValueError, match="_auto_extracted"):
        reviewer.preview("_auto_extracted")


def test_preview_raises_on_a_malformed_drafting_response(files):
    files.remember_import("_auto_extracted", "Some content.")
    index = _FakeIndex(hits=[])
    provider = Mock()
    provider.chat = Mock(return_value=LLMResponse(text="not the expected format", finish_reason="stop"))
    reviewer = ImportReviewer(index, files, provider, agent_id="main")

    with pytest.raises(ValueError, match="separator"):
        reviewer.preview("_auto_extracted")


def test_preview_shows_no_existing_claims_for_a_new_topic(files):
    files.remember_import("_auto_extracted", "Some content.")
    index = _FakeIndex(hits=[])
    provider = _draft_response()
    reviewer = ImportReviewer(index, files, provider, agent_id="main")

    reviewer.preview("_auto_extracted")

    sent = provider.chat.call_args.kwargs["messages"][0]["content"]
    assert "Existing claims in this note:\n(none)" in sent


def test_preview_shows_existing_claims_when_revising(files):
    files.remember("project-goals", "Existing goal: ship v1.")
    files.remember_import("_auto_extracted", "Some content.")
    claims = [{"id": "c-abc", "status": "supported", "text": "Ship v1 by June."}]
    index = _FakeIndex(
        hits=[{"rel_path": "memory/topics/project-goals.md", "score": 1.0}],
        claims_by_rel_path={"memory/topics/project-goals.md": claims},
    )
    provider = _draft_response()
    reviewer = ImportReviewer(index, files, provider, agent_id="main")

    reviewer.preview("_auto_extracted")

    sent = provider.chat.call_args.kwargs["messages"][0]["content"]
    assert "c-abc (supported): Ship v1 by June." in sent


def test_preview_passes_the_import_key_to_the_prompt(files):
    files.remember_import("_auto_extracted", "Some content.")
    index = _FakeIndex(hits=[])
    provider = _draft_response()
    reviewer = ImportReviewer(index, files, provider, agent_id="main")

    reviewer.preview("_auto_extracted")

    sent = provider.chat.call_args.kwargs["messages"][0]["content"]
    assert "import key: _auto_extracted" in sent


def test_preview_passes_a_pool_of_new_claim_ids_to_the_prompt(files):
    files.remember_import("_auto_extracted", "Some content.")
    index = _FakeIndex(hits=[])
    provider = _draft_response()
    reviewer = ImportReviewer(index, files, provider, agent_id="main")

    reviewer.preview("_auto_extracted")

    sent = provider.chat.call_args.kwargs["messages"][0]["content"]
    ids_line = next(ln for ln in sent.splitlines() if ln.startswith("Available new claim ids"))
    ids = [i.strip() for i in ids_line.split(":", 1)[1].split(",")]
    assert len(ids) == 10
    assert len(set(ids)) == 10  # all distinct


def test_preview_passes_todays_date_to_the_prompt(files):
    import datetime as _dt

    files.remember_import("_auto_extracted", "Some content.")
    index = _FakeIndex(hits=[])
    provider = _draft_response()
    reviewer = ImportReviewer(index, files, provider, agent_id="main")

    reviewer.preview("_auto_extracted")

    sent = provider.chat.call_args.kwargs["messages"][0]["content"]
    assert f"Today's date: {_dt.date.today().isoformat()}" in sent


# ---------------------------------------------------------------------------
# ImportReviewer.approve
# ---------------------------------------------------------------------------

def test_approve_writes_the_drafted_content_to_disk(files):
    files.remember_import("_auto_extracted", "Some content.")
    index = _FakeIndex(hits=[])
    provider = _draft_response(key="dark-mode", content="User prefers dark mode.")
    reviewer = ImportReviewer(index, files, provider, agent_id="main")
    preview = reviewer.preview("_auto_extracted")

    reviewer.approve(preview["id"])

    assert files.load("dark-mode") == "User prefers dark mode."


def test_approve_reindexes_the_written_file(files):
    files.remember_import("_auto_extracted", "Some content.")
    index = _FakeIndex(hits=[])
    provider = _draft_response(key="dark-mode", content="Drafted.")
    reviewer = ImportReviewer(index, files, provider, agent_id="main")
    preview = reviewer.preview("_auto_extracted")

    reviewer.approve(preview["id"])

    assert index.reindexed_files == [("main", "memory/topics/dark-mode.md", "durable", "Drafted.")]


def test_approve_deletes_the_reviewed_import(files):
    files.remember_import("_auto_extracted", "Some content.")
    index = _FakeIndex(hits=[])
    provider = _draft_response(key="dark-mode")
    reviewer = ImportReviewer(index, files, provider, agent_id="main")
    preview = reviewer.preview("_auto_extracted")

    reviewer.approve(preview["id"])

    assert files.load_import("_auto_extracted") is None


def test_approve_removes_the_imports_index_entries(files):
    files.remember_import("_auto_extracted", "Some content.")
    index = _FakeIndex(hits=[])
    provider = _draft_response(key="dark-mode")
    reviewer = ImportReviewer(index, files, provider, agent_id="main")
    preview = reviewer.preview("_auto_extracted")

    reviewer.approve(preview["id"])

    assert index.removed_files == [("main", "memory/imports/_auto_extracted.md")]


def test_approve_returns_the_import_key_target_key_and_rel_path(files):
    files.remember_import("_auto_extracted", "Some content.")
    index = _FakeIndex(hits=[])
    provider = _draft_response(key="dark-mode", content="Drafted.")
    reviewer = ImportReviewer(index, files, provider, agent_id="main")
    preview = reviewer.preview("_auto_extracted")

    result = reviewer.approve(preview["id"])

    assert result == {
        "import_key": "_auto_extracted",
        "target_key": "dark-mode",
        "rel_path": "memory/topics/dark-mode.md",
    }


def test_approve_raises_for_an_unknown_preview_id(files):
    reviewer = ImportReviewer(_FakeIndex(), files, Mock(), agent_id="main")

    with pytest.raises(ValueError, match="999"):
        reviewer.approve(999)


def test_approve_refuses_to_apply_over_a_stale_target(files):
    # Someone hand-edits the note after the preview was drafted but before
    # it's approved — the human edit must win, never get clobbered.
    files.remember("project-goals", "Existing goal: ship v1.")
    files.remember_import("_auto_extracted", "Goal moved to March.")
    index = _FakeIndex(hits=[{"rel_path": "memory/topics/project-goals.md", "score": 1.0}])
    provider = _draft_response(content="Merged content.")
    reviewer = ImportReviewer(index, files, provider, agent_id="main")
    preview = reviewer.preview("_auto_extracted")

    files.remember("project-goals", "Human-edited content since the preview.")

    with pytest.raises(StaleImportError):
        reviewer.approve(preview["id"])
    # And the human's edit must survive the attempt.
    assert files.load("project-goals") == "Human-edited content since the preview."
    # The import is not retired by a refused apply.
    assert files.load_import("_auto_extracted") == "Goal moved to March."


# ---------------------------------------------------------------------------
# ImportReviewer.reject
# ---------------------------------------------------------------------------

def test_reject_deletes_the_import(files):
    files.remember_import("_auto_extracted", "Some content.")
    reviewer = ImportReviewer(_FakeIndex(), files, Mock(), agent_id="main")

    reviewer.reject("_auto_extracted", reason="Not actually useful.")

    assert files.load_import("_auto_extracted") is None


def test_reject_removes_the_imports_index_entries(files):
    files.remember_import("_auto_extracted", "Some content.")
    index = _FakeIndex()
    reviewer = ImportReviewer(index, files, Mock(), agent_id="main")

    reviewer.reject("_auto_extracted")

    assert index.removed_files == [("main", "memory/imports/_auto_extracted.md")]


def test_reject_never_writes_a_topic_note(files):
    files.remember_import("_auto_extracted", "Some content.")
    index = _FakeIndex()
    reviewer = ImportReviewer(index, files, Mock(), agent_id="main")

    reviewer.reject("_auto_extracted")

    assert index.reindexed_files == []
    assert files.list_keys() == []


def test_reject_raises_for_an_unknown_import_key(files):
    reviewer = ImportReviewer(_FakeIndex(), files, Mock(), agent_id="main")

    with pytest.raises(ValueError, match="unknown-key"):
        reviewer.reject("unknown-key")


# ---------------------------------------------------------------------------
# format_import_preview_report / is_import_preview_stale
# ---------------------------------------------------------------------------

def test_format_import_preview_report_includes_key_fields():
    preview = {
        "import_key": "_auto_extracted", "target_kind": "new_topic", "target_key": "dark-mode",
        "rationale": "New preference.", "drafted_content": "User prefers dark mode.",
    }

    report = format_import_preview_report(preview)

    assert "_auto_extracted" in report
    assert "Create new topic: dark-mode" in report
    assert "New preference." in report
    assert "User prefers dark mode." in report


def test_format_import_preview_report_labels_a_revision_correctly():
    preview = {
        "import_key": "_auto_extracted", "target_kind": "revise_topic", "target_key": "project-goals",
        "rationale": "Updated.", "drafted_content": "Content.",
    }

    report = format_import_preview_report(preview)

    assert "Revise topic: project-goals" in report


def test_is_import_preview_stale_false_when_unchanged(files):
    files.remember("project-goals", "Existing content.")
    preview = {"target_key": "project-goals", "based_on_content_hash": _hash_text("Existing content.")}

    assert is_import_preview_stale(files, preview) is False


def test_is_import_preview_stale_true_when_changed(files):
    files.remember("project-goals", "Existing content.")
    preview = {"target_key": "project-goals", "based_on_content_hash": _hash_text("Old hash source.")}

    assert is_import_preview_stale(files, preview) is True

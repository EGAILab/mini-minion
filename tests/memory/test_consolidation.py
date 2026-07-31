"""Tests for memory/consolidation.py: ranking + preview drafting (Stage One Phase 5, slice C).

Uses fakes for SessionDB/PostgresMemoryIndex (no live database needed — this
module only calls a handful of narrow methods on each) and a real, tmp_path-
backed MemoryFileRepository so "never writes to disk" can be verified for
real rather than assumed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from minion_assist.memory.consolidation import (
    MemoryConsolidator,
    _hash_text,
    _parse_draft_response,
    _topic_key_from_rel_path,
    format_preview_report,
    rank_proposals,
)
from minion_assist.memory.files import MemoryFileRepository
from minion_assist.providers.base import LLMResponse


def _proposal(**overrides) -> dict:
    base = {
        "id": 1, "job_id": 1, "agent_id": "main",
        "claim_text": "User prefers dark mode.", "created_at": 100.0, "status": "pending",
    }
    base.update(overrides)
    return base


class _FakeDB:
    """Fakes just the two SessionDB methods this module calls."""

    def __init__(self, proposals: list[dict]):
        self._proposals = {p["id"]: p for p in proposals}

    def list_pending_proposals(self, agent_id: str) -> list[dict]:
        return [p for p in self._proposals.values()
                if p["agent_id"] == agent_id and p["status"] == "pending"]

    def get_proposal(self, proposal_id: int) -> dict | None:
        return self._proposals.get(proposal_id)


_EMPTY_STATS = {"recall_count": 0, "unique_queries": 0, "injected_count": 0, "last_recalled_at": None}


class _FakeIndex:
    """Fakes just the PostgresMemoryIndex methods this module calls."""

    def __init__(self, recall_stats: dict[str, dict] | None = None, hits: list[dict] | None = None):
        self._recall_stats = recall_stats or {}
        self._hits = hits or []
        self.recorded_previews: list[dict] = []

    def recall_stats(self, agent_id: str, rel_path: str) -> dict:
        return self._recall_stats.get(rel_path, dict(_EMPTY_STATS))

    def hybrid_search(self, agent_id: str, query: str, *, corpus: str | None = None,
                       max_results: int = 20) -> list[dict]:
        return self._hits

    def record_consolidation_preview(
        self, agent_id, proposal_id, target_kind, target_key,
        based_on_content_hash, drafted_content, rationale,
    ) -> int:
        self.recorded_previews.append({
            "agent_id": agent_id, "proposal_id": proposal_id, "target_kind": target_kind,
            "target_key": target_key, "based_on_content_hash": based_on_content_hash,
            "drafted_content": drafted_content, "rationale": rationale,
        })
        return len(self.recorded_previews)


def _draft_response(key: str = "some-key", rationale: str = "Because reasons.",
                     content: str = "Drafted content.") -> Mock:
    provider = Mock()
    provider.chat = Mock(
        return_value=LLMResponse(
            text=f"KEY: {key}\nRATIONALE: {rationale}\n---\n{content}", finish_reason="stop"
        )
    )
    return provider


# ---------------------------------------------------------------------------
# _hash_text / _topic_key_from_rel_path / _parse_draft_response
# ---------------------------------------------------------------------------

def test_hash_text_is_deterministic():
    assert _hash_text("hello") == _hash_text("hello")


def test_hash_text_differs_for_different_content():
    assert _hash_text("hello") != _hash_text("world")


def test_topic_key_from_rel_path_extracts_the_key():
    assert _topic_key_from_rel_path("memory/topics/project-goals.md") == "project-goals"


def test_topic_key_from_rel_path_is_none_for_memory_md():
    assert _topic_key_from_rel_path("MEMORY.md") is None


def test_topic_key_from_rel_path_is_none_for_a_daily_note():
    assert _topic_key_from_rel_path("memory/2026-01-01.md") is None


def test_topic_key_from_rel_path_is_none_for_an_import():
    assert _topic_key_from_rel_path("memory/imports/scraped.md") is None


def test_parse_draft_response_extracts_key_rationale_and_content():
    key, rationale, content = _parse_draft_response(
        "KEY: coding-preferences\nRATIONALE: New preference noted.\n---\nUser likes tabs."
    )
    assert key == "coding-preferences"
    assert rationale == "New preference noted."
    assert content == "User likes tabs."


def test_parse_draft_response_raises_without_a_separator():
    with pytest.raises(ValueError, match="separator"):
        _parse_draft_response("KEY: x\nRATIONALE: y\nNo separator here.")


# ---------------------------------------------------------------------------
# rank_proposals
# ---------------------------------------------------------------------------

def test_rank_proposals_orders_by_injected_count_first():
    db = _FakeDB([_proposal(id=1), _proposal(id=2)])
    index = _FakeIndex(recall_stats={
        # score = 5*injected + 2*recall + unique_queries
        "proposals/1": {"recall_count": 3, "unique_queries": 1, "injected_count": 0, "last_recalled_at": 1.0},  # 7
        "proposals/2": {"recall_count": 1, "unique_queries": 1, "injected_count": 1, "last_recalled_at": 1.0},  # 8
    })

    ranked = rank_proposals(db, index, "main")

    assert [p["id"] for p in ranked] == [2, 1]  # injected beats merely-recalled, despite fewer recalls


def test_rank_proposals_includes_never_recalled_proposals_at_the_bottom():
    db = _FakeDB([_proposal(id=1), _proposal(id=2)])
    index = _FakeIndex(recall_stats={
        "proposals/1": {"recall_count": 3, "unique_queries": 2, "injected_count": 1, "last_recalled_at": 1.0},
    })

    ranked = rank_proposals(db, index, "main")

    assert [p["id"] for p in ranked] == [1, 2]
    assert ranked[1]["score"] == 0


def test_rank_proposals_ties_broken_by_proposal_id():
    db = _FakeDB([_proposal(id=2), _proposal(id=1)])
    index = _FakeIndex()  # no recall stats for either — both score 0

    ranked = rank_proposals(db, index, "main")

    assert [p["id"] for p in ranked] == [1, 2]


def test_rank_proposals_excludes_non_pending_proposals():
    db = _FakeDB([_proposal(id=1, status="promoted"), _proposal(id=2, status="pending")])
    index = _FakeIndex()

    ranked = rank_proposals(db, index, "main")

    assert [p["id"] for p in ranked] == [2]


def test_rank_proposals_scopes_to_the_given_agent():
    db = _FakeDB([_proposal(id=1, agent_id="main"), _proposal(id=2, agent_id="researcher")])
    index = _FakeIndex()

    ranked = rank_proposals(db, index, "main")

    assert [p["id"] for p in ranked] == [1]


# ---------------------------------------------------------------------------
# format_preview_report
# ---------------------------------------------------------------------------

def test_format_preview_report_shows_pending_review_decision():
    report = format_preview_report({
        "proposal_id": 1, "target_kind": "new_topic", "target_key": "coding-preferences",
        "rationale": "New preference noted.", "drafted_content": "User likes tabs.",
    })

    assert "Decision: pending review" in report
    assert "Create new topic: coding-preferences" in report
    assert "User likes tabs." in report


def test_format_preview_report_labels_a_revision_differently_from_a_new_topic():
    report = format_preview_report({
        "proposal_id": 1, "target_kind": "revise_topic", "target_key": "project-goals",
        "rationale": "Merged in.", "drafted_content": "Updated content.",
    })

    assert "Revise topic: project-goals" in report


# ---------------------------------------------------------------------------
# MemoryConsolidator.preview
# ---------------------------------------------------------------------------

@pytest.fixture
def files(tmp_path: Path) -> MemoryFileRepository:
    return MemoryFileRepository(tmp_path)


def test_preview_creates_a_new_topic_when_no_existing_note_matches(files):
    db = _FakeDB([_proposal(id=1, claim_text="User prefers dark mode.")])
    index = _FakeIndex(hits=[])  # nothing found
    provider = _draft_response(key="dark-mode", rationale="New preference.", content="User prefers dark mode.")
    consolidator = MemoryConsolidator(db, index, files, provider)

    preview = consolidator.preview("main", 1)

    assert preview["target_kind"] == "new_topic"
    assert preview["target_key"] == "dark-mode"
    assert preview["based_on_content_hash"] == _hash_text("")


def test_preview_revises_an_existing_topic_when_a_topic_hit_matches(files):
    files.remember("project-goals", "Existing goal: ship v1.")
    db = _FakeDB([_proposal(id=1, claim_text="Goal deadline moved to March.")])
    index = _FakeIndex(hits=[{"rel_path": "memory/topics/project-goals.md", "score": 1.0}])
    provider = _draft_response(key="ignored-by-consolidator", content="Merged content.")
    consolidator = MemoryConsolidator(db, index, files, provider)

    preview = consolidator.preview("main", 1)

    assert preview["target_kind"] == "revise_topic"
    # The known match's key wins over whatever the model echoed back.
    assert preview["target_key"] == "project-goals"
    assert preview["based_on_content_hash"] == _hash_text("Existing goal: ship v1.")


def test_preview_ignores_a_memory_md_hit_and_falls_back_to_new_topic(files):
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[{"rel_path": "MEMORY.md", "score": 5.0}])
    provider = _draft_response(key="new-key")
    consolidator = MemoryConsolidator(db, index, files, provider)

    preview = consolidator.preview("main", 1)

    assert preview["target_kind"] == "new_topic"
    assert preview["target_key"] == "new-key"


def test_preview_never_writes_to_disk(files):
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[])
    provider = _draft_response(key="dark-mode", content="Drafted content.")
    consolidator = MemoryConsolidator(db, index, files, provider)

    consolidator.preview("main", 1)

    assert files.load("dark-mode") is None  # drafted, never saved


def test_preview_leaves_an_existing_target_notes_content_unchanged(files):
    files.remember("project-goals", "Existing goal: ship v1.")
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[{"rel_path": "memory/topics/project-goals.md", "score": 1.0}])
    provider = _draft_response(content="Some drafted merge nobody approved yet.")
    consolidator = MemoryConsolidator(db, index, files, provider)

    consolidator.preview("main", 1)

    assert files.load("project-goals") == "Existing goal: ship v1."


def test_preview_records_the_draft_via_the_index(files):
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[])
    provider = _draft_response(key="dark-mode", rationale="New preference.", content="Drafted.")
    consolidator = MemoryConsolidator(db, index, files, provider)

    consolidator.preview("main", 1)

    assert len(index.recorded_previews) == 1
    recorded = index.recorded_previews[0]
    assert recorded["target_key"] == "dark-mode"
    assert recorded["rationale"] == "New preference."
    assert recorded["drafted_content"] == "Drafted."


def test_preview_raises_for_an_unknown_proposal_id(files):
    db = _FakeDB([])
    index = _FakeIndex()
    consolidator = MemoryConsolidator(db, index, files, Mock())

    with pytest.raises(ValueError, match="999"):
        consolidator.preview("main", 999)


def test_preview_raises_on_a_malformed_drafting_response(files):
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[])
    provider = Mock()
    provider.chat = Mock(return_value=LLMResponse(text="not the expected format", finish_reason="stop"))
    consolidator = MemoryConsolidator(db, index, files, provider)

    with pytest.raises(ValueError, match="separator"):
        consolidator.preview("main", 1)


# ---------------------------------------------------------------------------
# Evidence provenance (Task 8) — diary/generated prose must never feed a draft
# ---------------------------------------------------------------------------

def test_draft_prompt_contains_only_the_claim_and_the_merge_targets_content(files):
    db = _FakeDB([_proposal(id=1, claim_text="User prefers dark mode.")])
    index = _FakeIndex(hits=[])
    provider = _draft_response()
    consolidator = MemoryConsolidator(db, index, files, provider)

    consolidator.preview("main", 1)

    sent_messages = provider.chat.call_args.kwargs["messages"]
    assert sent_messages == [{
        "role": "user",
        "content": "New claim:\nUser prefers dark mode.\n\n"
                    "Existing note content:\n(no existing note — propose a new one)",
    }]


def test_draft_prompt_never_mentions_dreams_md(files):
    # DREAMS.md is never read by this module at all — there is no code path
    # that could pull diary text into a draft (see the module docstring's
    # "Evidence provenance" section). This test pins that down structurally:
    # the prompt is built purely from claim_text + existing note content.
    files.remember("project-goals", "Existing goal: ship v1.")
    db = _FakeDB([_proposal(id=1, claim_text="Goal deadline moved to March.")])
    index = _FakeIndex(hits=[{"rel_path": "memory/topics/project-goals.md", "score": 1.0}])
    provider = _draft_response()
    consolidator = MemoryConsolidator(db, index, files, provider)

    consolidator.preview("main", 1)

    sent_messages = provider.chat.call_args.kwargs["messages"]
    assert "DREAMS" not in sent_messages[0]["content"]
    assert "dream" not in sent_messages[0]["content"].lower()

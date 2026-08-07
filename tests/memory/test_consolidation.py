"""Tests for memory/consolidation.py: ranking, preview drafting, apply/reject/rollback,
and historical backfill (Stage One Phase 5, slices C-D).

Uses fakes for SessionDB/PostgresMemoryIndex (no live database needed — this
module only calls a handful of narrow methods on each) and a real, tmp_path-
backed MemoryFileRepository so "never writes to disk" (slice C) and "writes
exactly the expected content" (slice D) can be verified for real rather than
assumed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from minion_assist.memory.consolidation import (
    MemoryConsolidator,
    StaleProposalError,
    _hash_text,
    _topic_key_from_rel_path,
    backfill_agent,
    find_merge_target,
    format_preview_report,
    is_preview_stale,
    parse_draft_response,
    rank_proposals,
)
from minion_assist.memory.files import MemoryFileRepository
from minion_assist.session.db import _chunk_run, compute_backfill_windows
from minion_assist.providers.base import LLMResponse


def _proposal(**overrides) -> dict:
    base = {
        "id": 1, "job_id": 1, "agent_id": "main",
        "claim_text": "User prefers dark mode.", "created_at": 100.0, "status": "pending",
        "rejected_reason": "",
    }
    base.update(overrides)
    return base


class _FakeDB:
    """Fakes just the SessionDB methods this module calls."""

    def __init__(self, proposals: list[dict] | None = None):
        self._proposals = {p["id"]: p for p in (proposals or [])}
        self._next_id = max(self._proposals) + 1 if self._proposals else 1
        # session_id -> list of message ids
        self._sessions: dict[str, list[int]] = {}
        # session_id -> list of (from_id, to_id)
        self._capture_jobs: dict[str, list[tuple[int, int]]] = {}
        self._agent_sessions: dict[str, list[str]] = {}
        self.enqueued: list[tuple] = []

    # -- proposals --

    def list_pending_proposals(self, agent_id: str) -> list[dict]:
        return [p for p in self._proposals.values()
                if p["agent_id"] == agent_id and p["status"] == "pending"]

    def get_proposal(self, proposal_id: int) -> dict | None:
        return self._proposals.get(proposal_id)

    def set_proposal_status(self, proposal_id: int, status: str, reason: str = "") -> None:
        self._proposals[proposal_id]["status"] = status
        self._proposals[proposal_id]["rejected_reason"] = reason

    # -- backfill helpers --

    def add_session(self, agent_id: str, session_id: str, message_ids: list[int]) -> None:
        self._sessions[session_id] = message_ids
        self._agent_sessions.setdefault(agent_id, []).append(session_id)

    def add_capture_job_range(self, session_id: str, from_id: int, to_id: int) -> None:
        self._capture_jobs.setdefault(session_id, []).append((from_id, to_id))

    def list_session_ids_for_agent(self, agent_id: str) -> list[str]:
        return list(self._agent_sessions.get(agent_id, []))

    def list_message_ids(self, session_id: str) -> list[int]:
        return list(self._sessions.get(session_id, []))

    def list_capture_job_ranges(self, session_id: str) -> list[tuple[int, int]]:
        return list(self._capture_jobs.get(session_id, []))

    def enqueue_capture_job(self, agent_id, session_id, from_id, to_id, idempotency_key) -> int | None:
        if idempotency_key in {e[4] for e in self.enqueued}:
            return None
        self.enqueued.append((agent_id, session_id, from_id, to_id, idempotency_key))
        return len(self.enqueued)


_EMPTY_STATS = {"recall_count": 0, "unique_queries": 0, "injected_count": 0, "last_recalled_at": None}


class _FakeIndex:
    """Fakes just the PostgresMemoryIndex methods this module calls."""

    def __init__(
        self,
        recall_stats: dict[str, dict] | None = None,
        hits: list[dict] | None = None,
        claims_by_rel_path: dict[str, list[dict]] | None = None,
    ):
        self._recall_stats = recall_stats or {}
        self._hits = hits or []
        self._claims_by_rel_path = claims_by_rel_path or {}
        self.recorded_previews: list[dict] = []
        self.recorded_revisions: list[dict] = []
        self.removed_proposals: list[tuple] = []
        self.reindexed_proposals: list[tuple] = []
        self.reindexed_files: list[tuple] = []
        self.removed_files: list[tuple] = []

    def recall_stats(self, agent_id: str, rel_path: str) -> dict:
        return self._recall_stats.get(rel_path, dict(_EMPTY_STATS))

    def hybrid_search(self, agent_id: str, query: str, *, corpus: str | None = None,
                       max_results: int = 20) -> list[dict]:
        return self._hits

    def list_claims(self, agent_id: str, rel_path: str | None = None, status=None) -> list[dict]:
        return self._claims_by_rel_path.get(rel_path, [])

    def record_consolidation_preview(
        self, agent_id, proposal_id, target_kind, target_key,
        based_on_content_hash, drafted_content, rationale,
    ) -> int:
        preview_id = len(self.recorded_previews) + 1
        self.recorded_previews.append({
            "id": preview_id, "agent_id": agent_id, "proposal_id": proposal_id,
            "target_kind": target_kind, "target_key": target_key,
            "based_on_content_hash": based_on_content_hash,
            "drafted_content": drafted_content, "rationale": rationale,
        })
        return preview_id

    def get_consolidation_preview(self, preview_id: int) -> dict | None:
        for p in self.recorded_previews:
            if p["id"] == preview_id:
                return dict(p)
        return None

    def record_topic_revision(self, agent_id, target_key, proposal_id, prior_content) -> int:
        revision_id = len(self.recorded_revisions) + 1
        self.recorded_revisions.append({
            "id": revision_id, "agent_id": agent_id, "target_key": target_key,
            "proposal_id": proposal_id, "prior_content": prior_content,
        })
        return revision_id

    def latest_topic_revision(self, agent_id: str, target_key: str) -> dict | None:
        matches = [r for r in self.recorded_revisions
                   if r["agent_id"] == agent_id and r["target_key"] == target_key]
        return dict(matches[-1]) if matches else None

    def delete_topic_revision(self, revision_id: int) -> None:
        self.recorded_revisions = [r for r in self.recorded_revisions if r["id"] != revision_id]

    def remove_proposal(self, agent_id: str, proposal_id: int) -> None:
        self.removed_proposals.append((agent_id, proposal_id))

    def reindex_proposal(self, agent_id: str, proposal_id: int, claim_text: str) -> int:
        self.reindexed_proposals.append((agent_id, proposal_id, claim_text))
        return 1

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
# _hash_text / _topic_key_from_rel_path / parse_draft_response
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


def test_find_merge_target_returns_the_first_topic_hit():
    index = Mock()
    index.hybrid_search.return_value = [
        {"rel_path": "MEMORY.md"},
        {"rel_path": "memory/topics/coding-preferences.md"},
    ]

    result = find_merge_target(index, "main", "User likes tabs.")

    assert result == "coding-preferences"
    index.hybrid_search.assert_called_once_with(
        "main", "User likes tabs.", corpus="durable", max_results=5
    )


def test_find_merge_target_returns_none_with_no_topic_hits():
    index = Mock()
    index.hybrid_search.return_value = [{"rel_path": "MEMORY.md"}]

    assert find_merge_target(index, "main", "User likes tabs.") is None


def test_parse_draft_response_extracts_key_rationale_and_content():
    key, rationale, content = parse_draft_response(
        "KEY: coding-preferences\nRATIONALE: New preference noted.\n---\nUser likes tabs."
    )
    assert key == "coding-preferences"
    assert rationale == "New preference noted."
    assert content == "User likes tabs."


def test_parse_draft_response_raises_without_a_separator():
    with pytest.raises(ValueError, match="separator"):
        parse_draft_response("KEY: x\nRATIONALE: y\nNo separator here.")


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

def test_preview_creates_a_new_topic_when_no_existing_note_matches(files):
    db = _FakeDB([_proposal(id=1, claim_text="User prefers dark mode.")])
    index = _FakeIndex(hits=[])  # nothing found
    provider = _draft_response(key="dark-mode", rationale="New preference.", content="User prefers dark mode.")
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")

    preview = consolidator.preview(1)

    assert preview["target_kind"] == "new_topic"
    assert preview["target_key"] == "dark-mode"
    assert preview["based_on_content_hash"] == _hash_text("")


def test_preview_revises_an_existing_topic_when_a_topic_hit_matches(files):
    files.remember("project-goals", "Existing goal: ship v1.")
    db = _FakeDB([_proposal(id=1, claim_text="Goal deadline moved to March.")])
    index = _FakeIndex(hits=[{"rel_path": "memory/topics/project-goals.md", "score": 1.0}])
    provider = _draft_response(key="ignored-by-consolidator", content="Merged content.")
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")

    preview = consolidator.preview(1)

    assert preview["target_kind"] == "revise_topic"
    # The known match's key wins over whatever the model echoed back.
    assert preview["target_key"] == "project-goals"
    assert preview["based_on_content_hash"] == _hash_text("Existing goal: ship v1.")


def test_preview_ignores_a_memory_md_hit_and_falls_back_to_new_topic(files):
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[{"rel_path": "MEMORY.md", "score": 5.0}])
    provider = _draft_response(key="new-key")
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")

    preview = consolidator.preview(1)

    assert preview["target_kind"] == "new_topic"
    assert preview["target_key"] == "new-key"


def test_preview_never_writes_to_disk(files):
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[])
    provider = _draft_response(key="dark-mode", content="Drafted content.")
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")

    consolidator.preview(1)

    assert files.load("dark-mode") is None  # drafted, never saved


def test_preview_leaves_an_existing_target_notes_content_unchanged(files):
    files.remember("project-goals", "Existing goal: ship v1.")
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[{"rel_path": "memory/topics/project-goals.md", "score": 1.0}])
    provider = _draft_response(content="Some drafted merge nobody approved yet.")
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")

    consolidator.preview(1)

    assert files.load("project-goals") == "Existing goal: ship v1."


def test_preview_records_the_draft_via_the_index(files):
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[])
    provider = _draft_response(key="dark-mode", rationale="New preference.", content="Drafted.")
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")

    consolidator.preview(1)

    assert len(index.recorded_previews) == 1
    recorded = index.recorded_previews[0]
    assert recorded["target_key"] == "dark-mode"
    assert recorded["rationale"] == "New preference."
    assert recorded["drafted_content"] == "Drafted."


def test_preview_raises_for_an_unknown_proposal_id(files):
    db = _FakeDB([])
    index = _FakeIndex()
    consolidator = MemoryConsolidator(db, index, files, Mock(), agent_id="main")

    with pytest.raises(ValueError, match="999"):
        consolidator.preview(999)


def test_preview_raises_on_a_malformed_drafting_response(files):
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[])
    provider = Mock()
    provider.chat = Mock(return_value=LLMResponse(text="not the expected format", finish_reason="stop"))
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")

    with pytest.raises(ValueError, match="separator"):
        consolidator.preview(1)


# ---------------------------------------------------------------------------
# Evidence provenance (Task 8) — diary/generated prose must never feed a draft
# ---------------------------------------------------------------------------

def test_draft_prompt_contains_only_the_claim_and_the_merge_targets_content(files):
    db = _FakeDB([_proposal(id=1, claim_text="User prefers dark mode.")])
    index = _FakeIndex(hits=[])
    provider = _draft_response()
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")

    consolidator.preview(1)

    sent_messages = provider.chat.call_args.kwargs["messages"]
    assert len(sent_messages) == 1
    content = sent_messages[0]["content"]
    assert "User prefers dark mode." in content
    assert "(no existing note — propose a new one)" in content
    assert "Existing claims in this note:\n(none)" in content


def test_draft_prompt_never_mentions_dreams_md(files):
    # DREAMS.md is never read by this module at all — there is no code path
    # that could pull diary text into a draft (see the module docstring's
    # "Evidence provenance" section). This test pins that down structurally:
    # the prompt is built purely from claim_text + existing note content.
    files.remember("project-goals", "Existing goal: ship v1.")
    db = _FakeDB([_proposal(id=1, claim_text="Goal deadline moved to March.")])
    index = _FakeIndex(hits=[{"rel_path": "memory/topics/project-goals.md", "score": 1.0}])
    provider = _draft_response()
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")

    consolidator.preview(1)

    sent_messages = provider.chat.call_args.kwargs["messages"]
    assert "DREAMS" not in sent_messages[0]["content"]
    assert "dream" not in sent_messages[0]["content"].lower()


# ---------------------------------------------------------------------------
# Claim markers in the drafting prompt (Stage One Phase 7, slice B)
# ---------------------------------------------------------------------------

def test_draft_system_prompt_explains_claim_markers():
    from minion_assist.memory.consolidation import _DRAFT_SYSTEM

    assert "claim:" in _DRAFT_SYSTEM
    assert "contradicts=" in _DRAFT_SYSTEM
    assert "status=contested" in _DRAFT_SYSTEM


def test_preview_shows_no_existing_claims_for_a_new_topic(files):
    db = _FakeDB([_proposal(id=1, claim_text="User prefers dark mode.")])
    index = _FakeIndex(hits=[])
    provider = _draft_response()
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")

    consolidator.preview(1)

    content = provider.chat.call_args.kwargs["messages"][0]["content"]
    assert "Existing claims in this note:\n(none)" in content


def test_preview_shows_existing_claims_when_revising(files):
    files.remember("project-goals", "Existing goal: ship v1.")
    db = _FakeDB([_proposal(id=1, claim_text="Goal deadline moved to March.")])
    index = _FakeIndex(
        hits=[{"rel_path": "memory/topics/project-goals.md", "score": 1.0}],
        claims_by_rel_path={
            "memory/topics/project-goals.md": [
                {"id": "c-old1", "status": "supported", "text": "Ship v1 by June."},
            ]
        },
    )
    provider = _draft_response()
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")

    consolidator.preview(1)

    content = provider.chat.call_args.kwargs["messages"][0]["content"]
    assert "c-old1 (supported): Ship v1 by June." in content


def test_preview_looks_up_claims_for_the_correct_rel_path(files):
    files.remember("project-goals", "Existing goal: ship v1.")
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[{"rel_path": "memory/topics/project-goals.md", "score": 1.0}])
    calls = []
    index.list_claims = lambda agent_id, rel_path=None, status=None: calls.append(
        (agent_id, rel_path)
    ) or []
    provider = _draft_response()
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")

    consolidator.preview(1)

    assert calls == [("main", "memory/topics/project-goals.md")]


def test_preview_never_calls_list_claims_for_a_new_topic(files):
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[])
    called = []
    index.list_claims = lambda *a, **kw: called.append(True) or []
    provider = _draft_response()
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")

    consolidator.preview(1)

    assert called == []


def test_preview_generates_a_new_claim_id_and_passes_it_to_the_prompt(files):
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[])
    provider = _draft_response()
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")

    consolidator.preview(1)

    content = provider.chat.call_args.kwargs["messages"][0]["content"]
    assert "New claim id to use: c-" in content


def test_preview_generates_a_different_claim_id_each_call(files):
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[])
    provider = _draft_response()
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")

    consolidator.preview(1)
    first_content = provider.chat.call_args.kwargs["messages"][0]["content"]
    provider2 = _draft_response()
    consolidator2 = MemoryConsolidator(db, index, files, provider2, agent_id="main")
    consolidator2.preview(1)
    second_content = provider2.chat.call_args.kwargs["messages"][0]["content"]

    def _extract_id(content: str) -> str:
        for line in content.splitlines():
            if line.startswith("New claim id to use: "):
                return line.split(": ", 1)[1]
        return ""

    assert _extract_id(first_content) != _extract_id(second_content)


def test_preview_passes_the_proposal_id_to_the_prompt(files):
    db = _FakeDB([_proposal(id=7)])
    index = _FakeIndex(hits=[])
    provider = _draft_response()
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")

    consolidator.preview(7)

    content = provider.chat.call_args.kwargs["messages"][0]["content"]
    assert "Proposal id (for evidence=proposal:...): 7" in content


def test_preview_passes_todays_date_to_the_prompt(files):
    from datetime import date as _date

    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[])
    provider = _draft_response()
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")

    consolidator.preview(1)

    content = provider.chat.call_args.kwargs["messages"][0]["content"]
    assert f"Today's date: {_date.today().isoformat()}" in content


# ---------------------------------------------------------------------------
# MemoryConsolidator.approve
# ---------------------------------------------------------------------------

def test_approve_writes_the_drafted_content_to_disk(files):
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[])
    provider = _draft_response(key="dark-mode", content="User prefers dark mode.")
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")
    preview = consolidator.preview(1)

    consolidator.approve(preview["id"])

    assert files.load("dark-mode") == "User prefers dark mode."


def test_approve_reindexes_the_written_file(files):
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[])
    provider = _draft_response(key="dark-mode", content="Drafted.")
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")
    preview = consolidator.preview(1)

    consolidator.approve(preview["id"])

    assert index.reindexed_files == [("main", "memory/topics/dark-mode.md", "durable", "Drafted.")]


def test_approve_marks_the_proposal_promoted(files):
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[])
    provider = _draft_response(key="dark-mode")
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")
    preview = consolidator.preview(1)

    consolidator.approve(preview["id"])

    assert db.get_proposal(1)["status"] == "promoted"


def test_approve_removes_the_raw_proposal_chunk(files):
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[])
    provider = _draft_response(key="dark-mode")
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")
    preview = consolidator.preview(1)

    consolidator.approve(preview["id"])

    assert index.removed_proposals == [("main", 1)]


def test_approve_records_a_revision_snapshot_of_the_prior_content(files):
    files.remember("project-goals", "Existing goal: ship v1.")
    db = _FakeDB([_proposal(id=1, claim_text="Goal moved to March.")])
    index = _FakeIndex(hits=[{"rel_path": "memory/topics/project-goals.md", "score": 1.0}])
    provider = _draft_response(content="Merged content.")
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")
    preview = consolidator.preview(1)

    consolidator.approve(preview["id"])

    assert len(index.recorded_revisions) == 1
    assert index.recorded_revisions[0]["prior_content"] == "Existing goal: ship v1."


def test_approve_records_an_empty_prior_content_for_a_brand_new_topic(files):
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[])
    provider = _draft_response(key="dark-mode")
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")
    preview = consolidator.preview(1)

    consolidator.approve(preview["id"])

    assert index.recorded_revisions[0]["prior_content"] == ""


def test_approve_raises_for_an_unknown_preview_id(files):
    db = _FakeDB([])
    index = _FakeIndex()
    consolidator = MemoryConsolidator(db, index, files, Mock(), agent_id="main")

    with pytest.raises(ValueError, match="999"):
        consolidator.approve(999)


def test_approve_refuses_to_apply_over_a_stale_target(files):
    # Someone hand-edits the note after the preview was drafted but before
    # it's approved — the human edit must win, never get clobbered.
    files.remember("project-goals", "Existing goal: ship v1.")
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[{"rel_path": "memory/topics/project-goals.md", "score": 1.0}])
    provider = _draft_response(content="Merged content.")
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")
    preview = consolidator.preview(1)

    files.remember("project-goals", "Human-edited content since the preview.")

    with pytest.raises(StaleProposalError):
        consolidator.approve(preview["id"])
    # And the human's edit must survive the attempt.
    assert files.load("project-goals") == "Human-edited content since the preview."


# ---------------------------------------------------------------------------
# MemoryConsolidator.reject
# ---------------------------------------------------------------------------

def test_reject_marks_the_proposal_rejected(files):
    db = _FakeDB([_proposal(id=1)])
    consolidator = MemoryConsolidator(db, _FakeIndex(), files, Mock(), agent_id="main")

    consolidator.reject(1, reason="Not actually useful.")

    proposal = db.get_proposal(1)
    assert proposal["status"] == "rejected"
    assert proposal["rejected_reason"] == "Not actually useful."


def test_reject_never_touches_the_index_or_disk(files):
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex()
    consolidator = MemoryConsolidator(db, index, files, Mock(), agent_id="main")

    consolidator.reject(1)

    assert index.removed_proposals == []
    assert index.reindexed_files == []


# ---------------------------------------------------------------------------
# MemoryConsolidator.rollback
# ---------------------------------------------------------------------------

def test_rollback_restores_the_prior_content_of_a_revised_topic(files):
    files.remember("project-goals", "Existing goal: ship v1.")
    db = _FakeDB([_proposal(id=1, claim_text="Goal moved to March.")])
    index = _FakeIndex(hits=[{"rel_path": "memory/topics/project-goals.md", "score": 1.0}])
    provider = _draft_response(content="Merged content.")
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")
    preview = consolidator.preview(1)
    consolidator.approve(preview["id"])

    consolidator.rollback("project-goals")

    assert files.load("project-goals") == "Existing goal: ship v1."


def test_rollback_deletes_a_brand_new_topic_entirely(files):
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[])
    provider = _draft_response(key="dark-mode")
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")
    preview = consolidator.preview(1)
    consolidator.approve(preview["id"])
    assert files.load("dark-mode") is not None  # sanity: it was written

    consolidator.rollback("dark-mode")

    assert files.load("dark-mode") is None


def test_rollback_restores_the_proposal_to_pending(files):
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[])
    provider = _draft_response(key="dark-mode")
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")
    preview = consolidator.preview(1)
    consolidator.approve(preview["id"])
    assert db.get_proposal(1)["status"] == "promoted"

    consolidator.rollback("dark-mode")

    assert db.get_proposal(1)["status"] == "pending"


def test_rollback_reindexes_the_proposals_raw_chunk_again(files):
    db = _FakeDB([_proposal(id=1, claim_text="User prefers dark mode.")])
    index = _FakeIndex(hits=[])
    provider = _draft_response(key="dark-mode")
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")
    preview = consolidator.preview(1)
    consolidator.approve(preview["id"])

    consolidator.rollback("dark-mode")

    assert index.reindexed_proposals == [("main", 1, "User prefers dark mode.")]


def test_rollback_consumes_the_revision_so_a_second_rollback_has_nothing_left(files):
    db = _FakeDB([_proposal(id=1)])
    index = _FakeIndex(hits=[])
    provider = _draft_response(key="dark-mode")
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id="main")
    preview = consolidator.preview(1)
    consolidator.approve(preview["id"])

    consolidator.rollback("dark-mode")

    with pytest.raises(ValueError, match="dark-mode"):
        consolidator.rollback("dark-mode")


def test_rollback_raises_for_a_topic_with_no_revision_history(files):
    db = _FakeDB([])
    consolidator = MemoryConsolidator(db, _FakeIndex(), files, Mock(), agent_id="main")

    with pytest.raises(ValueError, match="never-applied"):
        consolidator.rollback("never-applied")


# ---------------------------------------------------------------------------
# rollback — explicit-write revisions (MEM-GAP-020, proposal_id=None)
# ---------------------------------------------------------------------------
# MemoryService.remember()/delete() record a revision the same way
# approve() does, but with proposal_id=None (no proposal triggered an
# explicit write) — these prove rollback() handles that shape without
# assuming a proposal always exists.

def test_rollback_restores_content_from_an_explicit_write_revision(files):
    files.remember("project-goals", "Delayed to next quarter.")
    db = _FakeDB([])
    index = _FakeIndex(hits=[])
    index.record_topic_revision("main", "project-goals", None, "Ship it this quarter.")
    consolidator = MemoryConsolidator(db, index, files, None, agent_id="main")

    result = consolidator.rollback("project-goals")

    assert files.load("project-goals") == "Ship it this quarter."
    assert result["proposal_id"] is None


def test_rollback_deletes_a_brand_new_topic_from_an_explicit_write_revision(files):
    files.remember("dark-mode", "User prefers dark mode.")
    db = _FakeDB([])
    index = _FakeIndex(hits=[])
    index.record_topic_revision("main", "dark-mode", None, "")  # "" = didn't exist before
    consolidator = MemoryConsolidator(db, index, files, None, agent_id="main")

    consolidator.rollback("dark-mode")

    assert files.load("dark-mode") is None


def test_rollback_never_touches_proposal_status_for_an_explicit_write_revision(files):
    # _FakeDB.set_proposal_status(None, ...) would KeyError if rollback()
    # ever called it for a proposal_id=None revision — this proves it's
    # skipped, not just that it happens not to be reached.
    files.remember("project-goals", "Delayed to next quarter.")
    db = _FakeDB([])
    index = _FakeIndex(hits=[])
    index.record_topic_revision("main", "project-goals", None, "Ship it this quarter.")
    consolidator = MemoryConsolidator(db, index, files, None, agent_id="main")

    consolidator.rollback("project-goals")  # must not raise


def test_rollback_consumes_an_explicit_write_revision_so_a_second_rollback_steps_further(files):
    files.remember("project-goals", "Third draft.")
    db = _FakeDB([])
    index = _FakeIndex(hits=[])
    index.record_topic_revision("main", "project-goals", None, "First draft.")
    index.record_topic_revision("main", "project-goals", None, "Second draft.")
    consolidator = MemoryConsolidator(db, index, files, None, agent_id="main")

    consolidator.rollback("project-goals")
    assert files.load("project-goals") == "Second draft."

    consolidator.rollback("project-goals")
    assert files.load("project-goals") == "First draft."


# ---------------------------------------------------------------------------
# is_preview_stale
# ---------------------------------------------------------------------------

def test_is_preview_stale_is_false_right_after_drafting(files):
    files.remember("project-goals", "Existing content.")
    preview = {"target_key": "project-goals", "based_on_content_hash": _hash_text("Existing content.")}

    assert is_preview_stale(files, preview) is False


def test_is_preview_stale_is_true_after_a_hand_edit(files):
    files.remember("project-goals", "Existing content.")
    preview = {"target_key": "project-goals", "based_on_content_hash": _hash_text("Existing content.")}
    files.remember("project-goals", "Edited content.")

    assert is_preview_stale(files, preview) is True


# ---------------------------------------------------------------------------
# _compute_backfill_windows / _chunk_run
# ---------------------------------------------------------------------------

def test_chunk_run_splits_a_long_run_into_bounded_windows():
    run = list(range(1, 45))  # 44 ids, window size 20 -> 20, 20, 4

    windows = _chunk_run(run)

    assert windows == [(1, 20), (21, 40), (41, 44)]


def test_compute_backfill_windows_covers_a_never_captured_session():
    message_ids = [10, 11, 12, 13]

    windows = compute_backfill_windows(message_ids, covered_ranges=[])

    assert windows == [(10, 13)]


def test_compute_backfill_windows_skips_a_fully_covered_session():
    message_ids = [10, 11, 12, 13]

    windows = compute_backfill_windows(message_ids, covered_ranges=[(10, 13)])

    assert windows == []


def test_compute_backfill_windows_finds_the_gap_in_a_partially_covered_session():
    # A session that started before the durable queue existed (uncaptured
    # early messages) and continued after (captured live, per turn).
    message_ids = [1, 2, 3, 4, 5, 6]

    windows = compute_backfill_windows(message_ids, covered_ranges=[(4, 5), (6, 6)])

    assert windows == [(1, 3)]


def test_compute_backfill_windows_finds_multiple_separate_gaps():
    message_ids = [1, 2, 3, 4, 5, 6, 7]

    windows = compute_backfill_windows(message_ids, covered_ranges=[(3, 5)])

    assert windows == [(1, 2), (6, 7)]


def test_compute_backfill_windows_ignores_other_sessions_ids_interleaved_in_the_range():
    # messages is one shared table/sequence — ids 1,2,5,6 belonging to THIS
    # session are not numerically consecutive (3,4 belong to another
    # session), but they are positionally consecutive in this session's own
    # history and must be treated as one contiguous gap.
    message_ids = [1, 2, 5, 6]

    windows = compute_backfill_windows(message_ids, covered_ranges=[])

    assert windows == [(1, 6)]


# ---------------------------------------------------------------------------
# backfill_agent
# ---------------------------------------------------------------------------

def test_backfill_agent_enqueues_a_job_for_a_never_captured_session():
    db = _FakeDB()
    db.add_session("main", "sess-1", [1, 2, 3])

    enqueued = backfill_agent(db, "main", "some-model")

    assert enqueued == 1
    assert db.enqueued[0][:4] == ("main", "sess-1", 1, 3)


def test_backfill_agent_skips_a_fully_covered_session():
    db = _FakeDB()
    db.add_session("main", "sess-1", [1, 2, 3])
    db.add_capture_job_range("sess-1", 1, 3)

    enqueued = backfill_agent(db, "main", "some-model")

    assert enqueued == 0


def test_backfill_agent_only_covers_the_gap_in_a_partially_covered_session():
    db = _FakeDB()
    db.add_session("main", "sess-1", [1, 2, 3, 4, 5])
    db.add_capture_job_range("sess-1", 4, 5)

    enqueued = backfill_agent(db, "main", "some-model")

    assert enqueued == 1
    assert db.enqueued[0][2:4] == (1, 3)


def test_backfill_agent_covers_every_session_for_the_agent():
    db = _FakeDB()
    db.add_session("main", "sess-1", [1, 2])
    db.add_session("main", "sess-2", [10, 11])

    enqueued = backfill_agent(db, "main", "some-model")

    assert enqueued == 2


def test_backfill_agent_is_idempotent_on_a_second_run():
    db = _FakeDB()
    db.add_session("main", "sess-1", [1, 2, 3])

    first_run = backfill_agent(db, "main", "some-model")
    second_run = backfill_agent(db, "main", "some-model")

    assert first_run == 1
    assert second_run == 0


def test_backfill_agent_skips_a_session_with_no_messages():
    db = _FakeDB()
    db.add_session("main", "sess-1", [])

    enqueued = backfill_agent(db, "main", "some-model")

    assert enqueued == 0

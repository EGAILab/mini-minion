"""Tests for memory/forgetting.py: forget_source (Stage One Phase 7, slice F).

Uses a fake PostgresMemoryIndex (no live database needed) and a real,
tmp_path-backed MemoryFileRepository so the actual marker edit written to
disk can be verified for real rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minion_assist.memory.files import MemoryFileRepository
from minion_assist.memory.forgetting import forget_source


class _FakeIndex:
    """Fakes just the PostgresMemoryIndex methods forget_source calls."""

    def __init__(self, claims: list[dict] | None = None):
        self._claims = claims or []
        self.reindexed_files: list[tuple] = []

    def list_claims_citing_evidence(self, agent_id: str, source_kind: str, source_ref: str) -> list[dict]:
        return self._claims

    def reindex_file(self, agent_id: str, rel_path: str, source_kind: str, content: str) -> int:
        self.reindexed_files.append((agent_id, rel_path, source_kind, content))
        return 1


@pytest.fixture
def files(tmp_path: Path) -> MemoryFileRepository:
    return MemoryFileRepository(tmp_path)


def test_forget_source_rewrites_the_marker_on_disk(files):
    files.remember("dark-mode", "- User prefers dark mode.\n  <!-- claim:c-1 evidence=proposal:42 -->")
    index = _FakeIndex(claims=[{"id": "c-1", "rel_path": "memory/topics/dark-mode.md"}])

    forget_source(index, files, "main", "proposal", "42")

    content = files.load("dark-mode")
    assert "evidence=" not in content
    assert "status=unknown" in content
    assert "User prefers dark mode." in content  # prose preserved


def test_forget_source_reindexes_the_edited_file(files):
    files.remember("dark-mode", "- User prefers dark mode.\n  <!-- claim:c-1 evidence=proposal:42 -->")
    index = _FakeIndex(claims=[{"id": "c-1", "rel_path": "memory/topics/dark-mode.md"}])

    forget_source(index, files, "main", "proposal", "42")

    assert len(index.reindexed_files) == 1
    agent_id, rel_path, source_kind, content = index.reindexed_files[0]
    assert agent_id == "main"
    assert rel_path == "memory/topics/dark-mode.md"
    assert source_kind == "durable"
    assert "status=unknown" in content


def test_forget_source_reports_reevaluated_claims(files):
    files.remember("dark-mode", "- User prefers dark mode.\n  <!-- claim:c-1 evidence=proposal:42 -->")
    index = _FakeIndex(claims=[{"id": "c-1", "rel_path": "memory/topics/dark-mode.md"}])

    result = forget_source(index, files, "main", "proposal", "42")

    assert result["reevaluated"] == ["c-1"]
    assert result["still_grounded"] == []
    assert result["skipped_manual_review"] == []


def test_forget_source_reports_still_grounded_claims_with_other_evidence(files):
    files.remember(
        "dark-mode",
        "- User prefers dark mode.\n  <!-- claim:c-1 evidence=proposal:42,message:1189 -->",
    )
    index = _FakeIndex(claims=[{"id": "c-1", "rel_path": "memory/topics/dark-mode.md"}])

    result = forget_source(index, files, "main", "proposal", "42")

    assert result["reevaluated"] == []
    assert result["still_grounded"] == ["c-1"]
    content = files.load("dark-mode")
    assert "evidence=message:1189" in content


def test_forget_source_handles_multiple_claims_in_the_same_page(files):
    files.remember(
        "dark-mode",
        "- Claim one.\n  <!-- claim:c-1 evidence=proposal:42 -->\n\n"
        "- Claim two.\n  <!-- claim:c-2 evidence=proposal:42 -->",
    )
    index = _FakeIndex(claims=[
        {"id": "c-1", "rel_path": "memory/topics/dark-mode.md"},
        {"id": "c-2", "rel_path": "memory/topics/dark-mode.md"},
    ])

    result = forget_source(index, files, "main", "proposal", "42")

    assert set(result["reevaluated"]) == {"c-1", "c-2"}
    content = files.load("dark-mode")
    assert content.count("status=unknown") == 2


def test_forget_source_handles_claims_across_multiple_pages(files):
    files.remember("dark-mode", "- Claim one.\n  <!-- claim:c-1 evidence=proposal:42 -->")
    files.remember("project-goals", "- Claim two.\n  <!-- claim:c-2 evidence=proposal:42 -->")
    index = _FakeIndex(claims=[
        {"id": "c-1", "rel_path": "memory/topics/dark-mode.md"},
        {"id": "c-2", "rel_path": "memory/topics/project-goals.md"},
    ])

    result = forget_source(index, files, "main", "proposal", "42")

    assert set(result["reevaluated"]) == {"c-1", "c-2"}
    assert len(index.reindexed_files) == 2
    assert "status=unknown" in files.load("dark-mode")
    assert "status=unknown" in files.load("project-goals")


def test_forget_source_skips_memory_md_for_manual_review(files):
    index = _FakeIndex(claims=[{"id": "c-1", "rel_path": "MEMORY.md"}])

    result = forget_source(index, files, "main", "proposal", "42")

    assert result["reevaluated"] == []
    assert result["still_grounded"] == []
    assert result["skipped_manual_review"] == [{"claim_id": "c-1", "rel_path": "MEMORY.md"}]
    assert index.reindexed_files == []


def test_forget_source_is_a_harmless_no_op_when_nothing_cites_the_source(files):
    index = _FakeIndex(claims=[])

    result = forget_source(index, files, "main", "proposal", "999")

    assert result == {
        "source_kind": "proposal", "source_ref": "999",
        "reevaluated": [], "still_grounded": [], "skipped_manual_review": [],
    }
    assert index.reindexed_files == []

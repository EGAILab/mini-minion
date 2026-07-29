"""Tests for PinMemoryTool — Stage One Phase 4, slice B."""

from __future__ import annotations

from unittest.mock import Mock

from minion_assist.memory.files import MemoryFileRepository
from minion_assist.memory.service import MemoryService
from minion_assist.tools.memory import PinMemoryTool
from minion_assist.tools.policy import PermissionPolicy


def _service(tmp_path) -> MemoryService:
    """A MemoryService with no index — pin/unpin will raise RuntimeError."""
    return MemoryService(MemoryFileRepository(tmp_path))


def _indexed_service(tmp_path):
    """A MemoryService wired to a mock index, plus the mock for assertions."""
    mock_index = Mock()
    mock_index.is_pinned.return_value = False
    svc = MemoryService(MemoryFileRepository(tmp_path), index=mock_index, agent_id="main")
    return svc, mock_index


def test_schema_name_and_required_fields():
    mem = MemoryService.__new__(MemoryService)
    tool = PinMemoryTool(mem)
    schema = tool.schema
    assert schema.name == "pin_memory"
    assert "key" in schema.parameters["properties"]
    assert "pinned" in schema.parameters["properties"]
    assert schema.parameters["required"] == ["key", "pinned"]


def test_pin_without_an_index_returns_an_error(tmp_path):
    mem = _service(tmp_path)
    mem.remember("project-goals", "content")
    tool = PinMemoryTool(mem)

    result = tool.execute(key="project-goals", pinned=True)

    assert "Error" in result
    assert "No lexical index configured" in result


def test_pin_for_a_missing_note_returns_an_error(tmp_path):
    svc, _mock_index = _indexed_service(tmp_path)
    tool = PinMemoryTool(svc)

    result = tool.execute(key="never-saved", pinned=True)

    assert "Error" in result
    assert "No note found" in result


def test_pin_an_existing_note_succeeds(tmp_path):
    svc, mock_index = _indexed_service(tmp_path)
    svc.remember("project-goals", "content")
    tool = PinMemoryTool(svc)

    result = tool.execute(key="project-goals", pinned=True)

    assert result == "Pinned memory: project-goals"
    mock_index.pin_file.assert_called_once_with("main", "memory/topics/project-goals.md")


def test_unpin_succeeds_even_for_a_missing_note(tmp_path):
    svc, mock_index = _indexed_service(tmp_path)
    tool = PinMemoryTool(svc)

    result = tool.execute(key="never-saved", pinned=False)

    assert result == "Unpinned memory: never-saved"
    mock_index.unpin_file.assert_called_once_with("main", "memory/topics/never-saved.md")


def test_pin_respects_read_only_mode(tmp_path):
    svc, mock_index = _indexed_service(tmp_path)
    svc.remember("project-goals", "content")
    policy = PermissionPolicy(read_only_mode=True)
    tool = PinMemoryTool(svc, policy=policy)

    result = tool.execute(key="project-goals", pinned=True)

    assert "Error" in result
    assert "read-only mode" in result
    mock_index.pin_file.assert_not_called()


def test_pin_proceeds_when_read_only_mode_is_false(tmp_path):
    svc, mock_index = _indexed_service(tmp_path)
    svc.remember("project-goals", "content")
    policy = PermissionPolicy(read_only_mode=False)
    tool = PinMemoryTool(svc, policy=policy)

    result = tool.execute(key="project-goals", pinned=True)

    assert result == "Pinned memory: project-goals"
    mock_index.pin_file.assert_called_once()

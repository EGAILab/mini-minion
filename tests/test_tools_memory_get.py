"""Tests for MemoryGetTool — Stage One Phase 1, slice 5."""

from minion_assist.memory.files import MemoryFileRepository
from minion_assist.memory.service import MemoryService
from minion_assist.tools.memory import MemoryGetTool


def _service(tmp_path) -> MemoryService:
    return MemoryService(MemoryFileRepository(tmp_path))


def test_schema_name_and_required_path():
    mem = MemoryService.__new__(MemoryService)
    tool = MemoryGetTool(mem)
    schema = tool.schema
    assert schema.name == "memory_get"
    assert schema.is_read_only is True
    assert "path" in schema.parameters["properties"]
    assert "path" in schema.parameters["required"]


def test_get_whole_file_by_default(tmp_path):
    mem = _service(tmp_path)
    mem.remember("note", "line1\nline2\nline3")
    tool = MemoryGetTool(mem)
    result = tool.execute(path="memory/topics/note.md")
    assert "line1" in result
    assert "line2" in result
    assert "line3" in result
    assert "lines 1-3 of 3" in result


def test_get_respects_from_line_and_lines(tmp_path):
    mem = _service(tmp_path)
    mem.remember("note", "line1\nline2\nline3\nline4")
    tool = MemoryGetTool(mem)
    result = tool.execute(path="memory/topics/note.md", from_line=2, lines=2)
    assert "line2" in result
    assert "line3" in result
    assert "line1" not in result
    assert "line4" not in result
    assert "lines 2-3 of 4" in result


def test_get_rejects_path_outside_root(tmp_path):
    mem = _service(tmp_path)
    tool = MemoryGetTool(mem)
    result = tool.execute(path="../../etc/passwd")
    assert "Error" in result
    assert "outside the memory root" in result


def test_get_reports_missing_file(tmp_path):
    mem = _service(tmp_path)
    tool = MemoryGetTool(mem)
    result = tool.execute(path="memory/topics/missing.md")
    assert "Error" in result
    assert "not found" in result.lower()


def test_get_includes_path_citation(tmp_path):
    mem = _service(tmp_path)
    mem.remember("note", "content")
    tool = MemoryGetTool(mem)
    result = tool.execute(path="memory/topics/note.md")
    assert "note.md" in result

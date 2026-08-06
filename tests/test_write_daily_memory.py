"""Tests for WriteDailyMemoryTool."""

from datetime import date

from minion_assist.memory.files import MemoryFileRepository
from minion_assist.memory.service import MemoryService
from minion_assist.tools.write_daily_memory import WriteDailyMemoryTool


def _tool(tmp_path) -> WriteDailyMemoryTool:
    return WriteDailyMemoryTool(MemoryService(MemoryFileRepository(tmp_path)))


def test_execute_creates_memory_dir(tmp_path):
    tool = _tool(tmp_path)
    tool.execute(content="Test entry")
    assert (tmp_path / "memory").is_dir()


def test_execute_creates_dated_file(tmp_path):
    tool = _tool(tmp_path)
    tool.execute(content="Today's note")
    today = date.today().isoformat()
    assert (tmp_path / "memory" / f"{today}.md").exists()


def test_execute_appends_content(tmp_path):
    tool = _tool(tmp_path)
    tool.execute(content="First note")
    tool.execute(content="Second note")
    today = date.today().isoformat()
    text = (tmp_path / "memory" / f"{today}.md").read_text()
    assert "First note" in text
    assert "Second note" in text


def test_execute_content_in_file(tmp_path):
    tool = _tool(tmp_path)
    tool.execute(content="My important note")
    today = date.today().isoformat()
    text = (tmp_path / "memory" / f"{today}.md").read_text()
    assert "My important note" in text


def test_execute_returns_path_info(tmp_path):
    tool = _tool(tmp_path)
    result = tool.execute(content="Some content")
    assert "memory" in result
    assert ".md" in result


def test_execute_empty_content_does_not_write(tmp_path):
    tool = _tool(tmp_path)
    result = tool.execute(content="")
    assert "Empty" in result or "empty" in result
    today = date.today().isoformat()
    assert not (tmp_path / "memory" / f"{today}.md").exists()


def test_schema_name(tmp_path):
    tool = _tool(tmp_path)
    assert tool.schema.name == "write_daily_memory"


def test_schema_not_read_only(tmp_path):
    tool = _tool(tmp_path)
    assert tool.schema.is_read_only is False


def test_schema_has_content_parameter(tmp_path):
    tool = _tool(tmp_path)
    params = tool.schema.parameters
    assert "content" in params["properties"]
    assert "content" in params["required"]


def test_schema_description_includes_absolute_memory_dir(tmp_path):
    """The description must give the absolute memory directory, not just the
    relative pattern — a relative path resolves against whatever cwd a
    shell/read tool happens to use, which is not reliably this directory."""
    tool = _tool(tmp_path)
    expected_dir = str((tmp_path / "memory").resolve())
    assert expected_dir in tool.schema.description

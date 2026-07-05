"""Tests for WriteDailyMemoryTool."""

from datetime import date
from pathlib import Path

import pytest

from minion_assist.tools.write_daily_memory import WriteDailyMemoryTool


def test_execute_creates_memory_dir(tmp_path):
    tool = WriteDailyMemoryTool(tmp_path)
    tool.execute(content="Test entry")
    assert (tmp_path / "memory").is_dir()


def test_execute_creates_dated_file(tmp_path):
    tool = WriteDailyMemoryTool(tmp_path)
    tool.execute(content="Today's note")
    today = date.today().isoformat()
    assert (tmp_path / "memory" / f"{today}.md").exists()


def test_execute_appends_content(tmp_path):
    tool = WriteDailyMemoryTool(tmp_path)
    tool.execute(content="First note")
    tool.execute(content="Second note")
    today = date.today().isoformat()
    text = (tmp_path / "memory" / f"{today}.md").read_text()
    assert "First note" in text
    assert "Second note" in text


def test_execute_content_in_file(tmp_path):
    tool = WriteDailyMemoryTool(tmp_path)
    tool.execute(content="My important note")
    today = date.today().isoformat()
    text = (tmp_path / "memory" / f"{today}.md").read_text()
    assert "My important note" in text


def test_execute_returns_path_info(tmp_path):
    tool = WriteDailyMemoryTool(tmp_path)
    result = tool.execute(content="Some content")
    assert "memory" in result
    assert ".md" in result


def test_execute_empty_content_does_not_write(tmp_path):
    tool = WriteDailyMemoryTool(tmp_path)
    result = tool.execute(content="")
    assert "Empty" in result or "empty" in result
    today = date.today().isoformat()
    assert not (tmp_path / "memory" / f"{today}.md").exists()


def test_schema_name():
    tool = WriteDailyMemoryTool(Path("/tmp"))
    assert tool.schema.name == "write_daily_memory"


def test_schema_not_read_only():
    tool = WriteDailyMemoryTool(Path("/tmp"))
    assert tool.schema.is_read_only is False


def test_schema_has_content_parameter():
    tool = WriteDailyMemoryTool(Path("/tmp"))
    params = tool.schema.parameters
    assert "content" in params["properties"]
    assert "content" in params["required"]

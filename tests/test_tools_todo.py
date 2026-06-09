"""Tests for TodoWriteTool and TodoReadTool."""

from pathlib import Path

import pytest

from mini_minion.tools.policy import PermissionPolicy
from mini_minion.tools.todo import TodoReadTool, TodoWriteTool


def test_write_and_read_round_trip(tmp_path):
    todo_path = tmp_path / "todos.json"
    writer = TodoWriteTool(todo_path)
    reader = TodoReadTool(todo_path)

    result = writer.execute(todos=["Buy milk", "Write tests"])
    assert "2" in result

    read_result = reader.execute()
    assert "Buy milk" in read_result
    assert "Write tests" in read_result


def test_read_empty_list_when_no_file(tmp_path):
    reader = TodoReadTool(tmp_path / "missing.json")
    result = reader.execute()
    assert "empty" in result.lower()


def test_write_empty_list_clears_todos(tmp_path):
    todo_path = tmp_path / "todos.json"
    writer = TodoWriteTool(todo_path)
    reader = TodoReadTool(todo_path)

    writer.execute(todos=["item1"])
    writer.execute(todos=[])
    result = reader.execute()
    assert "empty" in result.lower()


def test_write_creates_parent_dirs(tmp_path):
    todo_path = tmp_path / "deep" / "dir" / "todos.json"
    writer = TodoWriteTool(todo_path)
    writer.execute(todos=["test"])
    assert todo_path.exists()


def test_read_numbered_list(tmp_path):
    todo_path = tmp_path / "todos.json"
    writer = TodoWriteTool(todo_path)
    reader = TodoReadTool(todo_path)

    writer.execute(todos=["alpha", "beta", "gamma"])
    result = reader.execute()
    assert "1." in result
    assert "2." in result
    assert "3." in result


def test_write_blocked_in_read_only_mode(tmp_path):
    todo_path = tmp_path / "todos.json"
    policy = PermissionPolicy(workspace=tmp_path, read_only_mode=True)
    writer = TodoWriteTool(todo_path, policy=policy)
    result = writer.execute(todos=["blocked item"])
    assert "read-only" in result.lower()
    assert not todo_path.exists()


def test_write_overwrites_previous_list(tmp_path):
    todo_path = tmp_path / "todos.json"
    writer = TodoWriteTool(todo_path)
    reader = TodoReadTool(todo_path)

    writer.execute(todos=["old item"])
    writer.execute(todos=["new item"])
    result = reader.execute()
    assert "new item" in result
    assert "old item" not in result

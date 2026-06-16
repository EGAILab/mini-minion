"""Tests for FindDefinitionTool."""

from pathlib import Path

import pytest

from minion_assist.tools.find_definition import FindDefinitionTool
from minion_assist.tools.policy import PermissionPolicy


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_finds_function_definition(tmp_path):
    _write(tmp_path / "mod.py", "def my_func():\n    pass\n")
    tool = FindDefinitionTool(root=tmp_path)
    result = tool.execute(symbol="my_func")
    assert "my_func" in result
    assert "1" in result  # line number


def test_finds_class_definition(tmp_path):
    _write(tmp_path / "cls.py", "class MyClass:\n    pass\n")
    tool = FindDefinitionTool(root=tmp_path)
    result = tool.execute(symbol="MyClass")
    assert "MyClass" in result


def test_finds_assignment(tmp_path):
    _write(tmp_path / "consts.py", "MY_CONST = 42\n")
    tool = FindDefinitionTool(root=tmp_path)
    result = tool.execute(symbol="MY_CONST")
    assert "MY_CONST" in result


def test_finds_annotated_assignment(tmp_path):
    _write(tmp_path / "typed.py", "my_var: int = 5\n")
    tool = FindDefinitionTool(root=tmp_path)
    result = tool.execute(symbol="my_var")
    assert "my_var" in result


def test_returns_not_found_message(tmp_path):
    _write(tmp_path / "empty.py", "x = 1\n")
    tool = FindDefinitionTool(root=tmp_path)
    result = tool.execute(symbol="nonexistent_symbol_xyz")
    assert "no definition" in result.lower() or "not found" in result.lower()


def test_skips_unparseable_file(tmp_path):
    """Files with syntax errors are silently skipped."""
    _write(tmp_path / "broken.py", "def bad syntax:\n")
    _write(tmp_path / "good.py", "def good_func():\n    pass\n")
    tool = FindDefinitionTool(root=tmp_path)
    result = tool.execute(symbol="good_func")
    assert "good_func" in result


def test_searches_subdirectory_path(tmp_path):
    sub = tmp_path / "sub"
    _write(sub / "file.py", "def sub_func():\n    pass\n")
    tool = FindDefinitionTool(root=tmp_path)
    result = tool.execute(symbol="sub_func", path="sub")
    assert "sub_func" in result


def test_multiple_definitions_across_files(tmp_path):
    _write(tmp_path / "a.py", "def shared():\n    pass\n")
    _write(tmp_path / "b.py", "def shared():\n    pass\n")
    tool = FindDefinitionTool(root=tmp_path)
    result = tool.execute(symbol="shared")
    assert result.count("shared") >= 2


def test_path_check_applied_when_policy_set(tmp_path):
    restricted = tmp_path / "restricted"
    policy = PermissionPolicy(workspace=restricted)
    tool = FindDefinitionTool(root=tmp_path, policy=policy)
    # Searching outside workspace should be denied.
    result = tool.execute(symbol="anything")
    assert "outside" in result.lower() or "error" in result.lower()


def test_async_function_definition(tmp_path):
    _write(tmp_path / "async_mod.py", "async def async_func():\n    pass\n")
    tool = FindDefinitionTool(root=tmp_path)
    result = tool.execute(symbol="async_func")
    assert "async_func" in result

"""Tests for EditTool."""

from pathlib import Path

import pytest

from minion_assistant.tools.edit import EditTool
from minion_assistant.tools.policy import PermissionPolicy


def _tool(tmp_path: Path) -> EditTool:
    """Return an EditTool scoped to tmp_path."""
    return EditTool(PermissionPolicy.default(workspace=tmp_path))


# ---------------------------------------------------------------------------
# Basic replacement
# ---------------------------------------------------------------------------

def test_edit_replaces_string(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("hello world")
    result = _tool(tmp_path).execute(path=str(f), old_string="world", new_string="there")
    assert f.read_text() == "hello there"
    assert "Replaced 1" in result


def test_edit_returns_success_message(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("foo bar")
    result = _tool(tmp_path).execute(path=str(f), old_string="foo", new_string="baz")
    assert "Replaced" in result
    assert str(f) in result


def test_edit_preserves_rest_of_file(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("def foo():\n    pass\n\ndef bar():\n    pass\n")
    _tool(tmp_path).execute(
        path=str(f), old_string="    pass\n\ndef bar", new_string="    return 1\n\ndef bar"
    )
    content = f.read_text()
    assert "def foo():" in content
    assert "def bar():" in content
    assert "    return 1" in content


# ---------------------------------------------------------------------------
# Not found
# ---------------------------------------------------------------------------

def test_edit_not_found_returns_error(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("hello world")
    result = _tool(tmp_path).execute(path=str(f), old_string="missing", new_string="x")
    assert "Error" in result
    assert "not found" in result.lower()
    # File must be unchanged.
    assert f.read_text() == "hello world"


# ---------------------------------------------------------------------------
# Multiple matches
# ---------------------------------------------------------------------------

def test_edit_multiple_matches_without_replace_all_returns_error(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("foo foo foo")
    result = _tool(tmp_path).execute(path=str(f), old_string="foo", new_string="bar")
    assert "Error" in result
    assert "3" in result  # reports count
    # File unchanged.
    assert f.read_text() == "foo foo foo"


def test_edit_replace_all_replaces_all_occurrences(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("foo foo foo")
    result = _tool(tmp_path).execute(
        path=str(f), old_string="foo", new_string="bar", replace_all=True
    )
    assert f.read_text() == "bar bar bar"
    assert "3" in result


# ---------------------------------------------------------------------------
# File not found
# ---------------------------------------------------------------------------

def test_edit_missing_file_returns_error(tmp_path):
    result = _tool(tmp_path).execute(
        path=str(tmp_path / "does_not_exist.txt"),
        old_string="x",
        new_string="y",
    )
    assert "Error" in result
    assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

def test_edit_blocks_path_outside_workspace(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")
    result = EditTool(PermissionPolicy.default(workspace=tmp_path)).execute(
        path=str(outside), old_string="secret", new_string="pwned"
    )
    assert "Error" in result
    assert outside.read_text() == "secret"  # untouched


def test_edit_blocks_sensitive_path(tmp_path):
    ssh_file = Path.home() / ".ssh" / "id_rsa"
    result = EditTool().execute(
        path=str(ssh_file), old_string="anything", new_string="anything"
    )
    assert "Error" in result
    assert "protected" in result.lower() or "credential" in result.lower()

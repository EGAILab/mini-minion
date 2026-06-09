"""Tests for GrepTool."""

from pathlib import Path

import pytest

from mini_minion.tools.grep import GrepTool
from mini_minion.tools.policy import PermissionPolicy


def _tool(tmp_path: Path) -> GrepTool:
    return GrepTool(PermissionPolicy.default(workspace=tmp_path))


# ---------------------------------------------------------------------------
# Basic search
# ---------------------------------------------------------------------------

def test_grep_finds_match_in_file(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("hello world\nfoo bar\n")
    result = _tool(tmp_path).execute(pattern="hello", path=str(f))
    assert "hello world" in result
    assert ":1:" in result  # line 1


def test_grep_returns_no_matches_message_when_nothing_found(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("apple banana cherry\n")
    result = _tool(tmp_path).execute(pattern="mango", path=str(f))
    assert "No matches" in result


def test_grep_includes_file_path_in_output(tmp_path):
    f = tmp_path / "source.py"
    f.write_text("def my_func():\n    pass\n")
    result = _tool(tmp_path).execute(pattern="my_func", path=str(f))
    assert "source.py" in result


# ---------------------------------------------------------------------------
# Directory search
# ---------------------------------------------------------------------------

def test_grep_searches_directory_recursively(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "a.py").write_text("import os\n")
    (sub / "b.py").write_text("import sys\n")
    result = _tool(tmp_path).execute(pattern="import", path=str(tmp_path))
    assert "import os" in result
    assert "import sys" in result


def test_grep_include_filter_limits_to_matching_files(tmp_path):
    (tmp_path / "code.py").write_text("hello python\n")
    (tmp_path / "notes.txt").write_text("hello notes\n")
    result = _tool(tmp_path).execute(pattern="hello", path=str(tmp_path), include="*.py")
    assert "code.py" in result
    assert "notes.txt" not in result


# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------

def test_grep_uses_regex(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("abc\n123\nfoo456bar\n")
    result = _tool(tmp_path).execute(pattern=r"\d+", path=str(f))
    assert "123" in result
    assert "foo456bar" in result


def test_grep_invalid_regex_returns_error(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("test\n")
    result = _tool(tmp_path).execute(pattern="[invalid", path=str(f))
    assert "Error" in result
    assert "regex" in result.lower()


# ---------------------------------------------------------------------------
# Case insensitive
# ---------------------------------------------------------------------------

def test_grep_ignore_case(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("Hello World\n")
    result = _tool(tmp_path).execute(pattern="hello", path=str(f), ignore_case=True)
    assert "Hello World" in result


def test_grep_case_sensitive_by_default(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("Hello World\n")
    result = _tool(tmp_path).execute(pattern="hello", path=str(f))
    assert "No matches" in result


# ---------------------------------------------------------------------------
# Context lines
# ---------------------------------------------------------------------------

def test_grep_context_lines_shows_surrounding(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("line1\nTARGET\nline3\n")
    result = _tool(tmp_path).execute(pattern="TARGET", path=str(f), context_lines=1)
    assert "line1" in result
    assert "TARGET" in result
    assert "line3" in result


def test_grep_context_uses_dash_separator_for_non_match_lines(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("before\nMATCH\nafter\n")
    result = _tool(tmp_path).execute(pattern="MATCH", path=str(f), context_lines=1)
    # Context lines use '-' separator; match line uses ':'
    assert "-before" in result or ":before" not in result
    assert ":2:MATCH" in result


# ---------------------------------------------------------------------------
# max_results
# ---------------------------------------------------------------------------

def test_grep_truncates_at_max_results(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("\n".join(f"line{i}" for i in range(200)) + "\n")
    result = _tool(tmp_path).execute(pattern="line", path=str(f), max_results=10)
    assert "Truncated" in result


# ---------------------------------------------------------------------------
# Non-existent path
# ---------------------------------------------------------------------------

def test_grep_nonexistent_path_returns_error(tmp_path):
    result = _tool(tmp_path).execute(
        pattern="test", path=str(tmp_path / "no_such_dir")
    )
    assert "Error" in result


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

def test_grep_blocks_path_outside_workspace(tmp_path):
    outside = tmp_path.parent
    result = GrepTool(PermissionPolicy.default(workspace=tmp_path)).execute(
        pattern="anything", path=str(outside)
    )
    assert "Error" in result

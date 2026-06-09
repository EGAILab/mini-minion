"""Tests for PatchPreviewTool."""

from pathlib import Path

import pytest

from mini_minion.tools.patch import PatchPreviewTool
from mini_minion.tools.policy import PermissionPolicy


def _tool(tmp_path: Path) -> PatchPreviewTool:
    return PatchPreviewTool(PermissionPolicy.default(workspace=tmp_path))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_patch_preview_schema_name(tmp_path):
    assert _tool(tmp_path).schema.name == "patch_preview"


def test_patch_preview_schema_is_read_only(tmp_path):
    assert _tool(tmp_path).schema.is_read_only is True


def test_patch_preview_schema_required_params(tmp_path):
    required = _tool(tmp_path).schema.parameters["required"]
    assert "path" in required
    assert "old_string" in required
    assert "new_string" in required


# ---------------------------------------------------------------------------
# Happy path — diff output
# ---------------------------------------------------------------------------


def test_patch_preview_returns_unified_diff(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("hello world")

    result = _tool(tmp_path).execute(path=str(f), old_string="world", new_string="there")

    assert "---" in result
    assert "+++" in result
    assert "-hello world" in result
    assert "+hello there" in result


def test_patch_preview_does_not_modify_file(tmp_path):
    f = tmp_path / "unchanged.txt"
    original = "original content"
    f.write_text(original)

    _tool(tmp_path).execute(path=str(f), old_string="original", new_string="new")

    assert f.read_text() == original


def test_patch_preview_multiline_file(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("def foo():\n    pass\n\ndef bar():\n    pass\n")

    result = _tool(tmp_path).execute(
        path=str(f), old_string="    pass\n\ndef bar", new_string="    return 1\n\ndef bar"
    )

    assert "---" in result
    assert "return 1" in result


def test_patch_preview_identical_strings_returns_no_changes(tmp_path):
    f = tmp_path / "same.txt"
    f.write_text("no change here")

    result = _tool(tmp_path).execute(path=str(f), old_string="no change", new_string="no change")

    assert "no changes" in result.lower()


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_patch_preview_old_string_not_found(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("hello world")

    result = _tool(tmp_path).execute(path=str(f), old_string="missing", new_string="x")

    assert "Error" in result
    assert "not found" in result.lower()


def test_patch_preview_ambiguous_without_replace_all(tmp_path):
    f = tmp_path / "dup.txt"
    f.write_text("foo foo foo")

    result = _tool(tmp_path).execute(path=str(f), old_string="foo", new_string="bar")

    assert "Error" in result
    assert "3" in result  # mentions the count


def test_patch_preview_replace_all_previews_all_occurrences(tmp_path):
    f = tmp_path / "dup.txt"
    f.write_text("foo foo foo")

    result = _tool(tmp_path).execute(
        path=str(f), old_string="foo", new_string="bar", replace_all=True
    )

    assert "---" in result and "+++" in result
    assert "bar bar bar" in result


def test_patch_preview_file_not_found(tmp_path):
    result = _tool(tmp_path).execute(
        path=str(tmp_path / "nonexistent.txt"),
        old_string="x",
        new_string="y",
    )

    assert "Error" in result
    assert "not found" in result.lower()


def test_patch_preview_blocks_path_outside_root(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")

    result = _tool(tmp_path).execute(
        path=str(outside), old_string="secret", new_string="x"
    )

    assert "outside the workspace root" in result


# ---------------------------------------------------------------------------
# Diff format specifics
# ---------------------------------------------------------------------------


def test_patch_preview_diff_uses_filename(tmp_path):
    f = tmp_path / "myfile.py"
    f.write_text("x = 1")

    result = _tool(tmp_path).execute(path=str(f), old_string="x = 1", new_string="x = 2")

    assert "myfile.py" in result

"""Tests for GitStatusTool, GitDiffTool, and GitCommitTool."""

import subprocess

import pytest

from minion_assistant.tools.git import GitCommitTool, GitDiffTool, GitStatusTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_repo(path):
    """Create a minimal git repo at path with an initial commit."""
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, capture_output=True)
    (path / "init.txt").write_text("initial", encoding="utf-8")
    subprocess.run(["git", "add", "init.txt"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, capture_output=True, check=True)


# ---------------------------------------------------------------------------
# GitStatusTool
# ---------------------------------------------------------------------------


def test_git_status_schema():
    tool = GitStatusTool()
    assert tool.schema.name == "git_status"
    assert tool.schema.is_read_only is True


def test_git_status_shows_untracked(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "new_file.txt").write_text("hello", encoding="utf-8")

    result = GitStatusTool(cwd=tmp_path).execute()

    assert "new_file.txt" in result


def test_git_status_shows_modified(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "init.txt").write_text("changed", encoding="utf-8")

    result = GitStatusTool(cwd=tmp_path).execute()

    assert "init.txt" in result


def test_git_status_clean_repo(tmp_path):
    _init_repo(tmp_path)

    result = GitStatusTool(cwd=tmp_path).execute()

    # Clean repo has no changed files; should not error and should mention branch.
    assert "Error" not in result


def test_git_status_no_git_repo(tmp_path):
    result = GitStatusTool(cwd=tmp_path).execute()

    # Not a git repo — git exits non-zero and writes to stderr.
    # Result should contain some output (the git error), not crash.
    assert result  # non-empty


# ---------------------------------------------------------------------------
# GitDiffTool
# ---------------------------------------------------------------------------


def test_git_diff_schema():
    tool = GitDiffTool()
    assert tool.schema.name == "git_diff"
    assert tool.schema.is_read_only is True


def test_git_diff_unstaged_changes(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "init.txt").write_text("modified content", encoding="utf-8")

    result = GitDiffTool(cwd=tmp_path).execute()

    assert "modified content" in result or "+" in result


def test_git_diff_staged_changes(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "init.txt").write_text("staged change", encoding="utf-8")
    subprocess.run(["git", "add", "init.txt"], cwd=tmp_path, capture_output=True)

    result = GitDiffTool(cwd=tmp_path).execute(staged=True)

    assert "staged change" in result or "+" in result


def test_git_diff_unstaged_does_not_show_staged(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "init.txt").write_text("staged", encoding="utf-8")
    subprocess.run(["git", "add", "init.txt"], cwd=tmp_path, capture_output=True)

    # Unstaged diff should be empty (change was staged, not modified after staging).
    result = GitDiffTool(cwd=tmp_path).execute()

    assert "(no output)" in result or result.strip() == ""


def test_git_diff_with_path(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)

    result = GitDiffTool(cwd=tmp_path).execute(staged=True, path=str(tmp_path / "a.txt"))

    assert "alpha" in result
    assert "beta" not in result


# ---------------------------------------------------------------------------
# GitCommitTool
# ---------------------------------------------------------------------------


def test_git_commit_schema():
    tool = GitCommitTool()
    assert tool.schema.name == "git_commit"
    assert "message" in tool.schema.parameters["required"]


def test_git_commit_creates_commit(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "new.txt").write_text("content", encoding="utf-8")
    subprocess.run(["git", "add", "new.txt"], cwd=tmp_path, capture_output=True)

    result = GitCommitTool(cwd=tmp_path, confirm=None).execute(message="test commit")

    assert "Error" not in result
    # Verify commit actually happened.
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"], cwd=tmp_path, capture_output=True, text=True
    )
    assert "test commit" in log.stdout


def test_git_commit_stages_files(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "staged_by_tool.txt").write_text("hello", encoding="utf-8")

    result = GitCommitTool(cwd=tmp_path, confirm=None).execute(
        message="commit via tool",
        files=["staged_by_tool.txt"],
    )

    assert "Error" not in result
    log = subprocess.run(
        ["git", "log", "--oneline", "-1"], cwd=tmp_path, capture_output=True, text=True
    )
    assert "commit via tool" in log.stdout


def test_git_commit_cancelled_by_confirm(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "c.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "c.txt"], cwd=tmp_path, capture_output=True)

    result = GitCommitTool(cwd=tmp_path, confirm=lambda _: False).execute(
        message="should not commit"
    )

    assert "cancelled" in result.lower()
    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=tmp_path, capture_output=True, text=True
    )
    assert "should not commit" not in log.stdout


def test_git_commit_confirm_receives_command_preview(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "d.txt").write_text("y", encoding="utf-8")
    subprocess.run(["git", "add", "d.txt"], cwd=tmp_path, capture_output=True)

    previews: list[str] = []
    GitCommitTool(cwd=tmp_path, confirm=lambda p: previews.append(p) or False).execute(
        message="preview test"
    )

    assert previews and "preview test" in previews[0]


def test_git_commit_confirm_includes_files_in_preview(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "e.txt").write_text("z", encoding="utf-8")

    previews: list[str] = []
    GitCommitTool(cwd=tmp_path, confirm=lambda p: previews.append(p) or False).execute(
        message="with files",
        files=["e.txt"],
    )

    assert previews and "e.txt" in previews[0]

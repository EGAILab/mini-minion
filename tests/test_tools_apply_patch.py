"""Tests for ApplyPatchTool."""

import subprocess
from pathlib import Path

import pytest

from mini_minion.tools.apply_patch import ApplyPatchTool
from mini_minion.tools.policy import PermissionPolicy


def _git_available() -> bool:
    """Return True if git is on PATH."""
    try:
        subprocess.run(["git", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _init_git_repo(path: Path) -> None:
    """Initialise a minimal git repo with one commit so git apply works."""
    subprocess.run(["git", "init"], cwd=path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, capture_output=True)
    (path / "initial.txt").write_text("initial\n")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, capture_output=True)


requires_git = pytest.mark.skipif(not _git_available(), reason="git not installed")


@requires_git
def test_apply_patch_check_only_valid(tmp_path):
    """check_only=True returns success without touching files."""
    _init_git_repo(tmp_path)
    target = tmp_path / "file.txt"
    target.write_text("hello\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add file"], cwd=tmp_path, capture_output=True)

    patch = (
        "--- a/file.txt\n"
        "+++ b/file.txt\n"
        "@@ -1 +1 @@\n"
        "-hello\n"
        "+world\n"
    )
    tool = ApplyPatchTool(cwd=tmp_path)
    result = tool.execute(patch=patch, check_only=True)
    assert "cleanly" in result.lower() or "dry run" in result.lower()
    # File must be unchanged.
    assert target.read_text() == "hello\n"


@requires_git
def test_apply_patch_applies_changes(tmp_path):
    """apply_patch changes the file on disk."""
    _init_git_repo(tmp_path)
    target = tmp_path / "file.txt"
    target.write_text("hello\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add file"], cwd=tmp_path, capture_output=True)

    patch = (
        "--- a/file.txt\n"
        "+++ b/file.txt\n"
        "@@ -1 +1 @@\n"
        "-hello\n"
        "+world\n"
    )
    tool = ApplyPatchTool(cwd=tmp_path)
    result = tool.execute(patch=patch)
    assert "applied" in result.lower() or "success" in result.lower()
    assert target.read_text() == "world\n"


@requires_git
def test_apply_patch_invalid_patch_returns_error(tmp_path):
    """An invalid patch string returns an error message."""
    _init_git_repo(tmp_path)
    tool = ApplyPatchTool(cwd=tmp_path)
    result = tool.execute(patch="this is not a valid patch")
    assert "error" in result.lower()


def test_apply_patch_blocks_in_read_only_mode(tmp_path):
    """read_only_mode blocks the apply (not check_only) path."""
    policy = PermissionPolicy(workspace=tmp_path, read_only_mode=True)
    tool = ApplyPatchTool(cwd=tmp_path, policy=policy)
    result = tool.execute(patch="--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n")
    assert "read-only" in result.lower()


def test_apply_patch_check_only_allowed_in_read_only_mode(tmp_path):
    """check_only skips the policy check (it's a dry run)."""
    # We don't have a real git repo here, so this will fail at the git level,
    # but the policy check should not be the reason.
    policy = PermissionPolicy(workspace=tmp_path, read_only_mode=True)
    tool = ApplyPatchTool(cwd=tmp_path, policy=policy)
    result = tool.execute(patch="invalid patch", check_only=True)
    # Should reach git, not be blocked by policy.
    assert "read-only" not in result.lower()


def test_apply_patch_git_not_found_returns_error(monkeypatch):
    """When git is not on PATH, execute() returns a clear error before touching any file."""
    import mini_minion.tools.apply_patch as ap_module
    # Pretend git is missing from PATH.
    monkeypatch.setattr(ap_module, "_GIT_PATH", None)
    # Also make shutil.which return None so the runtime re-check agrees.
    monkeypatch.setattr(ap_module.shutil, "which", lambda _: None)

    tool = ApplyPatchTool()
    result = tool.execute(patch="--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n")
    assert "git" in result.lower()
    assert "Error" in result or "error" in result.lower()

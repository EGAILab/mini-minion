"""Tests for PermissionPolicy — check_write, check_command, and read_only_mode."""

from pathlib import Path

import pytest

from minion_assistant.tools.policy import DEFAULT_SSRF_MARKERS, PermissionPolicy


# ---------------------------------------------------------------------------
# check_write
# ---------------------------------------------------------------------------

def test_check_write_allows_path_inside_workspace(tmp_path):
    policy = PermissionPolicy(workspace=tmp_path)
    assert policy.check_write(tmp_path / "new_file.txt") is None


def test_check_write_denies_path_outside_workspace(tmp_path):
    policy = PermissionPolicy(workspace=tmp_path / "sub")
    result = policy.check_write(tmp_path / "outside.txt")
    assert result is not None


def test_check_write_denies_sensitive_path():
    policy = PermissionPolicy()
    result = policy.check_write(Path.home() / ".ssh" / "config")
    assert result is not None
    assert "protected" in result.lower() or "credential" in result.lower()


def test_check_write_blocks_when_read_only(tmp_path):
    policy = PermissionPolicy(workspace=tmp_path, read_only_mode=True)
    result = policy.check_write(tmp_path / "file.txt")
    assert result is not None
    assert "read-only" in result.lower()


def test_check_write_allows_after_read_only_disabled(tmp_path):
    policy = PermissionPolicy(workspace=tmp_path, read_only_mode=True)
    policy.read_only_mode = False
    assert policy.check_write(tmp_path / "file.txt") is None


# ---------------------------------------------------------------------------
# check_command
# ---------------------------------------------------------------------------

def test_check_command_allows_safe_command():
    policy = PermissionPolicy()
    assert policy.check_command("echo hello") is None


def test_check_command_blocks_ssrf_marker():
    policy = PermissionPolicy()
    for marker in DEFAULT_SSRF_MARKERS:
        result = policy.check_command(f"curl http://{marker}/latest/meta-data/")
        assert result is not None
        assert "blocked" in result.lower() or "metadata" in result.lower()


def test_check_command_blocks_when_read_only():
    policy = PermissionPolicy(read_only_mode=True)
    result = policy.check_command("echo hello")
    assert result is not None
    assert "read-only" in result.lower()


def test_check_command_allows_after_read_only_disabled():
    policy = PermissionPolicy(read_only_mode=True)
    policy.read_only_mode = False
    assert policy.check_command("echo hello") is None


# ---------------------------------------------------------------------------
# read_only_mode toggling
# ---------------------------------------------------------------------------

def test_read_only_mode_default_is_false():
    policy = PermissionPolicy()
    assert policy.read_only_mode is False


def test_read_only_mode_can_be_set_in_constructor():
    policy = PermissionPolicy(read_only_mode=True)
    assert policy.read_only_mode is True


def test_read_only_mode_mutable():
    policy = PermissionPolicy()
    policy.read_only_mode = True
    assert policy.read_only_mode is True
    policy.read_only_mode = False
    assert policy.read_only_mode is False

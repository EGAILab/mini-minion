"""Tests for PermissionPolicy."""

from pathlib import Path

import pytest

from minion_assistant.tools.policy import DEFAULT_SSRF_MARKERS, PermissionPolicy


# ---------------------------------------------------------------------------
# check_path
# ---------------------------------------------------------------------------

def test_check_path_allows_file_inside_workspace(tmp_path):
    policy = PermissionPolicy(workspace=tmp_path)
    result = policy.check_path(tmp_path / "file.txt")
    assert result is None


def test_check_path_denies_file_outside_workspace(tmp_path):
    policy = PermissionPolicy(workspace=tmp_path / "sub")
    result = policy.check_path(tmp_path / "outside.txt")
    assert result is not None
    assert "outside" in result.lower() or "workspace" in result.lower()


def test_check_path_no_workspace_allows_any_safe_path(tmp_path):
    policy = PermissionPolicy(workspace=None)
    result = policy.check_path(tmp_path / "any.txt")
    assert result is None


def test_check_path_denies_ssh_dir():
    policy = PermissionPolicy()
    ssh_file = Path.home() / ".ssh" / "id_rsa"
    result = policy.check_path(ssh_file)
    assert result is not None
    assert "protected" in result.lower() or "credential" in result.lower()


def test_check_path_denies_aws_dir():
    policy = PermissionPolicy()
    result = policy.check_path(Path.home() / ".aws" / "credentials")
    assert result is not None


# ---------------------------------------------------------------------------
# check_url
# ---------------------------------------------------------------------------

def test_check_url_allows_normal_url():
    policy = PermissionPolicy()
    result = policy.check_url("https://example.com/page")
    assert result is None


def test_check_url_blocks_aws_metadata():
    policy = PermissionPolicy()
    result = policy.check_url("http://169.254.169.254/latest/meta-data/")
    assert result is not None
    assert "blocked" in result.lower()


def test_check_url_blocks_gcp_metadata():
    policy = PermissionPolicy()
    result = policy.check_url("http://metadata.google.internal/")
    assert result is not None


def test_check_url_blocks_ecs_metadata():
    policy = PermissionPolicy()
    result = policy.check_url("http://169.254.170.2/v2/metadata")
    assert result is not None


def test_check_url_custom_markers():
    policy = PermissionPolicy(ssrf_markers=frozenset({"internal.corp"}))
    assert policy.check_url("https://internal.corp/api") is not None
    assert policy.check_url("https://external.com/api") is None


# ---------------------------------------------------------------------------
# default() classmethod
# ---------------------------------------------------------------------------

def test_default_policy_has_ssrf_markers():
    policy = PermissionPolicy.default()
    assert "169.254.169.254" in policy.ssrf_markers


def test_default_policy_with_workspace(tmp_path):
    policy = PermissionPolicy.default(workspace=tmp_path)
    assert policy.workspace == tmp_path
    assert policy.check_path(tmp_path / "file.txt") is None


def test_default_ssrf_markers_set_exported():
    assert "169.254.169.254" in DEFAULT_SSRF_MARKERS
    assert "metadata.google.internal" in DEFAULT_SSRF_MARKERS

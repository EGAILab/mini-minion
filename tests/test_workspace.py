"""Tests for workspace.py: directory creation, marker attestation, WorkspaceVanishedError.

All tests are self-contained and use tmp_path — no filesystem side effects.
"""

from __future__ import annotations

import pytest

from minion_assist.workspace import (
    MARKER_FILENAME,
    WorkspaceVanishedError,
    _compute_marker,
    agent_workspace_root,
    check_workspace,
    ensure_workspace,
)


# ---------------------------------------------------------------------------
# ensure_workspace
# ---------------------------------------------------------------------------

def test_ensure_workspace_creates_directory(tmp_path):
    """ensure_workspace creates the target directory if absent."""
    root = tmp_path / "workspace"
    assert not root.exists()
    ensure_workspace(root)
    assert root.exists()


def test_ensure_workspace_creates_marker(tmp_path):
    """ensure_workspace writes a .workspace-marker file."""
    root = tmp_path / "ws"
    ensure_workspace(root)
    assert (root / MARKER_FILENAME).exists()


def test_ensure_workspace_idempotent(tmp_path):
    """Calling ensure_workspace twice does not raise or corrupt the marker."""
    root = tmp_path / "ws"
    ensure_workspace(root)
    marker_content = (root / MARKER_FILENAME).read_text()
    ensure_workspace(root)  # second call
    assert (root / MARKER_FILENAME).read_text() == marker_content


def test_ensure_workspace_does_not_overwrite_existing_marker(tmp_path):
    """If a marker already exists, ensure_workspace leaves it untouched."""
    root = tmp_path / "ws"
    root.mkdir()
    (root / "AGENTS.md").write_text("agents")
    ensure_workspace(root)
    first_hash = (root / MARKER_FILENAME).read_text()

    # Add another file and call again — marker must not change.
    (root / "SOUL.md").write_text("soul")
    ensure_workspace(root)
    assert (root / MARKER_FILENAME).read_text() == first_hash


def test_ensure_workspace_creates_parent_directories(tmp_path):
    """ensure_workspace creates nested parents (like mkdir -p)."""
    root = tmp_path / "deep" / "nested" / "ws"
    ensure_workspace(root)
    assert root.exists()
    assert (root / MARKER_FILENAME).exists()


# ---------------------------------------------------------------------------
# check_workspace
# ---------------------------------------------------------------------------

def test_check_workspace_passes_on_healthy_workspace(tmp_path):
    """check_workspace does not raise when dir and marker both exist."""
    root = tmp_path / "ws"
    ensure_workspace(root)
    check_workspace(root)  # should not raise


def test_check_workspace_raises_when_dir_missing(tmp_path):
    """check_workspace raises WorkspaceVanishedError when directory is gone."""
    root = tmp_path / "ws"
    ensure_workspace(root)
    import shutil
    shutil.rmtree(root)
    with pytest.raises(WorkspaceVanishedError, match="vanished"):
        check_workspace(root)


def test_check_workspace_raises_when_marker_missing(tmp_path):
    """check_workspace raises WorkspaceVanishedError when marker file is deleted."""
    root = tmp_path / "ws"
    ensure_workspace(root)
    (root / MARKER_FILENAME).unlink()
    with pytest.raises(WorkspaceVanishedError, match="vanished"):
        check_workspace(root)


def test_workspace_vanished_error_message_contains_path(tmp_path):
    """WorkspaceVanishedError message includes the vanished path."""
    root = tmp_path / "my_workspace"
    ensure_workspace(root)
    import shutil
    shutil.rmtree(root)
    with pytest.raises(WorkspaceVanishedError) as exc_info:
        check_workspace(root)
    assert "my_workspace" in str(exc_info.value)


# ---------------------------------------------------------------------------
# agent_workspace_root
# ---------------------------------------------------------------------------

def test_agent_workspace_root_returns_per_agent_dir(tmp_path):
    """Returns {workspace}/workspaces/{agent_id}/ when it exists."""
    per_agent = tmp_path / "workspaces" / "researcher"
    per_agent.mkdir(parents=True)
    result = agent_workspace_root(tmp_path, "researcher")
    assert result == per_agent


def test_agent_workspace_root_falls_back_to_main(tmp_path):
    """Falls back to workspaces/main/ when per-agent dir does not exist."""
    main_ws = tmp_path / "workspaces" / "main"
    main_ws.mkdir(parents=True)
    result = agent_workspace_root(tmp_path, "researcher")
    assert result == main_ws


def test_agent_workspace_root_prefers_per_agent_over_main(tmp_path):
    """Per-agent dir takes priority over main even when main exists."""
    per_agent = tmp_path / "workspaces" / "researcher"
    per_agent.mkdir(parents=True)
    main_ws = tmp_path / "workspaces" / "main"
    main_ws.mkdir(parents=True)
    result = agent_workspace_root(tmp_path, "researcher")
    assert result == per_agent


def test_agent_workspace_root_returns_none_when_no_workspace(tmp_path):
    """Returns None when neither per-agent nor main workspace dirs exist."""
    result = agent_workspace_root(tmp_path, "researcher")
    assert result is None


def test_agent_workspace_root_main_agent_uses_per_agent_dir(tmp_path):
    """'main' agent resolves its own per-agent dir when present."""
    per_main = tmp_path / "workspaces" / "main"
    per_main.mkdir(parents=True)
    result = agent_workspace_root(tmp_path, "main")
    assert result == per_main


# ---------------------------------------------------------------------------
# _compute_marker
# ---------------------------------------------------------------------------

def test_compute_marker_is_deterministic(tmp_path):
    """Same file set produces the same hash on repeated calls."""
    (tmp_path / "AGENTS.md").write_text("agents")
    h1 = _compute_marker(tmp_path)
    h2 = _compute_marker(tmp_path)
    assert h1 == h2


def test_compute_marker_excludes_marker_file_itself(tmp_path):
    """The marker file is excluded from the hash so ensure_workspace is idempotent."""
    (tmp_path / "AGENTS.md").write_text("agents")
    h_before = _compute_marker(tmp_path)
    # Write the marker file itself
    (tmp_path / MARKER_FILENAME).write_text("some-hash")
    h_after = _compute_marker(tmp_path)
    assert h_before == h_after


def test_compute_marker_changes_when_files_change(tmp_path):
    """Hash changes when a new file is added."""
    (tmp_path / "AGENTS.md").write_text("agents")
    h1 = _compute_marker(tmp_path)
    (tmp_path / "SOUL.md").write_text("soul")
    h2 = _compute_marker(tmp_path)
    assert h1 != h2

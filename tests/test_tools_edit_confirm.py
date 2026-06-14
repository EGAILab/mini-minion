"""Tests for EditTool 'confirm' callback integration."""

from minion_assistant.tools.edit import EditTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path, content):
    """Write content to a Path object."""
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_edit_no_confirm_edits(tmp_path):
    """When no confirm callback is passed, EditTool edits unconditionally."""
    target = tmp_path / "f.txt"
    _write(target, "hello world")
    EditTool().execute(path=str(target), old_string="world", new_string="earth")
    assert target.read_text() == "hello earth"


def test_edit_confirm_approved_edits(tmp_path):
    """When confirm returns True, the edit proceeds."""
    target = tmp_path / "f.txt"
    _write(target, "foo bar")
    EditTool(confirm=lambda desc: True).execute(path=str(target), old_string="bar", new_string="baz")
    assert target.read_text() == "foo baz"


def test_edit_confirm_rejected_does_not_edit(tmp_path):
    """When confirm returns False, the file is left unchanged."""
    target = tmp_path / "f.txt"
    _write(target, "original content")
    result = EditTool(confirm=lambda desc: False).execute(
        path=str(target), old_string="original", new_string="modified"
    )
    assert target.read_text() == "original content"
    assert "cancelled" in result.lower()


def test_edit_confirm_receives_description(tmp_path):
    """The confirm callback receives a human-readable description of the edit."""
    target = tmp_path / "f.txt"
    _write(target, "alpha beta gamma")
    received: list[str] = []

    def capturing(desc: str) -> bool:
        received.append(desc)
        return True

    EditTool(confirm=capturing).execute(path=str(target), old_string="beta", new_string="omega")
    assert len(received) == 1
    # Description should identify the file or the replaced string.
    assert "beta" in received[0] or target.name in received[0]


def test_edit_confirm_policy_read_only_blocks_before_confirm(tmp_path):
    """Policy read_only_mode fires before the confirm callback."""
    from minion_assistant.tools.policy import PermissionPolicy
    policy = PermissionPolicy(read_only_mode=True)

    target = tmp_path / "f.txt"
    _write(target, "content here")
    called: list[bool] = []
    result = EditTool(policy, confirm=lambda d: called.append(True) or True).execute(
        path=str(target), old_string="content", new_string="replaced"
    )
    assert not called
    assert "Error" in result or "read-only" in result.lower()
    assert target.read_text() == "content here"

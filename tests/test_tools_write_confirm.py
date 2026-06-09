"""Tests for WriteTool 'confirm' callback integration."""

from mini_minion.tools.write import WriteTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _approving() -> bool:
    """Confirmation callback that always approves."""
    return True


def _rejecting(description: str) -> bool:
    """Confirmation callback that always rejects."""
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_write_no_confirm_writes(tmp_path):
    """When no confirm callback is passed, WriteTool writes unconditionally."""
    target = tmp_path / "out.txt"
    WriteTool().execute(path=str(target), content="hello")
    assert target.read_text() == "hello"


def test_write_confirm_approved_writes(tmp_path):
    """When confirm callback returns True, the write proceeds."""
    target = tmp_path / "out.txt"
    WriteTool(confirm=lambda desc: True).execute(path=str(target), content="approved")
    assert target.read_text() == "approved"


def test_write_confirm_rejected_does_not_write(tmp_path):
    """When confirm callback returns False, the file is not created."""
    target = tmp_path / "out.txt"
    result = WriteTool(confirm=lambda desc: False).execute(path=str(target), content="rejected")
    assert not target.exists()
    assert "cancelled" in result.lower()


def test_write_confirm_receives_description(tmp_path):
    """The confirm callback receives a human-readable description string."""
    target = tmp_path / "out.txt"
    received: list[str] = []

    def capturing_confirm(desc: str) -> bool:
        received.append(desc)
        return True

    WriteTool(confirm=capturing_confirm).execute(path=str(target), content="abc")
    assert len(received) == 1
    # Description should mention the file path.
    assert str(target) in received[0] or target.name in received[0]


def test_write_confirm_description_includes_char_count(tmp_path):
    """The description passed to confirm includes the character count."""
    target = tmp_path / "out.txt"
    received: list[str] = []

    def capturing_confirm(desc: str) -> bool:
        received.append(desc)
        return True

    content = "x" * 50
    WriteTool(confirm=capturing_confirm).execute(path=str(target), content=content)
    assert "50" in received[0]


def test_write_confirm_policy_read_only_blocks_before_confirm(tmp_path):
    """Policy read_only_mode check fires before the confirm callback."""
    from mini_minion.tools.policy import PermissionPolicy
    policy = PermissionPolicy(read_only_mode=True)

    called: list[bool] = []
    target = tmp_path / "out.txt"
    result = WriteTool(policy=policy, confirm=lambda d: called.append(True) or True).execute(
        path=str(target), content="data"
    )
    # Policy blocks first — confirm should never be called.
    assert not called
    assert "Error" in result or "read-only" in result.lower()
    assert not target.exists()

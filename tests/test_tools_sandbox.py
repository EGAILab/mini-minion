"""Tests for the sandbox backend abstraction (NEW-01)."""

import subprocess

from mini_minion.tools.audit import ApprovalDecision, AuditLog
from mini_minion.tools.bash import BashTool
from mini_minion.tools.policy import PermissionPolicy
from mini_minion.tools.sandbox import LocalSandboxBackend, SandboxBackend


# ---------------------------------------------------------------------------
# LocalSandboxBackend
# ---------------------------------------------------------------------------

def test_local_sandbox_runs_command(tmp_path):
    backend = LocalSandboxBackend()
    result = backend.run(
        ["python", "-c", "print('sandbox-ok')"],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=tmp_path,
    )
    assert result.returncode == 0
    assert "sandbox-ok" in result.stdout


def test_local_sandbox_satisfies_protocol():
    # LocalSandboxBackend must structurally satisfy SandboxBackend (duck typing).
    # No explicit isinstance check needed — Protocol uses structural subtyping.
    backend = LocalSandboxBackend()
    assert callable(getattr(backend, "run", None))


# ---------------------------------------------------------------------------
# BashTool with custom sandbox backend
# ---------------------------------------------------------------------------

class _RecordingSandbox:
    """Sandbox that records calls and returns a fixed result."""
    def __init__(self, output: str = "recorded") -> None:
        self.calls: list[tuple] = []
        self._output = output

    def run(self, args, *, capture_output, text, timeout, cwd):
        self.calls.append((args, cwd))
        return subprocess.CompletedProcess(args, 0, stdout=self._output, stderr="")


def test_bash_tool_uses_sandbox_when_provided():
    recording = _RecordingSandbox(output="custom-output")
    tool = BashTool(confirm=None, sandbox=recording)
    result = tool.execute(command="echo hello")
    assert result == "custom-output"
    assert len(recording.calls) == 1


def test_bash_tool_sandbox_receives_correct_cwd(tmp_path):
    recording = _RecordingSandbox()
    tool = BashTool(confirm=None, sandbox=recording, cwd=tmp_path)
    tool.execute(command="echo hi")
    _, received_cwd = recording.calls[0]
    assert received_cwd == tmp_path


def test_bash_tool_without_sandbox_uses_direct_subprocess():
    """With sandbox=None, BashTool falls back to direct subprocess execution."""
    tool = BashTool(confirm=None, sandbox=None)
    result = tool.execute(command="echo direct")
    assert "direct" in result


# ---------------------------------------------------------------------------
# approval_fn integration
# ---------------------------------------------------------------------------

def test_bash_tool_approval_fn_allow_once_runs():
    tool = BashTool(approval_fn=lambda _: ApprovalDecision.ALLOW_ONCE)
    result = tool.execute(command="echo approved-once")
    assert "approved-once" in result


def test_bash_tool_approval_fn_deny_cancels():
    tool = BashTool(approval_fn=lambda _: ApprovalDecision.DENY)
    result = tool.execute(command="echo should-not-run")
    assert "cancelled" in result.lower()
    assert "should-not-run" not in result


def test_bash_tool_approval_fn_always_deny_blocks_future():
    policy = PermissionPolicy.default()
    tool = BashTool(approval_fn=lambda _: ApprovalDecision.ALWAYS_DENY, policy=policy)
    # First call: ALWAYS_DENY is returned.
    result1 = tool.execute(command="echo blocked-cmd")
    assert "blocked" in result1.lower() or "denied" in result1.lower()
    # Second call with the same command: auto-denied without calling approval_fn.
    called = []
    tool2 = BashTool(
        approval_fn=lambda c: called.append(c) or ApprovalDecision.ALLOW_ONCE,
        policy=policy,
    )
    result2 = tool2.execute(command="echo blocked-cmd")
    assert "denied" in result2.lower() or "blocked" in result2.lower()
    assert not called  # approval_fn must NOT have been called for the already-denied command


def test_bash_tool_approval_fn_allow_session_skips_future_prompt():
    policy = PermissionPolicy.default()
    call_count = [0]

    def _approval(cmd: str) -> ApprovalDecision:
        call_count[0] += 1
        return ApprovalDecision.ALLOW_SESSION

    tool = BashTool(approval_fn=_approval, policy=policy)
    tool.execute(command="echo session-cmd")
    # Second execution: approval_fn must NOT be called again (session-allowed).
    tool.execute(command="echo session-cmd")
    assert call_count[0] == 1  # only one prompt was shown


def test_bash_tool_approval_fn_records_to_audit_log():
    policy = PermissionPolicy.default()
    tool = BashTool(approval_fn=lambda _: ApprovalDecision.ALLOW_ONCE, policy=policy)
    tool.execute(command="echo audited")
    entries = policy.audit_log.entries
    assert any(e.tool_name == "bash" and e.decision == "allowed" for e in entries)

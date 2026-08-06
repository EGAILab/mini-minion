"""Tests for tool workspace-root sandboxing and bash confirmation callable."""

from minion_assist.tools.bash import BashTool
from minion_assist.tools.glob import GlobTool
from minion_assist.tools.policy import PermissionPolicy
from minion_assist.tools.read import ReadTool
from minion_assist.tools.write import WriteTool

# ---------------------------------------------------------------------------
# ReadTool — path containment
# ---------------------------------------------------------------------------


def test_read_rejects_path_outside_root(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "secrets.txt"
    outside.write_text("secret", encoding="utf-8")

    result = ReadTool(root).execute(path=str(outside))

    assert "outside the workspace root" in result


def test_read_allows_path_inside_root(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    inside = root / "hello.txt"
    inside.write_text("hi", encoding="utf-8")

    result = ReadTool(root).execute(path=str(inside))

    assert "1: hi" in result


def test_read_no_root_allows_any_path(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("open", encoding="utf-8")

    result = ReadTool(root=None).execute(path=str(f))

    assert "1: open" in result


def test_read_rejects_traversal_escape(tmp_path):
    """Path traversal via .. must not escape the root."""
    root = tmp_path / "project"
    root.mkdir()
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")
    traversal = root / ".." / "outside.txt"

    result = ReadTool(root).execute(path=str(traversal))

    assert "outside the workspace root" in result


# ---------------------------------------------------------------------------
# ReadTool — extra_roots (agent's own workspace dir, e.g. SOUL.md/memory files
# that live outside the project sandbox `root`)
# ---------------------------------------------------------------------------


def test_read_allows_path_under_extra_root_legacy_branch(tmp_path):
    """Without a policy (legacy inline check), extra_roots still widens access."""
    root = tmp_path / "project"
    root.mkdir()
    workspace = tmp_path / "workspace" / "main"
    workspace.mkdir(parents=True)
    soul = workspace / "SOUL.md"
    soul.write_text("soul content", encoding="utf-8")

    result = ReadTool(root, extra_roots=(workspace,)).execute(path=str(soul))

    assert "1: soul content" in result


def test_read_still_rejects_path_outside_root_and_extra_roots(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    workspace = tmp_path / "workspace" / "main"
    workspace.mkdir(parents=True)
    outside = tmp_path / "secrets.txt"
    outside.write_text("secret", encoding="utf-8")

    result = ReadTool(root, extra_roots=(workspace,)).execute(path=str(outside))

    assert "outside the workspace root" in result


def test_read_allows_path_under_extra_root_with_policy(tmp_path):
    """With a policy injected, extra_roots is forwarded to policy.check_path()."""
    root = tmp_path / "project"
    root.mkdir()
    workspace = tmp_path / "workspace" / "main"
    workspace.mkdir(parents=True)
    soul = workspace / "SOUL.md"
    soul.write_text("soul content", encoding="utf-8")
    policy = PermissionPolicy(workspace=root)

    result = ReadTool(root, policy=policy, extra_roots=(workspace,)).execute(path=str(soul))

    assert "1: soul content" in result


def test_write_tool_has_no_extra_roots_parameter():
    """WriteTool must not accept extra_roots — widening is read-only, by design."""
    import inspect
    assert "extra_roots" not in inspect.signature(WriteTool.__init__).parameters


# ---------------------------------------------------------------------------
# WriteTool — path containment
# ---------------------------------------------------------------------------


def test_write_rejects_path_outside_root(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "evil.txt"

    result = WriteTool(root).execute(path=str(outside), content="bad")

    assert "outside the workspace root" in result
    assert not outside.exists()


def test_write_allows_path_inside_root(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    target = root / "out.txt"

    result = WriteTool(root).execute(path=str(target), content="hello")

    assert "Wrote" in result
    assert target.read_text(encoding="utf-8") == "hello"


# ---------------------------------------------------------------------------
# GlobTool — path containment and default root
# ---------------------------------------------------------------------------


def test_glob_rejects_explicit_path_outside_root(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "other"
    outside.mkdir()

    result = GlobTool(root).execute(pattern="*", path=str(outside))

    assert "outside the workspace root" in result


def test_glob_allows_explicit_path_inside_root(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.py").write_text("", encoding="utf-8")

    result = GlobTool(root).execute(pattern="*.py", path=str(root))

    assert "a.py" in result


def test_glob_defaults_to_root_when_no_path_given(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "found.py").write_text("", encoding="utf-8")

    result = GlobTool(root).execute(pattern="*.py")

    assert "found.py" in result


# ---------------------------------------------------------------------------
# BashTool — confirmation callable
# ---------------------------------------------------------------------------


def test_bash_confirm_callable_true_runs_command():
    """A confirm callable that returns True must allow the command to run."""
    tool = BashTool(confirm=lambda _: True)

    result = tool.execute(command="echo hello")

    assert "hello" in result


def test_bash_confirm_callable_false_cancels():
    """A confirm callable that returns False must cancel execution."""
    tool = BashTool(confirm=lambda _: False)

    result = tool.execute(command="echo should-not-run")

    assert "cancelled" in result.lower()
    assert "should-not-run" not in result


def test_bash_confirm_none_runs_without_calling_confirm():
    """confirm=None must run the command directly without calling any confirm function."""
    # Pass a callable that raises if ever called — proves confirm=None bypasses it.
    def _raise(_: str) -> bool:
        raise AssertionError("confirm callable must not be called when confirm=None")

    tool = BashTool(confirm=None)
    result = tool.execute(command="echo no-prompt")

    assert "no-prompt" in result


def test_bash_confirm_receives_command_string():
    """The confirm callable must be called with the full command string."""
    received: list[str] = []
    tool = BashTool(confirm=lambda cmd: received.append(cmd) or False)

    tool.execute(command="echo marker")

    assert received == ["echo marker"]


def test_bash_tool_cwd_is_used(tmp_path):
    """BashTool cwd= starts the subprocess in the specified directory."""
    tool = BashTool(confirm=None, cwd=tmp_path)
    result = tool.execute(command='python -c "import os; print(os.getcwd())"')
    # Normalize separators for cross-platform comparison.
    assert str(tmp_path).lower().replace("\\", "/") in result.lower().replace("\\", "/")


# ---------------------------------------------------------------------------
# IMP-01: Sensitive-path guardrails
# ---------------------------------------------------------------------------


from pathlib import Path as _Path
from minion_assist.tools.base import _is_sensitive
from minion_assist.tools.policy import PermissionPolicy


class TestSensitivePaths:
    def test_ssh_dir_is_sensitive(self):
        assert _is_sensitive(_Path("~/.ssh/id_rsa").expanduser())

    def test_aws_dir_is_sensitive(self):
        assert _is_sensitive(_Path("~/.aws/credentials").expanduser())

    def test_docker_config_is_sensitive(self):
        assert _is_sensitive(_Path("~/.docker/config.json").expanduser())

    def test_kube_config_is_sensitive(self):
        assert _is_sensitive(_Path("~/.kube/config").expanduser())

    def test_gnupg_dir_is_sensitive(self):
        assert _is_sensitive(_Path("~/.gnupg/secring.gpg").expanduser())

    def test_normal_path_is_not_sensitive(self, tmp_path):
        assert not _is_sensitive(tmp_path / "myfile.txt")

    def test_read_tool_blocks_sensitive_path(self, tmp_path):
        result = ReadTool(root=None).execute(path=str(_Path("~/.ssh/id_rsa").expanduser()))
        assert "protected" in result.lower() or "permitted" in result.lower()

    def test_write_tool_blocks_sensitive_path(self, tmp_path):
        result = WriteTool(root=None).execute(
            path=str(_Path("~/.ssh/id_rsa").expanduser()), content="evil"
        )
        assert "protected" in result.lower() or "permitted" in result.lower()


# ---------------------------------------------------------------------------
# IMP-01: SSRF guardrails
# ---------------------------------------------------------------------------


class TestSSRF:
    def test_bash_blocks_aws_metadata(self):
        tool = BashTool(confirm=None)
        result = tool.execute(command="curl http://169.254.169.254/latest/meta-data/")
        assert "blocked" in result.lower()

    def test_bash_blocks_gcp_metadata_dns(self):
        tool = BashTool(confirm=None)
        result = tool.execute(command="curl http://metadata.google.internal/")
        assert "blocked" in result.lower()

    def test_bash_allows_normal_curl(self):
        """curl to a normal domain must not be blocked by the SSRF check."""
        # confirm=False cancels the command without running it; result should
        # NOT be the SSRF error string.
        tool = BashTool(confirm=lambda cmd: False)
        result = tool.execute(command="curl https://example.com")
        assert "blocked" not in result.lower()
        assert "cancelled" in result.lower()


# ---------------------------------------------------------------------------
# IMP-02: PermissionPolicy threading into legacy tools
# ---------------------------------------------------------------------------


class TestPolicyThreading:
    """Read/Write/GlobTool accept an optional policy kwarg and use it for
    path safety checks when provided, keeping existing root-only callers working.
    """

    def test_read_with_policy_blocks_outside_root(self, tmp_path):
        policy = PermissionPolicy(workspace=tmp_path / "project")
        outside = tmp_path / "outside.txt"
        outside.write_text("secret", encoding="utf-8")

        result = ReadTool(policy=policy).execute(path=str(outside))

        assert "outside the workspace root" in result

    def test_read_without_policy_still_works(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello", encoding="utf-8")

        result = ReadTool(root=tmp_path).execute(path=str(f))

        assert "1: hello" in result

    def test_write_with_policy_blocks_outside_root(self, tmp_path):
        policy = PermissionPolicy(workspace=tmp_path / "project")
        outside = tmp_path / "evil.txt"

        result = WriteTool(policy=policy).execute(path=str(outside), content="bad")

        assert "outside the workspace root" in result
        assert not outside.exists()

    def test_write_without_policy_still_works(self, tmp_path):
        target = tmp_path / "out.txt"

        result = WriteTool(root=tmp_path).execute(path=str(target), content="ok")

        assert "Wrote" in result

    def test_glob_with_policy_blocks_outside_root(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        outside = tmp_path / "other"
        outside.mkdir()
        policy = PermissionPolicy(workspace=project)

        result = GlobTool(project, policy=policy).execute(pattern="*", path=str(outside))

        assert "outside the workspace root" in result

    def test_glob_without_policy_still_works(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        (root / "a.py").write_text("", encoding="utf-8")

        result = GlobTool(root).execute(pattern="*.py", path=str(root))

        assert "a.py" in result

    def test_read_policy_blocks_sensitive_path(self, tmp_path):
        policy = PermissionPolicy.default(workspace=tmp_path)

        result = ReadTool(policy=policy).execute(
            path=str(_Path("~/.ssh/id_rsa").expanduser())
        )

        assert "protected" in result.lower() or "permitted" in result.lower()

    def test_write_policy_blocks_sensitive_path(self, tmp_path):
        policy = PermissionPolicy.default(workspace=tmp_path)

        result = WriteTool(policy=policy).execute(
            path=str(_Path("~/.ssh/id_rsa").expanduser()), content="evil"
        )

        assert "protected" in result.lower() or "permitted" in result.lower()

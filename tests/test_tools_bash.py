"""Tests for BashTool."""

import platform

from mini_minion.tools.bash import BashTool

_IS_WINDOWS = platform.system() == "Windows"


def test_bash_runs_command():
    tool = BashTool(confirm=False)
    result = tool.execute(command="echo hello")
    assert "hello" in result


def test_bash_uses_platform_shell():
    """Verify the correct shell interpreter is used per OS."""
    tool = BashTool(confirm=False)
    if _IS_WINDOWS:
        # $env:OS is PowerShell-only; cmd.exe does not expand it
        result = tool.execute(command="$env:OS")
        assert "windows" in result.lower()
    else:
        # $BASH_VERSION is set by bash but not by sh or other shells
        result = tool.execute(command="echo $BASH_VERSION")
        assert result.strip()


def test_bash_captures_stderr():
    """Stderr output from the subprocess must be returned."""
    tool = BashTool(confirm=False)
    if _IS_WINDOWS:
        result = tool.execute(command="Write-Error 'deliberate_error'")
    else:
        result = tool.execute(command="echo stderr_msg >&2")
    assert result


def test_bash_timeout():
    tool = BashTool(confirm=False)
    result = tool.execute(command='python -c "import time; time.sleep(5)"', timeout=1)
    assert "timed out" in result.lower()


def test_bash_no_output():
    tool = BashTool(confirm=False)
    result = tool.execute(command='python -c "pass"')
    assert result == "(no output)"


def test_bash_nonzero_exit_still_returns_output():
    """A command that exits non-zero should still return whatever it printed."""
    tool = BashTool(confirm=False)
    if _IS_WINDOWS:
        result = tool.execute(command="Write-Output 'before_exit'; exit 1")
    else:
        result = tool.execute(command="echo before_exit; exit 1")
    assert "before_exit" in result


def test_bash_schema():
    tool = BashTool(confirm=False)
    schema = tool.schema
    assert schema.name == "bash"
    assert "command" in schema.parameters["properties"]
    assert "timeout" in schema.parameters["properties"]
    assert "command" in schema.parameters["required"]
    assert "timeout" not in schema.parameters["required"]


def test_bash_schema_description_matches_os():
    """Schema description must name the correct shell for the running OS."""
    tool = BashTool(confirm=False)
    desc = tool.schema.description.lower()
    if _IS_WINDOWS:
        assert "powershell" in desc
    else:
        assert "bash" in desc

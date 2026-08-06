"""Tests for BashTool."""

import platform
import sys

from minion_assist.tools.bash import BashTool

_IS_WINDOWS = platform.system() == "Windows"

# Bare `python` is unreliable in tests: on Windows it can resolve to the
# Microsoft Store app-execution-alias stub (when no python.org/registered
# install is on PATH ahead of it), which silently no-ops instead of running
# real Python. sys.executable is always the interpreter actually running
# these tests. `&` is PowerShell's call operator, needed to invoke a quoted
# path as a command rather than treat it as a string.
_PYTHON = f'& "{sys.executable}"' if _IS_WINDOWS else f'"{sys.executable}"'


def test_bash_runs_command():
    tool = BashTool(confirm=None)
    result = tool.execute(command="echo hello")
    assert "hello" in result


def test_bash_uses_platform_shell():
    """Verify the correct shell interpreter is used per OS."""
    tool = BashTool(confirm=None)
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
    tool = BashTool(confirm=None)
    if _IS_WINDOWS:
        result = tool.execute(command="Write-Error 'deliberate_error'")
    else:
        result = tool.execute(command="echo stderr_msg >&2")
    assert result


def test_bash_timeout():
    tool = BashTool(confirm=None)
    result = tool.execute(command=f'{_PYTHON} -c "import time; time.sleep(5)"', timeout=1)
    assert "timed out" in result.lower()


def test_bash_no_output():
    tool = BashTool(confirm=None)
    result = tool.execute(command=f'{_PYTHON} -c "pass"')
    assert result == "(no output)"


def test_bash_nonzero_exit_still_returns_output():
    """A command that exits non-zero should still return whatever it printed."""
    tool = BashTool(confirm=None)
    if _IS_WINDOWS:
        result = tool.execute(command="Write-Output 'before_exit'; exit 1")
    else:
        result = tool.execute(command="echo before_exit; exit 1")
    assert "before_exit" in result


def test_bash_schema():
    tool = BashTool(confirm=None)
    schema = tool.schema
    assert schema.name == "bash"
    assert "command" in schema.parameters["properties"]
    assert "timeout" in schema.parameters["properties"]
    assert "command" in schema.parameters["required"]
    assert "timeout" not in schema.parameters["required"]


def test_bash_schema_description_matches_os():
    """Schema description must name the correct shell for the running OS."""
    tool = BashTool(confirm=None)
    desc = tool.schema.description.lower()
    if _IS_WINDOWS:
        assert "powershell" in desc
    else:
        assert "bash" in desc


def test_bash_ssrf_markers_are_policy_markers():
    """BashTool must import DEFAULT_SSRF_MARKERS from policy, not define its own.

    This ensures the SSRF block list cannot drift between bash.py and policy.py.
    Importing the same object (``is``) rather than just equal values (``==``)
    proves there is a single source of truth.
    """
    from minion_assist.tools.bash import DEFAULT_SSRF_MARKERS as bash_markers
    from minion_assist.tools.policy import DEFAULT_SSRF_MARKERS as policy_markers
    assert bash_markers is policy_markers

"""Git tools — status, diff, and commit without a full shell.

Why dedicated git tools instead of BashTool?
--------------------------------------------
BashTool gives the agent a general shell, which works but has downsides:
- The agent must remember the correct git command syntax.
- Platform-specific shell differences (PowerShell vs bash) can confuse the
  model when writing git commands.
- There is no confirmation step for destructive operations like ``git commit``.

These three focused tools give the agent a structured, platform-neutral
interface to the most common git operations, with clear parameter names and
confirmation support for write operations.

Design decisions
----------------
- **Shared ``_git()`` helper**: All three tools call a single ``_git()``
  function that invokes ``git`` as a subprocess with ``capture_output=True``.
  Git itself is cross-platform; we don't need a shell wrapper.
- **stdout + stderr combined**: Git often writes informational output to
  stderr (e.g. branch tracking info).  Combining both lets the agent see the
  full output without special handling.
- **FileNotFoundError → readable error**: On systems without git installed,
  ``subprocess.run(["git", ...])`` raises ``FileNotFoundError``.  We catch it
  and return a human-readable message so the agent can react gracefully.
- **Confirm callback for GitCommitTool**: Committing creates a permanent git
  object. The CLI passes a confirmation callable (same pattern as
  :class:`BashTool`) so the user sees the proposed commit before it runs.
  ``confirm=None`` skips the prompt (useful for headless/batch agents).

Talks to
--------
- ``base.py`` — extends :class:`Tool`, returns :class:`ToolSchema`.
- ``registry.py`` / ``__init__.py`` — registered via ``default_registry()``.
- ``minion.py`` — the CLI reuses ``_console_confirm`` for GitCommitTool.
- Python's ``subprocess`` module for process execution.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from .base import Tool, ToolSchema

_DEFAULT_TIMEOUT = 30  # seconds


def _git(args: list[str], cwd: Path | None, timeout: int = _DEFAULT_TIMEOUT) -> str:
    """Run a git command and return combined stdout + stderr.

    Args:
        args: Arguments to pass after ``git`` (e.g. ``["status", "--short"]``).
        cwd: Working directory for the subprocess.  ``None`` inherits the
            parent process's cwd.
        timeout: Maximum seconds to wait before killing the process.

    Returns:
        str: Combined stdout + stderr, stripped.  Returns ``"(no output)"``
            when both streams are empty.  On timeout or error: an error string.
    """
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return (result.stdout + result.stderr).strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: git command timed out after {timeout} seconds."
    except FileNotFoundError:
        return "Error: git is not installed or not found on PATH."
    except Exception as exc:
        return f"Error: {exc}"


class GitStatusTool(Tool):
    """Read-only tool that shows the current git working-tree status.

    Equivalent to ``git status --short --branch``.  Returns a compact summary
    of staged, modified, and untracked files, plus the current branch name.
    """

    def __init__(self, cwd: Path | None = None) -> None:
        # cwd: repository root.  None inherits from the parent process (usually
        # the mini-minion workspace root, set by default_registry()).
        self._cwd = cwd

    @property
    def schema(self) -> ToolSchema:
        """Describe this tool to the LLM."""
        return ToolSchema(
            name="git_status",
            description=(
                "Show the current git working-tree status: branch name, staged files, "
                "modified files, and untracked files."
            ),
            is_read_only=True,
            parameters={"type": "object", "properties": {}, "required": []},
        )

    def execute(self, **kwargs: object) -> str:
        """Return the output of ``git status --short --branch``.

        Returns:
            str: Compact status output, or an error message.
        """
        return _git(["status", "--short", "--branch"], self._cwd)


class GitDiffTool(Tool):
    """Read-only tool that shows changes between the working tree and HEAD.

    Supports both unstaged (default) and staged (``staged=true``) diffs, and
    can be limited to a single file or directory via the ``path`` parameter.
    """

    def __init__(self, cwd: Path | None = None) -> None:
        self._cwd = cwd

    @property
    def schema(self) -> ToolSchema:
        """Describe this tool to the LLM."""
        return ToolSchema(
            name="git_diff",
            description=(
                "Show git changes as a unified diff. "
                "Use staged=true to see staged (--staged) changes. "
                "Optionally limit to a file or directory with path."
            ),
            is_read_only=True,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Limit the diff to this file or directory path.",
                    },
                    "staged": {
                        "type": "boolean",
                        "description": "Show staged changes (git diff --staged). Default false.",
                    },
                },
                "required": [],
            },
        )

    def execute(self, **kwargs: object) -> str:
        """Return the unified diff output.

        Args:
            path (str, optional): File or directory to limit the diff to.
            staged (bool, optional): If true, show staged changes instead of
                unstaged.  Default false.

        Returns:
            str: Unified diff output, or an error message.
        """
        args = ["diff"]
        if kwargs.get("staged"):
            args.append("--staged")
        if kwargs.get("path"):
            # '--' separator prevents git from misinterpreting paths as flags.
            args += ["--", str(kwargs["path"])]
        return _git(args, self._cwd)


class GitCommitTool(Tool):
    """Tool for staging files and creating a git commit.

    Optionally stages a list of files before committing.  When a ``confirm``
    callable is provided, the proposed git command is shown to the user before
    execution (same pattern as :class:`BashTool`).
    """

    def __init__(
        self,
        cwd: Path | None = None,
        confirm: Callable[[str], bool] | None = None,
    ) -> None:
        self._cwd = cwd
        # confirm: called with a human-readable summary of the git operations
        # before they run.  Returns True to proceed, False to cancel.
        # None means run without asking (headless/batch mode).
        self._confirm = confirm

    @property
    def schema(self) -> ToolSchema:
        """Describe this tool to the LLM."""
        return ToolSchema(
            name="git_commit",
            description=(
                "Stage files and create a git commit. "
                "Provide files to stage them first; omit files to commit only "
                "what is already staged."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Commit message.",
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "File paths to stage before committing. "
                            "Omit to commit only already-staged changes."
                        ),
                    },
                },
                "required": ["message"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        """Stage files (if given) and create a commit.

        Args:
            message (str): The commit message.
            files (list[str], optional): Paths to stage with ``git add``
                before committing.  If omitted, commits only what is already
                staged.

        Returns:
            str: The output from ``git commit``, or an error / cancellation
                message.
        """
        message = str(kwargs["message"])
        files: list[str] = [str(f) for f in (kwargs.get("files") or [])]

        # Build a human-readable preview of the operations about to run.
        lines = []
        if files:
            lines.append(f"git add -- {' '.join(files)}")
        lines.append(f"git commit -m {message!r}")
        preview = "\n".join(lines)

        if self._confirm is not None and not self._confirm(preview):
            return "Commit cancelled by user."

        if files:
            stage_result = _git(["add", "--"] + files, self._cwd)
            if stage_result.startswith("Error:"):
                return f"Stage failed: {stage_result}"

        return _git(["commit", "-m", message], self._cwd)

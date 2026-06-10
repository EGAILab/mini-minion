"""BashTool — run shell commands in a subprocess.

This is the most powerful (and most dangerous) tool available to agents. It
gives the agent the ability to run any shell command: install packages, run
tests, call CLIs, query APIs, manipulate files, etc.

Platform handling
-----------------
Python runs on both Windows and Unix-like systems. The shell to use differs:
- **Windows**: PowerShell (``powershell -Command ...``). The tool description
  is customized to mention PowerShell cmdlets (Get-ChildItem, etc.).
- **Unix/Mac**: bash (``bash -c ...``). The description mentions Unix commands
  (ls, cat, grep, etc.).

The description sent to the LLM changes based on the platform so the model
knows which syntax to use. This is done at module load time (once) rather than
per-call, since the platform doesn't change at runtime.

Safety and limitations
-----------------------
- ``timeout`` prevents runaway commands from hanging forever.
- stdout and stderr are both captured and combined in the output.
- The process runs as the *same user* running the Python script — all their
  permissions apply. Agents can theoretically run destructive commands.
  This is an inherent tradeoff of giving an agent shell access.
- ``capture_output=True, text=True`` returns output as a string (decoded from
  bytes using the system's default encoding).

Talks to
--------
- ``base.py`` — extends :class:`Tool`, returns :class:`ToolSchema`.
- ``policy.py`` — accepts an optional :class:`PermissionPolicy` kwarg.  When
  provided, ``check_command()`` handles SSRF and read-only checks in one call.
  Falls back to importing :data:`DEFAULT_SSRF_MARKERS` directly when no policy
  is supplied so existing callers that omit the kwarg keep working.
- ``registry.py`` — registered via ``default_registry()`` in ``__init__.py``.
- Python's ``subprocess`` module for process execution.
- Python's ``platform`` module to detect Windows vs. Unix.
"""

from __future__ import annotations

import platform
import subprocess
from collections.abc import Callable
from pathlib import Path

from .audit import AuditEntry, ApprovalDecision, AuditLog, _utcnow
from .base import Tool, ToolSchema
from .policy import DEFAULT_SSRF_MARKERS, PermissionPolicy
from .sandbox import SandboxBackend

# Detect once at import time; doesn't change during execution.
_IS_WINDOWS = platform.system() == "Windows"

_DEFAULT_TIMEOUT = 30  # seconds

# The description is platform-specific so the model uses the right shell syntax.
_DESCRIPTION = (
    "Run a PowerShell command. Use PowerShell syntax and cmdlets: "
    "Get-ChildItem, Get-Content, Set-Location, git, python, uv, etc. "
    "Never use Unix commands like ls, cat, or grep."
    if _IS_WINDOWS else
    "Run a bash shell command. Use ls, cat, grep, git, python, uv, etc. "
    "Never use Windows commands like dir, type, or Get-ChildItem."
)


def _build_args(command: str) -> list[str]:
    """Build the subprocess argument list for the current platform.

    Args:
        command (str): The raw shell command string to execute.

    Returns:
        list[str]: Argument list for ``subprocess.run()``. The shell executable
            is the first element, followed by flags and the command string.
    """
    if _IS_WINDOWS:
        # -NoProfile: skip loading the PowerShell user profile (faster startup).
        # -NonInteractive: prevent the process from waiting for user input.
        # -Command: treat the next argument as a command string to execute.
        return ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
    return ["bash", "-c", command]


class BashTool(Tool):
    """Tool for running shell commands.

    Executes the command in a subprocess with a configurable timeout.
    Returns combined stdout + stderr as a string.
    """

    def __init__(
        self,
        confirm: Callable[[str], bool] | None = None,
        cwd: Path | None = None,
        policy: PermissionPolicy | None = None,
        sandbox: SandboxBackend | None = None,
        approval_fn: Callable[[str], ApprovalDecision] | None = None,
    ) -> None:
        # confirm: simple bool callback — True to proceed, False to cancel.
        # The CLI passes a callable that calls input(); tests pass a lambda.
        self._confirm = confirm
        # cwd: working directory for the subprocess; None inherits from parent.
        self._cwd = cwd
        # policy: SSRF + read_only_mode checks; also holds the session audit log.
        self._policy = policy
        # sandbox: optional backend that replaces subprocess.run().  When None,
        # subprocess.run() is called directly (LocalSandboxBackend behaviour).
        self._sandbox = sandbox
        # approval_fn: richer 4-option approval callback (ALLOW_ONCE, ALLOW_SESSION,
        # DENY, ALWAYS_DENY).  Takes priority over confirm when both are set.
        # Used by the CLI to offer session-level approve/deny; tests use confirm.
        self._approval_fn = approval_fn

    @property
    def schema(self) -> ToolSchema:
        """Describe this tool to the LLM (description is platform-specific)."""
        return ToolSchema(
            name="bash",
            description=_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": f"Timeout in seconds (default {_DEFAULT_TIMEOUT})",
                    },
                },
                "required": ["command"],  # timeout is optional
            },
        )

    def execute(self, **kwargs: object) -> str:
        """Execute a shell command and return its output.

        Args:
            command (str): The shell command to run.
            timeout (int, optional): Maximum seconds to wait before killing
                the process. Defaults to 30 seconds.

        Returns:
            str: Combined stdout and stderr output, stripped of leading/trailing
                whitespace. Returns ``"(no output)"`` if both stdout and stderr
                are empty. On timeout: a timeout error message. On other errors:
                a human-readable error string.
        """
        command = str(kwargs["command"])
        timeout = int(kwargs.get("timeout") or _DEFAULT_TIMEOUT)

        # --- Policy check (SSRF + read_only_mode) ---
        # Policy records the denial to the audit log via _record_deny().
        if self._policy is not None:
            error = self._policy.check_command(command)
            if error:
                return error
        elif any(marker in command for marker in DEFAULT_SSRF_MARKERS):
            return (
                "Error: command blocked — requests to cloud instance metadata "
                "endpoints (169.254.169.254 and equivalents) are not permitted."
            )

        # --- Session-level deny (ALWAYS_DENY from a prior confirmation prompt) ---
        _audit: AuditLog | None = self._policy.audit_log if self._policy else None
        if _audit is not None and _audit.is_session_denied(command):
            _audit.record(AuditEntry(
                timestamp=_utcnow(), tool_name="bash",
                args_repr=command[:120], decision="denied",
                reason="always-deny (set this session)",
            ))
            return "Command blocked — denied for the rest of this session."

        # --- Confirmation / approval ---
        # Precedence: approval_fn (4-option menu) > session_allowed > confirm (bool).
        if _audit is not None and _audit.is_session_allowed(command):
            # Previously approved for the full session — skip prompting.
            pass
        elif self._approval_fn is not None:
            decision = self._approval_fn(command)
            if decision == ApprovalDecision.DENY:
                if _audit:
                    _audit.record(AuditEntry(
                        timestamp=_utcnow(), tool_name="bash",
                        args_repr=command[:120], decision="denied", reason="user deny",
                    ))
                return "Command cancelled by user."
            if decision == ApprovalDecision.ALWAYS_DENY:
                if _audit:
                    _audit.set_session_denied(command)
                    _audit.record(AuditEntry(
                        timestamp=_utcnow(), tool_name="bash",
                        args_repr=command[:120], decision="denied", reason="always deny",
                    ))
                return "Command blocked — denied for this session."
            if decision == ApprovalDecision.ALLOW_SESSION and _audit:
                _audit.set_session_allowed(command)
        elif self._confirm is not None:
            if not self._confirm(command):
                if _audit:
                    _audit.record(AuditEntry(
                        timestamp=_utcnow(), tool_name="bash",
                        args_repr=command[:120], decision="denied", reason="user cancelled",
                    ))
                return "Command cancelled by user."

        # Record "allowed" decision before running.
        if _audit is not None:
            _audit.record(AuditEntry(
                timestamp=_utcnow(), tool_name="bash",
                args_repr=command[:120], decision="allowed",
            ))

        # --- Execute via sandbox backend or direct subprocess ---
        _args = _build_args(command)
        try:
            if self._sandbox is not None:
                result = self._sandbox.run(
                    _args, capture_output=True, text=True, timeout=timeout, cwd=self._cwd,
                )
            else:
                result = subprocess.run(
                    _args, capture_output=True, text=True, timeout=timeout, cwd=self._cwd,
                )
            # Combine stdout and stderr; strip whitespace so agent sees clean output.
            combined = (result.stdout + result.stderr).strip()
            return combined or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout} seconds."
        except Exception as exc:
            return f"Error: {exc}"

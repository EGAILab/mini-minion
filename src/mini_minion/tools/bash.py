"""BashTool — run shell commands (PowerShell on Windows, bash on Unix)."""

from __future__ import annotations

import platform
import subprocess

from .base import Tool, ToolSchema

_IS_WINDOWS = platform.system() == "Windows"
_DEFAULT_TIMEOUT = 30

_DESCRIPTION = (
    "Run a PowerShell command. Use PowerShell syntax and cmdlets: "
    "Get-ChildItem, Get-Content, Set-Location, git, python, uv, etc. "
    "Never use Unix commands like ls, cat, or grep."
    if _IS_WINDOWS else
    "Run a bash shell command. Use ls, cat, grep, git, python, uv, etc. "
    "Never use Windows commands like dir, type, or Get-ChildItem."
)


def _build_args(command: str) -> list[str]:
    if _IS_WINDOWS:
        return ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
    return ["bash", "-c", command]


class BashTool(Tool):
    @property
    def schema(self) -> ToolSchema:
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
                "required": ["command"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        command = str(kwargs["command"])
        timeout = int(kwargs.get("timeout") or _DEFAULT_TIMEOUT)
        try:
            result = subprocess.run(
                _build_args(command),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            combined = (result.stdout + result.stderr).strip()
            return combined or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout} seconds."
        except Exception as exc:
            return f"Error: {exc}"

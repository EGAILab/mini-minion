"""Sandbox backend abstraction for BashTool subprocess execution.

Provides a :class:`SandboxBackend` protocol and a default
:class:`LocalSandboxBackend` that calls subprocess.run() directly.

Why does this exist?
--------------------
Plugging in a different backend (Docker, firejail, nsjail, etc.) only requires
implementing the :class:`SandboxBackend` protocol and passing the instance to
:class:`~minion_assistant.tools.bash.BashTool`.  The policy-level checks (SSRF,
read-only mode) happen before the sandbox is invoked, so the sandbox only sees
already-approved commands.

Talks to
--------
- ``tools/bash.py`` — :class:`BashTool` accepts an optional ``sandbox`` kwarg
  and routes subprocess execution through it.
"""

from __future__ import annotations

import subprocess
from typing import Protocol


class SandboxBackend(Protocol):
    """Protocol for subprocess isolation backends.

    Any object with a ``run()`` method matching this signature satisfies the
    protocol — no inheritance required (structural typing).

    Implement this to route BashTool execution through a container or syscall
    sandbox instead of the local shell.
    """

    def run(
        self,
        args: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
        cwd: object,
    ) -> subprocess.CompletedProcess:
        """Execute a command and return its completed-process result.

        Args:
            args:           Argument list, e.g. ``["powershell", "-Command", "ls"]``.
            capture_output: If True, capture stdout and stderr (passed to subprocess.run).
            text:           If True, decode output bytes to strings.
            timeout:        Seconds before the process is forcibly killed.
            cwd:            Working directory for the subprocess (Path or None).

        Returns:
            subprocess.CompletedProcess with stdout, stderr, and returncode.
        """
        ...


class LocalSandboxBackend:
    """Direct subprocess execution — no additional isolation.

    This is the default backend.  It passes the argument list straight to
    ``subprocess.run()`` with no container or syscall filtering.

    Replace with a Docker or firejail backend to add isolation without
    touching :class:`BashTool` itself.
    """

    def run(
        self,
        args: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
        cwd: object,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            args,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
            cwd=cwd,
        )

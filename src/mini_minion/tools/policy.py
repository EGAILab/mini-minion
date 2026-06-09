"""PermissionPolicy — centralised safety rules for all tools.

Previously, safety checks were scattered across individual tool files:

- ``_within()`` and ``_is_sensitive()`` in ``base.py`` (for file paths)
- ``_SSRF_MARKERS`` in ``bash.py`` (for network requests)

PermissionPolicy gathers these rules into a single injectable object so
they can be configured once and shared by all tools that perform I/O.

Usage
-----
Create the default policy at startup and pass it to every I/O tool::

    policy = PermissionPolicy.default(workspace=Path.cwd())
    edit_tool = EditTool(policy)
    grep_tool = GrepTool(policy)
    web_fetch_tool = WebFetchTool(policy)

Custom policies (e.g. for tests with a temp directory) are easy to build::

    policy = PermissionPolicy(workspace=tmp_path)

Existing tools (ReadTool, WriteTool, GlobTool, BashTool) continue to use
``_within()`` and ``_is_sensitive()`` from ``base.py`` directly — they are
unchanged for backwards compatibility.  New tools take a PermissionPolicy.

Talks to
--------
- ``tools/edit.py``, ``tools/grep.py``, ``tools/web_fetch.py`` — instantiate
  a PermissionPolicy and call ``check_path()`` / ``check_url()`` before I/O.
- ``tools/__init__.py`` — ``default_registry()`` creates and passes the policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .base import _is_sensitive, _within

# Cloud instance metadata endpoints — same set as BashTool's _SSRF_MARKERS.
# Centralised here so both BashTool and WebFetchTool block the same addresses.
DEFAULT_SSRF_MARKERS: frozenset[str] = frozenset({
    "169.254.169.254",           # AWS / GCP / Azure instance metadata
    "metadata.google.internal",  # GCP metadata DNS alias
    "169.254.170.2",             # ECS task metadata endpoint
    "fd00:ec2::254",             # AWS IMDSv2 IPv6
})


@dataclass
class PermissionPolicy:
    """Encapsulates path and network safety rules for I/O tools.

    Attributes:
        workspace: Optional root directory.  Tool paths must stay inside it
            when set.  ``None`` disables the workspace boundary check.
        ssrf_markers: Set of strings that must NOT appear in a URL.
            Any URL containing one of these strings is blocked.
            Defaults to the standard cloud metadata endpoint list.
    """
    workspace: Path | None = None
    ssrf_markers: frozenset[str] = field(default_factory=lambda: DEFAULT_SSRF_MARKERS)

    def check_path(self, path: Path) -> str | None:
        """Check a file path against the policy.

        Returns None when the path is allowed, or an error string to return
        to the LLM when the path should be denied.

        Checks (in order):
        1. Sensitive credential paths (SSH keys, AWS config, etc.) — always denied.
        2. Workspace boundary — denied when ``workspace`` is set and path escapes it.

        Args:
            path: The file path to validate. Need not exist yet.

        Returns:
            None if the path is allowed, or an error message string.
        """
        if _is_sensitive(path):
            return (
                f"Error: '{path}' is a protected credential path and cannot be "
                "accessed. Reading or writing credential files is not permitted."
            )
        if self.workspace is not None and not _within(path, self.workspace):
            return f"Error: '{path}' is outside the workspace root '{self.workspace}'"
        return None

    def check_url(self, url: str) -> str | None:
        """Check a URL against the policy.

        Returns None when the URL is allowed, or an error string when it
        matches a blocked marker (e.g. cloud metadata endpoint).

        Args:
            url: The full URL string to validate.

        Returns:
            None if the URL is allowed, or an error message string.
        """
        for marker in self.ssrf_markers:
            if marker in url:
                return (
                    f"Error: URL blocked — requests to cloud instance metadata "
                    f"endpoints ({marker!r}) are not permitted."
                )
        return None

    @classmethod
    def default(cls, workspace: Path | None = None) -> "PermissionPolicy":
        """Build the standard policy used by default_registry().

        Args:
            workspace: The workspace root to restrict file I/O to.
                       Pass ``None`` to allow unrestricted file access.

        Returns:
            PermissionPolicy with the default SSRF marker set.
        """
        return cls(workspace=workspace)

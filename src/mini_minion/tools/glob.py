"""GlobTool — find files matching a glob pattern, sorted by modification time.

A glob pattern is a string with wildcards that matches file paths:
- ``*``  matches any text within a single directory level.
- ``**`` matches any number of directory levels (recursive search).
- ``?``  matches a single character.

Examples: ``**/*.py`` (all Python files), ``src/*.ts`` (TypeScript in src/),
``tests/test_*.py`` (files starting with "test_" in tests/).

This tool is how the agent explores a codebase without knowing filenames in
advance — it can discover what files exist before reading them.

Design decisions
----------------
- **Sorted by modification time (newest first)**: The most recently changed
  file is typically the most relevant one. This makes it easier for the agent
  to find recently edited files without extra filtering.
- **Files only**: Directories matching the pattern are excluded. The agent can
  use the ``read`` tool on a directory path to list its contents.
- **Prefix with root**: The pattern is joined with the root path so relative
  patterns work correctly regardless of the working directory.

Talks to
--------
- ``base.py`` — extends :class:`Tool`, returns :class:`ToolSchema`.
- ``registry.py`` — registered via ``default_registry()`` in ``__init__.py``.
"""

from __future__ import annotations

import glob as _glob  # renamed to avoid conflict with the class name GlobTool
import os
from pathlib import Path

from .base import Tool, ToolSchema, _within

_MAX_RESULTS = 200  # cap to avoid flooding the context window with thousands of paths
# Directories to exclude from results — traversed by glob but noisy and rarely useful.
_SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache"}


class GlobTool(Tool):
    """Tool for finding files by glob pattern.

    The agent calls this to discover which files exist before reading them.
    Results are sorted newest-first by modification time.
    """

    def __init__(self, root: Path | None = None) -> None:
        # root is the workspace boundary and the default search base.
        # root=None means unrestricted, defaulting search base to cwd.
        self._root = root.resolve() if root else None

    @property
    def schema(self) -> ToolSchema:
        """Describe this tool to the LLM."""
        return ToolSchema(
            name="glob",
            description=(
                "Find files matching a glob pattern, sorted by modification time (newest first). "
                "Use ** for recursive matching (e.g. '**/*.py')."
            ),
            is_read_only=True,
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g. '**/*.py', 'src/*.ts')",
                    },
                    "path": {
                        "type": "string",
                        "description": "Root directory to search. Defaults to current directory.",
                    },
                },
                "required": ["pattern"],  # "path" is optional; defaults to cwd
            },
        )

    def execute(self, **kwargs: object) -> str:
        """Find files matching the given glob pattern.

        Args:
            pattern (str): Glob pattern to match, e.g. ``"**/*.py"``.
            path (str, optional): Root directory to search from.
                Defaults to the current working directory.

        Returns:
            str: Newline-separated list of matching file paths, sorted newest-first.
                Returns ``"(no matches)"`` if nothing matches.
        """
        pattern = str(kwargs["pattern"])
        if kwargs.get("path"):
            root = Path(str(kwargs["path"]))
            if self._root and not _within(root, self._root):
                return f"Error: '{root}' is outside the workspace root '{self._root}'"
        else:
            # Default to workspace root when set; fall back to cwd otherwise.
            root = self._root or Path.cwd()

        # _glob.glob with recursive=True enables ** matching across directories.
        matches = _glob.glob(str(root / pattern), recursive=True)

        # Filter out directories and paths that pass through a skip directory.
        matches = [
            m for m in matches
            if os.path.isfile(m)
            and not any(part in _SKIP_DIRS for part in Path(m).parts)
        ]

        # Sort newest-first by modification timestamp.
        # os.path.exists() guard prevents a crash if a file disappears between
        # glob and sort (rare but possible with rapidly changing filesystems).
        matches.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)

        truncated = len(matches) > _MAX_RESULTS
        matches = matches[:_MAX_RESULTS]

        if not matches:
            return "(no matches)"
        result = "\n".join(matches)
        if truncated:
            result += f"\n\n(Results capped at {_MAX_RESULTS}. Narrow your pattern to see more.)"
        return result

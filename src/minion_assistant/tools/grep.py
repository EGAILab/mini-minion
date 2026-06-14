"""GrepTool — search files for a regex pattern without using the shell.

Why not just use bash + grep?
------------------------------
Using :class:`BashTool` for ``grep`` works, but it:

- Creates a subprocess on every call (slow for many small searches).
- Requires the agent to use the correct ``grep`` or PowerShell ``Select-String``
  syntax depending on the platform.
- Has less predictable output formatting (varies by OS and flags).

:class:`GrepTool` is a native Python alternative that runs directly in the
minion-assistant process using the ``re`` standard library module.  It is fast,
cross-platform, and produces consistent output with file paths and line numbers
in the ``path:line_no:line`` format that editors and humans recognise.

Output format
-------------
Each matching line is shown as::

    /absolute/path/to/file.py:42:    def my_function():

When ``context_lines > 0``, surrounding lines use ``-`` instead of ``:``::

    /path/file.py:41-    # comment before the function
    /path/file.py:42:    def my_function():    ← the match line (uses :)
    /path/file.py:43-        pass

Talks to
--------
- ``policy.py`` — :class:`PermissionPolicy` for path validation.
- ``base.py`` — extends :class:`Tool`, returns :class:`ToolSchema`.
- ``__init__.py`` — registered via ``default_registry()``.
"""

from __future__ import annotations

import re
from pathlib import Path

from .base import Tool, ToolSchema
from .policy import PermissionPolicy

_DEFAULT_MAX_RESULTS = 100


class GrepTool(Tool):
    """Tool for searching file content with a regex pattern.

    Returns matching lines with file path and line number, similar to
    ``grep -n`` or ripgrep output.  Runs entirely in Python — no subprocess.
    """

    def __init__(self, policy: PermissionPolicy | None = None) -> None:
        self._policy = policy or PermissionPolicy()

    @property
    def schema(self) -> ToolSchema:
        """Describe the grep tool to the LLM."""
        return ToolSchema(
            name="grep",
            description=(
                "Search files for a regex pattern. Returns matching lines with "
                "file paths and line numbers in 'path:line_no:line' format. "
                "Faster and more portable than using bash grep or Select-String."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "File or directory to search in. "
                            "When a directory, all files inside are searched recursively."
                        ),
                    },
                    "include": {
                        "type": "string",
                        "description": (
                            "Glob pattern to filter files when path is a directory. "
                            "Example: '*.py' to search only Python files. "
                            "Defaults to '*' (all files)."
                        ),
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": (
                            "Number of lines to show before and after each match "
                            "(like grep -C N). Default 0."
                        ),
                    },
                    "ignore_case": {
                        "type": "boolean",
                        "description": "Case-insensitive search. Default false.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": (
                            f"Maximum number of matching lines to return. "
                            f"Default {_DEFAULT_MAX_RESULTS}."
                        ),
                    },
                },
                "required": ["pattern", "path"],
            },
            is_read_only=True,
        )

    def execute(self, **kwargs: object) -> str:
        """Search files for a pattern and return matching lines.

        Args:
            pattern (str): Regex pattern to search for.
            path (str): File or directory path to search in.
            include (str, optional): Glob filter for filenames (default '*').
            context_lines (int, optional): Lines of context before/after match.
            ignore_case (bool, optional): Case-insensitive flag. Default False.
            max_results (int, optional): Cap on matching lines returned.

        Returns:
            str: Formatted match output, or a "no matches" message.
        """
        pattern = str(kwargs["pattern"])
        search_path = Path(str(kwargs["path"]))
        include = str(kwargs.get("include") or "*")
        context_lines = max(0, int(kwargs.get("context_lines") or 0))
        ignore_case = bool(kwargs.get("ignore_case", False))
        max_results = int(kwargs.get("max_results") or _DEFAULT_MAX_RESULTS)

        error = self._policy.check_path(search_path)
        if error:
            return error

        # Compile the regex once before iterating files.
        flags = re.IGNORECASE if ignore_case else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return f"Error: invalid regex pattern: {exc}"

        # Collect files to search.
        if search_path.is_file():
            files: list[Path] = [search_path]
        elif search_path.is_dir():
            # rglob("*") then filter by include pattern to avoid shell dependency.
            files = sorted(f for f in search_path.rglob(include) if f.is_file())
        elif not search_path.exists():
            return f"Error: path does not exist: {search_path}"
        else:
            return f"Error: path is neither a file nor a directory: {search_path}"

        output_lines: list[str] = []
        total_matches = 0

        for file_path in files:
            if total_matches >= max_results:
                break
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue  # skip unreadable files silently

            file_lines = content.splitlines()

            # Track which line indices have already been printed to avoid
            # duplicates when context windows of adjacent matches overlap.
            printed: set[int] = set()

            for i, line in enumerate(file_lines):
                if total_matches >= max_results:
                    break
                if not regex.search(line):
                    continue

                # Context window: show lines [start, end).
                start = max(0, i - context_lines)
                end = min(len(file_lines), i + context_lines + 1)

                for j in range(start, end):
                    if j in printed:
                        continue
                    printed.add(j)
                    # Use ':' separator for the match line, '-' for context.
                    sep = ":" if j == i else "-"
                    output_lines.append(f"{file_path}:{j + 1}{sep}{file_lines[j]}")

                total_matches += 1

        if not output_lines:
            return f"No matches for {pattern!r} in {search_path}"

        if total_matches >= max_results:
            output_lines.append(f"\n(Truncated — showing first {max_results} matches)")

        return "\n".join(output_lines)

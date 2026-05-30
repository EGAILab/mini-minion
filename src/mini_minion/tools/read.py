"""ReadTool — read a file with line numbers, or list a directory.

This tool lets the agent inspect the filesystem: read source code, config files,
log files, or see what's inside a folder. It's one of the most frequently used
tools in practice.

Design decisions
----------------
- **Line numbers in output**: Each line is prefixed with its line number so the
  agent can reference specific lines in follow-up questions or tool calls.
- **Pagination**: Large files are returned in chunks (default 200 lines). A hint
  at the bottom of the output tells the agent how to read the next chunk. This
  prevents the agent from filling its context window with one enormous file.
- **Safe errors**: File-not-found and OS errors are returned as strings, not
  raised, so the agent can see the error and adapt its approach.
- **Directory listing**: If the path is a directory, it lists the contents with
  ``/`` suffixes on folder names — giving the agent a way to explore the tree.

Talks to
--------
- ``base.py`` — extends :class:`Tool`, returns :class:`ToolSchema`.
- ``registry.py`` — registered via ``default_registry()`` in ``__init__.py``.
"""

from __future__ import annotations

import itertools
import pathlib
from pathlib import Path

from .base import Tool, ToolSchema, _within

_DEFAULT_LIMIT = 200   # maximum lines returned per call if no limit is specified
_MAX_BYTES = 50 * 1024  # stop streaming beyond 50 KB to bound memory and context usage
_BINARY_SAMPLE = 512   # bytes sampled for null-byte binary detection


def _is_binary(path: pathlib.Path) -> bool:
    """Return True if the file appears to be binary (null byte in first 512 bytes)."""
    try:
        return b"\x00" in path.read_bytes()[:_BINARY_SAMPLE]
    except OSError:
        return False


class ReadTool(Tool):
    """Tool for reading files or listing directory contents.

    The agent calls this with a path and optional pagination parameters.
    Returns the file content as a numbered-line string, or a sorted directory
    listing, or an error message.
    """

    def __init__(self, root: Path | None = None) -> None:
        # root=None means unrestricted; set to the project directory at startup.
        self._root = root.resolve() if root else None

    @property
    def schema(self) -> ToolSchema:
        """Describe this tool to the LLM."""
        return ToolSchema(
            name="read",
            description="Read a file with line numbers, or list the contents of a directory.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file or directory",
                    },
                    "offset": {
                        "type": "integer",
                        # 1-indexed: line 1 is the first line, matching the displayed numbers.
                        "description": "Start line, 1-indexed (default 1)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": f"Max lines to return (default {_DEFAULT_LIMIT})",
                    },
                },
                "required": ["path"],  # only "path" is mandatory; offset and limit are optional
            },
        )

    def execute(self, **kwargs: object) -> str:
        """Read the file or list the directory at the given path.

        Args:
            path (str): Absolute path to the file or directory to read.
            offset (int, optional): First line to return, 1-indexed. Default 1.
            limit (int, optional): Maximum number of lines to return. Default 200.

        Returns:
            str: For a file: numbered lines in the format ``"N: content"``,
                with a pagination hint if there are more lines beyond the limit.
                For a directory: sorted list of entries, folders marked with ``/``.
                On error: a human-readable error message.
        """
        path = pathlib.Path(str(kwargs["path"]))
        if self._root and not _within(path, self._root):
            return f"Error: '{path}' is outside the workspace root '{self._root}'"

        # Clamp offset and limit to at least 1 to avoid nonsensical values.
        offset = max(1, int(kwargs.get("offset") or 1))
        limit = max(1, int(kwargs.get("limit") or _DEFAULT_LIMIT))

        if not path.exists():
            return f"File not found: {path}"

        if path.is_dir():
            # List directory: sort entries, append "/" to subdirectory names.
            entries = sorted(e.name + ("/" if e.is_dir() else "") for e in path.iterdir())
            return "\n".join(entries) or "(empty directory)"

        if _is_binary(path):
            return f"Error: '{path}' appears to be a binary file and cannot be read as text."

        try:
            with path.open(encoding="utf-8", errors="replace") as fh:
                # Skip lines before offset without loading them into memory.
                for _ in itertools.islice(fh, offset - 1):
                    pass
                sliced: list[str] = []
                bytes_read = 0
                byte_capped = False
                for raw_line in itertools.islice(fh, limit):
                    bytes_read += len(raw_line.encode("utf-8", errors="replace"))
                    if bytes_read > _MAX_BYTES:
                        byte_capped = True
                        break
                    sliced.append(raw_line.rstrip("\n\r"))
        except OSError as exc:
            return f"Error: {exc}"

        end = offset + len(sliced) - 1
        # Format each line as "N: content" where N is the 1-indexed line number.
        output = "\n".join(f"{offset + i}: {line}" for i, line in enumerate(sliced))

        if byte_capped:
            output += f"\n\n(Stopped at {_MAX_BYTES // 1024} KB. Use offset={end + 1} to read more.)"
        elif sliced and len(sliced) == limit:
            # Got exactly limit lines — there may be more.
            output += f"\n\n(Lines {offset}–{end} shown. Use offset={end + 1} to read more.)"

        return output

"""WriteTool — write content to a file, creating parent directories as needed.

This tool lets the agent create or overwrite files. It always performs a full
overwrite — there is no "append" mode. If the agent wants to update just part
of a file, it must read the file first, modify the content, and write it back.

Design decisions
----------------
- **mkdir -p**: ``path.parent.mkdir(parents=True, exist_ok=True)`` creates the
  entire directory tree if it doesn't exist, just like the Unix ``mkdir -p``
  command. This means the agent doesn't need to create directories separately.
- **Safe errors**: OS errors (permission denied, path too long, etc.) are
  returned as strings rather than raised.
- **UTF-8**: Always writes as UTF-8. This is the safe universal default for
  text content.

Talks to
--------
- ``base.py`` — extends :class:`Tool`, returns :class:`ToolSchema`.
- ``policy.py`` — accepts an optional :class:`PermissionPolicy` to centralise
  path safety checks.  Falls back to ``_within`` + ``_is_sensitive`` when no
  policy is supplied.
- ``registry.py`` — registered via ``default_registry()`` in ``__init__.py``.
"""

from __future__ import annotations

import pathlib
from pathlib import Path

from .base import Tool, ToolSchema, _is_sensitive, _within
from .policy import PermissionPolicy


class WriteTool(Tool):
    """Tool for writing content to a file.

    The agent calls this with a path and the content to write. If the path's
    parent directories don't exist, they are created automatically.
    """

    def __init__(self, root: Path | None = None, *, policy: PermissionPolicy | None = None) -> None:
        # root=None means unrestricted; set to the project directory at startup.
        self._root = root.resolve() if root else None
        # When a policy is provided, check_path() replaces the inline checks.
        # policy=None keeps legacy behaviour for callers that pass only root.
        self._policy = policy

    @property
    def schema(self) -> ToolSchema:
        """Describe this tool to the LLM."""
        return ToolSchema(
            name="write",
            description="Write content to a file. Creates parent directories if needed.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write",
                    },
                },
                "required": ["path", "content"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        """Write content to a file, creating parent directories as needed.

        Args:
            path (str): Absolute path to the destination file.
                The file is created if it doesn't exist; overwritten if it does.
            content (str): The text content to write to the file.

        Returns:
            str: A success message with the character count, e.g.
                ``"Wrote 1234 chars to /path/to/file.py"``.
                On error: a human-readable error message.
        """
        path = pathlib.Path(str(kwargs["path"]))
        if self._policy is not None:
            error = self._policy.check_path(path)
            if error:
                return error
        else:
            if _is_sensitive(path):
                return (
                    f"Error: '{path}' is a protected system path and cannot be written. "
                    "Writing to credential files and secret directories is not permitted."
                )
            if self._root and not _within(path, self._root):
                return f"Error: '{path}' is outside the workspace root '{self._root}'"
        content = str(kwargs["content"])
        try:
            # Create all missing parent directories (equivalent to `mkdir -p`).
            # exist_ok=True means it won't fail if the directories already exist.
            path.parent.mkdir(parents=True, exist_ok=True)

            path.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} chars to {path}"
        except OSError as exc:
            return f"Error: {exc}"

"""EditTool — replace an exact string in a file with a new string.

Why this tool matters
---------------------
:class:`WriteTool` rewrites an entire file.  For large files (source code,
documents) that means the agent must first read the whole file, modify its
copy in memory, and send the entire content back in a tool call.  That
wastes context window tokens and risks overwriting unrelated sections if the
agent's copy is slightly stale.

:class:`EditTool` only sends the ``old_string`` (what to find) and
``new_string`` (what to replace it with).  The edit is applied in place; the
rest of the file is never sent over the wire.

Safety: unique-match enforcement
----------------------------------
When ``replace_all`` is ``False`` (the default), the tool refuses to edit a
file if ``old_string`` appears more than once.  This prevents accidental
double-replacements when the agent provides a string that happens to repeat
in the file.  The agent is asked to supply more surrounding context to make
the match unique.

Safety: path guard
------------------
All path checks are delegated to :class:`PermissionPolicy`, which blocks
credential files and paths outside the workspace root.

Talks to
--------
- ``policy.py`` — :class:`PermissionPolicy` for path validation.
- ``base.py`` — extends :class:`Tool`, returns :class:`ToolSchema`.
- ``__init__.py`` — registered via ``default_registry()``.
"""

from __future__ import annotations

from pathlib import Path

from .base import Tool, ToolSchema
from .policy import PermissionPolicy


class EditTool(Tool):
    """Tool for editing a file by replacing an exact string.

    Safer than WriteTool for partial file changes because it never rewrites
    sections that weren't meant to change.
    """

    def __init__(self, policy: PermissionPolicy | None = None) -> None:
        # Use the supplied policy or a default (unrestricted) policy.
        # default_registry() always passes a policy with the workspace root set.
        self._policy = policy or PermissionPolicy()

    @property
    def schema(self) -> ToolSchema:
        """Describe the edit tool to the LLM."""
        return ToolSchema(
            name="edit",
            description=(
                "Replace an exact string in a file with a new string. "
                "Safer than 'write' for partial changes — only the specified "
                "portion changes; everything else is untouched. "
                "The old_string must match exactly (including whitespace and indentation). "
                "If old_string appears more than once, use replace_all=true or provide "
                "more surrounding context to make it unique."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to edit.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": (
                            "The exact text to find. Must match byte-for-byte, "
                            "including whitespace and indentation."
                        ),
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement text.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": (
                            "When true, replace every occurrence of old_string. "
                            "When false (default), the tool errors if there are "
                            "multiple matches — use this to prevent accidental "
                            "double-replacements."
                        ),
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        """Edit a file by replacing an exact string.

        Args:
            path (str): Absolute path to the file.
            old_string (str): Exact text to find and replace.
            new_string (str): Replacement text.
            replace_all (bool, optional): Replace all occurrences. Default False.

        Returns:
            str: Success message with replacement count, or an error string.
        """
        path = Path(str(kwargs["path"]))
        old_string = str(kwargs["old_string"])
        new_string = str(kwargs["new_string"])
        replace_all = bool(kwargs.get("replace_all", False))

        # check_write: covers sensitive paths, workspace boundary, AND read_only_mode.
        error = self._policy.check_write(path)
        if error:
            return error

        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return f"Error: file not found: {path}"
        except OSError as exc:
            return f"Error reading {path}: {exc}"

        count = content.count(old_string)

        if count == 0:
            return (
                f"Error: string not found in {path}. No changes made.\n"
                "Tip: ensure whitespace and indentation match exactly."
            )

        # Multiple matches without replace_all is likely a bug — the agent
        # should provide more context so the match is unique.
        if count > 1 and not replace_all:
            return (
                f"Error: found {count} occurrences of the string in {path}. "
                "Use replace_all=true to replace all occurrences, or add more "
                "surrounding context to old_string to make the match unique."
            )

        # Replace: -1 means "replace all", 1 means "replace first only".
        new_content = content.replace(old_string, new_string, -1 if replace_all else 1)

        try:
            path.write_text(new_content, encoding="utf-8")
        except OSError as exc:
            return f"Error writing {path}: {exc}"

        replaced = count if replace_all else 1
        return f"Replaced {replaced} occurrence(s) in {path}"

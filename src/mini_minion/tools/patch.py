"""PatchPreviewTool — preview an edit as a unified diff without writing anything.

Why a preview tool?
--------------------
:class:`EditTool` applies an exact-string replacement immediately.  Before
committing an edit, the agent (or a human reviewer) may want to see the diff to
confirm the change is correct, especially in large files where the context
around the replaced string matters.

PatchPreviewTool accepts the same ``path``, ``old_string``, ``new_string``, and
``replace_all`` parameters as :class:`EditTool` but uses Python's
:mod:`difflib` to produce a unified diff and returns it as a string —
**without touching the file**.  This makes it safe to call multiple times
without side-effects.

Design decisions
----------------
- **Same parameters as EditTool**: The model can call ``patch_preview`` first,
  inspect the diff, and then call ``edit`` with identical arguments, confident
  the result matches the preview.
- **Unified diff format**: The ``---`` / ``+++`` / ``@@`` format is familiar to
  developers and LLMs alike.  ``difflib.unified_diff`` produces it in pure
  Python with no external dependencies.
- **Unique-match guard** (same as EditTool): If ``old_string`` matches more
  than once and ``replace_all`` is false, the tool returns an error instead of
  silently previewing a partial replacement.
- **Read-only**: ``is_read_only=True`` signals to the runner that this tool
  never mutates state and can run safely in any order.
- **PermissionPolicy**: Uses the same centralised policy as EditTool and
  GrepTool for consistent workspace-boundary and sensitive-path enforcement.

Talks to
--------
- ``base.py`` — extends :class:`Tool`, returns :class:`ToolSchema`.
- ``policy.py`` — calls :meth:`PermissionPolicy.check_path` before reading the
  file, so workspace and sensitive-path rules are always enforced.
- ``registry.py`` / ``__init__.py`` — registered via ``default_registry()``.
- Python's :mod:`difflib` — ``unified_diff`` generates the diff output.
"""

from __future__ import annotations

import difflib
from pathlib import Path

from .base import Tool, ToolSchema
from .policy import PermissionPolicy


class PatchPreviewTool(Tool):
    """Tool that shows a proposed edit as a unified diff without applying it.

    The agent calls ``patch_preview(path=..., old_string=..., new_string=...)``
    to see exactly what :class:`EditTool` would change before committing.
    The file on disk is never modified.
    """

    def __init__(self, policy: PermissionPolicy) -> None:
        self._policy = policy

    @property
    def schema(self) -> ToolSchema:
        """Describe this tool to the LLM."""
        return ToolSchema(
            name="patch_preview",
            description=(
                "Preview what an edit would look like as a unified diff, without applying it. "
                "Use before 'edit' to confirm the change is correct. "
                "Accepts the same path/old_string/new_string/replace_all parameters as 'edit'."
            ),
            is_read_only=True,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to preview.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Text to replace.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": (
                            "Preview all occurrences replaced (default false). "
                            "Required when old_string appears more than once."
                        ),
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        """Generate a unified diff showing what the edit would produce.

        Args:
            path (str): Absolute path to the file.
            old_string (str): Text that would be replaced.
            new_string (str): Text that would replace it.
            replace_all (bool, optional): If true, preview all occurrences
                replaced.  If false (default) and old_string appears more than
                once, returns an error.

        Returns:
            str: A unified diff string (``---`` / ``+++`` / ``@@`` format).
                Returns an error string when the path is denied, the file cannot
                be read, old_string is not found, or old_string is ambiguous
                without replace_all=true.
        """
        path = Path(str(kwargs["path"]))
        error = self._policy.check_path(path)
        if error:
            return error

        old_string = str(kwargs["old_string"])
        new_string = str(kwargs["new_string"])
        replace_all = bool(kwargs.get("replace_all", False))

        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return f"Error: file not found: '{path}'"
        except OSError as exc:
            return f"Error: {exc}"

        count = content.count(old_string)
        if count == 0:
            return f"Error: old_string not found in '{path}'"
        if count > 1 and not replace_all:
            return (
                f"Error: old_string appears {count} times in '{path}'. "
                "Pass replace_all=true to preview all replacements, or use a "
                "more specific old_string."
            )

        new_content = content.replace(old_string, new_string)

        if new_content == content:
            return "(no changes — old_string and new_string are identical)"

        diff = list(difflib.unified_diff(
            content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
        ))

        return "".join(diff)

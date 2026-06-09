"""TodoWriteTool / TodoReadTool — lightweight in-session todo list.

Why these tools?
----------------
The existing ``ReadTaskTool`` / ``UpdateTaskTool`` manage a *structured task
plan* stored as a JSON file on disk.  Todo items are different: they are
short-lived, session-scoped reminders ("check this edge case", "ask the user
about X") that the agent writes and reads within one conversation.

These tools store the list as a simple JSON file so it persists across turns
in the same session but is easy to inspect.  The file lives in the workspace
under ``.mini-minion/todos.json``.

Design
------
- :class:`TodoWriteTool`: replaces the full list in one call.  Keeping the
  write side simple (one array, no merging) avoids race conditions and keeps
  the implementation under 50 lines.
- :class:`TodoReadTool`: reads and formats the current list for the agent.
- Both tools are idempotent — writing the same list twice has no effect.

Talks to
--------
- ``base.py`` — extends :class:`Tool`, returns :class:`ToolSchema`.
- ``policy.py`` — ``TodoWriteTool`` calls ``check_write()`` before writing.
- ``registry.py`` / ``__init__.py`` — registered via ``default_registry()``.
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import Tool, ToolSchema
from .policy import PermissionPolicy


class TodoWriteTool(Tool):
    """Replace the current todo list with a new set of items.

    Overwrites the entire list rather than merging so the agent always knows
    exactly what's in the list after a write.
    """

    def __init__(
        self,
        todo_path: Path,
        policy: PermissionPolicy | None = None,
    ) -> None:
        # todo_path: absolute path to the todos.json file.
        self._todo_path = todo_path
        # policy: guards writes against read_only_mode and workspace rules.
        self._policy = policy

    @property
    def schema(self) -> ToolSchema:
        """Describe this tool to the LLM."""
        return ToolSchema(
            name="todo_write",
            description=(
                "Replace the current session todo list with a new list of items. "
                "Pass an empty list to clear all todos. "
                "Each item should be a short string (one line)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The new list of todo items (replaces the current list).",
                    },
                },
                "required": ["todos"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        """Overwrite the todo list.

        Args:
            todos (list[str]): The complete new list of todo items.

        Returns:
            str: Confirmation message, or an error string.
        """
        todos = [str(t) for t in (kwargs.get("todos") or [])]

        if self._policy is not None:
            error = self._policy.check_write(self._todo_path)
            if error:
                return error

        try:
            self._todo_path.parent.mkdir(parents=True, exist_ok=True)
            self._todo_path.write_text(json.dumps(todos, indent=2), encoding="utf-8")
            return f"Saved {len(todos)} todo item(s)."
        except OSError as exc:
            return f"Error: {exc}"


class TodoReadTool(Tool):
    """Read the current session todo list.

    Returns a numbered list so the agent can reference items by index.
    """

    def __init__(self, todo_path: Path) -> None:
        # todo_path: same path used by TodoWriteTool for this session.
        self._todo_path = todo_path

    @property
    def schema(self) -> ToolSchema:
        """Describe this tool to the LLM."""
        return ToolSchema(
            name="todo_read",
            description="Read the current session todo list.",
            is_read_only=True,
            parameters={"type": "object", "properties": {}, "required": []},
        )

    def execute(self, **kwargs: object) -> str:
        """Return the numbered todo list.

        Returns:
            str: Numbered list of todo items, or a message if the list is empty.
        """
        if not self._todo_path.exists():
            return "Todo list is empty."
        try:
            todos: list[str] = json.loads(self._todo_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return f"Error reading todo list: {exc}"

        if not todos:
            return "Todo list is empty."
        lines = [f"{i + 1}. {item}" for i, item in enumerate(todos)]
        return "\n".join(lines)

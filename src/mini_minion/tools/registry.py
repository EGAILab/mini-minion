"""Tool registry — stores and dispatches tool calls from the LLM.

The :class:`ToolRegistry` is the directory of all tools available to an agent.
It serves two purposes:
1. **Announce** available tools to the LLM via ``definitions`` — the list of
   schemas the model reads before deciding what to call.
2. **Dispatch** tool calls from the LLM — when the model says "call tool X
   with arguments Y", the registry looks up X and calls ``execute(**Y)``.

Design notes
------------
- Tools are stored in a plain ``dict`` keyed by name. Registering a tool with
  an existing name silently overwrites it (easy replacement, no duplicates).
- ``definitions`` is a Python ``@property`` that rebuilds the list on every
  access. For the typical use case (a small number of tools), this is fine.
- All errors from ``execute()`` are caught and returned as strings. This keeps
  the TAO loop running even if a tool crashes — the agent sees the error and
  can try something else.

Talks to
--------
- ``base.py`` — imports the :class:`Tool` base class.
- ``runner.py`` — calls ``registry.definitions`` to tell the LLM what's
  available, and ``registry.execute()`` to run tool calls.
- ``__init__.py`` — the ``default_registry()`` factory creates and populates
  a registry with all standard tools.
"""

from __future__ import annotations

from .base import Tool


class ToolRegistry:
    """A container for a set of tools, with LLM-friendly definitions and dispatch.

    Args: (none — start empty, then call ``register()``.)

    Attributes:
        _tools (dict[str, Tool]): Internal mapping of tool name → Tool instance.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add a tool to the registry.

        If a tool with the same name already exists, it is replaced. This
        allows overriding default tools with custom implementations.

        Args:
            tool (Tool): A concrete :class:`Tool` instance to register.
        """
        self._tools[tool.schema.name] = tool

    @property
    def definitions(self) -> list[dict]:
        """Return all tool schemas in OpenAI function-calling format.

        This is the list passed to the LLM provider's ``chat()`` method so
        the model knows which tools are available and how to call them.

        Each entry looks like:
        ::

            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file...",
                    "parameters": { ... JSON Schema ... }
                }
            }

        Returns:
            list[dict]: One entry per registered tool, in OpenAI format.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t.schema.name,
                    "description": t.schema.description,
                    "parameters": t.schema.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def execute(self, name: str, arguments: dict) -> str:
        """Look up a tool by name and run it with the given arguments.

        Args:
            name (str): The tool name as returned by the LLM, e.g. ``"read"``.
            arguments (dict): Keyword arguments parsed from the LLM's tool call.

        Returns:
            str: The tool's output string. If the tool name is unknown, returns
                an error string. If the tool raises an exception, wraps it in
                an error string so the agent can read and react to the failure.
        """
        tool = self._tools.get(name)
        if not tool:
            # Return an error string instead of raising — the agent can read
            # this and potentially correct the tool name on the next iteration.
            return f"Unknown tool: {name!r}"
        try:
            return tool.execute(**arguments)
        except Exception as exc:
            # Catch all exceptions and return them as strings.
            # This prevents a buggy tool from crashing the entire conversation.
            return f"Error: {exc}"

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
- ``commands.py`` — the ``/plan`` and ``/auto`` commands read and toggle
  ``registry.policy.read_only_mode`` to switch the agent between plan and auto modes.
"""

from __future__ import annotations

import time as _time
from collections.abc import Callable
from typing import TYPE_CHECKING

from .base import Tool

if TYPE_CHECKING:
    from .policy import PermissionPolicy


class ToolRegistry:
    """A container for a set of tools, with LLM-friendly definitions and dispatch.

    Args:
        before_execute: Optional legacy hook called before each tool execution.
            Signature: ``(name: str, arguments: dict) -> None``.
            Exceptions from this hook are silently swallowed so they never
            crash the tool loop.  Prefer :meth:`add_before_hook` for new code.
        after_execute: Optional legacy hook called after each tool execution.
            Signature: ``(name: str, arguments: dict, output: str, elapsed_ms: int) -> None``.
            Exceptions are silently swallowed.  Prefer :meth:`add_after_hook`.

    Attributes:
        _tools (dict[str, Tool]): Internal mapping of tool name → Tool instance.

    Plugin hooks
    ------------
    Plugins loaded from the local plugin manifest can register hooks by
    calling :meth:`add_before_hook` and :meth:`add_after_hook`.  Each hook
    receives a structured event object (:class:`ToolPreExecuteHookEvent` or
    :class:`ToolPostExecuteHookEvent`) rather than positional arguments.
    Multiple hooks may be registered; they are called in registration order.
    """

    def __init__(
        self,
        before_execute: "Callable[[str, dict], None] | None" = None,
        after_execute: "Callable[[str, dict, str, int], None] | None" = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        # Legacy single-callback hooks (backwards compatible).
        self._before_execute = before_execute
        self._after_execute = after_execute
        # Multi-hook lists for the plugin-facing API.
        # Each callable receives a ToolPreExecuteHookEvent / ToolPostExecuteHookEvent.
        self._before_hooks: list[Callable] = []
        self._after_hooks: list[Callable] = []
        # Shared PermissionPolicy for this registry.  Set by default_registry() after
        # construction so /plan and /auto can toggle policy.read_only_mode at runtime.
        # None until default_registry() assigns it.
        self.policy: "PermissionPolicy | None" = None

    def add_before_hook(self, hook: Callable) -> None:
        """Register a hook to run before every tool execution.

        The hook is called with a :class:`ToolPreExecuteHookEvent` instance.
        Exceptions from the hook are silently swallowed — hooks must never
        crash the tool execution loop.

        Example (in a plugin module)::

            def my_hook(event: ToolPreExecuteHookEvent) -> None:
                print(f"[plugin] About to call: {event.name}")

            registry.add_before_hook(my_hook)

        Args:
            hook: Callable that accepts a :class:`ToolPreExecuteHookEvent`.
        """
        self._before_hooks.append(hook)

    def add_after_hook(self, hook: Callable) -> None:
        """Register a hook to run after every tool execution.

        The hook is called with a :class:`ToolPostExecuteHookEvent` instance
        that includes the tool output and elapsed time.

        Args:
            hook: Callable that accepts a :class:`ToolPostExecuteHookEvent`.
        """
        self._after_hooks.append(hook)

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry by name.

        No-op if the name is not found.  Used by hot-reload to remove stale
        MCP adapters before registering fresh ones.

        Args:
            name: Tool name to remove, e.g. ``"mcp__playwright__screenshot"``.
        """
        self._tools.pop(name, None)

    def unregister_prefix(self, prefix: str) -> int:
        """Remove all tools whose names start with ``prefix``.

        Returns:
            int: Number of tools removed.
        """
        to_remove = [n for n in self._tools if n.startswith(prefix)]
        for n in to_remove:
            del self._tools[n]
        return len(to_remove)

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

        # --- Pre-execute hooks ---
        if self._before_execute is not None:
            try:
                self._before_execute(name, arguments)
            except Exception:
                pass  # hooks must never crash the tool loop

        if self._before_hooks:
            # Import lazily to avoid a circular import at module level.
            from ..agents.events import ToolPreExecuteHookEvent
            pre_event = ToolPreExecuteHookEvent(name=name, arguments=arguments)
            for hook in self._before_hooks:
                try:
                    hook(pre_event)
                except Exception:
                    pass

        _start = _time.monotonic()
        try:
            output = tool.execute(**arguments)
        except Exception as exc:
            # Catch all exceptions and return them as strings.
            # This prevents a buggy tool from crashing the entire conversation.
            output = f"Error: {exc}"

        elapsed_ms = int((_time.monotonic() - _start) * 1000)

        # --- Post-execute hooks ---
        if self._after_execute is not None:
            try:
                self._after_execute(name, arguments, output, elapsed_ms)
            except Exception:
                pass  # hooks must never crash the tool loop

        if self._after_hooks:
            from ..agents.events import ToolPostExecuteHookEvent
            post_event = ToolPostExecuteHookEvent(
                name=name, arguments=arguments, output=output, elapsed_ms=elapsed_ms
            )
            for hook in self._after_hooks:
                try:
                    hook(post_event)
                except Exception:
                    pass

        return output

    def is_read_only(self, name: str) -> bool:
        """Return True if the named tool declares itself as read-only.

        Used by the runner to identify tool-call batches that can execute
        concurrently. Unknown tools return False (safe default: treat as mutating).
        """
        tool = self._tools.get(name)
        return tool.schema.is_read_only if tool else False

"""spawn_subagent tool — delegate a subtask to a child AgentSession and return the result.

An agent calls ``spawn_subagent`` to hand off a self-contained task to a
subagent.  The tool blocks (in a daemon thread) until the subagent finishes
or times out, then returns the subagent's text response as a plain string.

Design overview
---------------
:class:`SpawnSubagentTool` is intentionally thin: all the heavy work
(provider creation, registry setup, session creation, threading, depth/child
limit enforcement) lives in the ``spawn_fn`` closure that ``minion.py``
constructs and injects at registration time.  This keeps the tool testable —
tests can pass a mock ``spawn_fn`` without standing up a full agent stack.

Phase 4 — event relay
----------------------
``relay_fn`` is an optional callback set by ``minion.py``.  When provided,
the subagent's ``on_event`` stream is forwarded to the parent's terminal
handler so the user sees subagent output in real time with a
``[subagent:{agent_id}]`` prefix rather than waiting silently.

Public API
----------
- :class:`SpawnSubagentTool` — the tool itself.
- :func:`_make_subagent_registry` — helper used by ``minion.py``'s
  ``spawn_fn`` to build a read-only tool registry for child sessions.

Talks to
--------
- ``minion.py``        — constructs ``spawn_fn`` and ``relay_fn``, then calls
                          ``registry.register(SpawnSubagentTool(...))``.
- ``agents/session.py``— ``spawn_fn`` creates and calls ``AgentSession.send()``.
- ``spawn_registry.py``— ``spawn_fn`` calls depth/child limit helpers.
- ``workspace.py``     — ``spawn_fn`` calls ``agent_workspace_root()``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from .base import Tool, ToolSchema
from .glob import GlobTool
from .grep import GrepTool
from .policy import PermissionPolicy
from .read import ReadTool
from .registry import ToolRegistry
from .web_fetch import WebFetchTool
from .web_search import WebSearchTool


def _make_subagent_registry(root: Path | None = None) -> ToolRegistry:
    """Build a minimal read-only tool registry for subagent sessions.

    Subagents receive: ReadTool, GlobTool, GrepTool, WebSearchTool, WebFetchTool.
    Write tools (WriteTool, EditTool, BashTool) are intentionally excluded so
    subagents cannot mutate the filesystem unless explicitly granted write access.

    Args:
        root: Optional workspace root for path-boundary checks.  ``None``
            disables the workspace boundary.

    Returns:
        ToolRegistry: A populated registry with read-only tools registered.
    """
    policy = PermissionPolicy.default(workspace=root)
    registry = ToolRegistry()
    for tool in [
        ReadTool(root, policy=policy),
        GlobTool(root, policy=policy),
        GrepTool(policy),
        WebSearchTool(policy=policy),
        WebFetchTool(policy),
    ]:
        registry.register(tool)
    registry.policy = policy
    return registry


class SpawnSubagentTool(Tool):
    """Delegate a self-contained subtask to a child agent and return its response.

    The tool blocks until the subagent finishes (or the timeout is hit), then
    returns the subagent's final text response.  All session creation, depth
    limit enforcement, and thread management are handled inside ``spawn_fn`` —
    a closure provided by ``minion.py`` at registration time.

    Args:
        spawn_fn: Callable with signature
            ``(task, agent_id, timeout_seconds, relay_fn) -> str``.
            Created by ``minion.py``; encapsulates provider, session store,
            short-term memory, and limit checks.
        relay_fn: Optional event relay callback (Phase 4).  When provided,
            the parent's ``_on_event`` handler is wrapped to tag subagent
            events with ``[subagent:{agent_id}]`` before printing.  ``None``
            runs the subagent silently (result returned when done).
    """

    def __init__(
        self,
        spawn_fn: Callable[[str, str, int, Callable | None], str],
        relay_fn: Callable | None = None,
    ) -> None:
        self._spawn_fn = spawn_fn
        self._relay_fn = relay_fn

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="spawn_subagent",
            description=(
                "Delegate a self-contained subtask to a subagent and return its response. "
                "Use when a task is independent, has a clear deliverable, and can run "
                "without interactive follow-up. The subagent runs to completion and "
                "its final text response is returned. "
                "Subagents have read access to files and web but cannot write files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "The task for the subagent. Be specific — include all "
                            "context the subagent needs since it has no shared history."
                        ),
                    },
                    "agent_id": {
                        "type": "string",
                        "description": (
                            "Agent ID to use (default: 'researcher'). "
                            "Must match an agent defined in AGENTS.md."
                        ),
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Maximum seconds to wait for the subagent (default: 120).",
                    },
                },
                "required": ["task"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        """Run the subagent and return its response.

        Args:
            task (str): The task to delegate.
            agent_id (str): Target agent ID (default: ``"researcher"``).
            timeout_seconds (int): Timeout in seconds (default: 120).

        Returns:
            str: The subagent's final text response, or an error string if
                the subagent timed out, exceeded depth/child limits, or failed.
        """
        task = str(kwargs.get("task", ""))
        agent_id = str(kwargs.get("agent_id", "researcher"))
        timeout_seconds = int(kwargs.get("timeout_seconds", 120))

        if not task.strip():
            return "[spawn_subagent] Error: 'task' must be a non-empty string."

        return self._spawn_fn(task, agent_id, timeout_seconds, self._relay_fn)

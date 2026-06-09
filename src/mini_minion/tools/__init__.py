"""Public API and factory for the tools subsystem.

This module re-exports the key tool types and provides the
:func:`default_registry` factory function that creates a ready-to-use
:class:`ToolRegistry` with all standard tools pre-registered.

Why a factory function?
-----------------------
:func:`default_registry` exists because several tools require instances to be
injected at construction:

- :class:`SaveMemoryTool` / :class:`SearchMemoryTool` need a
  :class:`LongTermMemory` instance.
- :class:`ReadTaskTool` / :class:`UpdateTaskTool` need the path to the agent's
  task JSON file (derived from ``tasks_dir`` + ``agent_id``).
- :class:`BashTool` accepts an optional confirmation callback and a working
  directory.

This is the **dependency injection** pattern — backends are provided from
outside rather than created inside the tool, which keeps tools testable with
mock dependencies and avoids hard-coded paths.

Talks to
--------
- ``minion.py`` calls ``default_registry(...)`` at startup.
- ``runner.py`` receives the :class:`ToolRegistry` and calls it during the
  TAO loop.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mcp.client import McpClientManager

from .base import Tool, ToolSchema
from .bash import BashTool
from .edit import EditTool
from .glob import GlobTool
from .grep import GrepTool
from .memory import SaveMemoryTool, SearchMemoryTool
from .policy import PermissionPolicy
from .read import ReadTool
from .registry import ToolRegistry
from .skill import SkillTool
from .task import ReadTaskTool, UpdateTaskTool
from .web_fetch import WebFetchTool
from .web_search import WebSearchTool
from .write import WriteTool


def default_registry(
    long_term: "LongTermMemory | None" = None,
    root: Path | None = None,
    bash_confirm: "Callable[[str], bool] | None" = None,
    skills: "SkillRegistry | None" = None,
    tasks_dir: Path | None = None,
    agent_id: str | None = None,
    mcp_manager: "McpClientManager | None" = None,
    policy: "PermissionPolicy | None" = None,
) -> ToolRegistry:
    """Build a :class:`ToolRegistry` with all standard tools registered.

    Args:
        long_term:   If provided, also registers memory tools with this backend.
        root:        Workspace root path. File tools reject paths outside this
                     boundary.  Pass ``None`` to disable path restriction.
        bash_confirm: Optional callable passed to :class:`BashTool`.  Called
                     with the command string before execution; returns ``True``
                     to proceed.  ``None`` runs without asking (headless).
        skills:      Optional skill registry.  When non-empty, also registers
                     a :class:`SkillTool`.
        tasks_dir:   Directory for per-agent task JSON files.  When provided
                     alongside ``agent_id``, registers
                     :class:`ReadTaskTool` + :class:`UpdateTaskTool`.
        agent_id:    The agent's ID, used to build the task file path
                     ``{tasks_dir}/{agent_id}.json``.
        mcp_manager: If provided, registers MCP status/resource tools and
                     one :class:`McpToolAdapter` per connected MCP tool.
        policy:      Optional :class:`PermissionPolicy` injected into edit,
                     grep, and web_fetch tools.  When ``None`` a default
                     policy is built from ``root`` (so path restrictions still
                     apply even when the caller omits this argument).

    Returns:
        :class:`ToolRegistry` populated and ready to pass to :func:`run_turn`.
    """
    # Build a PermissionPolicy from root when the caller didn't supply one.
    # This ensures EditTool/GrepTool/WebFetchTool always have a workspace
    # boundary even when minion.py doesn't pass an explicit policy object.
    _policy = policy or PermissionPolicy.default(workspace=root)

    registry = ToolRegistry()

    # Core file-system and shell tools — always registered.
    for tool in [
        ReadTool(root),
        WriteTool(root),
        GlobTool(root),
        BashTool(confirm=bash_confirm, cwd=root),
        # WebSearchTool has no external dependencies at construction time; it
        # fails gracefully at execute() if duckduckgo-search is not installed.
        WebSearchTool(),
        # New coding-agent tools — use PermissionPolicy for path/URL checks.
        EditTool(_policy),
        GrepTool(_policy),
        WebFetchTool(_policy),
    ]:
        registry.register(tool)

    # Memory tools — only when a backend is provided.
    if long_term is not None:
        registry.register(SaveMemoryTool(long_term))
        registry.register(SearchMemoryTool(long_term))

    # Skill tool — only when at least one skill was discovered.
    if skills:
        registry.register(SkillTool(skills))

    # Task tools — only when both tasks_dir and agent_id are provided.
    if tasks_dir is not None and agent_id is not None:
        task_path = Path(tasks_dir) / f"{agent_id}.json"
        registry.register(ReadTaskTool(task_path))
        registry.register(UpdateTaskTool(task_path))

    # MCP tools — only when a manager is provided.
    # Import here to avoid a circular import (mcp.types imports from config).
    if mcp_manager is not None:
        from .mcp import (
            GetMcpPromptTool,
            ListMcpPromptsTool,
            ListMcpResourcesTool,
            McpToolAdapter,
            McpStatusTool,
            ReadMcpResourceTool,
        )
        # Always register the five management tools so the agent can inspect
        # server status and browse the full MCP capability triad:
        # tools / resources / prompts.
        registry.register(McpStatusTool(mcp_manager))
        registry.register(ListMcpResourcesTool(mcp_manager))
        registry.register(ReadMcpResourceTool(mcp_manager))
        registry.register(ListMcpPromptsTool(mcp_manager))
        registry.register(GetMcpPromptTool(mcp_manager))
        # Register one adapter per tool discovered from connected servers.
        for tool_info in mcp_manager.list_tools():
            registry.register(McpToolAdapter(tool_info, mcp_manager))

    return registry


__all__ = [
    "Tool",
    "ToolSchema",
    "ToolRegistry",
    "PermissionPolicy",
    "EditTool",
    "GrepTool",
    "WebFetchTool",
    "SkillTool",
    "ReadTaskTool",
    "UpdateTaskTool",
    "default_registry",
]

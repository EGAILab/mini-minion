"""Public API and factory for the tools subsystem.

This module re-exports the key tool types and provides the
:func:`default_registry` factory function that creates a ready-to-use
:class:`ToolRegistry` with all standard tools pre-registered.

Why a factory function?
-----------------------
:func:`default_registry` exists because several tools require instances to be
injected at construction:

- :class:`SaveMemoryTool` / :class:`SearchMemoryTool` need a
  :class:`~minion_assist.memory.service.MemoryService` instance.
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
    from ..memory.service import MemoryService

from .apply_patch import ApplyPatchTool
from .browser import BrowserTool
from .ask_user import AskUserTool
from .audit import ApprovalDecision, AuditLog
from .base import Tool, ToolSchema
from .bash import BashTool
from .edit import EditTool
from .find_definition import FindDefinitionTool
from .git import GitCommitTool, GitDiffTool, GitStatusTool
from .glob import GlobTool
from .grep import GrepTool
from .memory import SaveMemoryTool, SearchMemoryTool
from .patch import PatchPreviewTool
from .session_search import SessionSearchTool
from .policy import PermissionPolicy
from .read import ReadTool
from .registry import ToolRegistry
from .sandbox import LocalSandboxBackend, SandboxBackend
from .skill import SkillTool
from .spawn_subagent import SpawnSubagentTool
from .task import ReadTaskTool, UpdateTaskTool
from .todo import TodoReadTool, TodoWriteTool
from .web_fetch import WebFetchTool
from .web_search import WebSearchTool
from .write import WriteTool
from .write_daily_memory import WriteDailyMemoryTool


def default_registry(
    memory: "MemoryService | None" = None,
    root: Path | None = None,
    bash_confirm: "Callable[[str], bool] | None" = None,
    bash_approval: "Callable[[str], ApprovalDecision] | None" = None,
    skills: "SkillRegistry | None" = None,
    tasks_dir: Path | None = None,
    agent_id: str | None = None,
    mcp_manager: "McpClientManager | None" = None,
    policy: "PermissionPolicy | None" = None,
    ask_user_fn: "Callable[[str], str] | None" = None,
    write_confirm: "Callable[[str], bool] | None" = None,
    db: object | None = None,
) -> ToolRegistry:
    """Build a :class:`ToolRegistry` with all standard tools registered.

    Args:
        memory:       If provided, also registers memory tools with this
                      :class:`~minion_assist.memory.service.MemoryService`
                      backend.
        root:         Workspace root path. File tools reject paths outside this
                      boundary.  Pass ``None`` to disable path restriction.
        bash_confirm: Optional callable passed to :class:`BashTool`.  Called
                      with the command string before execution; returns ``True``
                      to proceed.  ``None`` runs without asking (headless).
        skills:       Optional skill registry.  When non-empty, also registers
                      a :class:`SkillTool`.
        tasks_dir:    Directory for per-agent task JSON files.  When provided
                      alongside ``agent_id``, registers
                      :class:`ReadTaskTool` + :class:`UpdateTaskTool`.
        agent_id:     The agent's ID, used to build the task file path
                      ``{tasks_dir}/{agent_id}.json``.
        mcp_manager:  If provided, registers MCP status/resource tools and
                      one :class:`McpToolAdapter` per connected MCP tool.
        policy:       Optional :class:`PermissionPolicy` injected into all I/O
                      tools (read, write, glob, edit, grep, web_fetch).  When
                      ``None`` a default policy is built from ``root``.
        bash_approval: Optional 4-option approval callback for :class:`BashTool`.
                      Called with the command string; returns an
                      :class:`ApprovalDecision`.  Takes priority over
                      ``bash_confirm`` when both are provided.  The CLI wires
                      this to a menu that records decisions to the audit log.
        ask_user_fn:  Optional callable for :class:`AskUserTool`.  Called with
                      the agent's question string; returns the human's answer.
                      ``None`` registers the tool in headless mode (returns an
                      error when the agent calls it).
        write_confirm: Optional human approval callback passed to
                      :class:`WriteTool` and :class:`EditTool`.  Called with
                      a one-line description before each write; returning
                      ``False`` cancels the operation.  ``None`` (default)
                      means automatic writes without prompting.

    Returns:
        :class:`ToolRegistry` populated and ready to pass to :func:`run_turn`.
    """
    # Build a PermissionPolicy from root when the caller didn't supply one.
    # This ensures EditTool/GrepTool/WebFetchTool always have a workspace
    # boundary even when minion.py doesn't pass an explicit policy object.
    _policy = policy or PermissionPolicy.default(workspace=root)

    registry = ToolRegistry()

    # Browser tool — always registered; playwright is imported lazily inside the
    # tool itself so the package is optional (uv add playwright to activate).
    registry.register(BrowserTool())

    # Core file-system and shell tools — always registered.
    # Legacy tools (read/write/glob) now accept an optional policy so their
    # path safety checks stay in sync with the centralised PermissionPolicy.
    for tool in [
        ReadTool(root, policy=_policy),
        WriteTool(root, policy=_policy, confirm=write_confirm),
        GlobTool(root, policy=_policy),
        # BashTool: policy handles SSRF + read_only_mode; approval_fn provides the
        # 4-option confirm menu; confirm is the simpler bool fallback.
        BashTool(confirm=bash_confirm, cwd=root, policy=_policy, approval_fn=bash_approval),
        # WebSearchTool: policy enables SSRF marker checks on the query string.
        WebSearchTool(policy=_policy),
        # Coding-agent tools — use PermissionPolicy for path/URL checks.
        # write_confirm callback lets the caller (e.g. REPL) gate each edit on user approval.
        EditTool(_policy, confirm=write_confirm),
        GrepTool(_policy),
        WebFetchTool(_policy),
        PatchPreviewTool(_policy),
        # ApplyPatchTool: companion to PatchPreviewTool that actually applies the diff.
        ApplyPatchTool(cwd=root, policy=_policy),
        # FindDefinitionTool: AST-based symbol lookup across all .py files.
        FindDefinitionTool(root=root, policy=_policy),
        # Human-interaction tool — always registered; headless mode (ask_user_fn=None)
        # returns an informative error rather than hanging.
        AskUserTool(ask_user_fn),
        # Git tools — structured git interface without needing a shell.
        # GitCommitTool reuses bash_confirm so the user approves commits the same way.
        # All three now accept policy so read_only_mode and workspace rules apply.
        GitStatusTool(cwd=root, policy=_policy),
        GitDiffTool(cwd=root, policy=_policy),
        GitCommitTool(cwd=root, confirm=bash_confirm, policy=_policy),
    ]:
        registry.register(tool)

    # Session search tool — only when a PostgreSQL db is connected.
    if db is not None:
        registry.register(SessionSearchTool(db))

    # Memory tools — only when a backend is provided.
    # SaveMemoryTool and WriteDailyMemoryTool receive policy so /plan read-only
    # mode blocks writes. Note: only read_only_mode is checked (not check_write)
    # because memory files live under the agent's own workspace root, outside
    # the tool sandbox boundary.
    if memory is not None:
        registry.register(SaveMemoryTool(memory, policy=_policy))
        registry.register(SearchMemoryTool(memory))
        # Lets the agent append notes to memory/YYYY-MM-DD.md without needing
        # to read -> edit -> write the whole file.
        registry.register(WriteDailyMemoryTool(memory, policy=_policy))

    # Skill tool — only when at least one skill was discovered.
    if skills:
        registry.register(SkillTool(skills))

    # Task tools — only when both tasks_dir and agent_id are provided.
    if tasks_dir is not None and agent_id is not None:
        task_path = Path(tasks_dir) / f"{agent_id}.json"
        registry.register(ReadTaskTool(task_path))
        registry.register(UpdateTaskTool(task_path))
        # TodoWriteTool / TodoReadTool: per-agent session-scoped todo list.
        # Stored at tasks_dir/<agent_id>-todos.json (separate from the structured plan).
        todo_path = Path(tasks_dir) / f"{agent_id}-todos.json"
        registry.register(TodoWriteTool(todo_path, policy=_policy))
        registry.register(TodoReadTool(todo_path))

    # Expose the shared policy on the registry so /plan and /auto can toggle
    # read_only_mode at runtime without re-building the registry.
    registry.policy = _policy

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
    "BrowserTool",
    "PermissionPolicy",
    "ApprovalDecision",
    "AuditLog",
    "SandboxBackend",
    "LocalSandboxBackend",
    "ApplyPatchTool",
    "AskUserTool",
    "EditTool",
    "FindDefinitionTool",
    "GitCommitTool",
    "GitDiffTool",
    "GitStatusTool",
    "GrepTool",
    "PatchPreviewTool",
    "SaveMemoryTool",
    "SearchMemoryTool",
    "SessionSearchTool",
    "TodoReadTool",
    "TodoWriteTool",
    "WebFetchTool",
    "SkillTool",
    "SpawnSubagentTool",
    "ReadTaskTool",
    "UpdateTaskTool",
    "WriteDailyMemoryTool",
    "default_registry",
]

"""Public API and factory for the tools subsystem.

This module re-exports the key tool types and provides the
:func:`default_registry` factory function that creates a ready-to-use
:class:`ToolRegistry` with all standard tools pre-registered.

Why a factory function?
-----------------------
:func:`default_registry` exists because two of the tools (:class:`SaveMemoryTool`
and :class:`SearchMemoryTool`) require a :class:`LongTermMemory` instance to be
injected at construction. The factory handles this optional dependency cleanly:
- Without ``long_term``: 4 tools (read, write, glob, bash).
- With ``long_term``: 6 tools (+ save_memory, search_memory).

This is the **dependency injection** pattern — the memory backend is provided
from outside rather than created inside the tool, which makes the tools testable
with mock memory backends and avoids hard-coded paths.

Talks to
--------
- ``minion.py`` calls ``default_registry(long_term=long_term)`` at startup.
- ``runner.py`` receives the :class:`ToolRegistry` and calls it during the TAO loop.
"""

from pathlib import Path

from .base import Tool, ToolSchema
from .bash import BashTool
from .glob import GlobTool
from .memory import SaveMemoryTool, SearchMemoryTool
from .read import ReadTool
from .registry import ToolRegistry
from .write import WriteTool


def default_registry(
    long_term: "LongTermMemory | None" = None,
    root: Path | None = None,
    confirm_bash: bool = True,
) -> ToolRegistry:
    """Build a :class:`ToolRegistry` with all standard tools registered.

    Args:
        long_term: If provided, also registers memory tools with this backend.
        root: Workspace root path. File tools reject paths outside this boundary.
            Pass ``None`` to disable path restriction (unrestricted access).
        confirm_bash: If ``True`` (default), the bash tool prints each command
            and requires ``y`` confirmation before executing.

    Returns:
        ToolRegistry: A populated registry ready to pass to :func:`run_turn`.
    """
    registry = ToolRegistry()
    # Register the four core filesystem and shell tools unconditionally.
    for tool in [ReadTool(root), WriteTool(root), GlobTool(root), BashTool(confirm=confirm_bash, cwd=root)]:
        registry.register(tool)

    # Only add memory tools if a LongTermMemory backend was provided.
    # Both tools share the same instance so reads and writes use the same store.
    if long_term is not None:
        registry.register(SaveMemoryTool(long_term))
        registry.register(SearchMemoryTool(long_term))

    return registry


__all__ = ["Tool", "ToolSchema", "ToolRegistry", "default_registry"]

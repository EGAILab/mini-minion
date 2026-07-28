"""WriteDailyMemoryTool — append an entry to today's daily memory file.

Daily memory files live at ``memory/YYYY-MM-DD.md`` inside the agent's
workspace.  The agent can already *read* them via the standard ``read`` tool;
this tool provides a dedicated path for *appending* new entries so the agent
doesn't need to read → edit → write the whole file.

The tool is registered in the global ToolRegistry during startup so agents can
call it any time, not just during heartbeats.

History: absorbing the ``note`` tool (Stage One Phase 1, slice 4)
-------------------------------------------------------------------
This tool used to write directly to ``workspace_root/memory/{date}.md`` via
raw file I/O, while a separate ``note`` tool wrote a *different* daily file
(``memory/imports/_notes_{date}.md``, quarantined) through
:class:`~minion_assist.memory.long_term.LongTermMemory`. Both did almost the
same thing in different places. ``note`` is retired; this tool now delegates
to :meth:`MemoryService.append_daily`, which combines both tools' formats
(a ``## {date}`` header written once, then a timestamped bullet per entry —
see ``memory/files.py``'s ``append_daily`` docstring).

Talks to
--------
- ``memory/service.py`` — :meth:`MemoryService.append_daily` does the actual
  write.
- ``policy.py`` — :class:`PermissionPolicy` used to check ``read_only_mode``
  (the ``note`` tool checked this; this tool did not before the merge — that
  was a gap, now closed).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Tool, ToolSchema

if TYPE_CHECKING:
    from ..memory.service import MemoryService
    from .policy import PermissionPolicy


class WriteDailyMemoryTool(Tool):
    """Append a timestamped entry to today's ``memory/YYYY-MM-DD.md`` file.

    Args:
        memory (MemoryService): The memory backend. Injected at construction.
        policy (PermissionPolicy | None): Optional permission policy.  Only
            ``read_only_mode`` is checked (memory files live outside the tool
            sandbox boundary — see :class:`SaveMemoryTool`'s docstring for
            why ``check_write()`` doesn't apply here).
    """

    def __init__(
        self,
        memory: "MemoryService",
        policy: "PermissionPolicy | None" = None,
    ) -> None:
        self._memory = memory
        self._policy = policy

    @property
    def schema(self) -> ToolSchema:
        """Describe this tool to the LLM."""
        return ToolSchema(
            name="write_daily_memory",
            description=(
                "Append a note to today's daily memory file (memory/YYYY-MM-DD.md). "
                "Use this to log events, decisions, or context you want to remember "
                "in future sessions. The entry is timestamped automatically. "
                "For significant long-term learnings, also update MEMORY.md directly "
                "using the write or edit tools."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The note to append to today's memory file.",
                    },
                },
                "required": ["content"],
            },
            is_read_only=False,
        )

    def execute(self, **kwargs: object) -> str:
        """Append *content* to today's daily memory file.

        Args:
            content (str): Note text to append.

        Returns:
            str: Path of the file written to, or an error message.
        """
        if self._policy is not None and self._policy.read_only_mode:
            return (
                "Error: read-only mode is active — memory writes are not permitted. "
                "Use /auto to disable."
            )

        content = str(kwargs.get("content", "")).strip()
        if not content:
            return "[write_daily_memory] Empty content — nothing written."

        path = self._memory.append_daily(content)
        return f"[write_daily_memory] Appended to {path}."

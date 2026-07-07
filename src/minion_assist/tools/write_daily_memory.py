"""WriteDailyMemoryTool — append an entry to today's daily memory file.

Daily memory files live at ``memory/YYYY-MM-DD.md`` inside the agent's
workspace.  The agent can already *read* them via the standard ``read`` tool;
this tool provides a dedicated path for *appending* new entries so the agent
doesn't need to read → edit → write the whole file.

The tool is registered in the global ToolRegistry during startup so agents can
call it any time, not just during heartbeats.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from .base import Tool, ToolSchema


class WriteDailyMemoryTool(Tool):
    """Append a timestamped entry to today's ``memory/YYYY-MM-DD.md`` file.

    Args:
        workspace_root: The agent's workspace directory.  The ``memory/``
            subdirectory is created automatically when it doesn't exist.
    """

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root

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
            str: Path of the file written to and the number of bytes appended.
        """
        content = str(kwargs.get("content", "")).strip()
        if not content:
            return "[write_daily_memory] Empty content — nothing written."

        today = date.today().isoformat()  # "YYYY-MM-DD"
        memory_dir = self._workspace_root / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)

        target = memory_dir / f"{today}.md"
        entry = f"\n## {today}\n\n{content}\n"

        # Append mode — safe for concurrent writes (single process, single thread).
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(entry)

        rel = os.path.relpath(target, self._workspace_root)
        return f"[write_daily_memory] Appended {len(entry)} chars to {rel}."

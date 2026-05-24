"""Tools for saving and searching long-term memory."""

from __future__ import annotations

from .base import Tool, ToolSchema
from ..memory.long_term import LongTermMemory


class SaveMemoryTool(Tool):
    def __init__(self, memory: LongTermMemory) -> None:
        self._memory = memory

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="save_memory",
            description="Save a note to long-term memory under a given key.",
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Identifier for this memory (e.g. 'project-goals')"},
                    "content": {"type": "string", "description": "Markdown content to save"},
                },
                "required": ["key", "content"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        key = str(kwargs["key"])
        content = str(kwargs["content"])
        self._memory.save(key, content)
        return f"Saved memory: {key}"


class SearchMemoryTool(Tool):
    def __init__(self, memory: LongTermMemory) -> None:
        self._memory = memory

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="search_memory",
            description="Search long-term memory for notes containing a query string.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text to search for"},
                },
                "required": ["query"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        query = str(kwargs["query"])
        results = self._memory.search(query)
        if not results:
            return f"No memories found for: {query!r}"
        parts = []
        for key, content in results:
            parts.append(f"## {key}\n{content}")
        return "\n\n".join(parts)

"""WriteTool — write content to a file."""

from __future__ import annotations

import pathlib

from .base import Tool, ToolSchema


class WriteTool(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="write",
            description="Write content to a file. Creates parent directories if needed.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write",
                    },
                },
                "required": ["path", "content"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        path = pathlib.Path(str(kwargs["path"]))
        content = str(kwargs["content"])
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} chars to {path}"
        except OSError as exc:
            return f"Error: {exc}"

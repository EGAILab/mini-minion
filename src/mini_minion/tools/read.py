"""ReadTool — read a file with line numbers, or list a directory."""

from __future__ import annotations

import pathlib

from .base import Tool, ToolSchema

_DEFAULT_LIMIT = 200


class ReadTool(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="read",
            description="Read a file with line numbers, or list the contents of a directory.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file or directory",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Start line, 1-indexed (default 1)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": f"Max lines to return (default {_DEFAULT_LIMIT})",
                    },
                },
                "required": ["path"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        path = pathlib.Path(str(kwargs["path"]))
        offset = max(1, int(kwargs.get("offset") or 1))
        limit = max(1, int(kwargs.get("limit") or _DEFAULT_LIMIT))

        if not path.exists():
            return f"File not found: {path}"

        if path.is_dir():
            entries = sorted(e.name + ("/" if e.is_dir() else "") for e in path.iterdir())
            return "\n".join(entries) or "(empty directory)"

        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return f"Error: {exc}"

        start = offset - 1
        sliced = lines[start : start + limit]
        total = len(lines)
        output = "\n".join(f"{offset + i}: {line}" for i, line in enumerate(sliced))
        end = offset + len(sliced) - 1
        if end < total:
            output += f"\n\n(Lines {offset}–{end} of {total}. Use offset={end + 1} to continue.)"
        return output

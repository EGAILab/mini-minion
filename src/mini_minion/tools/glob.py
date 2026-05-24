"""GlobTool — find files by glob pattern, sorted by modification time."""

from __future__ import annotations

import glob as _glob
import os
from pathlib import Path

from .base import Tool, ToolSchema


class GlobTool(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="glob",
            description=(
                "Find files matching a glob pattern, sorted by modification time (newest first). "
                "Use ** for recursive matching (e.g. '**/*.py')."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g. '**/*.py', 'src/*.ts')",
                    },
                    "path": {
                        "type": "string",
                        "description": "Root directory to search. Defaults to current directory.",
                    },
                },
                "required": ["pattern"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        pattern = str(kwargs["pattern"])
        root = Path(str(kwargs["path"])) if kwargs.get("path") else Path.cwd()
        matches = _glob.glob(str(root / pattern), recursive=True)
        matches = [m for m in matches if os.path.isfile(m)]
        matches.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)
        return "\n".join(matches) if matches else "(no matches)"

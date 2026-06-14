"""FindDefinitionTool — find where a symbol is defined in the workspace.

Why this tool?
--------------
Agents frequently need to navigate to where a function, class, or variable is
defined.  Without this tool they must ``grep`` for the symbol and manually
filter out call sites, imports, and comments.  ``FindDefinitionTool`` uses
Python's built-in ``ast`` module to parse source files and return only genuine
*definition* lines (``def``, ``class``, or top-level assignment).

Scope
-----
This tool searches ``.py`` files only.  For other languages the agent can fall
back to :class:`GrepTool`.

Design
------
- Walk the workspace directory with ``os.walk``, parse each ``.py`` file with
  ``ast.parse``, and collect nodes whose name matches the target symbol.
- Returns file path, line number, and a short context snippet so the agent can
  navigate directly without a second ``read`` call.
- Skips files that cannot be parsed (syntax errors, binary garbage) silently,
  because most workspaces contain at least one unparseable file and it would
  be unhelpful to abort the whole search.

Talks to
--------
- ``base.py`` — extends :class:`Tool`, returns :class:`ToolSchema`.
- ``policy.py`` — uses ``check_path()`` for workspace boundary.
- ``registry.py`` / ``__init__.py`` — registered via ``default_registry()``.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

from .base import Tool, ToolSchema
from .policy import PermissionPolicy

# Max files to parse before stopping to avoid hanging on huge monorepos.
_MAX_FILES = 500


def _matches(node: ast.AST, symbol: str) -> bool:
    """Return True if the AST node defines ``symbol``."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name == symbol
    if isinstance(node, ast.Assign):
        # top-level assignments: ``CONSTANT = ...``
        return any(
            isinstance(t, ast.Name) and t.id == symbol
            for t in node.targets
        )
    if isinstance(node, ast.AnnAssign):
        # annotated assignments: ``name: Type = ...``
        return isinstance(node.target, ast.Name) and node.target.id == symbol
    return False


class FindDefinitionTool(Tool):
    """Search Python source files for the definition of a named symbol.

    Returns the file path, line number, and a one-line snippet for each
    definition site found.
    """

    def __init__(
        self,
        root: Path | None = None,
        policy: PermissionPolicy | None = None,
    ) -> None:
        # root: where to start the recursive search.  None falls back to cwd.
        self._root = root
        # policy: workspace boundary check on the search root.
        self._policy = policy

    @property
    def schema(self) -> ToolSchema:
        """Describe this tool to the LLM."""
        return ToolSchema(
            name="find_definition",
            description=(
                "Find where a Python symbol (function, class, or variable) is defined "
                "in the workspace. Searches all .py files under the workspace root "
                f"(up to {_MAX_FILES} files). Returns file path, line number, and a "
                "one-line snippet for each definition site."
            ),
            is_read_only=True,
            parameters={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The exact symbol name to look for (e.g. 'MyClass', 'my_function').",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Subdirectory to search within the workspace. "
                            "Defaults to the full workspace root."
                        ),
                    },
                },
                "required": ["symbol"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        """Search for the definition of a symbol in .py files.

        Args:
            symbol (str): Exact symbol name to find.
            path (str, optional): Subdirectory to search within.

        Returns:
            str: One result per line in the format ``path:lineno: <snippet>``,
                or a message indicating nothing was found.
        """
        symbol = str(kwargs["symbol"])
        search_root = self._root or Path.cwd()

        if kwargs.get("path"):
            sub = Path(str(kwargs["path"]))
            # Resolve relative paths against the workspace root.
            if not sub.is_absolute():
                sub = search_root / sub
            search_root = sub

        # Workspace boundary check.
        if self._policy is not None:
            error = self._policy.check_path(search_root)
            if error:
                return error

        results: list[str] = []
        files_checked = 0

        for dirpath, _dirs, filenames in os.walk(search_root):
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                if files_checked >= _MAX_FILES:
                    results.append(f"(search capped at {_MAX_FILES} files)")
                    break
                filepath = Path(dirpath) / filename
                files_checked += 1
                try:
                    source = filepath.read_text(encoding="utf-8", errors="replace")
                    tree = ast.parse(source, filename=str(filepath))
                except SyntaxError:
                    # Skip files that cannot be parsed (e.g. Python 2, template files).
                    continue
                except OSError:
                    continue

                lines = source.splitlines()
                for node in ast.walk(tree):
                    if _matches(node, symbol):
                        lineno = node.lineno  # type: ignore[attr-defined]
                        snippet = lines[lineno - 1].strip() if lineno <= len(lines) else ""
                        results.append(f"{filepath}:{lineno}: {snippet}")
            else:
                continue
            break  # files_checked >= _MAX_FILES — also stop the outer loop

        if not results:
            return f"No definition of '{symbol}' found in {search_root}."
        return "\n".join(results)

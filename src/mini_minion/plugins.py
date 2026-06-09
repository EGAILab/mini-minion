"""Local plugin manifest loader.

This module allows users to extend mini-minion with custom tools and registry
hooks without editing the core package source.  Extension points are declared
in a ``plugins.json`` manifest file that lives in the user's workspace or in
the project's ``.mini-minion/`` directory.

Why a file-based manifest?
---------------------------
The alternative — dropping Python files into a fixed directory and
auto-scanning them — is convenient but has two problems:

1. Any Python file in the directory gets imported, which makes it easy to
   accidentally load untrusted code (e.g. if the user clones a repo that
   contains a ``.mini-minion/`` directory with malicious files).
2. Files have no metadata: there is no way to tell whether a file is meant
   to be a plugin, a scratch file, or a leftover from a previous experiment.

An explicit manifest puts the user in control: only files the user has
deliberately listed are loaded.

Manifest format
---------------
Create one or both of these files:

- **Global** (applies to all projects): ``~/.mini-minion/plugins.json``
- **Local** (applies only to the current project):
  ``.mini-minion/plugins.json`` (relative to the working directory)

Both files are loaded when present.  Local tools are registered *after*
global tools so a local plugin can override a global one with the same name.

Example ``plugins.json``::

    {
        "tools": [
            "~/.mini-minion/tools/my_grep_tool.py",
            "./tools/project_specific_tool.py"
        ]
    }

Each entry in ``"tools"`` is a path to a Python module file.  The module
must expose its tools in one of two ways:

1. **Explicit list** (preferred)::

       from mini_minion.tools.base import Tool, ToolSchema

       class MyTool(Tool):
           @property
           def schema(self):
               return ToolSchema(name="my_tool", ...)
           def execute(self, **kwargs):
               ...

       TOOLS = [MyTool()]   # ← explicit export list

2. **Auto-discovery** (fallback): if the module has no ``TOOLS`` attribute,
   the loader scans for concrete subclasses of :class:`Tool` (non-abstract,
   instantiable with no constructor arguments) and instantiates them.

Security
--------
Only paths explicitly listed in the manifest are imported.  No automatic
directory scanning.  If a module file does not exist, it is skipped with a
warning rather than raising an error — this handles the case where the
manifest was written on one machine and is used on another.

Talks to
--------
- ``tools/base.py`` — uses :class:`Tool` to identify tool subclasses.
- ``tools/registry.py`` — calls :meth:`ToolRegistry.register` to add tools.
- ``minion.py`` — calls :func:`load_plugins` after ``default_registry()``
  to extend the tool set with user-defined tools.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tools.registry import ToolRegistry


def load_plugins(registry: "ToolRegistry", workspace: Path) -> int:
    """Load plugins from manifest files and register their tools.

    Searches two manifest locations (in order):

    1. ``{workspace}/plugins.json`` — user-global plugins.
    2. ``.mini-minion/plugins.json`` — project-local plugins (cwd-relative).

    Project-local tools are registered after global ones so a local plugin
    with the same name as a global plugin silently overrides it.

    Args:
        registry: The :class:`ToolRegistry` to register loaded tools into.
        workspace: The user's workspace root (typically ``~/.mini-minion``).

    Returns:
        int: Total number of tools registered from all manifests.
    """
    manifest_paths = [
        workspace / "plugins.json",
        Path.cwd() / ".mini-minion" / "plugins.json",
    ]

    total = 0
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            continue
        total += _load_manifest(registry, manifest_path)
    return total


def _load_manifest(registry: "ToolRegistry", manifest_path: Path) -> int:
    """Load one manifest file and register the tools it declares.

    Args:
        registry:      The tool registry to register tools into.
        manifest_path: Absolute path to the ``plugins.json`` file.

    Returns:
        int: Number of tools registered from this manifest.
    """
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [plugins] Warning: could not read {manifest_path}: {exc}")
        return 0

    if not isinstance(raw, dict):
        print(f"  [plugins] Warning: {manifest_path} must be a JSON object.")
        return 0

    count = 0
    for tool_path_str in raw.get("tools", []):
        # Resolve relative paths against the manifest file's directory, then
        # expand ~ for home-directory paths.
        tool_path = Path(tool_path_str).expanduser()
        if not tool_path.is_absolute():
            tool_path = (manifest_path.parent / tool_path).resolve()

        tools = _load_tool_module(tool_path)
        if not tools:
            print(f"  [plugins] Warning: no tools found in {tool_path}")
            continue

        for tool in tools:
            registry.register(tool)
            count += 1
            print(f"  [plugins] Registered tool '{tool.schema.name}' from {tool_path.name}")

    return count


def _load_tool_module(module_path: Path) -> list:
    """Dynamically import a Python file and return all Tool instances from it.

    Looks for a ``TOOLS`` list attribute first (explicit export).  Falls back
    to scanning for concrete :class:`Tool` subclasses if ``TOOLS`` is absent.

    Args:
        module_path: Absolute path to the ``.py`` file to import.

    Returns:
        list: Tool instances found in the module.  Empty list on any error.
    """
    if not module_path.exists():
        print(f"  [plugins] Warning: tool module not found: {module_path}")
        return []

    try:
        spec = importlib.util.spec_from_file_location(
            # Use a unique module name to avoid collisions in sys.modules.
            f"mini_minion_plugin_{module_path.stem}",
            module_path,
        )
        if spec is None or spec.loader is None:
            return []
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as exc:
        print(f"  [plugins] Warning: could not import {module_path}: {exc}")
        return []

    # Prefer explicit TOOLS list — clear, predictable, test-friendly.
    if hasattr(module, "TOOLS"):
        return list(module.TOOLS)

    # Auto-discover: find all concrete Tool subclasses that can be instantiated
    # with no arguments.  Abstract classes (those with __abstractmethods__) are
    # skipped — they're base classes, not usable tools.
    from .tools.base import Tool

    discovered = []
    for _name, obj in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(obj, Tool)
            and obj is not Tool
            and not getattr(obj, "__abstractmethods__", None)
        ):
            try:
                discovered.append(obj())
            except TypeError:
                # Constructor requires arguments — skip and suggest explicit TOOLS.
                print(
                    f"  [plugins] Warning: {obj.__name__} requires constructor arguments. "
                    f"Add a TOOLS = [...] list to {module_path.name} to export it explicitly."
                )
    return discovered

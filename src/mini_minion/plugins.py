"""Local plugin manifest loader.

This module allows users to extend mini-minion with custom tools, registry
hooks, and skills without editing the core package source.  Extension points
are declared in a ``plugins.json`` manifest file that lives in the user's
workspace or in the project's ``.mini-minion/`` directory.

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
        "trust": "trusted",
        "tools": [
            "~/.mini-minion/tools/my_grep_tool.py",
            "./tools/project_specific_tool.py"
        ],
        "hooks": [
            "./hooks/logging_hook.py"
        ],
        "skills": [
            "./custom-skills/"
        ]
    }

``"tools"`` section
-------------------
Each entry is a path to a Python module file.  The module must expose its
tools in one of two ways:

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

``"hooks"`` section
--------------------
Each entry is a path to a Python module file.  The module may expose:

- ``BEFORE_HOOKS``: list of callables registered via
  :meth:`ToolRegistry.add_before_hook`.  Each callable receives a
  :class:`ToolPreExecuteHookEvent`.
- ``AFTER_HOOKS``: list of callables registered via
  :meth:`ToolRegistry.add_after_hook`.  Each callable receives a
  :class:`ToolPostExecuteHookEvent`.

``"skills"`` section
---------------------
Each entry is a path to a directory that contains ``SKILL.md`` files.  The
loader passes these paths to :func:`~mini_minion.skills.discover_skills` and
merges any discovered skills into the optional skill registry argument.

``"commands"`` section
----------------------
Each entry declares a new slash command that the plugin provides:

.. code-block:: json

    "commands": [
        {
            "name": "/my-command",
            "description": "Do something useful.",
            "handler": "./handlers/my_command.py"
        }
    ]

The ``handler`` path points to a Python file that must expose a function::

    def handle(ctx: CommandContext) -> CommandResult: ...

Loaded commands are registered via :func:`~mini_minion.commands.register_plugin_command`
and are available in the REPL alongside built-in commands.

``"trust"`` field
-----------------
Optional string: ``"trusted"`` (default, for your own plugins) or
``"external"`` (plugins installed from a third party).  External plugins
print a warning at load time as a reminder that untrusted code is running.
The field has no runtime enforcement effect — it is metadata for the user.

Security
--------
Only paths explicitly listed in the manifest are imported.  No automatic
directory scanning.  If a module file does not exist, it is skipped with a
warning rather than raising an error — this handles the case where the
manifest was written on one machine and is used on another.

Talks to
--------
- ``tools/base.py`` — uses :class:`Tool` to identify tool subclasses.
- ``tools/registry.py`` — calls :meth:`ToolRegistry.register` + hook methods.
- ``skills/__init__.py`` — calls :func:`discover_skills` for skill paths.
- ``commands.py`` — calls :func:`register_plugin_command` for each ``"commands"`` entry.
- ``minion.py`` — calls :func:`load_plugins` after ``default_registry()``
  to extend the tool set with user-defined tools, hooks, skills, and commands.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .skills import SkillRegistry
    from .tools.registry import ToolRegistry


def load_plugins(
    registry: "ToolRegistry",
    workspace: Path,
    skills: "SkillRegistry | None" = None,
    extra_manifests: tuple[str, ...] = (),
) -> int:
    """Load plugins from manifest files and register their tools, hooks, skills, and commands.

    Searches manifest locations in order:

    1. ``{workspace}/plugins.json`` — user-global plugins.
    2. ``.mini-minion/plugins.json`` — project-local plugins (cwd-relative).
    3. Any paths in ``extra_manifests`` (from config.json ``"extra_plugin_manifests"``).

    Later manifests are loaded after earlier ones so project-local or explicitly
    listed plugins can override global ones with the same tool name.

    Args:
        registry:        The :class:`ToolRegistry` to register loaded tools and hooks into.
        workspace:       The user's workspace root (typically ``~/.mini-minion``).
        skills:          Optional mutable :data:`SkillRegistry` dict.  When provided,
                         skills discovered from manifest ``"skills"`` paths are merged in.
        extra_manifests: Additional manifest file paths from config.json
                         ``"extra_plugin_manifests"``.  ``~`` is expanded.

    Returns:
        int: Total number of tools registered from all manifests.
    """
    manifest_paths: list[Path] = [
        workspace / "plugins.json",
        Path.cwd() / ".mini-minion" / "plugins.json",
    ]
    # Expand user-provided extra manifest paths (~ expansion, str → Path).
    for extra in extra_manifests:
        manifest_paths.append(Path(extra).expanduser())

    total = 0
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            continue
        total += _load_manifest(registry, manifest_path, skills)
    return total


def _load_manifest(
    registry: "ToolRegistry",
    manifest_path: Path,
    skills: "SkillRegistry | None" = None,
) -> int:
    """Load one manifest file and register the tools, hooks, and skills it declares.

    Args:
        registry:      The tool registry to register tools and hooks into.
        manifest_path: Absolute path to the ``plugins.json`` file.
        skills:        Optional skill registry to merge discovered skills into.

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

    # Trust metadata: warn when loading external plugins so the user is aware.
    trust = raw.get("trust", "trusted")
    if trust == "external":
        print(
            f"  [plugins] Warning: loading EXTERNAL plugin manifest {manifest_path}. "
            "Only load external plugins from sources you trust."
        )

    count = 0

    # --- tools section ---
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

    # --- hooks section ---
    for hook_path_str in raw.get("hooks", []):
        hook_path = Path(hook_path_str).expanduser()
        if not hook_path.is_absolute():
            hook_path = (manifest_path.parent / hook_path).resolve()

        _load_hook_module(registry, hook_path)

    # --- skills section ---
    if skills is not None:
        _skill_paths: list[Path] = []
        for skill_path_str in raw.get("skills", []):
            skill_path = Path(skill_path_str).expanduser()
            if not skill_path.is_absolute():
                skill_path = (manifest_path.parent / skill_path).resolve()
            _skill_paths.append(skill_path)

        if _skill_paths:
            from .skills import discover_skills
            extra = discover_skills(_skill_paths)
            merged = 0
            for name, info in extra.items():
                skills[name] = info
                merged += 1
            if merged:
                print(f"  [plugins] Loaded {merged} skill(s) from manifest skill paths.")

    # --- commands section ---
    # Each entry: {"name": "/cmd", "description": "...", "handler": "./handler.py"}
    # The handler file must expose handle(ctx: CommandContext) -> CommandResult.
    for cmd_entry in raw.get("commands", []):
        if not isinstance(cmd_entry, dict):
            print(f"  [plugins] Warning: command entry must be an object, got {type(cmd_entry).__name__}.")
            continue
        cmd_name = cmd_entry.get("name", "").strip()
        cmd_desc = cmd_entry.get("description", "")
        handler_str = cmd_entry.get("handler", "")
        if not cmd_name or not handler_str:
            print(f"  [plugins] Warning: command entry missing 'name' or 'handler': {cmd_entry!r}.")
            continue

        handler_path = Path(handler_str).expanduser()
        if not handler_path.is_absolute():
            handler_path = (manifest_path.parent / handler_path).resolve()

        _load_command_handler(cmd_name, cmd_desc, handler_path)

    return count


def _load_command_handler(name: str, description: str, module_path: Path) -> None:
    """Dynamically import a command handler and register it as a plugin command.

    The handler module must expose a ``handle(ctx) -> CommandResult`` function.
    Registration is via :func:`~mini_minion.commands.register_plugin_command`.

    Args:
        name:        The slash command name, e.g. ``"/my-command"``.
        description: Short description shown in ``/help``.
        module_path: Absolute path to the handler Python file.
    """
    if not module_path.exists():
        print(f"  [plugins] Warning: command handler not found: {module_path}")
        return

    try:
        spec = importlib.util.spec_from_file_location(
            f"mini_minion_cmd_{module_path.stem}",
            module_path,
        )
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as exc:
        print(f"  [plugins] Warning: could not import command handler {module_path}: {exc}")
        return

    handler = getattr(module, "handle", None)
    if handler is None or not callable(handler):
        print(f"  [plugins] Warning: {module_path.name} has no callable 'handle' function.")
        return

    from .commands import CommandSpec, register_plugin_command
    spec_obj = CommandSpec(name=name, description=description)
    register_plugin_command(spec_obj, handler)
    print(f"  [plugins] Registered plugin command '{name}' from {module_path.name}")


def _load_hook_module(registry: "ToolRegistry", module_path: Path) -> None:
    """Dynamically import a Python file and register its hooks into the registry.

    Looks for ``BEFORE_HOOKS`` and ``AFTER_HOOKS`` list attributes in the module.
    Each callable in those lists is registered via :meth:`ToolRegistry.add_before_hook`
    and :meth:`ToolRegistry.add_after_hook` respectively.

    Args:
        registry:    The tool registry to add hooks to.
        module_path: Absolute path to the ``.py`` hook module file.
    """
    if not module_path.exists():
        print(f"  [plugins] Warning: hook module not found: {module_path}")
        return

    try:
        spec = importlib.util.spec_from_file_location(
            f"mini_minion_hook_{module_path.stem}",
            module_path,
        )
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as exc:
        print(f"  [plugins] Warning: could not import hook module {module_path}: {exc}")
        return

    before_count = 0
    after_count = 0

    for hook in getattr(module, "BEFORE_HOOKS", []):
        registry.add_before_hook(hook)
        before_count += 1

    for hook in getattr(module, "AFTER_HOOKS", []):
        registry.add_after_hook(hook)
        after_count += 1

    if before_count or after_count:
        print(
            f"  [plugins] Registered {before_count} before-hook(s) and "
            f"{after_count} after-hook(s) from {module_path.name}"
        )
    else:
        print(f"  [plugins] Warning: no hooks (BEFORE_HOOKS/AFTER_HOOKS) found in {module_path}")


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

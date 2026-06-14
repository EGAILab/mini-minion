"""Tests for the plugin manifest loader (plugins.py)."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from minion_assistant.plugins import load_plugins, _load_tool_module
from minion_assistant.tools.base import Tool, ToolSchema
from minion_assistant.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry() -> ToolRegistry:
    return ToolRegistry()


def _write_manifest(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content), encoding="utf-8")


def _write_tool_module(path: Path, tool_name: str = "plugin_test") -> None:
    """Write a minimal tool module that exports a TOOLS list."""
    path.write_text(
        f"""
from minion_assistant.tools.base import Tool, ToolSchema

class _PluginTool(Tool):
    @property
    def schema(self):
        return ToolSchema(
            name="{tool_name}",
            description="A test plugin tool.",
            parameters={{"type": "object", "properties": {{}}}},
        )
    def execute(self, **kwargs):
        return "plugin result"

TOOLS = [_PluginTool()]
""",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# load_plugins
# ---------------------------------------------------------------------------

def test_load_plugins_no_manifest_returns_zero(tmp_path):
    registry = _make_registry()
    count = load_plugins(registry, workspace=tmp_path)
    assert count == 0


def test_load_plugins_loads_tool_from_global_manifest(tmp_path):
    tool_file = tmp_path / "my_tool.py"
    _write_tool_module(tool_file, "my_plugin")
    _write_manifest(tmp_path / "plugins.json", {"tools": [str(tool_file)]})

    registry = _make_registry()
    count = load_plugins(registry, workspace=tmp_path)
    assert count == 1
    # Tool should be callable via the registry.
    assert "my_plugin" in [t.schema.name for t in registry._tools.values()]


def test_load_plugins_returns_total_count(tmp_path):
    tool_file1 = tmp_path / "tool1.py"
    tool_file2 = tmp_path / "tool2.py"
    _write_tool_module(tool_file1, "plugin_a")
    _write_tool_module(tool_file2, "plugin_b")
    _write_manifest(tmp_path / "plugins.json", {"tools": [str(tool_file1), str(tool_file2)]})

    registry = _make_registry()
    count = load_plugins(registry, workspace=tmp_path)
    assert count == 2


def test_load_plugins_project_local_overrides_global(tmp_path, monkeypatch):
    # Global manifest declares plugin_x
    global_tool = tmp_path / "global_tool.py"
    _write_tool_module(global_tool, "shared_name")
    _write_manifest(tmp_path / "plugins.json", {"tools": [str(global_tool)]})

    # Local manifest declares a different tool with the same name
    local_dir = tmp_path / ".minion-assistant"
    local_dir.mkdir()
    local_tool = local_dir / "local_tool.py"
    local_tool.write_text(
        """
from minion_assistant.tools.base import Tool, ToolSchema

class _LocalTool(Tool):
    @property
    def schema(self):
        return ToolSchema(
            name="shared_name",
            description="Local override",
            parameters={"type": "object", "properties": {}},
        )
    def execute(self, **kwargs):
        return "local result"

TOOLS = [_LocalTool()]
""",
        encoding="utf-8",
    )
    _write_manifest(local_dir / "plugins.json", {"tools": [str(local_tool)]})

    # Monkeypatch cwd to tmp_path so the local manifest is found.
    monkeypatch.chdir(tmp_path)

    registry = _make_registry()
    load_plugins(registry, workspace=tmp_path)

    # The registry should have exactly one tool named "shared_name" — the local one.
    tool = registry._tools["shared_name"]
    assert tool.execute() == "local result"


def test_load_plugins_skips_missing_module(tmp_path, capsys):
    _write_manifest(
        tmp_path / "plugins.json",
        {"tools": [str(tmp_path / "nonexistent.py")]},
    )
    registry = _make_registry()
    count = load_plugins(registry, workspace=tmp_path)
    assert count == 0
    out = capsys.readouterr().out
    assert "Warning" in out


def test_load_plugins_handles_corrupt_manifest(tmp_path):
    (tmp_path / "plugins.json").write_text("{bad json", encoding="utf-8")
    registry = _make_registry()
    count = load_plugins(registry, workspace=tmp_path)
    assert count == 0


# ---------------------------------------------------------------------------
# _load_tool_module
# ---------------------------------------------------------------------------

def test_load_tool_module_with_explicit_tools_list(tmp_path):
    tool_file = tmp_path / "tool.py"
    _write_tool_module(tool_file, "explicit_tool")
    tools = _load_tool_module(tool_file)
    assert len(tools) == 1
    assert tools[0].schema.name == "explicit_tool"


def test_load_tool_module_autodiscovery_without_tools_list(tmp_path):
    """Modules without a TOOLS attribute should have their Tool subclasses auto-discovered."""
    tool_file = tmp_path / "auto.py"
    tool_file.write_text(
        """
from minion_assistant.tools.base import Tool, ToolSchema

class AutoTool(Tool):
    @property
    def schema(self):
        return ToolSchema(
            name="auto_discovered",
            description="Auto-discovered tool",
            parameters={"type": "object", "properties": {}},
        )
    def execute(self, **kwargs):
        return "auto"
""",
        encoding="utf-8",
    )
    tools = _load_tool_module(tool_file)
    assert len(tools) == 1
    assert tools[0].schema.name == "auto_discovered"


def test_load_tool_module_nonexistent_returns_empty(tmp_path, capsys):
    tools = _load_tool_module(tmp_path / "ghost.py")
    assert tools == []
    out = capsys.readouterr().out
    assert "Warning" in out

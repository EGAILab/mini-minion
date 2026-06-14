"""Tests for plugin 'commands' section parsing and registration.

Plugin manifests can include a 'commands' array where each entry maps a
slash-command name to a Python handler file that exposes handle(ctx) -> CommandResult.
These tests verify that _load_command_handler and the commands section of
_load_manifest work correctly.
"""

import json
import pytest
from pathlib import Path

import minion_assistant.commands as commands_module
from minion_assistant.commands import register_plugin_command, CommandSpec


# ---------------------------------------------------------------------------
# Setup / teardown helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_plugin_registry():
    """Clear the plugin command registry before and after each test.

    _PLUGIN_COMMAND_REGISTRY is module-level state, so tests can pollute each
    other without this fixture.
    """
    commands_module._PLUGIN_COMMAND_REGISTRY.clear()
    yield
    commands_module._PLUGIN_COMMAND_REGISTRY.clear()


def _write_handler(path: Path, return_msg: str = "handler ran") -> None:
    """Write a minimal valid command handler Python file to path."""
    path.write_text(
        "from minion_assistant.commands import CommandResult\n"
        f"def handle(ctx):\n"
        f"    return CommandResult(handled=True, message={return_msg!r})\n",
        encoding="utf-8",
    )


def _write_manifest(manifest_path: Path, entries: list[dict]) -> None:
    """Write a plugins.json with a 'commands' section."""
    manifest_path.write_text(
        json.dumps({"trust": "trusted", "commands": entries}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# register_plugin_command
# ---------------------------------------------------------------------------

def test_register_plugin_command_adds_to_registry():
    """register_plugin_command must add the (spec, handler) pair to the registry."""
    spec = CommandSpec(name="/my-cmd", description="does something")
    register_plugin_command(spec, lambda ctx: None)
    assert any(s.name == "/my-cmd" for s, _ in commands_module._PLUGIN_COMMAND_REGISTRY)


def test_register_plugin_command_deduplicates_by_name():
    """Registering the same name twice should keep only the latest entry."""
    spec = CommandSpec(name="/dup", description="first")
    spec2 = CommandSpec(name="/dup", description="second")
    handler1 = lambda ctx: "first"
    handler2 = lambda ctx: "second"
    register_plugin_command(spec, handler1)
    register_plugin_command(spec2, handler2)
    matches = [h for s, h in commands_module._PLUGIN_COMMAND_REGISTRY if s.name == "/dup"]
    assert len(matches) == 1
    # Latest registration wins.
    assert matches[0] is handler2


# ---------------------------------------------------------------------------
# _load_command_handler
# ---------------------------------------------------------------------------

def test_load_command_handler_registers_command(tmp_path):
    """_load_command_handler must register a valid handler file as a plugin command."""
    from minion_assistant.plugins import _load_command_handler
    handler_path = tmp_path / "my_cmd.py"
    _write_handler(handler_path)
    _load_command_handler("/my-cmd", "A test command", handler_path)
    assert any(s.name == "/my-cmd" for s, _ in commands_module._PLUGIN_COMMAND_REGISTRY)


def test_load_command_handler_missing_file_warns(tmp_path, capsys):
    """When the handler file doesn't exist, a warning is printed and nothing is registered."""
    from minion_assistant.plugins import _load_command_handler
    _load_command_handler("/missing", "Missing handler", tmp_path / "nope.py")
    out = capsys.readouterr().out
    assert "Warning" in out or "not found" in out.lower()
    assert not commands_module._PLUGIN_COMMAND_REGISTRY


def test_load_command_handler_no_handle_function_warns(tmp_path, capsys):
    """When the handler file has no 'handle' function, a warning is printed."""
    from minion_assistant.plugins import _load_command_handler
    bad_handler = tmp_path / "bad.py"
    bad_handler.write_text("x = 1\n", encoding="utf-8")
    _load_command_handler("/bad", "Bad handler", bad_handler)
    out = capsys.readouterr().out
    assert "handle" in out.lower() or "Warning" in out
    assert not commands_module._PLUGIN_COMMAND_REGISTRY


# ---------------------------------------------------------------------------
# _load_manifest commands section
# ---------------------------------------------------------------------------

def test_load_manifest_commands_section(tmp_path):
    """_load_manifest must register commands declared in the 'commands' section."""
    from minion_assistant.plugins import _load_manifest
    from minion_assistant.tools.registry import ToolRegistry

    handler_path = tmp_path / "cmd.py"
    _write_handler(handler_path, "plugin command ran")

    manifest_path = tmp_path / "plugins.json"
    _write_manifest(manifest_path, [
        {"name": "/my-plugin-cmd", "description": "Plugin test", "handler": str(handler_path)},
    ])

    registry = ToolRegistry()
    _load_manifest(registry, manifest_path)

    assert any(s.name == "/my-plugin-cmd" for s, _ in commands_module._PLUGIN_COMMAND_REGISTRY)


def test_load_manifest_commands_missing_name_warns(tmp_path, capsys):
    """Command entries without 'name' are skipped with a warning."""
    from minion_assistant.plugins import _load_manifest
    from minion_assistant.tools.registry import ToolRegistry

    handler_path = tmp_path / "cmd.py"
    _write_handler(handler_path)
    manifest_path = tmp_path / "plugins.json"
    _write_manifest(manifest_path, [
        {"description": "No name", "handler": str(handler_path)},
    ])

    registry = ToolRegistry()
    _load_manifest(registry, manifest_path)
    out = capsys.readouterr().out
    assert "Warning" in out or "missing" in out.lower()
    assert not commands_module._PLUGIN_COMMAND_REGISTRY


def test_load_manifest_commands_missing_handler_warns(tmp_path, capsys):
    """Command entries without 'handler' are skipped with a warning."""
    from minion_assistant.plugins import _load_manifest
    from minion_assistant.tools.registry import ToolRegistry

    manifest_path = tmp_path / "plugins.json"
    _write_manifest(manifest_path, [
        {"name": "/no-handler", "description": "No handler"},
    ])

    registry = ToolRegistry()
    _load_manifest(registry, manifest_path)
    out = capsys.readouterr().out
    assert "Warning" in out or "missing" in out.lower()
    assert not commands_module._PLUGIN_COMMAND_REGISTRY

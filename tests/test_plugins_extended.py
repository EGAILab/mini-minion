"""Tests for expanded plugin manifest: hooks, skills, and trust."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from minion_assistant.plugins import load_plugins, _load_hook_module
from minion_assistant.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry() -> ToolRegistry:
    return ToolRegistry()


def _write_manifest(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content), encoding="utf-8")


def _write_hook_module(path: Path, before: bool = True, after: bool = True) -> None:
    """Write a minimal hook module exposing BEFORE_HOOKS and/or AFTER_HOOKS."""
    lines = []
    if before:
        lines.append("def _before_hook(event): pass")
        lines.append("BEFORE_HOOKS = [_before_hook]")
    if after:
        lines.append("def _after_hook(event): pass")
        lines.append("AFTER_HOOKS = [_after_hook]")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Hooks section
# ---------------------------------------------------------------------------

def test_load_hook_module_registers_before_hooks(tmp_path):
    hook_path = tmp_path / "hook.py"
    _write_hook_module(hook_path, before=True, after=False)
    registry = _make_registry()
    _load_hook_module(registry, hook_path)
    assert len(registry._before_hooks) == 1


def test_load_hook_module_registers_after_hooks(tmp_path):
    hook_path = tmp_path / "hook.py"
    _write_hook_module(hook_path, before=False, after=True)
    registry = _make_registry()
    _load_hook_module(registry, hook_path)
    assert len(registry._after_hooks) == 1


def test_load_hook_module_registers_both(tmp_path):
    hook_path = tmp_path / "hook.py"
    _write_hook_module(hook_path, before=True, after=True)
    registry = _make_registry()
    _load_hook_module(registry, hook_path)
    assert len(registry._before_hooks) == 1
    assert len(registry._after_hooks) == 1


def test_manifest_hooks_section_registers_hooks(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    hook_path = tmp_path / "hook.py"
    _write_hook_module(hook_path, before=True, after=False)

    manifest_path = workspace / "plugins.json"
    _write_manifest(manifest_path, {"hooks": [str(hook_path)]})

    registry = _make_registry()
    load_plugins(registry, workspace)
    assert len(registry._before_hooks) == 1


def test_manifest_hooks_missing_file_prints_warning(tmp_path, capsys):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest_path = workspace / "plugins.json"
    _write_manifest(manifest_path, {"hooks": ["/nonexistent/hook.py"]})

    registry = _make_registry()
    load_plugins(registry, workspace)
    captured = capsys.readouterr()
    assert "warning" in captured.out.lower() or "not found" in captured.out.lower()


def test_load_hook_module_missing_file_prints_warning(tmp_path, capsys):
    registry = _make_registry()
    _load_hook_module(registry, tmp_path / "missing.py")
    captured = capsys.readouterr()
    assert "not found" in captured.out.lower() or "warning" in captured.out.lower()


def test_load_hook_module_no_hooks_prints_warning(tmp_path, capsys):
    hook_path = tmp_path / "empty_hook.py"
    hook_path.write_text("# no hooks here\n", encoding="utf-8")
    registry = _make_registry()
    _load_hook_module(registry, hook_path)
    captured = capsys.readouterr()
    assert "warning" in captured.out.lower() or "no hooks" in captured.out.lower()


# ---------------------------------------------------------------------------
# Skills section
# ---------------------------------------------------------------------------

def test_manifest_skills_section_adds_to_registry(tmp_path):
    """Skills discovered from manifest skill paths are merged into the dict."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create a minimal SKILL.md file in a skill directory.
    skill_dir = tmp_path / "custom-skills" / "my-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: A custom skill.\n---\nDo something.\n",
        encoding="utf-8",
    )

    manifest_path = workspace / "plugins.json"
    _write_manifest(manifest_path, {"skills": [str(skill_dir.parent)]})

    registry = _make_registry()
    skills: dict = {}
    load_plugins(registry, workspace, skills=skills)
    assert "my-skill" in skills


def test_manifest_skills_section_ignored_when_skills_arg_is_none(tmp_path):
    """When skills=None, the skills section is silently ignored (no crash)."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest_path = workspace / "plugins.json"
    _write_manifest(manifest_path, {"skills": ["/some/path"]})

    registry = _make_registry()
    # Should not raise.
    load_plugins(registry, workspace, skills=None)


# ---------------------------------------------------------------------------
# Trust field
# ---------------------------------------------------------------------------

def test_external_trust_prints_warning(tmp_path, capsys):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest_path = workspace / "plugins.json"
    _write_manifest(manifest_path, {"trust": "external"})

    registry = _make_registry()
    load_plugins(registry, workspace)
    captured = capsys.readouterr()
    assert "external" in captured.out.lower() or "warning" in captured.out.lower()


def test_trusted_trust_no_extra_warning(tmp_path, capsys):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest_path = workspace / "plugins.json"
    _write_manifest(manifest_path, {"trust": "trusted"})

    registry = _make_registry()
    load_plugins(registry, workspace)
    captured = capsys.readouterr()
    # Should not print a trust warning for trusted plugins.
    assert "external" not in captured.out.lower()


def test_missing_trust_field_defaults_to_trusted(tmp_path, capsys):
    """Omitting 'trust' should behave the same as 'trusted'."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest_path = workspace / "plugins.json"
    _write_manifest(manifest_path, {})

    registry = _make_registry()
    load_plugins(registry, workspace)
    captured = capsys.readouterr()
    assert "external" not in captured.out.lower()

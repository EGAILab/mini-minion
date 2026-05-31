"""Tests for SkillTool."""

from pathlib import Path

import pytest

from mini_minion.skills import SkillInfo
from mini_minion.tools.skill import SkillTool


def _registry(*skills: SkillInfo) -> dict:
    return {s.name: s for s in skills}


def _skill(tmp_path, name="my-skill", description="Does X.", content="# Instructions\nDo this."):
    skill_dir = tmp_path / name
    skill_dir.mkdir(exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(f"---\nname: {name}\n---\n\n{content}", encoding="utf-8")
    return SkillInfo(name=name, description=description, path=skill_file, content=content)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_skill_tool_schema_name():
    tool = SkillTool({})
    assert tool.schema.name == "skill"


def test_skill_tool_schema_lists_available_names(tmp_path):
    reg = _registry(_skill(tmp_path, "alpha"), _skill(tmp_path, "beta"))
    tool = SkillTool(reg)
    desc = tool.schema.description
    assert "alpha" in desc
    assert "beta" in desc


def test_skill_tool_schema_shows_none_when_empty():
    tool = SkillTool({})
    assert "none" in tool.schema.description.lower()


def test_skill_tool_schema_requires_name_parameter():
    tool = SkillTool({})
    assert "name" in tool.schema.parameters["required"]


# ---------------------------------------------------------------------------
# execute — happy path
# ---------------------------------------------------------------------------

def test_skill_tool_returns_skill_content(tmp_path):
    skill = _skill(tmp_path, content="Do exactly this.")
    tool = SkillTool(_registry(skill))
    result = tool.execute(name="my-skill")
    assert "Do exactly this." in result


def test_skill_tool_wraps_content_in_skill_tag(tmp_path):
    skill = _skill(tmp_path)
    tool = SkillTool(_registry(skill))
    result = tool.execute(name="my-skill")
    assert result.startswith('<skill name="my-skill">')
    assert result.endswith("</skill>")


def test_skill_tool_lists_companion_files(tmp_path):
    skill = _skill(tmp_path)
    # Add companion files alongside SKILL.md
    (tmp_path / "my-skill" / "EXAMPLES.md").write_text("examples", encoding="utf-8")
    (tmp_path / "my-skill" / "template.py").write_text("code", encoding="utf-8")

    tool = SkillTool(_registry(skill))
    result = tool.execute(name="my-skill")

    assert "EXAMPLES.md" in result
    assert "template.py" in result


def test_skill_tool_does_not_list_skill_md_itself(tmp_path):
    skill = _skill(tmp_path)
    tool = SkillTool(_registry(skill))
    result = tool.execute(name="my-skill")
    # SKILL.md should not appear in the files listing
    lines_with_skill_md = [
        line for line in result.splitlines()
        if "SKILL.md" in line and line.strip().startswith("-")
    ]
    assert lines_with_skill_md == []


def test_skill_tool_does_not_list_hidden_files(tmp_path):
    skill = _skill(tmp_path)
    (tmp_path / "my-skill" / ".hidden").write_text("hidden", encoding="utf-8")
    tool = SkillTool(_registry(skill))
    result = tool.execute(name="my-skill")
    assert ".hidden" not in result


def test_skill_tool_no_companion_section_when_no_companions(tmp_path):
    skill = _skill(tmp_path)
    tool = SkillTool(_registry(skill))
    result = tool.execute(name="my-skill")
    # No "Files available:" section when there are no companion files
    assert "Files available" not in result


# ---------------------------------------------------------------------------
# execute — error handling
# ---------------------------------------------------------------------------

def test_skill_tool_unknown_name_returns_error(tmp_path):
    skill = _skill(tmp_path, name="known")
    tool = SkillTool(_registry(skill))
    result = tool.execute(name="unknown")
    assert "unknown" in result.lower() or "Unknown" in result
    assert "known" in result  # lists available skills


def test_skill_tool_empty_registry_returns_error():
    tool = SkillTool({})
    result = tool.execute(name="anything")
    assert "unknown" in result.lower() or "Unknown" in result


def test_skill_tool_caps_companion_file_listing(tmp_path):
    """When there are more than 10 companion files, only 10 are listed."""
    skill = _skill(tmp_path)
    skill_dir = tmp_path / "my-skill"
    for i in range(15):
        (skill_dir / f"file{i:02d}.txt").write_text("x", encoding="utf-8")

    tool = SkillTool(_registry(skill))
    result = tool.execute(name="my-skill")

    listed = [line for line in result.splitlines() if line.strip().startswith("- file")]
    assert len(listed) == 10
    assert "more" in result  # overflow notice

"""Tests for skill discovery and the format_skills_prompt helper."""

import pytest

from minion_assist.skills import SkillInfo, discover_skills, format_skills_prompt


def _write_skill(base, name, description="Does something useful.", body="# Instructions"):
    """Write a minimal SKILL.md under base/<name>/SKILL.md."""
    skill_dir = base / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = f"---\nname: {name}\ndescription: {description}\n---\n\n{body}"
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


# ---------------------------------------------------------------------------
# discover_skills — basic loading
# ---------------------------------------------------------------------------

def test_discover_skills_returns_empty_for_nonexistent_path(tmp_path):
    result = discover_skills([tmp_path / "no-such-dir"])
    assert result == {}


def test_discover_skills_returns_empty_for_empty_dir(tmp_path):
    result = discover_skills([tmp_path])
    assert result == {}


def test_discover_skills_loads_valid_skill(tmp_path):
    _write_skill(tmp_path, "my-skill", "Does X when Y.", "## Body")
    result = discover_skills([tmp_path])
    assert "my-skill" in result
    skill = result["my-skill"]
    assert skill.name == "my-skill"
    assert skill.description == "Does X when Y."
    assert "## Body" in skill.content


def test_discover_skills_content_excludes_frontmatter(tmp_path):
    """The loaded content must not contain the frontmatter block."""
    _write_skill(tmp_path, "clean", "Desc.", "Just the body.")
    skill = discover_skills([tmp_path])["clean"]
    assert "---" not in skill.content
    assert "name:" not in skill.content
    assert "Just the body." in skill.content


def test_discover_skills_path_points_to_skill_md(tmp_path):
    """SkillInfo.path must be the path to SKILL.md."""
    _write_skill(tmp_path, "located")
    skill = discover_skills([tmp_path])["located"]
    assert skill.path.name == "SKILL.md"
    assert skill.path.exists()


def test_discover_skills_skips_file_missing_name(tmp_path):
    """A SKILL.md without a name: key must be silently skipped."""
    skill_dir = tmp_path / "unnamed"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\ndescription: No name here.\n---\n\nBody.", encoding="utf-8")
    result = discover_skills([tmp_path])
    assert result == {}


def test_discover_skills_loads_skill_without_description(tmp_path):
    """A skill missing description: is loaded but has an empty description string."""
    skill_dir = tmp_path / "no-desc"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: no-desc\n---\n\nBody.", encoding="utf-8")
    result = discover_skills([tmp_path])
    assert "no-desc" in result
    assert result["no-desc"].description == ""


def test_discover_skills_skips_malformed_frontmatter(tmp_path):
    """A SKILL.md that cannot be parsed at all must be silently skipped."""
    skill_dir = tmp_path / "bad"
    skill_dir.mkdir()
    # Write a file that isn't valid frontmatter at all — but python-frontmatter
    # is lenient, so write content that IS clearly un-parseable.
    (skill_dir / "SKILL.md").write_text("", encoding="utf-8")
    # An empty file has no name, so it should be skipped.
    result = discover_skills([tmp_path])
    assert result == {}


def test_discover_skills_multiple_skills(tmp_path):
    """All valid skills in a directory are loaded."""
    _write_skill(tmp_path, "alpha")
    _write_skill(tmp_path, "beta")
    _write_skill(tmp_path, "gamma")
    result = discover_skills([tmp_path])
    assert set(result) == {"alpha", "beta", "gamma"}


# ---------------------------------------------------------------------------
# discover_skills — name collision / override order
# ---------------------------------------------------------------------------

def test_discover_skills_project_overrides_global(tmp_path):
    """Later path entries override earlier ones — project (second) beats global (first)."""
    global_dir = tmp_path / "global"
    project_dir = tmp_path / "project"
    _write_skill(global_dir, "shared", "Global description.")
    _write_skill(project_dir, "shared", "Project description.")

    # global first, project second → project wins
    result = discover_skills([global_dir, project_dir])
    assert result["shared"].description == "Project description."


def test_discover_skills_global_used_when_no_project_override(tmp_path):
    """Skills only in global dir are still available when no project override exists."""
    global_dir = tmp_path / "global"
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_skill(global_dir, "global-only")

    result = discover_skills([global_dir, project_dir])
    assert "global-only" in result


def test_discover_skills_merged_from_both_paths(tmp_path):
    """Skills from both global and project dirs are merged into one registry."""
    global_dir = tmp_path / "global"
    project_dir = tmp_path / "project"
    _write_skill(global_dir, "from-global")
    _write_skill(project_dir, "from-project")

    result = discover_skills([global_dir, project_dir])
    assert "from-global" in result
    assert "from-project" in result


# ---------------------------------------------------------------------------
# format_skills_prompt
# ---------------------------------------------------------------------------

def test_format_skills_prompt_empty_registry_returns_empty_string():
    assert format_skills_prompt({}) == ""


def test_format_skills_prompt_skips_skills_without_description(tmp_path):
    """Skills without a description must not appear in the system prompt block."""
    registry = {
        "no-desc": SkillInfo(name="no-desc", description="", path=tmp_path / "SKILL.md", content=""),
    }
    assert format_skills_prompt(registry) == ""


def test_format_skills_prompt_contains_skill_name_and_description(tmp_path):
    registry = {
        "my-skill": SkillInfo(
            name="my-skill",
            description="Does X when Y.",
            path=tmp_path / "SKILL.md",
            content="",
        ),
    }
    result = format_skills_prompt(registry)
    assert "my-skill" in result
    assert "Does X when Y." in result


def test_format_skills_prompt_wraps_in_available_skills_tag(tmp_path):
    registry = {
        "s": SkillInfo(name="s", description="Desc.", path=tmp_path / "SKILL.md", content=""),
    }
    result = format_skills_prompt(registry)
    assert result.startswith("<available_skills>")
    assert result.endswith("</available_skills>")


def test_format_skills_prompt_multiple_skills_sorted(tmp_path):
    """Skills appear in alphabetical order."""
    registry = {
        "zzz": SkillInfo(name="zzz", description="Last.", path=tmp_path / "SKILL.md", content=""),
        "aaa": SkillInfo(name="aaa", description="First.", path=tmp_path / "SKILL.md", content=""),
    }
    result = format_skills_prompt(registry)
    assert result.index("aaa") < result.index("zzz")

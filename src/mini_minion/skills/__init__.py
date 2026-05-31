"""Skill discovery and loading — domain-specific instruction modules.

A *skill* is a ``SKILL.md`` file with YAML frontmatter (``name`` and
``description``) followed by a markdown body of domain instructions.  Skills
teach the agent how to handle a specific class of task without having to embed
that knowledge in the static system prompt.

How skills work at runtime
--------------------------
1. At startup, :func:`discover_skills` scans two locations and builds a
   :data:`SkillRegistry` (a dict keyed by skill name).
2. The registry is formatted into an ``<available_skills>`` block and appended
   to each agent's system prompt so the model knows which skills exist.
3. When the user's task matches a skill's description, the model calls the
   ``skill`` tool (see ``tools/skill.py``) with the skill name.
4. The tool returns the full markdown body plus a listing of companion files in
   the skill's directory.  The model reads this content and uses it to guide
   its next actions.

Discovery order (following OpenCode)
-------------------------------------
Global skills (``~/.mini-minion/skills/``) are scanned first, then project
skills (``<cwd>/.mini-minion/skills/``).  When the same name appears in both
locations, the **project-level skill wins** (later entry overwrites earlier).
This mirrors OpenCode's convention: users can override global skills per-project.

Skill file format
-----------------
Each skill lives in its own directory::

    .mini-minion/skills/
      my-skill/
        SKILL.md          ← required: frontmatter + body
        EXAMPLES.md       ← optional companion files the model can read
        template.py       ← ...

``SKILL.md`` format::

    ---
    name: my-skill
    description: What this skill does and when to use it.
    ---

    # My Skill

    Full markdown instructions here…

Skills without a ``name:`` key are silently skipped.  Skills without a
``description:`` are loaded but will not appear in the system prompt listing
(the model won't know to call them).

Talks to
--------
- ``tools/skill.py`` — :class:`SkillTool` receives a :data:`SkillRegistry`.
- ``minion.py`` — calls :func:`discover_skills` at startup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import frontmatter

logger = logging.getLogger(__name__)


@dataclass
class SkillInfo:
    """All information about one loaded skill.

    Attributes:
        name (str): The skill's identifier, matching the ``name:`` frontmatter
            key.  Used as the lookup key in :data:`SkillRegistry`.
        description (str): One-sentence summary of what the skill does and when
            to use it.  Shown in the ``<available_skills>`` system prompt block.
            Empty string if the frontmatter omits ``description:``.
        path (Path): Absolute path to the ``SKILL.md`` file.  Used by
            :class:`SkillTool` to locate companion files in the same directory.
        content (str): The markdown body of the skill file with frontmatter
            stripped.  This is what the model receives when it invokes the skill.
    """
    name: str
    description: str
    path: Path
    content: str


# A dict keyed by skill name (e.g. "my-skill") mapping to loaded skill data.
SkillRegistry = dict[str, SkillInfo]


def discover_skills(paths: list[Path]) -> SkillRegistry:
    """Scan directories for ``SKILL.md`` files and return a populated registry.

    Searches each path in order using :meth:`~pathlib.Path.rglob`.  When the
    same skill ``name`` appears in multiple locations, **later entries win** —
    so pass global paths first and project paths last to achieve project-level
    override.

    Args:
        paths: Ordered list of base directories to scan.  Non-existent paths
            are silently ignored (skills directory may not exist yet).

    Returns:
        SkillRegistry: A dict of ``{name: SkillInfo}``.  Empty if no valid
        skills are found.
    """
    registry: SkillRegistry = {}
    for base in paths:
        if not base.is_dir():
            continue
        for skill_file in sorted(base.rglob("SKILL.md")):
            try:
                post = frontmatter.load(str(skill_file))
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", skill_file, exc)
                continue

            name = str(post.get("name", "") or "").strip()
            if not name:
                logger.warning("Skipping %s: missing 'name' in frontmatter", skill_file)
                continue

            description = str(post.get("description", "") or "").strip()
            content = (post.content or "").strip()

            if name in registry:
                logger.debug("Skill '%s' from %s overrides earlier definition", name, skill_file)

            registry[name] = SkillInfo(
                name=name,
                description=description,
                path=skill_file,
                content=content,
            )

    return registry


def format_skills_prompt(registry: SkillRegistry) -> str:
    """Format the skill registry as an ``<available_skills>`` system prompt block.

    Returns an empty string when the registry is empty so callers can
    do ``if suffix: soul += "\\n\\n" + suffix`` without special-casing.

    Args:
        registry: The registry returned by :func:`discover_skills`.

    Returns:
        str: XML-style block listing each skill name and description, or ``""``
        if the registry is empty or all skills lack descriptions.
    """
    # Only include skills with a description — ones without it are unusable
    # because the model has no way to know when to invoke them.
    describable = [s for s in sorted(registry.values(), key=lambda s: s.name) if s.description]
    if not describable:
        return ""

    lines = [
        "<available_skills>",
        "Use the skill tool to load one when the user task matches its description.",
    ]
    for skill in describable:
        lines.append(f'<skill name="{skill.name}">{skill.description}</skill>')
    lines.append("</available_skills>")
    return "\n".join(lines)

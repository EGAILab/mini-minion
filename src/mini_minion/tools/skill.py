"""SkillTool — loads a skill's instructions into the conversation.

When the model identifies that a user task matches a skill's description, it
calls this tool with the skill name.  The tool returns the skill's full
markdown body plus a listing of companion files in the skill's directory so the
model can ``read`` them if needed.

Output format (mirrors OpenCode's skill tool)::

    <skill name="my-skill">
    [full markdown body]

    Skill directory: /path/to/.mini-minion/skills/my-skill/
    Files available:
      - EXAMPLES.md
      - template.py
    </skill>

The model receives this content as a tool observation and uses it to guide its
next actions — the skill does not change the system prompt, available tools, or
the model being used.

Talks to
--------
- ``skills/__init__.py`` — receives a :data:`SkillRegistry`.
- ``tools/__init__.py`` — registered via ``default_registry()`` when the
  registry is non-empty.
- ``tools/base.py`` — extends :class:`Tool`, returns :class:`ToolSchema`.
"""

from __future__ import annotations

from pathlib import Path

from ..skills import SkillRegistry
from .base import Tool, ToolSchema

# Max companion files to list so the output doesn't overflow context.
_MAX_COMPANION_FILES = 10


class SkillTool(Tool):
    """Tool that loads a skill's instruction content on demand.

    The available skill names are embedded in the tool's description so the
    model can verify a name before calling — matching OpenCode's approach of
    making the listing discoverable without a separate list-skills call.

    Args:
        registry: The populated :data:`SkillRegistry` from :func:`discover_skills`.
    """

    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    @property
    def schema(self) -> ToolSchema:
        """Describe this tool to the LLM, listing available skill names."""
        names = ", ".join(sorted(self._registry)) if self._registry else "none loaded"
        return ToolSchema(
            name="skill",
            description=(
                f"Load a skill's domain instructions into the conversation. "
                f"Call this when the user's task matches a skill's description. "
                f"Available skills: {names}."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The exact skill name to load (case-sensitive).",
                    },
                },
                "required": ["name"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        """Load and return the named skill's content.

        Args:
            name (str): The skill name to load.

        Returns:
            str: XML-wrapped skill content with companion file listing, or an
                error message if the name is not found.
        """
        name = str(kwargs["name"]).strip()
        skill = self._registry.get(name)

        if skill is None:
            available = sorted(self._registry)
            return f"Unknown skill '{name}'. Available: {available or ['none']}"

        skill_dir: Path = skill.path.parent

        # List companion files (everything except SKILL.md and hidden files).
        companions = sorted(
            p for p in skill_dir.iterdir()
            if p.name != "SKILL.md" and not p.name.startswith(".")
        )

        output = f'<skill name="{name}">\n{skill.content}\n'
        if companions:
            output += f"\nSkill directory: {skill_dir}\nFiles available:\n"
            for p in companions[:_MAX_COMPANION_FILES]:
                output += f"  - {p.name}\n"
            if len(companions) > _MAX_COMPANION_FILES:
                output += f"  ... and {len(companions) - _MAX_COMPANION_FILES} more\n"
        output += "</skill>"
        return output

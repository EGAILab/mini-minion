"""Agent personalities — who the agents are and what they care about.

This module defines the static "soul" of each agent: their name, system prompt
(``soul``), and behavioural configuration such as ``max_tool_rounds``.

Concepts explained
------------------
- **System prompt**: Instructions sent to the LLM before the user's messages.
  The model treats these as high-priority rules for how to behave.
- **Agent**: In this project, an agent is an LLM instance with a specific
  identity (name + soul) and its own conversation history.  Having multiple
  agents lets the user route different kinds of questions to specialists.
- **max_tool_rounds**: The maximum number of LLM→tool-call iterations allowed
  per user turn.  Conversational agents need fewer rounds; task-focused agents
  that work through multi-step problems need more.

What this module exposes
------------------------
- :class:`AgentConfig` — a data holder for an agent's identity and config.
- :data:`AGENTS` — a dict mapping agent IDs to their :class:`AgentConfig`.
  This is the authoritative list of all agents in the system.

Talks to
--------
- ``minion.py`` reads ``AGENTS`` to know which agents exist.
- ``runner.py`` uses ``agent.name``, ``agent.soul``, and ``agent.max_tool_rounds``.
- ``router.py`` references agent IDs (like ``"researcher"``) when routing.

To add a new agent, add an entry to ``AGENTS`` here and a matching entry
under ``"agents"`` in ``config.json``. See README.md for the full walkthrough.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """A container for an agent's identity and behavioural configuration.

    Args:
        name (str): The human-readable display name of the agent.
            Shown in the terminal before each response, e.g. ``"Ada: Hello!"``.
        soul (str): The system prompt that defines the agent's personality,
            expertise, and behavioral rules. Sent to the LLM on every turn.
        max_tool_rounds (int): Maximum LLM→tool-call iterations per user turn.
            Default 10 is suitable for conversational agents.  Task-focused
            agents benefit from a higher value (e.g. 20–25) so they can work
            through multi-step problems without hitting the cap.
    """
    name: str
    soul: str
    max_tool_rounds: int = field(default=10)


# ---------------------------------------------------------------------------
# Shared soul fragment injected into every agent for long-running task support.
# Kept as a module-level constant so it is easy to locate and update.
# ---------------------------------------------------------------------------
_TASK_SOUL_SUFFIX = (
    "\nAt the start of each session, call read_task to check for an active task "
    "and continue from where you left off. After completing each step, call "
    "update_task to record progress. This ensures no work is lost if the context "
    "window fills or the session restarts."
)

# ---------------------------------------------------------------------------
# The authoritative registry of all agents.
#
# Keys (e.g. "main", "researcher") are the IDs used throughout the codebase
# to refer to agents. They must also appear in config.json under "agents"
# so the config loader knows which model to assign to each.
# ---------------------------------------------------------------------------
AGENTS: dict[str, AgentConfig] = {
    "main": AgentConfig(
        name="Ada",
        max_tool_rounds=20,
        soul=(
            # Ada is the default catch-all agent. She handles general questions,
            # API development topics, and anything not specifically routed elsewhere.
            "You are Ada, an AI assistant specializing in API development.\n"
            "Be genuinely helpful. Skip the pleasantries. Have opinions.\n"
            "You have tools — use them proactively but efficiently: once you have "
            "enough context, stop searching and answer. Don't keep running tools "
            "when you already know what to say.\n"
            "Use search_memory to recall past notes — always search memory first "
            "when you encounter an unfamiliar name, person, or topic before responding.\n"
            "Use save_memory to persist: (1) important research findings, and "
            "(2) any personal context the user shares — names, ages, relationships, "
            "preferences, background, location, city, address, zip code.\n"
            "Use the skill tool to load domain expertise when a task matches a "
            "skill's description.\n"
            "After completing tool calls, always respond directly with your answer "
            "to the user — do not summarise or explain what tools you ran, just give "
            "the answer. If tools returned no useful result, say so and provide your "
            "best answer anyway."
            + _TASK_SOUL_SUFFIX
        ),
    ),
    "researcher": AgentConfig(
        name="Elizabeth",
        max_tool_rounds=15,
        soul=(
            # Elizabeth is the research specialist. She is activated via the
            # /research command. Her soul enforces citation-backed responses
            # and discourages re-doing work already stored in memory.
            "You are Elizabeth, a research specialist for API development.\n"
            "Your job: find information and cite sources. Every claim needs evidence.\n"
            "Use tools to gather data. Be thorough but concise.\n"
            "Use save_memory to persist research findings. "
            "Use search_memory to avoid re-doing work.\n"
            "Use the skill tool to load domain expertise when a task matches a "
            "skill's description.\n"
            "After completing tool calls, always respond directly with your answer "
            "to the user — do not summarise what tools you ran, just give the answer. "
            "If tools returned no useful result, say so and provide your best answer "
            "anyway."
            + _TASK_SOUL_SUFFIX
        ),
    ),
}

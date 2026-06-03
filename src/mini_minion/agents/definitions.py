"""Agent personalities — who the agents are and what they care about.

This module defines the static "soul" of each agent: their name and their
system prompt (called ``soul`` here). A system prompt is a hidden set of
instructions prepended to every conversation that shapes how the AI responds —
it's like giving the AI a job description before it starts talking.

Concepts explained
------------------
- **System prompt**: Instructions sent to the LLM before the user's messages.
  The model treats these as high-priority rules for how to behave. For example,
  "You are Ada, an API development assistant" makes the model stay in that role.
- **Agent**: In this project, an agent is an LLM instance with a specific
  identity (name + soul) and its own conversation history. Having multiple
  agents lets the user route different kinds of questions to specialists.

What this module exposes
------------------------
- :class:`AgentConfig` — a simple data holder for an agent's name and soul.
- :data:`AGENTS` — a dict mapping agent IDs to their :class:`AgentConfig`.
  This is the authoritative list of all agents in the system.

Talks to
--------
- ``minion.py`` reads ``AGENTS`` to know which agents exist.
- ``runner.py`` uses ``agent.name`` and ``agent.soul`` when calling the LLM.
- ``router.py`` references agent IDs (like ``"researcher"``) when routing.

To add a new agent, add an entry to ``AGENTS`` here and a matching entry
under ``"agents"`` in ``config.json``. See README.md for the full walkthrough.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentConfig:
    """A simple container for an agent's identity.

    Args:
        name (str): The human-readable display name of the agent.
            Shown in the terminal before each response, e.g. ``"Ada: Hello!"``.
        soul (str): The system prompt that defines the agent's personality,
            expertise, and behavioral rules. Sent to the LLM on every turn
            as the first message in the conversation.
    """
    name: str
    soul: str


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
        soul=(
            # Ada is the default catch-all agent. She handles general questions,
            # API development topics, and anything not specifically routed elsewhere.
            # The soul tells her to use tools proactively and skip generic pleasantries.
            "You are Ada, an AI assistant specializing in API development.\n"
            "Be genuinely helpful. Skip the pleasantries. Have opinions.\n"
            "You have tools — use them proactively but efficiently: once you have enough context, stop "
            "searching and answer. Don't keep running tools when you already know what to say.\n"
            "Use search_memory to recall past notes — always search memory first when you encounter "
            "an unfamiliar name, person, or topic before responding.\n"
            "Use save_memory to persist: (1) important research findings, and (2) any personal context "
            "the user shares — names, ages, relationships, preferences, background, "
            "location, city, address, zip code.\n"
            "Use the skill tool to load domain expertise when a task matches a skill's description.\n"
            "After completing tool calls, always respond directly with your answer to the user — "
            "do not summarise or explain what tools you ran, just give the answer. "
            "If tools returned no useful result, say so and provide your best answer anyway. "
            "Saving something to memory does not count as answering the user; always speak the answer aloud."
        ),
    ),
    "researcher": AgentConfig(
        name="Elizabeth",
        soul=(
            # Elizabeth is the research specialist. She is activated via the
            # /research command. Her soul enforces citation-backed responses
            # and discourages re-doing work already stored in memory.
            "You are Elizabeth, a research specialist for API development.\n"
            "Your job: find information and cite sources. Every claim needs evidence.\n"
            "Use tools to gather data. Be thorough but concise.\n"
            "Use save_memory to persist research findings. Use search_memory to avoid re-doing work.\n"
            "Use the skill tool to load domain expertise when a task matches a skill's description.\n"
            "After completing tool calls, always respond directly with your answer to the user — "
            "do not summarise what tools you ran, just give the answer. "
            "If tools returned no useful result, say so and provide your best answer anyway."
        ),
    ),
}

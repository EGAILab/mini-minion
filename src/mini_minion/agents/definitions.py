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
    # Both task instructions are now code-enforced and no longer need to live
    # in the soul:
    #
    # "call read_task at session start" → replaced by AgentSession auto-injecting
    # an <active_task> block into the system prompt before every turn.
    #
    # "call update_task after each step" → replaced by _format_task_context()
    # appending a targeted "Action: call update_task" line inside <active_task>
    # whenever a step is currently in_progress — conditional and specific.
    #
    # This constant is kept as an empty string so existing code that appends it
    # continues to compile; its content is intentionally blank.
    ""
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
            "You have tools — use them proactively: gather what you need, then "
            "answer. If you see a <context_budget> warning above, be concise.\n"
            "Relevant memories are automatically injected above — call search_memory "
            "only when you need something specific not already shown there.\n"
            "For queries that depend on personal context (location, preferences, "
            "relationships), check the <user_context> block above first. "
            "If the needed context is not shown, call search_memory with relevant "
            "terms (e.g. 'user location', 'home address', 'preferences') before "
            "making external tool calls like web_search.\n"
            "Use save_memory for structured notes (research findings, key decisions, "
            "reference info) — personal context is captured automatically.\n"
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

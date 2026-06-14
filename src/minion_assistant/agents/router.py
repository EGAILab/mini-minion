"""Message routing — decide which agent should handle a user's input.

Routing rules are driven by ``config.json`` rather than hard-coded here.
Each agent entry can declare a ``"route_prefix"`` (e.g. ``"/research"``) in
``config.json``. When a user message starts with that prefix followed by a
space, it is routed to that agent and the prefix+space are stripped.
Messages matching no prefix go to the default agent — the one whose config
entry has no ``route_prefix`` (or ``null``).

Example ``config.json`` fragment::

    "agents": {
      "main":       {"model": "lmstudio/qwen-qwen3.5-9b"},
      "researcher": {"model": "aliyuncs/glm-5", "route_prefix": "/research"}
    }

With this config:
- ``/research find benchmarks`` → ``("researcher", "find benchmarks")``
- ``anything else``            → ``("main", "anything else")``

Adding a third agent only requires a new entry in ``config.json`` with its own
``route_prefix``; no Python changes are needed.

Talks to
--------
- ``config`` is imported at module load to build the routing table once.
- ``minion.py`` calls :func:`resolve` on every user input.
"""

from __future__ import annotations

from minion_assistant.config import agents as _agents_cfg


def _build_routes() -> tuple[list[tuple[str, str]], str]:
    """Build the routing table from the loaded agent config.

    Iterates over all agents in config. Agents with a ``route_prefix`` get an
    entry in the routing table; the agent without one becomes the fallback.

    The trailing space is appended to each stored prefix so that ``/research``
    only matches ``/research <message>`` and not ``/researchfoo``.

    Routes are sorted longest-prefix first to prevent a shorter prefix from
    shadowing a longer one if two agents share a common prefix start.

    Returns:
        tuple: A pair of ``(routes, default_id)`` where:

        - ``routes`` — list of ``(prefix_with_space, agent_id)`` pairs.
        - ``default_id`` — agent_id of the catch-all fallback. If every agent
          has a ``route_prefix`` a ``RuntimeError`` is raised — config validation
          in ``_validate()`` prevents this from occurring at normal startup.
    """
    routes: list[tuple[str, str]] = []
    default_id = ""  # set below; config validation guarantees one un-prefixed agent exists

    for agent_id, cfg in _agents_cfg.items():
        if cfg.route_prefix:
            # Append a space so "/research" matches "/research foo" but not "/researchfoo".
            routes.append((cfg.route_prefix + " ", agent_id))
        else:
            # No prefix configured → this agent is the catch-all.
            # If multiple agents lack a prefix, the last one wins.
            default_id = agent_id

    if not default_id:
        # Config validation (_validate in config.py) should have prevented this.
        # Defensive guard in case router.py is exercised outside the normal startup path.
        raise RuntimeError(
            "No default agent in config — every agent has a route_prefix. "
            "Add at least one agent without a route_prefix."
        )

    # Longest prefix first prevents shadowing (e.g. "/code advanced" before "/code").
    routes.sort(key=lambda pair: len(pair[0]), reverse=True)
    return routes, default_id


# Build the routing table once at module load — no work on each resolve() call.
_ROUTES, _DEFAULT_AGENT = _build_routes()


def resolve(message: str) -> tuple[str, str]:
    """Route a raw user message to the appropriate agent.

    Checks the message against the config-driven routing table and returns the
    matching agent ID along with the message cleaned of its prefix.

    Args:
        message (str): The raw text the user typed at the prompt,
            e.g. ``"/research latest FastAPI benchmarks"`` or
            ``"what is dependency injection?"``.

    Returns:
        tuple[str, str]: A pair of ``(agent_id, cleaned_message)`` where:

        - ``agent_id`` is a key in ``AGENTS`` (e.g. ``"main"`` or
          ``"researcher"``).
        - ``cleaned_message`` is the text the agent will actually receive,
          with any command prefix stripped off.

    Examples:
        >>> resolve("/research find benchmarks")
        ('researcher', 'find benchmarks')

        >>> resolve("what is async?")
        ('main', 'what is async?')
    """
    for prefix, agent_id in _ROUTES:
        if message.startswith(prefix):
            # Strip the prefix (including its trailing space) from the message.
            return agent_id, message[len(prefix):]

    # No prefix matched → send to the default catch-all agent.
    return _DEFAULT_AGENT, message

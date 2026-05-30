"""Entry point — wires all subsystems together and runs the interactive REPL.

This is the last file you should read when exploring the codebase. By the time
you get here, you should already understand: config loading (``config.py``),
agents (``agents/``), providers (``providers/``), tools (``tools/``), memory
(``memory/``), context compaction (``context.py``), and the TAO loop
(``agents/runner.py``). This file just wires them all together.

What is a REPL?
---------------
REPL stands for *Read–Evaluate–Print Loop*. It is the simplest possible
interactive program:

1. **Read** — call ``input()`` to wait for the user to type something and press Enter.
2. **Evaluate** — process the input (in our case, send it to an LLM agent).
3. **Print** — show the response (done inside ``run_turn()``).
4. **Loop** — go back to step 1 until the user types ``exit``.

What is wired together here
----------------------------
``main()`` creates exactly one of each subsystem per agent:

- :class:`ShortTermMemory` — one shared instance (handles all agents' JSONL files).
- :class:`LongTermMemory` — one instance per agent (isolated note folders).
- :class:`SessionStore` — one shared instance (the ``sessions.json`` metadata file).
- :class:`ToolRegistry` — one per agent (each wired to that agent's memory backend).
- LLM provider — one per agent (each pointing at the model in ``config.json``).
- Conversation history — one ``list[dict]`` per agent, loaded from disk at startup.
- :class:`Compactor` — one per agent (sized to that model's context window).

Per-turn execution order
------------------------
For each message the user types:

1. ``resolve()`` — route to the right agent, strip any prefix.
2. Append the user message to history — **before** compaction sees it.
3. ``compact()`` — summarise old history if the context window is nearly full.
4. ``short_term.save()`` — persist the user message before the provider call.
5. ``run_turn()`` — TAO loop: THINK → ACT → OBSERVE until a final answer.
6. ``session_store.touch()`` — update turn count.
7. ``short_term.save()`` (in ``finally``) — persist the full updated history.

Error recovery
--------------
If ``run_turn()`` raises, the ``except`` block rolls back any partial assistant
or tool messages that were appended mid-turn (``del histories[agent_id][snapshot_len:]``),
records the error as an assistant message so the model has context on the next
turn, and continues the REPL. The user can try again without restarting.

The ``finally`` block always persists — even on failure, the user message and
any error record are written to disk.

Talks to
--------
Every module in the package. This is the integration layer.
"""

import sys
from pathlib import Path

from .agents import AGENTS, resolve, run_turn
from .config import agents as agents_cfg
from .config import compaction as compaction_cfg
from .config import streaming, workspace
from .context import Compactor
from .memory import LongTermMemory, ShortTermMemory
from .providers import create_provider
from .session import SessionStore
from .tools import default_registry


def main() -> None:
    """Start the interactive mini-minion chat session.

    This function is the application's entry point. It sets up all subsystems,
    then runs an infinite input loop until the user types ``exit`` or ``quit``.

    The conversation history for each agent is kept in memory (as a Python
    list of message dicts) during the session and written to disk after every
    turn, so it survives restarts.

    Streaming is enabled when ``config.json`` has ``"streaming": {"chat_mode": true}``.
    Context window compaction is enabled when the token count approaches
    ``config.json`` ``"compaction": {"context_window": ...}``.
    """
    # Validate that AGENTS (definitions.py) and agents_cfg (config.json) have the same keys.
    # A mismatch means an agent can be routed to by config but crash at AGENTS[agent_id].
    _cfg_ids = set(agents_cfg)
    _def_ids = set(AGENTS)
    if _cfg_ids != _def_ids:
        _parts = []
        if _cfg_ids - _def_ids:
            _parts.append(f"in config.json but missing from AGENTS: {sorted(_cfg_ids - _def_ids)}")
        if _def_ids - _cfg_ids:
            _parts.append(f"in AGENTS but missing from config.json: {sorted(_def_ids - _cfg_ids)}")
        raise SystemExit(
            f"Agent identity mismatch — {'; '.join(_parts)}.\n"
            "AGENTS in definitions.py and 'agents' in config.json must have the same keys."
        )

    # --- Memory setup ---
    # Short-term: stores full conversation history as JSONL files.
    # One file per agent: ~/.mini-minion/sessions/main.jsonl, etc.
    short_term = ShortTermMemory(workspace / "sessions")

    # Long-term: one memory store per agent, each isolated in its own subdirectory.
    # ~/.mini-minion/memory/main/     → Ada's notes
    # ~/.mini-minion/memory/researcher/ → Elizabeth's notes
    long_terms = {
        agent_id: LongTermMemory(workspace / "memory" / agent_id)
        for agent_id in AGENTS
    }

    # Session store: a single JSON file tracking turn counts + timestamps.
    session_store = SessionStore(workspace / "sessions.json")

    # --- Tool setup ---
    # Lock in the project directory at startup as the workspace root.
    # File tools (read, write, glob) reject paths outside this boundary.
    _tool_root = Path.cwd()

    # One registry per agent, each wired to that agent's own LongTermMemory.
    # This prevents agents from reading or overwriting each other's notes.
    tool_registries = {
        agent_id: default_registry(long_term=long_terms[agent_id], root=_tool_root)
        for agent_id in AGENTS
    }

    # --- Provider setup ---
    # One LLM client per agent, each pointing at the model specified in config.json.
    # providers["main"]      → talks to LM Studio with Qwen 3.5
    # providers["researcher"] → talks to Aliyun with GLM-5
    providers = {
        agent_id: create_provider(
            api=cfg.provider.api,
            base_url=cfg.provider.base_url,
            api_key=cfg.provider.api_key,
            model=cfg.model.id,
        )
        for agent_id, cfg in agents_cfg.items()
    }

    # --- History setup ---
    # Load each agent's conversation history from disk into a Python list.
    # Each element is a dict like {"role": "user", "content": "hello"}.
    # run_turn() will append to these lists, and we save them back to disk
    # after each turn (so progress is never lost even on an abrupt exit).
    histories: dict[str, list[dict]] = {
        agent_id: short_term.load(agent_id) for agent_id in AGENTS
    }

    # Ensure every agent has a session record (creates one if this is first run).
    for agent_id in AGENTS:
        session_store.get_or_create(agent_id)

    # Determine whether to stream responses in this interactive session.
    # streaming.chat_mode comes from the "streaming.chat_mode" key in config.json.
    use_streaming = streaming.chat_mode

    # Context window compaction: one Compactor per agent, each using its model's
    # own context_window. preserve_tokens is shared (from the "compaction" block).
    compactors = {
        agent_id: Compactor(
            context_window=agents_cfg[agent_id].model.context_window,
            preserve_tokens=compaction_cfg.preserve_tokens,
        )
        for agent_id in AGENTS
    }

    # --- REPL loop ---
    print("Mini-Minion ready. Type 'exit' or '/quit' to quit.")
    # Print one hint per agent that has a route_prefix configured in config.json.
    # This is driven by config so it stays accurate when agents are added/removed.
    for agent_id, cfg_entry in agents_cfg.items():
        if cfg_entry.route_prefix:
            agent_name = AGENTS[agent_id].name
            print(f"  {cfg_entry.route_prefix} <message>  → {agent_name}")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "/quit"):
            break

        # Route the message: "/research ..." → researcher, anything else → main.
        # resolve() strips the command prefix and returns the clean message.
        agent_id, message = resolve(user_input)
        agent = AGENTS[agent_id]
        cfg = agents_cfg[agent_id]

        # Append the user's message first so compaction sees the complete
        # message list (including this new message) when checking the budget.
        histories[agent_id].append({"role": "user", "content": message})

        # Compact the history if it's approaching the model's context limit.
        # compact() is a no-op when under budget; it summarises the older half
        # and prunes large tool outputs when over budget.
        histories[agent_id] = compactors[agent_id].compact(histories[agent_id], providers[agent_id])

        # Save immediately — the user turn is on disk even if the provider call
        # fails before the finally block runs.
        short_term.save(agent_id, histories[agent_id])

        # Snapshot the history length so we can roll back any partial assistant/
        # tool messages that run_turn appends before a mid-turn crash.
        snapshot_len = len(histories[agent_id])

        try:
            # Run the TAO loop: the model thinks, optionally calls tools, then replies.
            # run_turn() mutates histories[agent_id] in place, adding all assistant
            # messages and tool results.
            # stream=use_streaming enables token-by-token output when chat_mode is True.
            run_turn(
                providers[agent_id],
                agent.name,           # display name printed before responses (e.g. "Ada")
                agent.soul,           # the system prompt that defines the agent's personality
                cfg.model.max_output_tokens,
                tool_registries[agent_id],
                histories[agent_id],
                stream=use_streaming,
            )
            # Update turn count only on success — failed turns don't count.
            session_store.touch(agent_id, increment_turns=True)
        except Exception as exc:
            # Roll back any partial messages appended before the crash so the
            # history ends at a clean turn boundary.
            del histories[agent_id][snapshot_len:]
            # Record the failure so the model has context on the next turn.
            histories[agent_id].append({
                "role": "assistant",
                "content": f"[Provider error: {exc.__class__.__name__}: {exc}]",
            })
            print(f"\n[Error] {agent.name} failed to respond: {exc}", file=sys.stderr)
            print("  Your message was kept. Try again or rephrase.", file=sys.stderr)
        finally:
            # Always persist — user message and any error message survive a crash.
            short_term.save(agent_id, histories[agent_id])


if __name__ == "__main__":
    main()

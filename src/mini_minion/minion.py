"""Entry point — wires all subsystems together and runs the interactive REPL.

This is the last file you should read when exploring the codebase. By the time
you get here, you should already understand: config loading (``config.py``),
agents (``agents/``), providers (``providers/``), tools (``tools/``), memory
(``memory/``), context compaction (``context.py``), and the TAO loop
(``agents/runner.py``).  This file's job is to wire everything together and
provide the one presentation layer that uses a terminal.

What is a REPL?
---------------
REPL stands for *Read–Evaluate–Print Loop*:

1. **Read** — call ``input()`` to wait for the user to type something.
2. **Evaluate** — route the message and call ``session.send()``.
3. **Print** — the ``_on_event`` handler renders events as terminal output.
4. **Loop** — go back to step 1 until the user types ``exit``.

How the CLI adapter works
--------------------------
The agent runtime (``runner.py``, ``context.py``, ``bash.py``) no longer calls
``print()`` or ``input()`` directly.  Instead it emits structured event objects.
This file defines two presentation callbacks:

- ``_on_event(event)`` — renders events as terminal output (prefix + text,
  tool status lines, streaming tokens, compaction notice).
- ``_console_confirm(command)`` — called by :class:`BashTool` before running a
  shell command; prints the command and prompts the user for ``y/N``.

Any other presentation (JSON logs, web SSE, silent batch) only needs to replace
these two functions — the rest of the stack is unchanged.

AgentSession
------------
All per-agent state (history, provider, tools, compactor) lives in an
:class:`AgentSession` instance.  ``main()`` creates one per agent and stores
them in a ``sessions`` dict keyed by agent ID.  The REPL calls
``sessions[agent_id].send(message, on_event=..., stream=...)`` each turn.

Error recovery
--------------
:meth:`AgentSession.send` re-raises on provider failure.  ``main()`` catches
the exception, prints a user-friendly error, and continues the REPL.
The history rollback and error record persistence happen inside the session
before the exception propagates — so the caller only needs to handle display.

Streaming
---------
``streaming.chat_mode`` from ``config.json`` controls whether tokens are emitted
one-by-one via :class:`StreamingStarted` / :class:`TokenStreamed` events.
When ``False``, the full response arrives in a single :class:`FinalAnswer` event.
The ``_on_event`` handler handles both cases transparently by tracking whether
streaming is currently active.

Talks to
--------
Every module in the package.  This is the integration layer.
"""

import signal
import sys
from pathlib import Path

from .agents import AGENTS, AgentSession, resolve
from .agents.events import (
    CompactionFailed,
    CompactionStarted,
    FinalAnswer,
    MaxRoundsReached,
    StreamingStarted,
    ThoughtEmitted,
    TokenStreamed,
    ToolCalled,
)
from .config import agents as agents_cfg
from .config import compaction as compaction_cfg
from .config import streaming, workspace
from .context import Compactor, _SNIP_SAFETY_BUFFER
from .memory import LongTermMemory, ShortTermMemory
from .providers import create_provider
from .session import SessionStore
from .skills import discover_skills, format_skills_prompt
from .tools import default_registry


def main() -> None:
    """Start the interactive mini-minion chat session.

    Sets up all subsystems (one :class:`AgentSession` per agent), defines the
    console event handler and bash confirmation callback, then runs the REPL
    until the user types ``exit``.

    Streaming is enabled when ``config.json`` has ``"streaming": {"chat_mode": true}``.
    Context compaction is per-agent, sized to each model's ``context_window``.
    """
    # Validate that AGENTS (definitions.py) and agents_cfg (config.json) have the same keys.
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

    # --- Skill discovery ---
    # Global skills (~/.mini-minion/skills/) are scanned first so project-level
    # skills (.mini-minion/skills/) can override them (later entry wins).
    _skill_paths = [workspace / "skills", Path.cwd() / ".mini-minion" / "skills"]
    skills = discover_skills(_skill_paths)
    _skills_suffix = format_skills_prompt(skills)

    # --- Shared resources ---
    short_term = ShortTermMemory(workspace / "sessions")
    session_store = SessionStore(workspace / "sessions.json")
    _tool_root = Path.cwd()
    _tasks_dir = workspace / "tasks"

    # --- Console callbacks (the only two places that call print/input) ---

    def _console_confirm(command: str) -> bool:
        """Called by BashTool before running a shell command.

        Prints the command, prompts for y/N, returns True only on "y".
        """
        print(f"\n[bash] {command}")
        return input("Run this command? [y/N]: ").strip().lower() == "y"

    # --- Session setup — one AgentSession per agent ---
    sessions: dict[str, AgentSession] = {}
    for agent_id, cfg in agents_cfg.items():
        long_term = LongTermMemory(workspace / "memory" / agent_id)
        tools = default_registry(
            long_term=long_term,
            root=_tool_root,
            bash_confirm=_console_confirm,
            skills=skills,
            tasks_dir=_tasks_dir,
            agent_id=agent_id,
        )
        provider = create_provider(
            api=cfg.provider.api,
            base_url=cfg.provider.base_url,
            api_key=cfg.provider.api_key,
            model=cfg.model.id,
        )
        # preserve_tokens: use explicit config override when present; otherwise
        # auto-compute as max_output_tokens + _SNIP_SAFETY_BUFFER.
        # The safety buffer (1 024 tokens, from nanobot's _SNIP_SAFETY_BUFFER)
        # covers system-prompt tokens, tool-definition JSON overhead, and token-
        # estimation inaccuracies that raw max_output_tokens does not account for.
        _preserve = (
            compaction_cfg.preserve_tokens
            if compaction_cfg.preserve_tokens is not None
            else cfg.model.max_output_tokens + _SNIP_SAFETY_BUFFER
        )
        compactor = Compactor(
            context_window=cfg.model.context_window,
            preserve_tokens=_preserve,
        )
        sessions[agent_id] = AgentSession(
            agent_id=agent_id,
            agent=AGENTS[agent_id],
            provider=provider,
            max_output_tokens=cfg.model.max_output_tokens,
            tools=tools,
            compactor=compactor,
            short_term=short_term,
            session_store=session_store,
            soul_suffix=_skills_suffix,
            long_term=long_term,
            tasks_dir=_tasks_dir,
        )

    use_streaming = streaming.chat_mode

    # Install SIGTERM handler (POSIX only) so `kill <pid>` exits cleanly.
    # History is already persisted from the last turn's finally block.
    def _sigterm_handler(signum: int, frame: object) -> None:
        print("\nShutdown signal received. Goodbye.")
        sys.exit(0)

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sigterm_handler)

    # --- Console event handler ---
    # Tracks whether a streaming response is currently in progress so that the
    # correct newline is printed at the end and FinalAnswer doesn't re-print
    # text that was already streamed token-by-token.
    _streaming_active = False

    def _on_event(event: object) -> None:
        """Render one agent runtime event as terminal output.

        Handles all event types emitted by the runner, compactor, and
        (indirectly) the bash tool confirm flow.
        """
        nonlocal _streaming_active

        if isinstance(event, StreamingStarted):
            print(f"\n{event.agent_name}: ", end="", flush=True)
            _streaming_active = True

        elif isinstance(event, TokenStreamed):
            print(event.token, end="", flush=True)

        else:
            was_streaming = _streaming_active
            if _streaming_active:
                print()
                _streaming_active = False

            if isinstance(event, ThoughtEmitted):
                print(f"\n{event.agent_name}: {event.text}")
            elif isinstance(event, FinalAnswer):
                if event.text and not was_streaming:
                    print(f"\n{event.agent_name}: {event.text}")
            elif isinstance(event, ToolCalled):
                print(f"  [tool: {event.name}({event.args})]")
            elif isinstance(event, MaxRoundsReached):
                print(f"\n{event.agent_name}: {event.message}")
            elif isinstance(event, CompactionStarted):
                print("\n  Compacting session history...")
            elif isinstance(event, CompactionFailed):
                print(f"\n  [Warning] Compaction failed: {event.error}")

    # --- REPL loop ---
    print("Mini-Minion ready. Type 'exit' or '/quit' to quit.")
    for agent_id, cfg_entry in agents_cfg.items():
        if cfg_entry.route_prefix:
            agent_name = AGENTS[agent_id].name
            print(f"  {cfg_entry.route_prefix} <message>  → {agent_name}")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            # Ctrl+C or Ctrl+D at the prompt — exit cleanly.
            # History was already persisted from the last turn's finally block.
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "/quit"):
            break

        agent_id, message = resolve(user_input)

        _streaming_active = False

        try:
            sessions[agent_id].send(message, on_event=_on_event, stream=use_streaming)
        except KeyboardInterrupt:
            # Ctrl+C during a turn. The finally block in AgentSession.send() already
            # saved history. Notify the user and continue the REPL.
            agent_name = AGENTS[agent_id].name
            print(f"\n  Turn interrupted. {agent_name}'s history has been saved.")
        except Exception as exc:
            agent_name = AGENTS[agent_id].name
            print(f"\n[Error] {agent_name} failed to respond: {exc}", file=sys.stderr)
            print("  Your message was kept. Try again or rephrase.", file=sys.stderr)


if __name__ == "__main__":
    main()

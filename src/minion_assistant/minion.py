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
from .cli_input import PromptReader
from .commands import CommandContext, dispatch_command, parse_command
from .media import MediaAttachment, describe_attachment, stage_attachment
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
from .config import extra_plugin_manifests
from .config import mcp as mcp_cfg
from .config import memory as memory_cfg
from .plugins import load_plugins
from .config import streaming, workspace
from .mcp import McpClientManager
from .context import Compactor, _SNIP_SAFETY_BUFFER
from .memory import LongTermMemory, ShortTermMemory
from .providers import create_provider
from .session import SessionStore
from .skills import discover_skills, format_skills_prompt
from .tools import default_registry


def main() -> None:
    """Start the interactive minion-assistant chat session.

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
    # Global skills (~/.minion-assistant/skills/) are scanned first so project-level
    # skills (.minion-assistant/skills/) can override them (later entry wins).
    _skill_paths = [workspace / "skills", Path.cwd() / ".minion-assistant" / "skills"]
    skills = discover_skills(_skill_paths)
    _skills_suffix = format_skills_prompt(skills)

    # --- Shared resources ---
    short_term = ShortTermMemory(workspace / "sessions")
    session_store = SessionStore(workspace / "sessions.json")
    _tool_root = Path.cwd()
    _tasks_dir = workspace / "tasks"

    # --- Console callbacks (the only two places that call print/input) ---

    def _console_confirm(command: str) -> bool:
        """Called by GitCommitTool before running a commit.

        Prints the command, prompts for y/N, returns True only on "y".
        """
        print(f"\n[git] {command}")
        return input("Run this command? [y/N]: ").strip().lower() == "y"

    def _console_approve(command: str) -> "ApprovalDecision":
        """Called by BashTool — shows a 4-option approval menu.

        Options:
          1 / Enter — Allow once (default)
          2         — Allow for this session (no re-prompting for same command)
          3         — Deny once
          4         — Always deny for this session

        The decision is recorded to the policy's audit log by BashTool so
        the user can review it with /audit.
        """
        from .tools.audit import ApprovalDecision
        print(f"\n[bash] {command}")
        print("  [1] Allow once   [2] Allow session   [3] Deny   [4] Always deny")
        choice = input("Choice [1]: ").strip()
        if choice == "2":
            return ApprovalDecision.ALLOW_SESSION
        if choice == "3":
            return ApprovalDecision.DENY
        if choice == "4":
            return ApprovalDecision.ALWAYS_DENY
        return ApprovalDecision.ALLOW_ONCE

    def _console_ask_user(question: str) -> str:
        """Called by AskUserTool when the agent needs a human response."""
        print(f"\n[ask_user] {question}")
        return input("Your answer: ").strip()

    # --- MCP setup — one shared manager for all agents ---
    # The manager owns a background asyncio loop and keeps sessions open for the
    # lifetime of the process. All agents share the same manager so they don't
    # open duplicate connections to the same MCP server.
    mcp_manager: McpClientManager | None = None
    if mcp_cfg.servers:
        # output_dir: where MCP image content (e.g. Playwright screenshots) is saved.
        # McpClientManager creates this directory the first time a screenshot
        # is taken, so we don't need to pre-create it here.
        # The path is workspace-relative so it's automatically namespaced per user
        # and doesn't collide with other minion-assistant workspaces.
        # Example: ~/.minion-assistant/playwright-output/screenshot-1718123456789.png
        _mcp_output_dir = workspace / "playwright-output"
        mcp_manager = McpClientManager(list(mcp_cfg.servers), output_dir=_mcp_output_dir)
        print("Connecting to MCP servers...")
        mcp_manager.connect_all_sync()
        for status in mcp_manager.list_statuses():
            state_str = "OK" if status.state == "connected" else f"FAILED: {status.detail}"
            print(f"  MCP [{status.name}]: {state_str}")

    # --- Session setup — one AgentSession per agent ---
    sessions: dict[str, AgentSession] = {}
    for agent_id, cfg in agents_cfg.items():
        long_term = LongTermMemory(workspace / "memory" / agent_id)
        tools = default_registry(
            long_term=long_term,
            root=_tool_root,
            bash_confirm=_console_confirm,
            bash_approval=_console_approve,
            skills=skills,
            tasks_dir=_tasks_dir,
            agent_id=agent_id,
            mcp_manager=mcp_manager,
            ask_user_fn=_console_ask_user,
        )
        # Load user-defined tools from plugins.json manifests.
        # Done after default_registry() so plugins can override built-in tools
        # by registering a tool with the same name.
        # extra_plugin_manifests: additional paths from config.json "extra_plugin_manifests".
        _plugin_count = load_plugins(tools, workspace, skills=skills, extra_manifests=extra_plugin_manifests)
        if _plugin_count:
            print(f"  Loaded {_plugin_count} plugin tool(s) for agent '{agent_id}'.")
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
            # Respect config.json "memory.enable_extraction": false to suppress
            # the background API call that extracts facts after each turn.
            enable_memory_extraction=memory_cfg.enable_extraction,
        )

    use_streaming = streaming.chat_mode

    # --- Attachment state ---
    # Per-agent staging area for files added with /attach but not yet sent.
    # Cleared after each send() call so attachments are tied to one message.
    _media_dir = workspace / "attachments"
    pending_attachments: dict[str, list[MediaAttachment]] = {
        aid: [] for aid in sessions
    }
    # Track which agent the user last interacted with, for targeting /attach etc.
    # Defaults to the first agent in config (the non-routed fallback).
    _default_agent_id = next(iter(sessions))
    active_agent_id: str = _default_agent_id

    # --- Prompt history reader (Plan 13) ---
    # PromptReader wraps prompt_toolkit for Up/Down arrow history navigation.
    # Falls back to plain input() automatically when not in a TTY (e.g. tests,
    # piped input), so no existing code paths break.
    prompt_reader = PromptReader(workspace / "prompt_history.txt")

    # Install SIGTERM handler (POSIX only) so `kill <pid>` exits cleanly.
    # History is already persisted from the last turn's finally block.
    def _sigterm_handler(signum: int, frame: object) -> None:
        print("\nShutdown signal received. Goodbye.")
        # Close MCP sessions before exiting so subprocess stdio transports shut down
        if mcp_manager is not None:
            mcp_manager.close_sync()
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
    print("Mini-Minion ready. Type /help for commands, /quit to quit.")
    for agent_id, cfg_entry in agents_cfg.items():
        if cfg_entry.route_prefix:
            agent_name = AGENTS[agent_id].name
            print(f"  {cfg_entry.route_prefix} <message>  → {agent_name}")

    try:
        while True:
            try:
                user_input = prompt_reader.read().strip()
            except (KeyboardInterrupt, EOFError):
                # Ctrl+C or Ctrl+D at the prompt — exit cleanly.
                # History was already persisted from the last turn's finally block.
                print("\nGoodbye.")
                break

            if not user_input:
                continue

            # Plain exit/quit (no slash) — kept for muscle-memory compatibility.
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye.")
                break

            # --- Attachment commands (handled before routing and command dispatcher) ---
            # These need direct access to pending_attachments and _media_dir, which
            # are not available in the CommandContext, so they live here in the REPL.

            # /attach <path> [path2 ...] — stage files for the next message.
            # We check for "/attach " (with space) to avoid matching "/attachments".
            if user_input.lower().startswith("/attach "):
                paths_str = user_input[len("/attach "):].strip()
                paths = paths_str.split()
                for p in paths:
                    try:
                        att = stage_attachment(Path(p), _media_dir)
                        pending_attachments[active_agent_id].append(att)
                        print(f"  Attached: {describe_attachment(att)}")
                    except (ValueError, FileNotFoundError) as e:
                        print(f"  Error: {e}")
                continue

            # /attachments — list what's staged for the active agent.
            if user_input.strip().lower() == "/attachments":
                atts = pending_attachments.get(active_agent_id, [])
                if atts:
                    print(f"Pending attachments for {active_agent_id}:")
                    for i, a in enumerate(atts, 1):
                        print(f"  [{i}] {describe_attachment(a)}")
                else:
                    print("No pending attachments.")
                continue

            # /clear-attachments — discard staged files without sending.
            if user_input.strip().lower() == "/clear-attachments":
                pending_attachments[active_agent_id] = []
                print("Cleared pending attachments.")
                continue

            # --- Slash command dispatch (Plan 14) ---
            # Resolve routing first so "/research /new" targets the researcher agent.
            # Then check if the routed payload (or the full input) is a slash command.
            agent_id, message = resolve(user_input)

            # Determine the text to examine for slash commands.
            # If routing stripped a prefix, examine the remaining payload; otherwise
            # examine the full input (handles "/new", "/help", etc. without a prefix).
            cmd_text = message if message.startswith("/") else user_input

            parsed = parse_command(cmd_text)
            if parsed is not None:
                cmd_token, cmd_args = parsed

                # Check whether this slash token is actually a route prefix used
                # as a standalone word (e.g. "/research" with no trailing message).
                # In that case it is NOT a command — it's an incomplete route.
                _is_lone_route_prefix = any(
                    user_input.strip() == cfg.route_prefix
                    for cfg in agents_cfg.values()
                    if cfg.route_prefix
                )

                if not _is_lone_route_prefix:
                    ctx = CommandContext(
                        raw=user_input,
                        command=cmd_token,
                        args=cmd_args,
                        target_agent_id=agent_id,
                        sessions=sessions,
                        agents_cfg=agents_cfg,
                        session_store=session_store,
                        mcp_manager=mcp_manager,
                    )
                    result = dispatch_command(ctx)
                    if result.handled:
                        if result.message:
                            print(result.message)
                        # /resume sets activate_agent_id to switch the default agent.
                        if result.activate_agent_id:
                            active_agent_id = result.activate_agent_id
                        if result.should_exit:
                            break
                        continue
                    else:
                        # Unknown slash command — warn the user instead of forwarding
                        # to the LLM, which would treat "/typo" as a message.
                        print(f"Unknown command '{cmd_token}'. Type /help for commands.")
                        continue

            # --- Normal agent routing ---
            _streaming_active = False
            # Track which agent handled this message so /attach targets the right session.
            active_agent_id = agent_id

            # Collect any staged attachments for this agent and clear the queue.
            # Attachments are one-shot: they go with the next send() and are then cleared.
            _atts = pending_attachments.get(agent_id, [])
            pending_attachments[agent_id] = []

            try:
                sessions[agent_id].send(
                    message,
                    attachments=_atts or None,
                    on_event=_on_event,
                    stream=use_streaming,
                )
            except KeyboardInterrupt:
                # Ctrl+C during a turn. The finally block in AgentSession.send() already
                # saved history. Notify the user and continue the REPL.
                agent_name = AGENTS[agent_id].name
                print(f"\n  Turn interrupted. {agent_name}'s history has been saved.")
            except Exception as exc:
                agent_name = AGENTS[agent_id].name
                print(f"\n[Error] {agent_name} failed to respond: {exc}", file=sys.stderr)
                print("  Your message was kept. Try again or rephrase.", file=sys.stderr)
    finally:
        # Always close MCP sessions on exit — cleans up subprocess stdio transports.
        if mcp_manager is not None:
            mcp_manager.close_sync()


if __name__ == "__main__":
    main()

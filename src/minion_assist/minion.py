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
import threading
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from .agents import AGENTS, AgentSession, resolve
from .cli_input import PromptReader
from .commands import CommandContext, build_completion_items, dispatch_command, parse_command
from .media import MediaAttachment, describe_attachment, stage_attachment
from .agents.events import (
    CompactionFailed,
    CompactionStarted,
    FinalAnswer,
    MaxRoundsReached,
    MemoryFlushed,
    StreamingStarted,
    ThoughtEmitted,
    TokenStreamed,
    ToolCalled,
)
from .bootstrap import build_bootstrap_prompt_block
from .config import agents as agents_cfg
from .config import bootstrap as bootstrap_cfg
from .config import channels as channels_cfg
from .config import commitments as commitments_cfg
from .config import compaction as compaction_cfg
from .config import extra_plugin_manifests
from .config import dreaming as dreaming_cfg
from .config import heartbeat as heartbeat_cfg
from .config import codex_cfg, logging_cfg
from .config import database as database_cfg
from .config import embeddings as embeddings_cfg
from .config import mcp as mcp_cfg
from .config import memory as memory_cfg
from .config import memory_consolidation as memory_consolidation_cfg
from .config import multi_agent as multi_agent_cfg
from .plugins import load_plugins
from .config import streaming, voice as voice_cfg, workspace
from .mcp import McpClientManager
from .context import Compactor, _SNIP_SAFETY_BUFFER
from .memory import MemoryFileRepository, MemoryService, ShortTermMemory
from .providers import create_provider
from .session import SessionStore
from .skills import discover_skills, format_skills_prompt
from .spawn_registry import count_active_children, get_spawn_depth
from .tools import ToolRegistry, default_registry
from .tools.spawn_subagent import SpawnSubagentTool, _make_subagent_registry
from .workspace import agent_workspace_root, ensure_workspace


def _build_reseed_context(messages: list[dict], max_chars: int) -> str | None:
    """Format prior session messages as a system-prompt block for a fresh session.

    Mirrors openclaw's ``buildCliSessionHistoryPrompt()``: only user and assistant
    text turns are included; tool calls and tool results are omitted for readability.
    Content exceeding ``max_chars`` is tail-truncated so recent turns survive.
    """
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ).strip()
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "user":
            lines.append(f"User: {content.strip()}")
        elif role == "assistant":
            lines.append(f"Assistant: {content.strip()}")
    if not lines:
        return None
    history_text = "\n\n".join(lines)
    if len(history_text) > max_chars:
        history_text = (
            "[Prior history truncated — older turns omitted]\n"
            + history_text[-max_chars:]
        )
    return "\n".join([
        "The following is a transcript of a prior session.",
        "Use it as context to continue naturally.",
        "",
        "<prior_session_history>",
        history_text,
        "</prior_session_history>",
    ])


def _resolve_session_id(
    agent_id: str,
    session_store: "SessionStore",
    short_term: "ShortTermMemory",
    reset_mode: str = "daily",
    reset_at_hour: int = 4,
    idle_minutes: int = 0,
    reseed_max_chars: int = 12_000,
) -> tuple[str, str | None]:
    """Return ``(session_id, reseed_context)`` for an agent.

    Mirrors openclaw's ``resolveSession()`` + ``evaluateSessionFreshness()``:

    - ``"daily"`` mode: stale when ``last_active`` predates today's ``reset_at_hour``.
    - ``"idle"`` mode: stale when ``last_active`` is older than ``idle_minutes``.
      ``idle_minutes=0`` always rotates.

    When stale, the old session's history is loaded and formatted as
    ``reseed_context`` (a ``<prior_session_history>`` block) so the new session
    can continue with context.  Returns ``(new_uuid, reseed_context_or_None)``.
    When fresh, returns ``(existing_uuid, None)``.

    Old session JSONL files are never deleted here; prune_sessions() handles that.
    """
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415
    info = session_store.get_or_create(agent_id)
    if not info.session_id:
        return session_store.new_session(agent_id), None

    now = datetime.now(UTC)
    stale = False

    if reset_mode == "daily":
        try:
            last_active = datetime.fromisoformat(info.last_active)
            boundary = now.replace(hour=reset_at_hour, minute=0, second=0, microsecond=0)
            if now < boundary:
                boundary -= timedelta(days=1)
            stale = last_active < boundary
        except (ValueError, TypeError):
            stale = True
    else:  # "idle"
        if idle_minutes == 0:
            stale = True
        else:
            try:
                last_active = datetime.fromisoformat(info.last_active)
                stale = (now - last_active).total_seconds() > idle_minutes * 60
            except (ValueError, TypeError):
                stale = True

    if not stale:
        return info.session_id, None

    # Build reseed context from the old session before rotating.
    old_messages = short_term.load(agent_id, info.session_id)
    reseed_context = _build_reseed_context(old_messages, max_chars=reseed_max_chars)
    return session_store.new_session(agent_id), reseed_context


def main() -> None:
    """Start the interactive minion-assist chat session.

    Sets up all subsystems (one :class:`AgentSession` per agent), defines the
    console event handler and bash confirmation callback, then runs the REPL
    until the user types ``exit``.

    Pass ``--voice`` on the command line to start in voice chat mode instead of
    the text REPL:  ``minion-assist --voice``.

    Streaming is enabled when ``config.json`` has ``"streaming": {"chat_mode": true}``.
    Context compaction is per-agent, sized to each model's ``context_window``.

    ``minion-assist memory ...`` is a separate, non-interactive CLI subcommand
    tree (see ``memory/cli.py``) — dispatched here, before any REPL setup, so
    it never pays the cost of building sessions/providers/tools just to run a
    migration report.
    """
    import sys as _sys  # noqa: PLC0415
    if len(_sys.argv) > 1 and _sys.argv[1] == "memory":
        from .memory.cli import main as _memory_cli_main  # noqa: PLC0415
        raise SystemExit(_memory_cli_main(_sys.argv[2:]))

    import argparse as _argparse  # noqa: PLC0415
    _parser = _argparse.ArgumentParser(prog="minion-assist", add_help=False)
    _parser.add_argument(
        "--voice",
        action="store_true",
        help="Start in voice chat mode (speech-to-speech).",
    )
    # parse_known_args so unrecognised flags (e.g. from test runners) don't error.
    _args, _ = _parser.parse_known_args()

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
    # Global skills (~/.minion-assist/skills/) are scanned first so project-level
    # skills (.minion-assist/skills/) can override them (later entry wins).
    _skill_paths = [workspace / "skills", Path.cwd() / ".minion-assist" / "skills"]
    skills = discover_skills(_skill_paths)
    _skills_suffix = format_skills_prompt(skills)

    # --- Shared resources ---
    short_term = ShortTermMemory(workspace / "sessions")
    session_store = SessionStore(workspace / "sessions.json")
    _tool_root = Path.cwd()
    _tasks_dir = workspace / "tasks"
    _log_dir = workspace / "logs" if logging_cfg.llm_requests else None

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

    def _console_approve_codex(method: str, params: dict) -> str:
        """Called by CodexProvider when Codex requests to execute a built-in tool.

        Runs in the Codex reader thread (main thread is blocked in done.wait).
        Returns "approve" or "deny".
        """
        import sys as _sys
        # Extract a human-readable command summary from params.
        cmd = (
            params.get("command")
            or params.get("cmd")
            or (params.get("arguments") or {}).get("command")
            or ""
        )
        print(f"\n[codex] Permission request: {method}")
        if cmd:
            print(f"  Command: {cmd[:200]}")
        elif params:
            import json as _json
            print(f"  Params:  {_json.dumps(params, ensure_ascii=False)[:200]}")
        _sys.stdout.flush()
        choice = input("Allow? [Y/n]: ").strip().lower()
        return "deny" if choice == "n" else "approve"

    if codex_cfg.allow_all_commands:
        _codex_approve: "Callable[[str, dict], str]" = lambda m, p: "approve"
    else:
        _codex_approve = _console_approve_codex

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
        # and doesn't collide with other minion-assist workspaces.
        # Example: ~/.minion-assist/playwright-output/screenshot-1718123456789.png
        _mcp_output_dir = workspace / "playwright-output"
        mcp_manager = McpClientManager(list(mcp_cfg.servers), output_dir=_mcp_output_dir)
        print("Connecting to MCP servers...")
        mcp_manager.connect_all_sync()
        for status in mcp_manager.list_statuses():
            state_str = "OK" if status.state == "connected" else f"FAILED: {status.detail}"
            print(f"  MCP [{status.name}]: {state_str}")

    # --- PostgreSQL session store (optional) ---
    _db = None
    if database_cfg.url:
        try:
            from .session.db import SessionDB  # noqa: PLC0415
            _db = SessionDB(database_cfg.url)
            print(f"Database connected: {database_cfg.url.split('@')[-1]}")
            # Reconcile every session's JSONL against message_mirrors — mirrors
            # exactly what's missing (including a partial mirror left by a prior
            # crash), never skips a whole session just because it has some rows.
            _reconciled = _db.reconcile_all_sessions(short_term, list(agents_cfg.keys()))
            if _reconciled:
                print(f"  Mirrored {_reconciled} messages from JSONL to database.")
        except Exception as _db_exc:
            print(f"[minion-assist] Warning: database unavailable ({_db_exc}). Session search disabled.")
            _db = None

    # --- Lexical memory index (optional — only when a database is configured) ---
    # Stage One Phase 3: rebuildable PostgreSQL FTS over memory files (MEMORY.md,
    # topic notes, daily notes, imports). Its own class/schema, separate from
    # SessionDB above — see memory/postgres_index.py's module docstring.
    _memory_index = None
    if _db is not None:
        try:
            from .memory.postgres_index import PostgresMemoryIndex  # noqa: PLC0415
            # Stage One Phase 4, slice A: pgvector's column width is fixed at
            # table-creation time, so the configured model's dimensions (if
            # any) must be known up front. embeddings_cfg is None when the
            # "embeddings" config.json section is absent — the vector lane
            # then simply never gets created.
            _embedding_dims = embeddings_cfg.dimensions if embeddings_cfg else None
            # Stage One Phase 4, slice C: constructed once here (not inside
            # PostgresMemoryIndex) so the index stays free of a direct
            # config.py dependency — same reasoning CaptureWorker's injected
            # provider_for_agent callable follows.
            _embedding_provider = None
            if embeddings_cfg is not None:
                from .providers.embeddings import EmbeddingProvider  # noqa: PLC0415
                _embedding_provider = EmbeddingProvider(
                    base_url=embeddings_cfg.provider.base_url,
                    api_key=embeddings_cfg.provider.api_key,
                    model=embeddings_cfg.model,
                    dimensions=embeddings_cfg.dimensions,
                )
            _memory_index = PostgresMemoryIndex(
                database_cfg.url,
                embedding_dimensions=_embedding_dims,
                embedding_provider=_embedding_provider,
            )
        except Exception as _idx_exc:
            print(
                f"[minion-assist] Warning: memory index unavailable ({_idx_exc}). "
                "Lexical memory search disabled."
            )
            _memory_index = None

    # --- Matrix channel setup (optional) ---
    # Runs in a background daemon thread; REPL continues on the main thread.
    # The channel is started after sessions are built so it can share them.
    _matrix_channel = None

    # --- Bootstrap root (global fallback) ---
    # bootstrap_cfg.path=None resolves to Path.cwd() so the working directory
    # is captured lazily rather than at startup.  Per-agent workspace directories
    # (if present) override this fallback for that agent's session.
    _bootstrap_root = (
        Path(bootstrap_cfg.path).expanduser()
        if bootstrap_cfg.path is not None
        else Path.cwd()
    )

    # --- Subagent spawn factory ---
    # Returns a spawn_fn closure bound to the per-agent parent context.
    # All state is captured by default-argument binding to avoid Python's
    # loop-closure gotcha (last-iteration capture).
    def _make_spawn_fn(
        parent_id: str,
        parent_cfg: object,
        parent_provider: object,
    ) -> "Callable[[str, str, int, object], str]":
        """Build the spawn callable for one parent agent session.

        The returned function runs a child AgentSession in a daemon thread,
        enforces depth/child limits, and returns the subagent's text response.

        Phase 4: event relay is wired when ``relay_fn`` is not None — the
        subagent's on_event stream is tagged with ``[sub:{agent_id}]`` and
        forwarded to the parent's terminal handler in real time.
        """
        def _spawn(
            task: str,
            subagent_id: str,
            timeout_seconds: int,
            relay_fn: object,
        ) -> str:
            # Depth and child-count limits.
            depth = get_spawn_depth(parent_id, session_store)
            if depth >= multi_agent_cfg.max_spawn_depth:
                return (
                    f"[spawn_subagent] Spawn depth limit reached "
                    f"({depth} ≥ {multi_agent_cfg.max_spawn_depth}). "
                    "Cannot spawn further subagents."
                )
            n_children = count_active_children(parent_id, session_store)
            if n_children >= multi_agent_cfg.max_children_per_agent:
                return (
                    f"[spawn_subagent] Child limit reached "
                    f"({n_children} ≥ {multi_agent_cfg.max_children_per_agent}). "
                    "Too many active subagents for this parent."
                )

            # Unique child session ID: sub-{parent}-{agenttype}-{token}
            child_id = f"sub-{parent_id}-{subagent_id}-{uuid.uuid4().hex[:8]}"
            session_store.get_or_create(child_id, parent_id=parent_id)

            # Agent definition (personality / soul).
            sub_agent_def = AGENTS.get(subagent_id) or AGENTS.get("researcher") or list(AGENTS.values())[0]

            # Provider for child: use subagent's configured model if available,
            # otherwise fall back to parent's provider.
            child_model_cfg = agents_cfg.get(subagent_id, parent_cfg)

            # Per-subagent workspace: resolves to main/ workspace if no per-agent dir.
            child_workspace = agent_workspace_root(workspace, subagent_id)
            if child_workspace is not None:
                ensure_workspace(child_workspace)

            # Read-only tool registry built BEFORE the provider so Codex
            # receives it at construction (same as openclaw's startup tool build).
            child_tools = _make_subagent_registry(root=_tool_root)

            child_provider = create_provider(
                api=child_model_cfg.provider.api,
                base_url=child_model_cfg.provider.base_url,
                api_key=child_model_cfg.provider.api_key,
                model=child_model_cfg.model.id,
                log_dir=_log_dir,
                registry=child_tools,
                approve_command=_codex_approve,
            )

            # Bootstrap context for subagent: filtered to AGENTS.md + TOOLS.md only.
            child_boot_root = child_workspace or _bootstrap_root
            child_bootstrap_context = (
                (
                    lambda root=child_boot_root: build_bootstrap_prompt_block(
                        root, bootstrap_cfg, session_type="subagent"
                    )
                )
                if bootstrap_cfg.enabled
                else None
            )

            # Compactor sized to the subagent's model context window.
            child_preserve = child_model_cfg.model.max_output_tokens + _SNIP_SAFETY_BUFFER
            child_compactor = Compactor(
                context_window=child_model_cfg.model.context_window,
                preserve_tokens=child_preserve,
            )

            # Subagents always get a fresh session UUID — they are ephemeral task
            # runners that should not carry history from a previous invocation.
            child_session_id = session_store.new_session(child_id)
            child_session = AgentSession(
                agent_id=child_id,
                session_id=child_session_id,
                agent=sub_agent_def,
                provider=child_provider,
                max_output_tokens=child_model_cfg.model.max_output_tokens,
                tools=child_tools,
                compactor=child_compactor,
                short_term=short_term,
                session_store=session_store,
                workspace_root=child_workspace,
                bootstrap_context=child_bootstrap_context,
                enable_memory_extraction=False,
                log_dir=_log_dir,
            )

            # Phase 4: relay subagent events to the parent's terminal in real time.
            # relay_fn wraps _on_event with a [sub:id] prefix on agent_name.
            def _child_on_event(event: object) -> None:
                if relay_fn is not None:
                    relay_fn(event)  # type: ignore[operator]

            result: list[str] = []
            error: list[str] = []

            def _run() -> None:
                try:
                    on_ev = _child_on_event if relay_fn is not None else None
                    text = child_session.send(task, on_event=on_ev)
                    result.append(text or "")
                except Exception as exc:
                    error.append(str(exc))

            t = threading.Thread(target=_run, daemon=True, name=f"subagent-{child_id}")
            t.start()
            t.join(timeout=timeout_seconds)

            if t.is_alive():
                return f"[spawn_subagent] Timed out after {timeout_seconds}s waiting for subagent '{subagent_id}'."
            if error:
                return f"[spawn_subagent] Subagent '{subagent_id}' error: {error[0]}"
            return result[0] if result else ""

        return _spawn

    # --- Session setup — one AgentSession per agent ---
    # _dream_session_factory is captured below inside the loop for the dreaming
    # agent and used later to start the DreamingScheduler.
    _dream_session_factory: "Callable[[], AgentSession] | None" = None
    _dream_workspace_dir: "Path | None" = None
    sessions: dict[str, AgentSession] = {}
    # Populated below, per agent — the durable capture worker (Stage One
    # Phase 2, slice C) looks up a provider by agent_id here.
    _providers_by_agent: dict[str, object] = {}
    # Populated below, per agent — the memory index watcher (Stage One
    # Phase 3, slice B) watches every agent's own workspace root.
    _agent_files_repos: dict[str, MemoryFileRepository] = {}
    for agent_id, cfg in agents_cfg.items():
        # Per-agent workspace root: resolved before building the memory service
        # and calling default_registry() so both point at the same directory
        # (and WriteDailyMemoryTool gets the correct workspace path too).
        _agent_workspace = agent_workspace_root(workspace, agent_id)
        if _agent_workspace is not None:
            ensure_workspace(_agent_workspace)
            _agent_bootstrap_root = _agent_workspace
        else:
            _agent_bootstrap_root = _bootstrap_root

        # Memory lives under the agent's own workspace root (merged Stage One
        # Phase 0 layout: memory/{topics,imports}/, dated daily files) rather
        # than the legacy shared ~/.minion-assist/memory/{agent_id}/ directory.
        # Falls back to _bootstrap_root (same fallback bootstrap context uses)
        # so memory tools are always available, even for an agent with no
        # dedicated or shared workspace directory configured.
        _agent_files_repo = MemoryFileRepository(_agent_bootstrap_root)
        _agent_files_repos[agent_id] = _agent_files_repo
        memory_service = MemoryService(_agent_files_repo, index=_memory_index, agent_id=agent_id)
        if _memory_index is not None:
            # Startup catch-up: reconcile by content hash rather than
            # unconditionally reindexing, so an unchanged file since the
            # last run costs one hash comparison, not a full rechunk.
            _reindexed = memory_service.reconcile_index()
            if _reindexed:
                print(f"  Reindexed {_reindexed} memory file(s) for agent '{agent_id}'.")

        tools = default_registry(
            memory=memory_service,
            root=_tool_root,
            bash_confirm=_console_confirm,
            bash_approval=_console_approve,
            skills=skills,
            tasks_dir=_tasks_dir,
            agent_id=agent_id,
            mcp_manager=mcp_manager,
            ask_user_fn=_console_ask_user,
            db=_db,
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
            log_dir=_log_dir,
            registry=tools,
            approve_command=_codex_approve,
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

        # Per-agent bootstrap context: uses the per-agent workspace root when
        # available so different agents can have different bootstrap files.
        # Root agents always receive the full file set (session_type="root").
        _agent_bootstrap_context = (
            (
                lambda root=_agent_bootstrap_root: build_bootstrap_prompt_block(
                    root, bootstrap_cfg, session_type="root"
                )
            )
            if bootstrap_cfg.enabled
            else None
        )

        # Session ID resolution — openclaw model:
        # Reuse the last session UUID if the agent was active within the freshness
        # window; otherwise rotate to a new UUID (new JSONL file, clean slate).
        # Old session files remain on disk for history / future resume.
        _reseed_max_chars = max(12_000, min(256_000, cfg.model.context_window * 32 // 100))
        _session_id, _reseed_context = _resolve_session_id(
            agent_id, session_store, short_term,
            reset_mode=cfg.session_reset_mode,
            reset_at_hour=cfg.session_reset_at_hour,
            idle_minutes=cfg.session_idle_minutes,
            reseed_max_chars=_reseed_max_chars,
        )
        sessions[agent_id] = AgentSession(
            agent_id=agent_id,
            session_id=_session_id,
            reseed_context=_reseed_context,
            agent=AGENTS[agent_id],
            provider=provider,
            max_output_tokens=cfg.model.max_output_tokens,
            tools=tools,
            compactor=compactor,
            short_term=short_term,
            session_store=session_store,
            soul_suffix=_skills_suffix,
            memory=memory_service,
            tasks_dir=_tasks_dir,
            enable_memory_extraction=memory_cfg.enable_extraction,
            enable_commitments=commitments_cfg.enabled,
            bootstrap_context=_agent_bootstrap_context,
            workspace_root=_agent_workspace,
            log_dir=_log_dir,
            db=_db,
            model_id=cfg.model.id,
        )
        # Tracked so the durable capture worker (Stage One Phase 2, slice C)
        # can look up the right provider per job — it isn't tied to one agent.
        _providers_by_agent[agent_id] = provider

        # Phase 4: build a relay function that tags subagent events with
        # [sub:{agent_id}] and forwards them to the parent's terminal handler.
        # dataclasses.replace() copies the event with a new agent_name so
        # the frozen event dataclasses are not mutated in place.
        def _make_relay(subagent_label: str) -> "Callable[[object], None]":
            def _relay(event: object) -> None:
                try:
                    if hasattr(event, "agent_name"):
                        event = replace(event, agent_name=f"[sub:{subagent_label}]")  # type: ignore[call-arg]
                except Exception:
                    pass
                _on_event(event)
            return _relay

        # Register SpawnSubagentTool on this agent's registry.
        # spawn_fn is bound via default-argument capture to avoid the loop closure gotcha.
        _spawn_fn = _make_spawn_fn(
            parent_id=agent_id,
            parent_cfg=cfg,
            parent_provider=provider,
        )
        _relay_fn = _make_relay(agent_id)
        tools.register(SpawnSubagentTool(spawn_fn=_spawn_fn, relay_fn=_relay_fn))

        # Capture the dream session factory for the configured dreaming agent.
        # Default-argument binding prevents the loop-closure gotcha — each variable
        # is frozen at the value it holds in this iteration.
        if dreaming_cfg.enabled and agent_id == dreaming_cfg.agent_id:
            _dream_workspace_dir = _agent_workspace or _bootstrap_root

            def _dream_factory(
                _provider=provider,
                _cfg=cfg,
                _aid=agent_id,
                _ws=_dream_workspace_dir,
                _st=short_term,
                _ss=session_store,
            ) -> AgentSession:
                """Create a fresh isolated AgentSession for one nightly dream turn."""
                from datetime import date as _date  # noqa: PLC0415
                from .dreaming import _read_dream_bootstrap  # noqa: PLC0415
                _session_id = f"dreaming-{_aid}-{_date.today().isoformat()}"
                _dream_tools = ToolRegistry()
                _dream_compactor = Compactor(
                    context_window=_cfg.model.context_window,
                    preserve_tokens=_cfg.model.max_output_tokens + _SNIP_SAFETY_BUFFER,
                )
                _dream_uuid = _ss.new_session(_session_id)
                return AgentSession(
                    agent_id=_session_id,
                    session_id=_dream_uuid,
                    agent=AGENTS[_aid],
                    provider=_provider,
                    max_output_tokens=_cfg.model.max_output_tokens,
                    tools=_dream_tools,
                    compactor=_dream_compactor,
                    short_term=_st,
                    session_store=_ss,
                    enable_memory_extraction=False,
                    bootstrap_context=lambda ws=_ws: _read_dream_bootstrap(ws),
                    log_dir=_log_dir,
                )

            _dream_session_factory = _dream_factory

    # --- Durable capture worker (optional — only when a database is configured) ---
    # Stage One Phase 2, slice C: replaces the per-turn daemon-thread extractor.
    # One worker for the whole process, not per agent — it looks up the right
    # provider per job via _providers_by_agent.
    _capture_worker = None
    if _db is not None:
        from .memory.capture_worker import CaptureWorker  # noqa: PLC0415
        # Stage One Phase 5, slice B: when a lexical index is configured,
        # every newly-extracted proposal gets indexed as searchable
        # (under corpus="proposal") right after it's recorded — see
        # capture_worker.py's module docstring for why this stays gated
        # out of normal per-turn search/injection.
        _index_proposal = _memory_index.reindex_proposal if _memory_index is not None else None
        _capture_worker = CaptureWorker(
            _db, lambda aid: _providers_by_agent[aid], index_proposal=_index_proposal
        )
        _capture_worker.start()

    # --- Durable commitment worker (optional — only when a database is
    # configured and commitments are enabled) ---
    # Stage One Phase 6, slice B. One worker for the whole process, same
    # shape as _capture_worker above (structurally identical, separate
    # table/queue — see session/db.py's module docstring for why).
    # min_due_seconds uses the configured heartbeat interval — a
    # commitment due before the next heartbeat tick could possibly check
    # for it would just sit expired-on-arrival (Task 4's "ensure the due
    # time is not immediate").
    _commitment_worker = None
    if _db is not None and commitments_cfg.enabled:
        from .memory.commitment_worker import CommitmentWorker  # noqa: PLC0415
        _commitment_worker = CommitmentWorker(
            _db, lambda aid: _providers_by_agent[aid],
            min_due_seconds=float(heartbeat_cfg.interval_seconds),
        )
        _commitment_worker.start()

    # --- Memory index watcher (optional — only when a database is configured) ---
    # Stage One Phase 3, slice B: catches on-disk edits made outside the app
    # (e.g. hand-editing MEMORY.md in a text editor). Write-path sync already
    # covers everything the app itself writes — see memory/watcher.py's
    # module docstring for why this is a narrow gap-closer, not the primary
    # sync mechanism.
    _memory_watcher = None
    if _memory_index is not None:
        try:
            from .memory.watcher import MemoryIndexWatcher  # noqa: PLC0415
            _memory_watcher = MemoryIndexWatcher(_memory_index, _agent_files_repos)
            _memory_watcher.start()
        except Exception as _watcher_exc:
            print(
                f"[minion-assist] Warning: memory index watcher unavailable ({_watcher_exc}). "
                "On-disk edits made outside the app won't be reindexed until next startup."
            )
            _memory_watcher = None

    if channels_cfg.matrix is not None:
        from .matrix.channel import MatrixChannel  # noqa: PLC0415 — optional dependency
        _matrix_channel = MatrixChannel(channels_cfg.matrix, workspace)
        _matrix_channel.start(
            sessions,
            agents_cfg=agents_cfg,
            session_store=session_store,
            mcp_manager=mcp_manager,
            skills=skills,
            short_term=short_term,
        )
        print("[matrix] Listener started.")

    # --- Heartbeat scheduler (optional) ---
    # Fires periodic background agent turns so the agent can check emails,
    # calendars, or other proactive tasks.  Disabled by default.
    _heartbeat: object = None
    if heartbeat_cfg.enabled:
        from .heartbeat import HeartbeatScheduler  # noqa: PLC0415
        _matrix_outbound = getattr(_matrix_channel, "_outbound", None) if _matrix_channel else None
        _matrix_loop = getattr(_matrix_channel, "_loop", None) if _matrix_channel else None
        _heartbeat = HeartbeatScheduler(
            config=heartbeat_cfg,
            sessions=sessions,
            matrix_outbound=_matrix_outbound,
            matrix_loop=_matrix_loop,
            db=_db,
        )
        _heartbeat.start()  # type: ignore[attr-defined]
        print(f"[heartbeat] Scheduler started (interval: {heartbeat_cfg.interval_seconds}s).")

    # --- Dreaming scheduler (optional) ---
    # Fires a nightly isolated agent turn that writes a poetic diary entry to
    # DREAMS.md.  Disabled by default; enabled via "dreaming": {"enabled": true}.
    _dreaming: object = None
    if dreaming_cfg.enabled:
        if _dream_session_factory is None or _dream_workspace_dir is None:
            print(
                f"[dreaming] Warning: agent '{dreaming_cfg.agent_id}' not found in config — "
                "dreaming disabled.",
                file=sys.stderr,
            )
        else:
            from .dreaming import DreamingScheduler  # noqa: PLC0415
            _dream_matrix_outbound = getattr(_matrix_channel, "_outbound", None) if _matrix_channel else None
            _dream_matrix_loop = getattr(_matrix_channel, "_loop", None) if _matrix_channel else None
            _dreaming = DreamingScheduler(
                cfg=dreaming_cfg,
                dream_session_factory=_dream_session_factory,
                workspace_dir=_dream_workspace_dir,
                matrix_outbound=_dream_matrix_outbound,
                matrix_loop=_dream_matrix_loop,
            )
            _dreaming.start()  # type: ignore[attr-defined]
            print(
                f"[dreaming] Scheduler started "
                f"(nightly at {dreaming_cfg.hour:02d}:{dreaming_cfg.minute:02d} "
                f"{dreaming_cfg.timezone})."
            )

    # --- Memory consolidation scheduler (optional) ---
    # Stage One Phase 5, slice D, Task 9: deliberately a separate schedule
    # from `dreaming` above, even though both use the same daily wall-clock
    # shape — this drafts consolidation previews (never applies/promotes
    # anything), keeping `memory consolidate list` populated with fresh
    # drafts. Requires a database, a lexical index, and the configured
    # agent to actually be running (its provider/files repo must already
    # exist from the per-agent loop above).
    _memory_consolidation: object = None
    if memory_consolidation_cfg.enabled:
        if _db is None or _memory_index is None:
            print(
                "[memory-consolidation] Warning: no database/index configured — "
                "consolidation scheduler disabled.",
                file=sys.stderr,
            )
        elif memory_consolidation_cfg.agent_id not in _providers_by_agent:
            print(
                f"[memory-consolidation] Warning: agent "
                f"'{memory_consolidation_cfg.agent_id}' not found in config — "
                "consolidation scheduler disabled.",
                file=sys.stderr,
            )
        else:
            from .memory.consolidation import MemoryConsolidator  # noqa: PLC0415
            from .memory.consolidation_scheduler import MemoryConsolidationScheduler  # noqa: PLC0415

            _consolidator = MemoryConsolidator(
                _db,
                _memory_index,
                _agent_files_repos[memory_consolidation_cfg.agent_id],
                _providers_by_agent[memory_consolidation_cfg.agent_id],
                agent_id=memory_consolidation_cfg.agent_id,
            )
            _memory_consolidation = MemoryConsolidationScheduler(
                memory_consolidation_cfg, _db, _memory_index, _consolidator
            )
            _memory_consolidation.start()  # type: ignore[attr-defined]
            print(
                f"[memory-consolidation] Scheduler started "
                f"(daily at {memory_consolidation_cfg.hour:02d}:{memory_consolidation_cfg.minute:02d} "
                f"{memory_consolidation_cfg.timezone})."
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
    prompt_reader = PromptReader(
        workspace / "prompt_history.txt",
        completion_items=build_completion_items(agents_cfg, skills),
    )

    # Install SIGTERM handler (POSIX only) so `kill <pid>` exits cleanly.
    # History is already persisted from the last turn's finally block.
    def _sigterm_handler(signum: int, frame: object) -> None:
        print("\nShutdown signal received. Goodbye.")
        # Close MCP sessions before exiting so subprocess stdio transports shut down
        if mcp_manager is not None:
            mcp_manager.close_sync()
        if _capture_worker is not None:
            _capture_worker.stop()
        if _commitment_worker is not None:
            _commitment_worker.stop()
        if _memory_watcher is not None:
            _memory_watcher.stop()
        sys.exit(0)

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sigterm_handler)

    # --- Console event handler ---
    # Tracks whether a streaming response is currently in progress so that the
    # correct newline is printed at the end and FinalAnswer doesn't re-print
    # text that was already streamed token-by-token.
    _streaming_active = False

    # ANSI escape codes for pink agent name in the terminal.
    _PINK = "\033[38;5;213m"   # bright pink (256-colour palette)
    _RESET = "\033[0m"          # resets all colour/style attributes

    def _pink(name: str) -> str:
        """Wrap *name* in ANSI pink colour codes for terminal display."""
        return f"{_PINK}{name}{_RESET}"

    def _on_event(event: object) -> None:
        """Render one agent runtime event as terminal output.

        Handles all event types emitted by the runner, compactor, and
        (indirectly) the bash tool confirm flow.
        """
        nonlocal _streaming_active

        if isinstance(event, StreamingStarted):
            print(f"\n{_pink(event.agent_name)}: ", end="", flush=True)
            _streaming_active = True

        elif isinstance(event, TokenStreamed):
            print(event.token, end="", flush=True)

        else:
            was_streaming = _streaming_active
            if _streaming_active:
                print()
                _streaming_active = False

            if isinstance(event, ThoughtEmitted):
                print(f"\n{_pink(event.agent_name)}: {event.text}")
            elif isinstance(event, FinalAnswer):
                if event.text and not was_streaming:
                    print(f"\n{_pink(event.agent_name)}: {event.text}")
            elif isinstance(event, ToolCalled):
                print(f"  [tool: {event.name}({event.args})]")
            elif isinstance(event, MaxRoundsReached):
                print(f"\n{_pink(event.agent_name)}: {event.message}")
            elif isinstance(event, CompactionStarted):
                print("\n  Compacting session history...")
            elif isinstance(event, CompactionFailed):
                print(f"\n  [Warning] Compaction failed: {event.error}")
            elif isinstance(event, MemoryFlushed) and event.status == "failed":
                print(f"\n  [Warning] Pre-compaction memory flush failed: {event.detail}")

    # --- Voice mode (--voice flag) ---
    # Placed here so _on_event is already defined when VoiceSession is built.
    # VoiceSession.run() blocks until Ctrl+C; we then fall through to cleanup.
    if _args.voice:
        from .voice.session import build_voice_session  # noqa: PLC0415
        _target_agent_id = next(iter(sessions))
        # Resolve the bootstrap root for the target agent (same logic as the
        # session setup loop above).  _agent_bootstrap_root is not reliable here
        # because it holds the last loop iteration's value, not the target agent's.
        _voice_agent_workspace = agent_workspace_root(workspace, _target_agent_id)
        _voice_bootstrap_root = _voice_agent_workspace if _voice_agent_workspace is not None else _bootstrap_root
        _voice_session = build_voice_session(
            agent_session=sessions[_target_agent_id],
            voice_config=voice_cfg,
            on_event=_on_event,
            bootstrap_root=_voice_bootstrap_root,
        )
        try:
            _voice_session.run()
        except KeyboardInterrupt:
            print("\n[voice] Goodbye.")
        finally:
            if _matrix_channel is not None:
                _matrix_channel.stop()
            if mcp_manager is not None:
                mcp_manager.close_sync()
            if _capture_worker is not None:
                _capture_worker.stop()
            if _commitment_worker is not None:
                _commitment_worker.stop()
            if _memory_watcher is not None:
                _memory_watcher.stop()
        return

    # --- REPL loop ---
    print("Mini-Minion ready. Type /help for commands, /quit to quit.")
    for agent_id, cfg_entry in agents_cfg.items():
        if cfg_entry.route_prefix:
            agent_name = AGENTS[agent_id].name
            print(f"  {cfg_entry.route_prefix} <message>  -> {agent_name}")

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
            # If routing stripped a prefix, only the routed payload can be a command.
            # Example:
            #   "/research /new"  -> command "/new" targeting researcher
            #   "/research hello" -> normal chat to researcher, not unknown "/research"
            # If no route matched, examine the full input so "/new", "/help", and
            # mistyped slash commands still behave normally.
            _route_matched = message != user_input
            cmd_text = message if _route_matched else user_input

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
                        skills=skills,
                        short_term=short_term,
                    )
                    result = dispatch_command(ctx)
                    if result.handled:
                        if result.message:
                            print(result.message)
                        # /switch sets activate_agent_id to switch the default agent.
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
        if _matrix_channel is not None:
            _matrix_channel.stop()
        # Always close MCP sessions on exit — cleans up subprocess stdio transports.
        if mcp_manager is not None:
            mcp_manager.close_sync()
        if _capture_worker is not None:
            _capture_worker.stop()
        if _commitment_worker is not None:
            _commitment_worker.stop()
        if _memory_watcher is not None:
            _memory_watcher.stop()


if __name__ == "__main__":
    main()

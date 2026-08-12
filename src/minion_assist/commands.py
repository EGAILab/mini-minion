"""Slash command dispatcher for the minion-assist interactive REPL.

Commands are handled BEFORE sending text to the LLM. The dispatcher returns
a CommandResult indicating whether the input was consumed (handled=True) so
minion.py can skip normal agent routing.

Supported commands:
  /help, /commands           — show command list
  /quit, /exit               — exit the REPL
  /new [all]                 — clear conversation history for the active (or all) agent(s)
  /compact                   — manually compact the active agent's history
  /status                    — show current agent/model info
  /agents                    — list all known agents with turn counts and last-active time
  /session [N|uuid-prefix]   — list past session files for the active agent; restore one by index or prefix
  /rename [N] <name>         — give a session a descriptive name (current session if N omitted)
  /switch [agent_id]         — switch the default routing target to the given agent
  /diagnose                  — show provider configuration and API key status for all agents
  /mcp-reload                — close and reconnect all MCP servers, refresh tool adapters
  /mcp-list                  — list connected MCP servers and their available tools
  /mcp-enable <server>       — reconnect a disabled MCP server
  /mcp-disable <server>      — disconnect an MCP server for this session
  /plan                      — enable read-only mode (agent can plan but not write/execute)
  /auto                      — disable read-only mode (agent runs with full tool access)
  /providers                 — list configured LLM provider and model info for all agents
  /provider test [agent_id]  — test provider connectivity with a minimal API call
  /audit [n]                 — show last n permission decisions from the audit log
  /fork <new_id>             — fork active session history to a new session
  /export [--md|--html] path — export session transcript to a file
  /plugin list               — list all registered tool names

Route-aware targeting:
  /research /new     — target the researcher agent specifically
  /research hello    — NOT a command (falls through to normal routing)
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class CommandSpec:
    """Metadata for one built-in slash command.

    Used to generate /help output and (later) autocomplete suggestions.
    """
    name: str          # canonical name e.g. "/new"
    description: str   # one-line description shown in /help
    arg_hint: str = ""                   # e.g. "[all]" — shown after name in help
    aliases: tuple[str, ...] = field(default_factory=tuple)  # e.g. ("/clear", "/reset")


@dataclass
class CommandContext:
    """Carries everything a command handler needs without exposing internals.

    Created by minion.py per-REPL-iteration and passed to dispatch_command().
    """
    raw: str           # the full original input string
    command: str       # normalized command token e.g. "/new"
    args: str          # everything after the command token, stripped
    target_agent_id: str              # which agent this command targets
    sessions: dict                    # dict[str, AgentSession] — keyed by agent_id
    agents_cfg: dict                  # dict[str, AgentModelConfig] from config
    session_store: object = None      # SessionStore instance (optional, for /agents)
    mcp_manager: object = None        # McpClientManager instance (optional, for /mcp-reload)
    skills: dict | None = None        # dict[str, SkillInfo] loaded at startup
    short_term: object = None         # ShortTermMemory instance (for /session)
    db: object = None                 # SessionDB instance (optional, for /delete-session's
                                       # cross-store cleanup — MEM-GAP-003)
    worker_health: dict | None = None # dict[str, WorkerHealth] — one entry per background
                                       # worker/scheduler actually running (for /status deep,
                                       # MEM-GAP-016)


@dataclass
class CommandResult:
    """What the command handler wants minion.py to do next."""
    handled: bool        # True = input was consumed, skip normal routing
    should_exit: bool = False    # True = break out of the REPL loop
    message: str | None = None  # text to print to the user (None = print nothing)
    activate_agent_id: str | None = None  # non-None → switch the REPL's active agent
    new_session_id: str | None = None  # non-None → /session <arg> switched the active
                                        # session's id in place (R2-GAP-004); the caller
                                        # is responsible for persisting that anywhere it
                                        # tracks a session_id binding outside this
                                        # session's own in-memory state — the CLI REPL
                                        # has nothing to persist here (SessionStore was
                                        # already updated by switch_session() itself),
                                        # but MatrixMessageHandler uses this to update
                                        # MatrixRoomSessionManager's (room_id, agent_id)
                                        # binding, or a restart would silently revert
                                        # the switch


# ---------------------------------------------------------------------------
# Built-in command catalog
# ---------------------------------------------------------------------------
# Keeping this as a list (not a dict) preserves display order in /help output.

BUILTIN_COMMANDS: list[CommandSpec] = [
    CommandSpec(
        name="/help",
        description="Show this command list.",
        aliases=("/commands",),
    ),
    CommandSpec(
        name="/quit",
        description="Exit minion-assist.",
        aliases=("/exit",),
    ),
    CommandSpec(
        name="/new",
        description="Start a fresh conversation for the active agent (long-term memory is kept).",
        arg_hint="[all]",
        aliases=("/clear", "/reset"),
    ),
    CommandSpec(
        name="/compact",
        description="Manually summarize the active agent's conversation history.",
    ),
    CommandSpec(
        name="/status",
        description="Show active agent, model, and history information.",
        arg_hint="[deep]",
    ),
    CommandSpec(
        name="/agents",
        description="List all known agents with turn counts and last-active timestamps.",
    ),
    CommandSpec(
        name="/session",
        description="List past session files for the active agent; restore one by index or UUID prefix.",
        arg_hint="[N | uuid-prefix]",
    ),
    CommandSpec(
        name="/rename",
        description="Give the current session (or session N from /session) a descriptive name.",
        arg_hint="[N] <name>",
    ),
    CommandSpec(
        name="/switch",
        description="Switch the default routing target to the given agent for the current session.",
        arg_hint="[agent_id]",
    ),
    CommandSpec(
        name="/diagnose",
        description="Show provider configuration and API key status for all configured agents.",
    ),
    CommandSpec(
        name="/mcp-reload",
        description="Close and reconnect all MCP servers, then refresh the tool adapters in every session.",
    ),
    CommandSpec(
        name="/mcp-list",
        description="List connected MCP servers and their available tools.",
    ),
    CommandSpec(
        name="/plan",
        description="Enable read-only mode — agent can reason and plan but not write files or run commands.",
    ),
    CommandSpec(
        name="/auto",
        description="Disable read-only mode — restore full tool access (write, bash, git commit).",
    ),
    CommandSpec(
        name="/providers",
        description="Show LLM provider and model configuration for all configured agents.",
    ),
    CommandSpec(
        name="/provider",
        description="Test provider connectivity for an agent with a minimal API call.",
        arg_hint="test [agent_id]",
    ),
    CommandSpec(
        name="/audit",
        description="Show recent permission decisions (allowed/denied tool calls) from the audit log.",
        arg_hint="[n]",
    ),
    CommandSpec(
        name="/fork",
        description="Fork the active session's history into a new session with a given ID.",
        arg_hint="<new_id>",
    ),
    CommandSpec(
        name="/export",
        description="Export the active session's conversation history to a file.",
        arg_hint="[--md|--html] <path>",
    ),
    CommandSpec(
        name="/mcp-enable",
        description="Reconnect a disabled MCP server and refresh its tool adapters.",
        arg_hint="<server>",
    ),
    CommandSpec(
        name="/mcp-disable",
        description="Disconnect an MCP server and mark it disabled for this session.",
        arg_hint="<server>",
    ),
    CommandSpec(
        name="/plugin",
        description="Manage plugins. Use 'list' to show all registered tools and their sources.",
        arg_hint="list",
    ),
    CommandSpec(
        name="/skills",
        description="List loaded skills available to the agent.",
        arg_hint="[name]",
    ),
    CommandSpec(
        name="/delete-session",
        description="Permanently delete a past session file for the active agent.",
        arg_hint="[N | uuid-prefix]",
    ),
]


# ---------------------------------------------------------------------------
# Plugin command registry
# ---------------------------------------------------------------------------
# Populated by plugins.py when a manifest declares a "commands" section.
# Each entry is a (CommandSpec, handler) pair.  The spec drives /help output;
# the handler is called by dispatch_command() after all built-in commands.
#
# This list is module-level mutable state — intentionally so, because plugins
# register commands at import time and the REPL loop reads the list on each
# dispatch.  Tests should clear this list in teardown to avoid cross-test
# pollution.

_PLUGIN_COMMAND_REGISTRY: list[tuple["CommandSpec", Callable]] = []


def register_plugin_command(spec: "CommandSpec", handler: Callable) -> None:
    """Register a plugin-provided slash command.

    Called by ``plugins.py`` when loading a manifest's ``"commands"`` section.
    The registered command is then available in the REPL alongside built-in
    commands and appears in ``/help`` output.

    Args:
        spec:    A :class:`CommandSpec` describing the command (name, description,
                 optional arg_hint and aliases).
        handler: A callable with signature ``(ctx: CommandContext) -> CommandResult``.
                 Receives the same :class:`CommandContext` as built-in handlers.
    """
    # Avoid duplicates: if a command with the same name was registered before
    # (e.g. the plugin was reloaded), replace the old entry rather than adding
    # a second one that would shadow the first.
    for i, (existing_spec, _) in enumerate(_PLUGIN_COMMAND_REGISTRY):
        if existing_spec.name == spec.name:
            _PLUGIN_COMMAND_REGISTRY[i] = (spec, handler)
            return
    _PLUGIN_COMMAND_REGISTRY.append((spec, handler))


def _all_names(spec: CommandSpec) -> tuple[str, ...]:
    """Return the canonical name plus all aliases for a command spec."""
    return (spec.name,) + spec.aliases


def format_help(agents_cfg: dict) -> str:
    """Generate a human-readable /help text from BUILTIN_COMMANDS and plugin commands."""
    lines = ["Built-in commands:"]
    for spec in BUILTIN_COMMANDS:
        names = ", ".join(_all_names(spec))
        hint = f" {spec.arg_hint}" if spec.arg_hint else ""
        lines.append(f"  {names}{hint}")
        lines.append(f"      {spec.description}")

    # Append plugin commands when any are registered.
    if _PLUGIN_COMMAND_REGISTRY:
        lines.append("")
        lines.append("Plugin commands:")
        for spec, _ in _PLUGIN_COMMAND_REGISTRY:
            names = ", ".join(_all_names(spec))
            hint = f" {spec.arg_hint}" if spec.arg_hint else ""
            lines.append(f"  {names}{hint}")
            lines.append(f"      {spec.description}")

    # Show route-targeting examples if there are multiple agents configured
    # with route prefixes (e.g. /research).
    prefixes = [(aid, cfg.route_prefix) for aid, cfg in agents_cfg.items() if cfg.route_prefix]
    if prefixes:
        lines.append("")
        lines.append("Route-targeted commands (prefix + command):")
        for aid, prefix in prefixes:
            lines.append(f"  {prefix} /new      — clear {aid} session only")
        lines.append("")
        lines.append("Route prefixes:")
        for aid, prefix in prefixes:
            lines.append(f"  {prefix} <message>  → routes to {aid} agent")

    return "\n".join(lines)


def build_completion_items(
    agents_cfg: dict,
    skills: dict | None = None,
) -> list[tuple[str, str]]:
    """Build prompt completion entries for commands, route prefixes, and skills.

    Returns:
        list of ``(insert_text, display_meta)`` tuples consumed by PromptReader.
    """
    items: list[tuple[str, str]] = []

    for spec in BUILTIN_COMMANDS:
        hint = f" {spec.arg_hint}" if spec.arg_hint else ""
        meta = f"{hint} - {spec.description}" if hint else spec.description
        for name in _all_names(spec):
            items.append((name, meta))

    for spec, _ in _PLUGIN_COMMAND_REGISTRY:
        hint = f" {spec.arg_hint}" if spec.arg_hint else ""
        meta = f"{hint} - {spec.description}" if hint else spec.description
        for name in _all_names(spec):
            items.append((name, meta))

    for aid, cfg in agents_cfg.items():
        if cfg.route_prefix:
            items.append((cfg.route_prefix, f"route to {aid} agent"))

    for name, info in sorted((skills or {}).items()):
        description = getattr(info, "description", "") or "loaded skill"
        items.append((f"/skills {name}", description))

    # Keep first occurrence when aliases/plugins collide.
    deduped: dict[str, str] = {}
    for value, meta in items:
        deduped.setdefault(value, meta)
    return [(value, deduped[value]) for value in sorted(deduped)]


def parse_command(text: str) -> tuple[str, str] | None:
    """Parse a slash command token and its arguments from user input.

    Returns (command_token, args) when input starts with "/" and looks like
    a command (the first token is the command, the rest is args).
    Returns None when input does NOT start with "/" (plain message).

    Examples:
        "/help"          → ("/help", "")
        "/new all"       → ("/new", "all")
        "/compact"       → ("/compact", "")
        "hello world"    → None            ← plain message, not a command
        "/research news" → ("/research", "news")   ← NOTE: this COULD be a route

    IMPORTANT — routing vs commands:
    ---------------------------------
    This function ONLY parses syntax. It does not know about agent routing.
    "/research news" parses to ("/research", "news") but minion.py knows
    "/research " is a route prefix and passes only "news" to the researcher
    session — it never reaches dispatch_command() as a command.

    The caller (minion.py) is responsible for deciding whether a parsed
    command token is a known route prefix (normal message) or a real command.
    """
    text = text.strip()
    if not text.startswith("/"):
        return None
    parts = text.split(None, 1)
    command = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""
    return command, args


# Canonical set of background workers/schedulers minion.py may construct
# (MEM-GAP-016) — listed explicitly (not just "whatever's in worker_health")
# so /status deep can show "not running" for a worker that isn't
# configured, rather than silently omitting it. Excludes the per-agent
# "session_writes:{agent_id}" entries (MEM-GAP-007), which are derived from
# ctx.agents_cfg instead, since the exact keys depend on which agents are
# configured.
_KNOWN_WORKER_NAMES = (
    "capture_worker",
    "commitment_worker",
    "message_embedding_worker",
    "image_caption_worker",
    "memory_watcher",
    "memory_reconciliation",
    "memory_consolidation",
    "knowledge_digest",
    "memory_retention",
    "dreaming",
    "heartbeat",
)


def _format_duration(age: float | None) -> str:
    """Render a duration in seconds (e.g. 'time since X') as a short string, or 'never'."""
    if age is None:
        return "never"
    age = max(0.0, age)
    if age < 60:
        return f"{int(age)}s"
    if age < 3600:
        return f"{int(age / 60)}m"
    if age < 86400:
        return f"{int(age / 3600)}h"
    return f"{int(age / 86400)}d"


def _format_age(ts: float | None, now: float) -> str:
    """Render a past epoch-seconds timestamp as a short relative age, or 'never'."""
    if ts is None:
        return "never"
    return f"{_format_duration(now - ts)} ago"


def _format_worker_line(name: str, health: object, now: float) -> str:
    """Format one /status deep worker row — 'not running' if health is None."""
    if health is None:
        return f"    {name}: not running"
    snap = health.snapshot()
    failures = snap["consecutive_failures"]
    status = f"{failures} consecutive failure(s)" if failures else "ok"
    error_note = f"  last_error={snap['last_error']!r}" if snap["last_error"] else ""
    return (
        f"    {name}: {status}  "
        f"last_poll={_format_age(snap['last_poll_at'], now)}  "
        f"last_success={_format_age(snap['last_success_at'], now)}"
        f"{error_note}"
    )


def _format_deep_status(ctx: CommandContext) -> list[str]:
    """Build the extra lines ``/status deep`` appends (MEM-GAP-016).

    Two independent sources, kept clearly labeled since they answer
    different questions:

    - Worker health (:class:`~minion_assist.worker_health.WorkerHealth`)
      is in-process-only liveness — only meaningful for *this* running
      process, never persisted.
    - Queue lag (:meth:`~minion_assist.session.db.SessionDB.queue_lag_summary`)
      and index summary (:meth:`~minion_assist.memory.service.MemoryService.deep_status`)
      are plain database facts, visible from any process with a connection
      (including the separate ``minion-assist memory status --deep`` CLI).
    """
    now = time.time()
    lines: list[str] = ["  Workers:"]
    worker_health = ctx.worker_health or {}
    for name in _KNOWN_WORKER_NAMES:
        lines.append(_format_worker_line(name, worker_health.get(name), now))
    # Per-agent session-write health (MEM-GAP-007) — keys depend on which
    # agents are configured, so derived from ctx.agents_cfg rather than a
    # fixed name list.
    for agent_id in ctx.agents_cfg:
        name = f"session_writes:{agent_id}"
        lines.append(_format_worker_line(name, worker_health.get(name), now))
    # Per-agent degraded-mode search health (MEM-GAP-008) — same reasoning:
    # only present for an agent whose MemoryService actually has an index
    # to fall back from, so absence already means "no index configured,"
    # not "unhealthy."
    for agent_id in ctx.agents_cfg:
        name = f"memory_search:{agent_id}"
        lines.append(_format_worker_line(name, worker_health.get(name), now))
    # Per-agent degraded-mode extraction health (MEM-GAP-013) — inverse of
    # the two above: only present for an agent with NO database configured
    # (the only time the daemon-thread extractor in memory/extractor.py
    # ever runs at all).
    for agent_id in ctx.agents_cfg:
        name = f"memory_extractor:{agent_id}"
        lines.append(_format_worker_line(name, worker_health.get(name), now))

    if ctx.db is None:
        lines.append("  Database: not configured — no queue lag / index data available.")
        return lines

    lines.append("  Queues and index (per agent):")
    for aid in ctx.agents_cfg:
        try:
            lag = ctx.db.queue_lag_summary(aid)
        except Exception as exc:
            lines.append(f"    [{aid}] queue lag unavailable: {exc}")
            continue
        capture = lag["capture"]
        commitment = lag["commitment"]
        message_embedding = lag["message_embedding"]
        lines.append(
            f"    [{aid}] capture_pending={capture['pending_count']}"
            f" (oldest {_format_duration(capture['oldest_pending_age_s'])})"
            f"  commitment_pending={commitment['pending_count']}"
            f" (oldest {_format_duration(commitment['oldest_pending_age_s'])})"
            f"  message_embedding_pending={message_embedding['pending_count']}"
            f" (oldest {_format_duration(message_embedding['oldest_pending_age_s'])})"
        )
        # R2-GAP-003/R2-GAP-014: running_count/oldest_running_age_s and
        # failed_count exist on `lag` since Phase 0, but were never actually
        # printed here — a stuck-running or permanently-failed job was
        # invisible in /status deep even though queue_lag_summary() already
        # reported it. Only shown when nonzero, so the common healthy case
        # doesn't get noisier.
        stuck_bits = []
        for lane_name, lane in (("capture", capture), ("commitment", commitment),
                                 ("message_embedding", message_embedding)):
            if lane["running_count"]:
                stuck_bits.append(
                    f"{lane_name}_running={lane['running_count']}"
                    f" (oldest {_format_duration(lane['oldest_running_age_s'])})"
                )
            if lane["failed_count"]:
                stuck_bits.append(f"{lane_name}_failed={lane['failed_count']}")
        if stuck_bits:
            lines.append(f"      {'  '.join(stuck_bits)}")
        session = ctx.sessions.get(aid)
        memory = getattr(session, "memory", None)
        if memory is not None:
            try:
                index_status = memory.deep_status()
            except Exception as exc:
                lines.append(f"      index status unavailable: {exc}")
                continue
            if index_status is None:
                lines.append("      index: not configured")
            else:
                lines.append(
                    f"      index: {index_status['total_chunks']} chunks, "
                    f"{index_status['file_count']} files, "
                    f"last_indexed={_format_age(index_status['last_indexed_at'], now)}"
                )
        # R2-GAP-014: how far behind this agent's embedding coverage is
        # overall (every session, not just what reconciliation would
        # enqueue in one bounded pass) — None when no vector lane is
        # configured, in which case there's nothing meaningful to report.
        try:
            coverage = ctx.db.embedding_coverage_summary(aid)
        except Exception as exc:
            lines.append(f"      embedding coverage unavailable: {exc}")
            continue
        if coverage is not None and coverage["missing_count"]:
            lines.append(
                f"      embedding coverage: {coverage['missing_count']} message(s) "
                f"missing an embedding under {coverage['model_identity']!r}"
            )
    return lines


def _format_history(messages: list[dict], max_content: int = 600) -> str:
    """Render a message list as a readable conversation transcript.

    Each turn is shown as ``User:`` / ``Assistant:`` with content truncated
    to *max_content* characters.  Tool calls and non-text content blocks are
    summarised rather than printed in full.
    """
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # LLM message content can be either a plain string OR a list of typed
        # content blocks (e.g. {"type": "text", "text": "..."} for text,
        # {"type": "tool_use", ...} for tool calls).  When it's a list we
        # need to extract just the text parts manually.
        if isinstance(content, list):
            # Gather text from all plain-text blocks; skip image / tool blocks.
            text_parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            # Count tool-related blocks so we can mention them without dumping
            # their full JSON (which is noisy and rarely useful to read).
            tool_count = sum(
                1 for b in content
                if isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result")
            )
            content = "\n".join(text_parts)
            if tool_count:
                # Append a compact note so the reader knows tool activity happened.
                suffix = f"\n[{tool_count} tool call(s) not shown]"
                content = (content + suffix).strip()

        # Guard against unexpected non-string values (shouldn't normally happen).
        if not isinstance(content, str):
            content = ""
        content = content.strip()

        # Truncate very long messages so the terminal output stays readable.
        if len(content) > max_content:
            content = content[:max_content] + " … [truncated]"

        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        # Roles like "tool" are intentionally skipped — they're included in the
        # tool_count summary above, not printed as separate turns.

    # Separate each turn with a blank line for readability.
    return "\n\n".join(lines)


def dispatch_command(ctx: CommandContext) -> CommandResult:
    """Execute a slash command and return what minion.py should do next.

    Returns CommandResult(handled=False) when the command token does not match
    any built-in, so the caller can fall through to normal agent routing.
    """
    cmd = ctx.command.lower()

    # Iterate over the command catalog to find a matching handler.
    # Using a loop over BUILTIN_COMMANDS (rather than a dict) keeps the
    # catalog as the single source of truth for both /help and dispatch.
    for spec in BUILTIN_COMMANDS:
        if cmd not in _all_names(spec):
            continue

        # --- /help, /commands ---
        if spec.name == "/help":
            return CommandResult(handled=True, message=format_help(ctx.agents_cfg))

        # --- /quit, /exit ---
        if spec.name == "/quit":
            return CommandResult(handled=True, should_exit=True, message="Goodbye.")

        # --- /new, /clear, /reset ---
        if spec.name == "/new":
            if ctx.args.strip().lower() == "all":
                # Clear every agent's history at once.
                for session in ctx.sessions.values():
                    session.reset()
                return CommandResult(
                    handled=True,
                    message="Cleared conversation history for all agents.",
                )
            else:
                # Clear only the currently targeted agent.
                session = ctx.sessions.get(ctx.target_agent_id)
                if session is None:
                    return CommandResult(handled=True, message=f"Unknown agent: {ctx.target_agent_id}")
                session.reset()
                return CommandResult(
                    handled=True,
                    message=f"Cleared conversation history for {ctx.target_agent_id}.",
                )

        # --- /compact ---
        if spec.name == "/compact":
            session = ctx.sessions.get(ctx.target_agent_id)
            if session is None:
                return CommandResult(handled=True, message=f"Unknown agent: {ctx.target_agent_id}")
            changed = session.compact_now()
            if changed:
                return CommandResult(handled=True, message=f"History compacted for {ctx.target_agent_id}.")
            else:
                return CommandResult(handled=True, message="Nothing to compact (history too short or already compact).")

        # --- /status ---
        if spec.name == "/status":
            from .config import streaming as _streaming_cfg
            lines = ["Status:"]
            lines.append(f"  Active agent: {ctx.target_agent_id}")
            for aid, cfg in ctx.agents_cfg.items():
                session = ctx.sessions.get(aid)
                # len() on history gives the raw message count across all roles.
                hist_len = len(session.history) if session else 0
                prefix = f" (route: {cfg.route_prefix})" if cfg.route_prefix else ""
                lines.append(
                    f"  [{aid}]{prefix}  model={cfg.model.id}  history={hist_len} messages"
                )
            lines.append(f"  Streaming: {'on' if _streaming_cfg.chat_mode else 'off'}")
            if ctx.args.strip().lower() == "deep":
                lines.extend(_format_deep_status(ctx))
            return CommandResult(handled=True, message="\n".join(lines))

        # --- /agents ---
        if spec.name == "/agents":
            if ctx.session_store is None:
                return CommandResult(handled=True, message="Session store not available.")
            sessions_list = ctx.session_store.list_sessions()
            if not sessions_list:
                return CommandResult(handled=True, message="No sessions recorded yet.")
            # Sort by most recently active first.
            sessions_list = sorted(sessions_list, key=lambda s: s.last_active, reverse=True)
            lines = [f"Sessions ({len(sessions_list)}):"]
            for s in sessions_list:
                # Trim microseconds from ISO timestamp for readability.
                last = s.last_active[:19].replace("T", " ")
                lines.append(
                    f"  [{s.agent_id}]  turns={s.turn_count}  last_active={last}"
                )
            return CommandResult(handled=True, message="\n".join(lines))

        # --- /session ---
        if spec.name == "/session":
            if ctx.short_term is None:
                return CommandResult(handled=True, message="Short-term memory not available.")
            agent_id = ctx.target_agent_id
            paths = list(reversed(ctx.short_term.list_sessions(agent_id)))
            if not paths:
                return CommandResult(
                    handled=True,
                    message=f"No session history found for '{agent_id}'.",
                )
            current_id = getattr(ctx.sessions.get(agent_id), "session_id", None)

            if not ctx.args:
                # ---- bare /session — print the session listing ----
                label = "session" if len(paths) == 1 else "sessions"
                lines = [f"History for {agent_id} ({len(paths)} {label}):"]
                for i, p in enumerate(paths, 1):
                    # Convert file modification time to a readable "YYYY-MM-DD HH:MM" string.
                    mtime = datetime.fromtimestamp(p.stat().st_mtime)
                    ts = mtime.strftime("%Y-%m-%d %H:%M")
                    # Show only the first 8 chars of the UUID — long enough to be
                    # unique in practice and short enough to read comfortably.
                    uuid_hint = p.stem[:8]
                    # Count non-blank lines in the JSONL file; each line = one message.
                    msg_count = sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())
                    # Mark the currently loaded session with * so the user knows
                    # where they are before picking a different one.
                    is_current = p.stem == current_id
                    marker = "*" if is_current else " "
                    # Prefer the human-readable name if one was set via /rename;
                    # otherwise fall back to the first user message as a preview.
                    name = ctx.short_term.get_name(agent_id, p.stem) if ctx.short_term else None
                    if name:
                        label = f"  [{name}]"
                    else:
                        label = ""
                        try:
                            # Scan lines until we find the first user message.
                            for ln in p.read_text(encoding="utf-8").splitlines():
                                m = json.loads(ln.strip())
                                if m.get("role") == "user":
                                    content = m.get("content", "")
                                    # Content may be a list of blocks — extract the first text block.
                                    if isinstance(content, list):
                                        content = next(
                                            (b.get("text", "") for b in content
                                             if isinstance(b, dict) and b.get("type") == "text"),
                                            "",
                                        )
                                    if content:
                                        label = f'  "{content[:50]}"'
                                    break
                        except Exception:
                            pass  # silently skip corrupt JSONL — session still appears in list
                    lines.append(f"{marker} [{i}] {ts}  msgs={msg_count}  {uuid_hint}{label}")
                lines.append("Use /session <N> or /session <uuid-prefix> to restore.")
                return CommandResult(handled=True, message="\n".join(lines))

            # ---- /session <arg> — load a specific session ----
            arg = ctx.args.strip()
            target_path = None
            try:
                # Try interpreting arg as a 1-based index into the listing above.
                idx = int(arg)
                if 1 <= idx <= len(paths):
                    target_path = paths[idx - 1]
                else:
                    return CommandResult(
                        handled=True,
                        message=f"Index {idx} out of range (1–{len(paths)}).",
                    )
            except ValueError:
                # Not a number — try it as a UUID prefix (first N chars of the session UUID).
                matches = [p for p in paths if p.stem.startswith(arg)]
                if not matches:
                    return CommandResult(handled=True, message=f"No session matching '{arg}'.")
                if len(matches) > 1:
                    # More than one session starts with this prefix — ask for more chars.
                    return CommandResult(
                        handled=True,
                        message=f"Ambiguous prefix '{arg}' matches {len(matches)} sessions.",
                    )
                target_path = matches[0]

            session = ctx.sessions.get(agent_id)
            if session is None:
                return CommandResult(handled=True, message=f"Agent '{agent_id}' not found.")
            # p.stem is the UUID without the .jsonl extension.
            target_id = target_path.stem
            if target_id == current_id:
                return CommandResult(handled=True, message="Already on that session.")
            # Swap history — loads messages from disk and updates the session store
            # so the next turn writes to the restored session file.
            session.switch_session(target_id)
            history = session.history
            # Use the session name in the header if one exists, otherwise show UUID prefix.
            name = ctx.short_term.get_name(agent_id, target_id) if ctx.short_term else None
            label = f"[{name}]" if name else target_id[:8]
            header = f"=== Session {label} ({len(history)} messages) ==="
            transcript = _format_history(history)
            body = transcript if transcript else "(empty session)"
            return CommandResult(
                handled=True,
                message=f"{header}\n\n{body}",
                new_session_id=target_id,
            )

        # --- /rename ---
        if spec.name == "/rename":
            if ctx.short_term is None:
                return CommandResult(handled=True, message="Short-term memory not available.")
            if not ctx.args:
                return CommandResult(
                    handled=True,
                    message="Usage: /rename <name>  or  /rename <N> <name>",
                )
            agent_id = ctx.target_agent_id
            # Split into at most two parts so multi-word names work:
            #   "/rename 2 Auth debugging"  → tokens = ["2", "Auth debugging"]
            #   "/rename Auth debugging"    → tokens = ["Auth", "debugging"] (no int prefix)
            tokens = ctx.args.split(None, 1)
            target_id: str | None = None
            name_part: str = ctx.args
            if len(tokens) >= 2:
                try:
                    # If the first token is a whole number, it's a /session index.
                    idx = int(tokens[0])
                    paths = list(reversed(ctx.short_term.list_sessions(agent_id)))
                    if not paths:
                        return CommandResult(handled=True, message=f"No sessions found for '{agent_id}'.")
                    if not (1 <= idx <= len(paths)):
                        return CommandResult(
                            handled=True,
                            message=f"Index {idx} out of range (1–{len(paths)}).",
                        )
                    # .stem strips the ".jsonl" extension to get the raw UUID.
                    target_id = paths[idx - 1].stem
                    name_part = tokens[1]
                except ValueError:
                    pass  # first token is not a number — treat whole args as the name
            if target_id is None:
                # No index prefix — rename the session that is currently active.
                session = ctx.sessions.get(agent_id)
                target_id = getattr(session, "session_id", None)
                if target_id is None:
                    return CommandResult(handled=True, message=f"Agent '{agent_id}' not found.")
            name_part = name_part.strip()
            if not name_part:
                return CommandResult(handled=True, message="Name cannot be empty.")
            # Write a sidecar .name file next to the session's .jsonl file.
            ctx.short_term.set_name(agent_id, target_id, name_part)
            return CommandResult(
                handled=True,
                message=f"Session {target_id[:8]} renamed to '{name_part}'.",
            )

        # --- /switch ---
        if spec.name == "/switch":
            # Resolve the target agent: use the argument if given, fall back to
            # the currently targeted agent (no-op switch, still prints confirmation).
            target = ctx.args.strip().lower() or ctx.target_agent_id
            if target not in ctx.sessions:
                known = ", ".join(sorted(ctx.sessions))
                return CommandResult(
                    handled=True,
                    message=f"Unknown agent '{target}'. Known agents: {known}",
                )
            # Reload the target agent's history from disk before switching.
            # This ensures /switch reflects any turns that happened in previous
            # process runs and picks up the latest state after /new was used.
            ctx.sessions[target].reload()
            return CommandResult(
                handled=True,
                activate_agent_id=target,
                message=(
                    f"Switched active agent to '{target}' and reloaded history from disk. "
                    f"Messages now go to {target} by default."
                ),
            )

        # --- /diagnose ---
        if spec.name == "/diagnose":
            lines = ["Provider diagnostics:"]
            for aid, cfg in ctx.agents_cfg.items():
                has_key = bool(cfg.provider.api_key) or cfg.provider.api == "lmstudio"
                endpoint = cfg.provider.base_url or "(SDK default)"
                key_hint = (
                    f"{cfg.provider.name.upper()}_API_KEY"
                    if cfg.provider.api != "lmstudio"
                    else "no key needed (local)"
                )
                status = "OK" if has_key else f"MISSING ({key_hint})"
                lines.append(f"\n  [{aid}]")
                lines.append(f"    Provider : {cfg.provider.name}  ({cfg.provider.api})")
                lines.append(f"    Model    : {cfg.model.id}")
                lines.append(f"    Endpoint : {endpoint}")
                lines.append(f"    Auth     : {status}")
            return CommandResult(handled=True, message="\n".join(lines))

        # --- /mcp-reload ---
        if spec.name == "/mcp-reload":
            if ctx.mcp_manager is None:
                return CommandResult(
                    handled=True,
                    message="No MCP manager configured. Add 'mcp.servers' to config.json first.",
                )
            # Close and reconnect all MCP servers.
            ctx.mcp_manager.reconnect_all_sync()

            # Refresh each session's tool adapters with the new connection state.
            refreshed = 0
            for session in ctx.sessions.values():
                refreshed += session.refresh_mcp_adapters(ctx.mcp_manager)

            # Build a status summary.
            lines = ["MCP servers reconnected:"]
            for status in ctx.mcp_manager.list_statuses():
                state_str = "OK" if status.state == "connected" else f"FAILED: {status.detail}"
                lines.append(f"  [{status.name}]: {state_str}")
            lines.append(f"Refreshed {refreshed} MCP tool adapter(s) across all sessions.")
            return CommandResult(handled=True, message="\n".join(lines))

        # --- /mcp-list ---
        if spec.name == "/mcp-list":
            if ctx.mcp_manager is None:
                return CommandResult(
                    handled=True,
                    message="No MCP manager configured. Add 'mcp.servers' to config.json first.",
                )
            statuses = ctx.mcp_manager.list_statuses()
            if not statuses:
                return CommandResult(handled=True, message="No MCP servers configured.")
            lines = ["MCP servers:"]
            for status in statuses:
                state_str = "connected" if status.state == "connected" else f"FAILED: {status.detail}"
                lines.append(f"\n  [{status.name}]  {state_str}")
            # List tools available from connected servers.
            tools = ctx.mcp_manager.list_tools()
            if tools:
                lines.append(f"\nAvailable tools ({len(tools)}):")
                for t in tools:
                    lines.append(f"  {t.name}")
            else:
                lines.append("\nNo tools available (no servers connected).")
            return CommandResult(handled=True, message="\n".join(lines))

        # --- /plan ---
        if spec.name == "/plan":
            # Enable read-only mode on the active agent's PermissionPolicy.
            # This blocks write/bash/commit tools while still allowing reads and reasoning.
            session = ctx.sessions.get(ctx.target_agent_id)
            if session is None:
                return CommandResult(handled=True, message=f"Unknown agent: {ctx.target_agent_id}")
            policy = session.registry.policy
            if policy is None:
                return CommandResult(handled=True, message="No permission policy configured for this agent.")
            policy.read_only_mode = True
            return CommandResult(
                handled=True,
                message=(
                    f"Read-only mode enabled for '{ctx.target_agent_id}'. "
                    "The agent can read and reason but write/bash/git-commit are blocked. "
                    "Use /auto to re-enable full access."
                ),
            )

        # --- /auto ---
        if spec.name == "/auto":
            # Disable read-only mode, restoring full tool access.
            session = ctx.sessions.get(ctx.target_agent_id)
            if session is None:
                return CommandResult(handled=True, message=f"Unknown agent: {ctx.target_agent_id}")
            policy = session.registry.policy
            if policy is None:
                return CommandResult(handled=True, message="No permission policy configured for this agent.")
            policy.read_only_mode = False
            return CommandResult(
                handled=True,
                message=(
                    f"Read-only mode disabled for '{ctx.target_agent_id}'. "
                    "Full tool access restored (write, bash, git commit)."
                ),
            )

        # --- /providers ---
        if spec.name == "/providers":
            lines = ["Configured providers:"]
            for aid, cfg in ctx.agents_cfg.items():
                lines.append(f"\n  [{aid}]")
                lines.append(f"    Provider : {cfg.provider.name}  (api={cfg.provider.api})")
                lines.append(f"    Model    : {cfg.model.id}")
                lines.append(f"    Context  : {cfg.model.context_window:,} tokens")
                lines.append(f"    Max out  : {cfg.model.max_output_tokens:,} tokens")
                endpoint = cfg.provider.base_url or "(SDK default)"
                lines.append(f"    Endpoint : {endpoint}")
            return CommandResult(handled=True, message="\n".join(lines))

        # --- /provider test ---
        if spec.name == "/provider":
            # Only "test" sub-command is supported for now.
            subcmd, _, sub_args = ctx.args.partition(" ")
            if subcmd.strip().lower() != "test":
                return CommandResult(
                    handled=True,
                    message="Usage: /provider test [agent_id]",
                )
            target = sub_args.strip().lower() or ctx.target_agent_id
            if target not in ctx.sessions:
                known = ", ".join(sorted(ctx.sessions))
                return CommandResult(
                    handled=True,
                    message=f"Unknown agent '{target}'. Known agents: {known}",
                )
            import time as _time
            session = ctx.sessions[target]
            _start_ms = _time.monotonic()
            try:
                # Minimal "hello" call — 5 output tokens is enough to confirm connectivity.
                resp = session.provider.chat(
                    system="You are a connectivity test.",
                    messages=[{"role": "user", "content": "Reply with one word: ok"}],
                    tools=[],
                    max_tokens=5,
                )
                _elapsed = int((_time.monotonic() - _start_ms) * 1000)
                _cfg_entry = ctx.agents_cfg.get(target)
                model_id = _cfg_entry.model.id if _cfg_entry else "(unknown)"
                return CommandResult(
                    handled=True,
                    message=(
                        f"Provider test for '{target}': OK ({_elapsed} ms)\n"
                        f"  Model: {model_id}\n"
                        f"  Response: {resp.text!r}"
                    ),
                )
            except Exception as exc:
                _elapsed = int((_time.monotonic() - _start_ms) * 1000)
                return CommandResult(
                    handled=True,
                    message=f"Provider test for '{target}': FAILED ({_elapsed} ms)\n  Error: {exc}",
                )

        # --- /audit ---
        if spec.name == "/audit":
            session = ctx.sessions.get(ctx.target_agent_id)
            if session is None:
                return CommandResult(handled=True, message=f"Unknown agent: {ctx.target_agent_id}")
            policy = session.registry.policy
            if policy is None:
                return CommandResult(handled=True, message="No permission policy configured.")
            # Parse optional count argument; default to 20.
            try:
                count = int(ctx.args.strip()) if ctx.args.strip() else 20
            except ValueError:
                count = 20
            entries = policy.audit_log.entries[-count:]
            if not entries:
                return CommandResult(handled=True, message="Audit log is empty.")
            lines = [f"Audit log (last {len(entries)} entries):"]
            for e in entries:
                ts = e.timestamp[:19].replace("T", " ")
                reason = f"  ({e.reason})" if e.reason else ""
                lines.append(f"  [{ts}] {e.decision:7s} {e.tool_name}: {e.args_repr}{reason}")
            return CommandResult(handled=True, message="\n".join(lines))

        # --- /fork ---
        if spec.name == "/fork":
            new_id = ctx.args.strip()
            if not new_id:
                return CommandResult(
                    handled=True, message="Usage: /fork <new_session_id>"
                )
            if new_id in ctx.sessions:
                return CommandResult(
                    handled=True,
                    message=f"Session '{new_id}' already exists. Choose a different ID.",
                )
            session = ctx.sessions.get(ctx.target_agent_id)
            if session is None:
                return CommandResult(handled=True, message=f"Unknown agent: {ctx.target_agent_id}")
            session.fork(new_id)
            return CommandResult(
                handled=True,
                message=(
                    f"Forked '{ctx.target_agent_id}' → '{new_id}' "
                    f"({len(session.history)} messages copied).\n"
                    f"Use /switch {new_id} to switch to the forked session."
                ),
            )

        # --- /export ---
        if spec.name == "/export":
            session = ctx.sessions.get(ctx.target_agent_id)
            if session is None:
                return CommandResult(handled=True, message=f"Unknown agent: {ctx.target_agent_id}")
            parts = ctx.args.split()
            fmt = "md"
            path_parts = parts
            if parts and parts[0] in ("--md", "--html"):
                fmt = parts[0][2:]   # strip "--"
                path_parts = parts[1:]
            if not path_parts:
                return CommandResult(
                    handled=True,
                    message="Usage: /export [--md|--html] <file_path>",
                )
            from pathlib import Path as _Path
            out_path = _Path(path_parts[0]).expanduser()
            content = session.export(format=fmt)
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(content, encoding="utf-8")
                return CommandResult(
                    handled=True,
                    message=f"Exported {len(session.history)} messages to {out_path}",
                )
            except OSError as exc:
                return CommandResult(handled=True, message=f"Export failed: {exc}")

        # --- /mcp-enable ---
        if spec.name == "/mcp-enable":
            if ctx.mcp_manager is None:
                return CommandResult(
                    handled=True,
                    message="No MCP manager configured. Add 'mcp.servers' to config.json first.",
                )
            server_name = ctx.args.strip()
            if not server_name:
                return CommandResult(handled=True, message="Usage: /mcp-enable <server_name>")
            try:
                ctx.mcp_manager.connect_server_sync(server_name)
                # Refresh tool adapters in all sessions.
                refreshed = sum(
                    s.refresh_mcp_adapters(ctx.mcp_manager) for s in ctx.sessions.values()
                )
                status = next(
                    (st for st in ctx.mcp_manager.list_statuses() if st.name == server_name),
                    None,
                )
                state_str = (status.state if status else "unknown")
                return CommandResult(
                    handled=True,
                    message=(
                        f"MCP server '{server_name}': {state_str}\n"
                        f"Refreshed {refreshed} tool adapter(s)."
                    ),
                )
            except Exception as exc:
                return CommandResult(handled=True, message=f"Failed to enable '{server_name}': {exc}")

        # --- /mcp-disable ---
        if spec.name == "/mcp-disable":
            if ctx.mcp_manager is None:
                return CommandResult(
                    handled=True,
                    message="No MCP manager configured. Add 'mcp.servers' to config.json first.",
                )
            server_name = ctx.args.strip()
            if not server_name:
                return CommandResult(handled=True, message="Usage: /mcp-disable <server_name>")
            try:
                ctx.mcp_manager.disconnect_server_sync(server_name)
                # Remove adapters for the disabled server from all sessions.
                for session in ctx.sessions.values():
                    session.registry.unregister_prefix(f"mcp__{server_name}__")
                return CommandResult(
                    handled=True,
                    message=f"MCP server '{server_name}' disconnected and its tools removed.",
                )
            except Exception as exc:
                return CommandResult(handled=True, message=f"Failed to disable '{server_name}': {exc}")

        # --- /plugin ---
        if spec.name == "/plugin":
            subcmd = ctx.args.strip().lower()
            if subcmd != "list":
                return CommandResult(handled=True, message="Usage: /plugin list")
            session = ctx.sessions.get(ctx.target_agent_id)
            if session is None:
                return CommandResult(handled=True, message=f"Unknown agent: {ctx.target_agent_id}")
            tool_names = sorted(
                t.schema.name for t in session.registry._tools.values()
            )
            lines = [f"Registered tools for '{ctx.target_agent_id}' ({len(tool_names)}):"]
            for name in tool_names:
                lines.append(f"  {name}")
            return CommandResult(handled=True, message="\n".join(lines))

        # --- /skills ---
        if spec.name == "/skills":
            skills = ctx.skills or {}
            if not skills:
                return CommandResult(handled=True, message="No skills loaded.")

            name = ctx.args.strip()
            if name:
                skill = skills.get(name)
                if skill is None:
                    known = ", ".join(sorted(skills))
                    return CommandResult(
                        handled=True,
                        message=f"Unknown skill '{name}'. Loaded skills: {known}",
                    )
                description = getattr(skill, "description", "") or "(no description)"
                path = getattr(skill, "path", "")
                return CommandResult(
                    handled=True,
                    message=f"{name}\n  {description}\n  Path: {path}",
                )

            lines = [f"Loaded skills ({len(skills)}):"]
            for skill_name, skill in sorted(skills.items()):
                description = getattr(skill, "description", "") or "(no description)"
                lines.append(f"  {skill_name} - {description}")
            return CommandResult(handled=True, message="\n".join(lines))

        # --- /delete-session ---
        if spec.name == "/delete-session":
            if ctx.short_term is None:
                return CommandResult(handled=True, message="Short-term memory not available.")
            agent_id = ctx.target_agent_id
            paths = list(reversed(ctx.short_term.list_sessions(agent_id)))
            if not paths:
                return CommandResult(
                    handled=True,
                    message=f"No session history found for '{agent_id}'.",
                )
            current_id = getattr(ctx.sessions.get(agent_id), "session_id", None)

            if not ctx.args:
                # Bare /delete-session — show the listing so the user can pick one.
                label = "session" if len(paths) == 1 else "sessions"
                lines = [f"History for {agent_id} ({len(paths)} {label}):"]
                for i, p in enumerate(paths, 1):
                    mtime = datetime.fromtimestamp(p.stat().st_mtime)
                    ts = mtime.strftime("%Y-%m-%d %H:%M")
                    uuid_hint = p.stem[:8]
                    msg_count = sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())
                    is_current = p.stem == current_id
                    marker = "*" if is_current else " "
                    name = ctx.short_term.get_name(agent_id, p.stem) if ctx.short_term else None
                    if name:
                        label = f"  [{name}]"
                    else:
                        label = ""
                        try:
                            for ln in p.read_text(encoding="utf-8").splitlines():
                                m = json.loads(ln.strip())
                                if m.get("role") == "user":
                                    content = m.get("content", "")
                                    if isinstance(content, list):
                                        content = next(
                                            (b.get("text", "") for b in content
                                             if isinstance(b, dict) and b.get("type") == "text"),
                                            "",
                                        )
                                    if content:
                                        label = f'  "{content[:50]}"'
                                    break
                        except Exception:
                            pass
                    lines.append(f"{marker} [{i}] {ts}  msgs={msg_count}  {uuid_hint}{label}")
                lines.append("Use /delete-session <N> or /delete-session <uuid-prefix> to delete.")
                return CommandResult(handled=True, message="\n".join(lines))

            # --- /delete-session <arg> — resolve the target then delete ---
            arg = ctx.args.strip()
            target_path = None
            try:
                idx = int(arg)
                if 1 <= idx <= len(paths):
                    target_path = paths[idx - 1]
                else:
                    return CommandResult(
                        handled=True,
                        message=f"Index {idx} out of range (1–{len(paths)}).",
                    )
            except ValueError:
                matches = [p for p in paths if p.stem.startswith(arg)]
                if not matches:
                    return CommandResult(handled=True, message=f"No session matching '{arg}'.")
                if len(matches) > 1:
                    return CommandResult(
                        handled=True,
                        message=f"Ambiguous prefix '{arg}' matches {len(matches)} sessions.",
                    )
                target_path = matches[0]

            target_id = target_path.stem
            if target_id == current_id:
                return CommandResult(
                    handled=True,
                    message=(
                        "Cannot delete the active session. "
                        "Use /session <N> to switch to a different session first."
                    ),
                )

            # Fetch name before deleting so we can include it in the confirmation.
            display_name = ctx.short_term.get_name(agent_id, target_id)
            hint = f"[{display_name}]" if display_name else target_id[:8]

            if ctx.db is None:
                # Local/basic profile — the JSONL file is this session's
                # only store, a single atomic operation with nothing to
                # resume if it fails.
                ctx.short_term.delete_session(agent_id, target_id)
                return CommandResult(handled=True, message=f"Deleted session {hint}.")

            # Cross-store cleanup (MEM-GAP-003), now tombstone-tracked
            # (R2-GAP-007): the JSONL file above was always this session's
            # only source of truth for the local/basic profile, but a
            # PostgreSQL-mirrored session also has messages, capture/
            # commitment jobs, proposals, and commitments living in
            # SessionDB, plus indexed evidence in PostgresMemoryIndex —
            # three independent operations against three stores, none of
            # which can share a transaction. start_deletion_tombstone()
            # records intent *before* anything is touched, and each phase
            # marks itself done as it completes, so a failure partway
            # through (e.g. a transient DB connection drop) leaves a
            # durable, discoverable record — see
            # `minion-assist memory verify-deletions` — instead of a
            # session that's half-deleted with no trace of what's left.
            ctx.db.start_deletion_tombstone(agent_id, target_id)

            ctx.short_term.delete_session(agent_id, target_id)
            ctx.db.mark_deletion_jsonl_done(agent_id, target_id)

            try:
                pg_result = ctx.db.delete_session(agent_id, target_id)
            except Exception as exc:
                # Surface this loudly rather than silently claiming a
                # complete deletion — the JSONL file is already gone, but
                # PostgreSQL records may still exist and be searchable.
                # Re-running /delete-session on this target won't work
                # (it's no longer in the listing), hence the CLI pointer.
                return CommandResult(
                    handled=True,
                    message=(
                        f"Deleted session {hint} locally. WARNING: database cleanup failed "
                        f"({exc}) — PostgreSQL records may still exist. Run "
                        "'minion-assist memory verify-deletions --retry' to finish it."
                    ),
                )
            proposal_ids = (pg_result.get("proposal_ids") or []) if pg_result is not None else []
            ctx.db.mark_deletion_db_done(agent_id, target_id, proposal_ids)
            # pg_result is None: this session_id was never mirrored to
            # PostgreSQL for this agent (e.g. it predates the database
            # being configured) — nothing more to clean up, not an error.

            forget_note = ""
            if proposal_ids:
                # Only SessionDB knows which proposal ids it just deleted;
                # only the matching agent's MemoryService can clean up
                # their indexed chunks, draft previews, and
                # knowledge-graph evidence citations (a separate
                # class/connection).
                agent_session = ctx.sessions.get(agent_id)
                memory = getattr(agent_session, "memory", None)
                if memory is not None:
                    try:
                        memory.forget_proposals(proposal_ids)
                    except Exception as exc:
                        forget_note = (
                            f" (evidence cleanup for {len(proposal_ids)} proposal(s) failed: "
                            f"{exc} — run 'minion-assist memory verify-deletions --retry' "
                            "to finish it)"
                        )
                    else:
                        ctx.db.mark_deletion_evidence_done(agent_id, target_id)
                else:
                    # No live MemoryService in this dispatch context (e.g.
                    # the target agent has no active session right now) —
                    # can't clean up evidence here; the tombstone stays
                    # incomplete for `verify-deletions --retry` to finish.
                    forget_note = (
                        f" ({len(proposal_ids)} proposal(s) still need evidence cleanup — "
                        "run 'minion-assist memory verify-deletions --retry' to finish it)"
                    )
            else:
                ctx.db.mark_deletion_evidence_done(agent_id, target_id)

            pg_note = (
                f" Also removed {pg_result['messages']} database message(s) "
                f"and {len(proposal_ids)} proposal(s).{forget_note}"
                if pg_result is not None else ""
            )
            return CommandResult(handled=True, message=f"Deleted session {hint}.{pg_note}")

    # --- Plugin commands ---
    # Check plugin-registered commands after all built-ins.  Plugins can shadow
    # built-ins only if they register a command with the same name, which is
    # intentional (a plugin can override /status with a richer implementation).
    for plugin_spec, plugin_handler in _PLUGIN_COMMAND_REGISTRY:
        if cmd in _all_names(plugin_spec):
            try:
                return plugin_handler(ctx)
            except Exception as exc:
                return CommandResult(
                    handled=True,
                    message=f"Error in plugin command '{plugin_spec.name}': {exc}",
                )

    # No built-in or plugin command matched — let the caller decide what to do.
    # Returning handled=False means the input falls through to normal routing.
    return CommandResult(handled=False)

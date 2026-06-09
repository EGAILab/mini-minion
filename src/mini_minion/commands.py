"""Slash command dispatcher for the mini-minion interactive REPL.

Commands are handled BEFORE sending text to the LLM. The dispatcher returns
a CommandResult indicating whether the input was consumed (handled=True) so
minion.py can skip normal agent routing.

Supported commands:
  /help, /commands   — show command list
  /quit, /exit       — exit the REPL
  /new [all]         — clear conversation history for the active (or all) agent(s)
  /compact           — manually compact the active agent's history
  /status            — show current agent/model info
  /sessions          — list all known sessions with turn counts and last-active time
  /resume [agent_id] — switch the default routing target to the given agent
  /diagnose          — show provider configuration and API key status for all agents
  /mcp-reload        — close and reconnect all MCP servers, refresh tool adapters
  /mcp-list          — list connected MCP servers and their available tools
  /plan              — enable read-only mode (agent can plan but not write/execute)
  /auto              — disable read-only mode (agent runs with full tool access)
  /providers         — list configured LLM provider and model info for all agents

Route-aware targeting:
  /research /new     — target the researcher agent specifically
  /research hello    — NOT a command (falls through to normal routing)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


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
    session_store: object = None      # SessionStore instance (optional, for /sessions)
    mcp_manager: object = None        # McpClientManager instance (optional, for /mcp-reload)


@dataclass
class CommandResult:
    """What the command handler wants minion.py to do next."""
    handled: bool        # True = input was consumed, skip normal routing
    should_exit: bool = False    # True = break out of the REPL loop
    message: str | None = None  # text to print to the user (None = print nothing)
    activate_agent_id: str | None = None  # non-None → switch the REPL's active agent


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
        description="Exit mini-minion.",
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
    ),
    CommandSpec(
        name="/sessions",
        description="List all known agent sessions with turn counts and last-active timestamps.",
    ),
    CommandSpec(
        name="/resume",
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
            return CommandResult(handled=True, message="\n".join(lines))

        # --- /sessions ---
        if spec.name == "/sessions":
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

        # --- /resume ---
        if spec.name == "/resume":
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
            # This ensures /resume reflects any turns that happened in previous
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

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

Route-aware targeting:
  /research /new     — target the researcher agent specifically
  /research hello    — NOT a command (falls through to normal routing)
"""

from __future__ import annotations

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


@dataclass
class CommandResult:
    """What the command handler wants minion.py to do next."""
    handled: bool        # True = input was consumed, skip normal routing
    should_exit: bool = False    # True = break out of the REPL loop
    message: str | None = None  # text to print to the user (None = print nothing)


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
]


def _all_names(spec: CommandSpec) -> tuple[str, ...]:
    """Return the canonical name plus all aliases for a command spec."""
    return (spec.name,) + spec.aliases


def format_help(agents_cfg: dict) -> str:
    """Generate a human-readable /help text from BUILTIN_COMMANDS."""
    lines = ["Built-in commands:"]
    for spec in BUILTIN_COMMANDS:
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

    # No built-in command matched — let the caller decide what to do.
    # Returning handled=False means the input falls through to normal routing.
    return CommandResult(handled=False)

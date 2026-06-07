"""Tests for commands.py — slash command parsing and dispatch."""

import pytest
from unittest.mock import MagicMock, patch

from mini_minion.commands import (
    CommandContext,
    CommandResult,
    CommandSpec,
    BUILTIN_COMMANDS,
    _all_names,
    format_help,
    parse_command,
    dispatch_command,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(history_len: int = 0, compact_result: bool = True) -> MagicMock:
    """Return a mock AgentSession with controllable history and compact_now."""
    session = MagicMock()
    # session.history is a property returning a list — simulate it with a list.
    session.history = [{"role": "user", "content": f"msg{i}"} for i in range(history_len)]
    session.reset = MagicMock()
    session.compact_now = MagicMock(return_value=compact_result)
    return session


def _make_agents_cfg(include_route: bool = False) -> dict:
    """Build a minimal agents_cfg dict matching the shape commands.py expects.

    Uses MagicMock objects that expose .route_prefix and .model.id.
    """
    main_cfg = MagicMock()
    main_cfg.route_prefix = None
    main_cfg.model.id = "test-model"

    cfg = {"main": main_cfg}

    if include_route:
        research_cfg = MagicMock()
        research_cfg.route_prefix = "/research"
        research_cfg.model.id = "research-model"
        cfg["researcher"] = research_cfg

    return cfg


def _make_ctx(
    command: str,
    args: str = "",
    target: str = "main",
    sessions: dict | None = None,
    agents_cfg: dict | None = None,
) -> CommandContext:
    """Build a CommandContext with sensible defaults for testing."""
    if sessions is None:
        sessions = {"main": _make_session()}
    if agents_cfg is None:
        agents_cfg = _make_agents_cfg()
    return CommandContext(
        raw=f"{command} {args}".strip(),
        command=command,
        args=args,
        target_agent_id=target,
        sessions=sessions,
        agents_cfg=agents_cfg,
    )


# ---------------------------------------------------------------------------
# CommandSpec / BUILTIN_COMMANDS
# ---------------------------------------------------------------------------

def test_builtin_commands_has_expected_names():
    """BUILTIN_COMMANDS must define all expected canonical command names."""
    names = {spec.name for spec in BUILTIN_COMMANDS}
    assert "/help" in names
    assert "/quit" in names
    assert "/new" in names
    assert "/compact" in names
    assert "/status" in names


def test_all_names_includes_canonical_and_aliases():
    """`_all_names` must return the canonical name plus its aliases."""
    spec = CommandSpec(name="/quit", description="exit", aliases=("/exit",))
    assert _all_names(spec) == ("/quit", "/exit")


# ---------------------------------------------------------------------------
# parse_command
# ---------------------------------------------------------------------------

def test_parse_command_returns_none_for_normal_text():
    """`parse_command` must return None for input that does not start with /."""
    assert parse_command("hello world") is None


def test_parse_command_returns_none_for_empty_string():
    assert parse_command("") is None


def test_parse_command_extracts_command_and_empty_args():
    """/help with no args must yield ('/help', '')."""
    result = parse_command("/help")
    assert result == ("/help", "")


def test_parse_command_extracts_command_and_args():
    """/new all must yield ('/new', 'all')."""
    result = parse_command("/new all")
    assert result == ("/new", "all")


def test_parse_command_lowercases_command_token():
    """/HELP must be normalized to /help."""
    result = parse_command("/HELP")
    assert result is not None
    assert result[0] == "/help"


def test_parse_command_strips_surrounding_whitespace():
    """Leading/trailing whitespace must be stripped before parsing."""
    result = parse_command("  /quit  ")
    assert result is not None
    assert result[0] == "/quit"


# ---------------------------------------------------------------------------
# format_help
# ---------------------------------------------------------------------------

def test_format_help_contains_all_command_names():
    """/help output must mention every canonical command and its aliases."""
    text = format_help(_make_agents_cfg())
    for spec in BUILTIN_COMMANDS:
        assert spec.name in text


def test_format_help_contains_descriptions():
    """Every command description must appear in /help output."""
    text = format_help(_make_agents_cfg())
    for spec in BUILTIN_COMMANDS:
        assert spec.description in text


def test_format_help_shows_route_prefixes_when_present():
    """/help must list route-targeting examples when agents have route_prefix."""
    text = format_help(_make_agents_cfg(include_route=True))
    assert "/research" in text


def test_format_help_no_route_section_when_no_prefixes():
    """When no agents have a route_prefix, the route section must not appear."""
    text = format_help(_make_agents_cfg(include_route=False))
    assert "Route prefixes" not in text


# ---------------------------------------------------------------------------
# dispatch_command — /help
# ---------------------------------------------------------------------------

def test_dispatch_help_returns_handled_true():
    result = dispatch_command(_make_ctx("/help"))
    assert result.handled is True


def test_dispatch_help_message_contains_commands():
    """/help result message must list command names."""
    result = dispatch_command(_make_ctx("/help"))
    assert result.message is not None
    assert "/new" in result.message
    assert "/quit" in result.message


def test_dispatch_commands_alias_works():
    """/commands alias must behave identically to /help."""
    result = dispatch_command(_make_ctx("/commands"))
    assert result.handled is True
    assert result.message is not None


# ---------------------------------------------------------------------------
# dispatch_command — /quit
# ---------------------------------------------------------------------------

def test_dispatch_quit_returns_should_exit_true():
    result = dispatch_command(_make_ctx("/quit"))
    assert result.handled is True
    assert result.should_exit is True


def test_dispatch_quit_has_goodbye_message():
    result = dispatch_command(_make_ctx("/quit"))
    assert result.message is not None
    assert "Goodbye" in result.message


def test_dispatch_exit_alias_works():
    """/exit alias must trigger the same exit as /quit."""
    result = dispatch_command(_make_ctx("/exit"))
    assert result.handled is True
    assert result.should_exit is True


# ---------------------------------------------------------------------------
# dispatch_command — /new
# ---------------------------------------------------------------------------

def test_dispatch_new_clears_active_session():
    """/new must call reset() on the targeted agent's session."""
    session = _make_session()
    sessions = {"main": session}
    ctx = _make_ctx("/new", target="main", sessions=sessions)
    result = dispatch_command(ctx)
    session.reset.assert_called_once()
    assert result.handled is True


def test_dispatch_new_all_clears_all_sessions():
    """/new all must call reset() on every session."""
    s1 = _make_session()
    s2 = _make_session()
    sessions = {"main": s1, "researcher": s2}
    ctx = _make_ctx("/new", args="all", target="main", sessions=sessions)
    result = dispatch_command(ctx)
    s1.reset.assert_called_once()
    s2.reset.assert_called_once()
    assert result.handled is True


def test_dispatch_new_message_includes_agent_id():
    """/new confirmation message must name the cleared agent."""
    session = _make_session()
    ctx = _make_ctx("/new", target="main", sessions={"main": session})
    result = dispatch_command(ctx)
    assert "main" in result.message


def test_dispatch_new_all_message_mentions_all():
    result = dispatch_command(_make_ctx("/new", args="all", sessions={"main": _make_session()}))
    assert "all" in result.message.lower()


def test_dispatch_clear_alias_works():
    """/clear alias must behave like /new."""
    session = _make_session()
    result = dispatch_command(_make_ctx("/clear", sessions={"main": session}))
    session.reset.assert_called_once()
    assert result.handled is True


# ---------------------------------------------------------------------------
# dispatch_command — /compact
# ---------------------------------------------------------------------------

def test_dispatch_compact_calls_compact_now():
    session = _make_session(compact_result=True)
    ctx = _make_ctx("/compact", sessions={"main": session})
    result = dispatch_command(ctx)
    session.compact_now.assert_called_once()
    assert result.handled is True


def test_dispatch_compact_reports_changed():
    """When compact_now returns True (history changed), message must confirm."""
    session = _make_session(compact_result=True)
    ctx = _make_ctx("/compact", sessions={"main": session})
    result = dispatch_command(ctx)
    assert "compacted" in result.message.lower()


def test_dispatch_compact_reports_nothing_to_compact():
    """When compact_now returns False (nothing changed), message must say so."""
    session = _make_session(compact_result=False)
    ctx = _make_ctx("/compact", sessions={"main": session})
    result = dispatch_command(ctx)
    assert "nothing" in result.message.lower() or "too short" in result.message.lower()


# ---------------------------------------------------------------------------
# dispatch_command — /status
# ---------------------------------------------------------------------------

def test_dispatch_status_is_handled():
    # Patch at the config module level because /status imports streaming lazily
    # inside the function body with `from .config import streaming as _streaming_cfg`.
    with patch("mini_minion.config.streaming") as mock_streaming:
        mock_streaming.chat_mode = False
        result = dispatch_command(_make_ctx("/status"))
    assert result.handled is True


def test_dispatch_status_message_contains_agent_id():
    """Status output must mention the active agent."""
    with patch("mini_minion.config.streaming") as mock_streaming:
        mock_streaming.chat_mode = True
        result = dispatch_command(_make_ctx("/status", target="main"))
    assert "main" in result.message


def test_dispatch_status_message_contains_model():
    """Status output must include the model ID."""
    with patch("mini_minion.config.streaming") as mock_streaming:
        mock_streaming.chat_mode = False
        result = dispatch_command(_make_ctx("/status"))
    assert "test-model" in result.message


def test_dispatch_status_shows_streaming_on():
    """Status must report 'on' when streaming.chat_mode is True."""
    with patch("mini_minion.config.streaming") as mock_streaming:
        mock_streaming.chat_mode = True
        result = dispatch_command(_make_ctx("/status"))
    assert "on" in result.message


# ---------------------------------------------------------------------------
# dispatch_command — unknown command
# ---------------------------------------------------------------------------

def test_dispatch_unknown_command_returns_handled_false():
    """/typo must return handled=False so the caller can warn the user."""
    result = dispatch_command(_make_ctx("/typo"))
    assert result.handled is False


def test_dispatch_unknown_command_no_should_exit():
    """An unrecognized command must never trigger an exit."""
    result = dispatch_command(_make_ctx("/doesnotexist"))
    assert result.should_exit is False

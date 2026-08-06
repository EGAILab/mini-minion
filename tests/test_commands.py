"""Tests for commands.py — slash command parsing and dispatch."""

import time

import pytest
from unittest.mock import MagicMock, patch

from minion_assist.commands import (
    CommandContext,
    CommandResult,
    CommandSpec,
    BUILTIN_COMMANDS,
    _all_names,
    build_completion_items,
    format_help,
    parse_command,
    dispatch_command,
)
from minion_assist.skills import SkillInfo


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
    # Matches AgentSession.memory's real default (None when unconfigured) —
    # without this, a bare MagicMock auto-creates a truthy child mock for
    # any attribute access, which would silently pass /status deep's
    # `memory is not None` check with fake data.
    session.memory = None
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
    db: object = None,
    worker_health: dict | None = None,
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
        db=db,
        worker_health=worker_health,
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


def test_completion_items_include_commands_routes_and_skills(tmp_path):
    agents_cfg = _make_agents_cfg(include_route=True)
    skill = SkillInfo(
        name="interview",
        description="Interview preparation.",
        path=tmp_path / "SKILL.md",
        content="",
    )

    items = build_completion_items(agents_cfg, {"interview": skill})
    values = {value for value, _ in items}

    assert "/help" in values
    assert "/research" in values
    assert "/skills interview" in values


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


def test_dispatch_skills_lists_loaded_skills(tmp_path):
    skill = SkillInfo(
        name="interview",
        description="Interview preparation.",
        path=tmp_path / "SKILL.md",
        content="",
    )
    ctx = _make_ctx("/skills")
    ctx.skills = {"interview": skill}

    result = dispatch_command(ctx)

    assert result.handled is True
    assert "interview" in result.message
    assert "Interview preparation" in result.message


def test_dispatch_skills_shows_single_skill_detail(tmp_path):
    skill = SkillInfo(
        name="interview",
        description="Interview preparation.",
        path=tmp_path / "SKILL.md",
        content="",
    )
    ctx = _make_ctx("/skills", args="interview")
    ctx.skills = {"interview": skill}

    result = dispatch_command(ctx)

    assert result.handled is True
    assert str(skill.path) in result.message


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
    with patch("minion_assist.config.streaming") as mock_streaming:
        mock_streaming.chat_mode = False
        result = dispatch_command(_make_ctx("/status"))
    assert result.handled is True


def test_dispatch_status_message_contains_agent_id():
    """Status output must mention the active agent."""
    with patch("minion_assist.config.streaming") as mock_streaming:
        mock_streaming.chat_mode = True
        result = dispatch_command(_make_ctx("/status", target="main"))
    assert "main" in result.message


def test_dispatch_status_message_contains_model():
    """Status output must include the model ID."""
    with patch("minion_assist.config.streaming") as mock_streaming:
        mock_streaming.chat_mode = False
        result = dispatch_command(_make_ctx("/status"))
    assert "test-model" in result.message


def test_dispatch_status_shows_streaming_on():
    """Status must report 'on' when streaming.chat_mode is True."""
    with patch("minion_assist.config.streaming") as mock_streaming:
        mock_streaming.chat_mode = True
        result = dispatch_command(_make_ctx("/status"))
    assert "on" in result.message


# ---------------------------------------------------------------------------
# dispatch_command — /status deep (MEM-GAP-016)
# ---------------------------------------------------------------------------

def test_dispatch_status_bare_does_not_include_worker_section():
    from minion_assist.worker_health import WorkerHealth

    health = {"capture_worker": WorkerHealth("capture_worker")}
    with patch("minion_assist.config.streaming") as mock_streaming:
        mock_streaming.chat_mode = False
        result = dispatch_command(_make_ctx("/status", worker_health=health))
    assert "Workers:" not in result.message


def test_dispatch_status_deep_lists_every_known_worker():
    with patch("minion_assist.config.streaming") as mock_streaming:
        mock_streaming.chat_mode = False
        result = dispatch_command(_make_ctx("/status", args="deep"))
    assert "capture_worker: not running" in result.message
    assert "heartbeat: not running" in result.message
    assert "dreaming: not running" in result.message
    assert "memory_reconciliation: not running" in result.message


def test_dispatch_status_deep_lists_a_session_writes_row_per_configured_agent():
    agents_cfg = _make_agents_cfg(include_route=True)  # main + researcher
    with patch("minion_assist.config.streaming") as mock_streaming:
        mock_streaming.chat_mode = False
        result = dispatch_command(_make_ctx("/status", args="deep", agents_cfg=agents_cfg))
    assert "session_writes:main: not running" in result.message
    assert "session_writes:researcher: not running" in result.message


def test_dispatch_status_deep_lists_a_memory_search_row_per_configured_agent():
    agents_cfg = _make_agents_cfg(include_route=True)  # main + researcher
    with patch("minion_assist.config.streaming") as mock_streaming:
        mock_streaming.chat_mode = False
        result = dispatch_command(_make_ctx("/status", args="deep", agents_cfg=agents_cfg))
    assert "memory_search:main: not running" in result.message
    assert "memory_search:researcher: not running" in result.message


def test_dispatch_status_deep_reports_a_running_memory_search_health():
    from minion_assist.worker_health import WorkerHealth

    health = WorkerHealth("memory_search:main")
    health.record_success()
    with patch("minion_assist.config.streaming") as mock_streaming:
        mock_streaming.chat_mode = False
        result = dispatch_command(
            _make_ctx("/status", args="deep", worker_health={"memory_search:main": health})
        )
    assert "memory_search:main: ok" in result.message


def test_dispatch_status_deep_reports_a_running_session_writes_health():
    from minion_assist.worker_health import WorkerHealth

    health = WorkerHealth("session_writes:main")
    health.record_success()
    with patch("minion_assist.config.streaming") as mock_streaming:
        mock_streaming.chat_mode = False
        result = dispatch_command(
            _make_ctx("/status", args="deep", worker_health={"session_writes:main": health})
        )
    assert "session_writes:main: ok" in result.message


def test_dispatch_status_deep_reports_a_running_workers_health():
    from minion_assist.worker_health import WorkerHealth

    health = WorkerHealth("capture_worker")
    health.record_poll()
    health.record_success()
    with patch("minion_assist.config.streaming") as mock_streaming:
        mock_streaming.chat_mode = False
        result = dispatch_command(
            _make_ctx("/status", args="deep", worker_health={"capture_worker": health})
        )
    assert "capture_worker: ok" in result.message
    assert "last_poll=" in result.message
    assert "last_success=" in result.message


def test_dispatch_status_deep_reports_consecutive_failures():
    from minion_assist.worker_health import WorkerHealth

    health = WorkerHealth("capture_worker")
    health.record_failure("connection refused")
    with patch("minion_assist.config.streaming") as mock_streaming:
        mock_streaming.chat_mode = False
        result = dispatch_command(
            _make_ctx("/status", args="deep", worker_health={"capture_worker": health})
        )
    assert "1 consecutive failure(s)" in result.message
    assert "connection refused" in result.message


def test_dispatch_status_deep_without_a_database_says_so():
    with patch("minion_assist.config.streaming") as mock_streaming:
        mock_streaming.chat_mode = False
        result = dispatch_command(_make_ctx("/status", args="deep"))
    assert "Database: not configured" in result.message


def test_dispatch_status_deep_reports_queue_lag_from_the_database():
    db = MagicMock()
    db.queue_lag_summary.return_value = {
        "capture": {"pending_count": 3, "oldest_pending_age_s": 120.0},
        "commitment": {"pending_count": 0, "oldest_pending_age_s": None},
        "message_embedding": {"pending_count": 2, "oldest_pending_age_s": 5.0},
    }
    with patch("minion_assist.config.streaming") as mock_streaming:
        mock_streaming.chat_mode = False
        result = dispatch_command(_make_ctx("/status", args="deep", db=db))
    db.queue_lag_summary.assert_called_once_with("main")
    assert "capture_pending=3" in result.message
    assert "commitment_pending=0" in result.message
    assert "message_embedding_pending=2" in result.message


def test_dispatch_status_deep_reports_index_summary_from_the_agents_memory():
    session = _make_session()
    session.memory = MagicMock()
    session.memory.deep_status.return_value = {
        "total_chunks": 42, "file_count": 7, "last_indexed_at": time.time() - 30,
    }
    db = MagicMock()
    db.queue_lag_summary.return_value = {
        "capture": {"pending_count": 0, "oldest_pending_age_s": None},
        "commitment": {"pending_count": 0, "oldest_pending_age_s": None},
        "message_embedding": {"pending_count": 0, "oldest_pending_age_s": None},
    }
    with patch("minion_assist.config.streaming") as mock_streaming:
        mock_streaming.chat_mode = False
        result = dispatch_command(
            _make_ctx("/status", args="deep", sessions={"main": session}, db=db)
        )
    assert "42 chunks" in result.message
    assert "7 files" in result.message


def test_dispatch_status_deep_survives_a_queue_lag_query_failure():
    db = MagicMock()
    db.queue_lag_summary.side_effect = Exception("connection lost")
    with patch("minion_assist.config.streaming") as mock_streaming:
        mock_streaming.chat_mode = False
        result = dispatch_command(_make_ctx("/status", args="deep", db=db))
    assert result.handled
    assert "queue lag unavailable" in result.message
    assert "connection lost" in result.message


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


# ---------------------------------------------------------------------------
# dispatch_command — /agents
# ---------------------------------------------------------------------------

def _make_ctx_with_store(command: str, args: str = "", target: str = "main", store=None):
    """Variant of _make_ctx that also injects a session_store."""
    sessions = {"main": _make_session()}
    agents_cfg = _make_agents_cfg()
    return CommandContext(
        raw=f"{command} {args}".strip(),
        command=command,
        args=args,
        target_agent_id=target,
        sessions=sessions,
        agents_cfg=agents_cfg,
        session_store=store,
    )


def test_dispatch_agents_no_store_returns_message():
    """/agents without a store must return a graceful message."""
    result = dispatch_command(_make_ctx_with_store("/agents", store=None))
    assert result.handled is True
    assert result.message is not None


def test_dispatch_agents_lists_agents():
    """/agents must list session records returned by session_store."""
    store = MagicMock()
    store.list_sessions.return_value = [
        MagicMock(agent_id="main", turn_count=5, last_active="2026-06-01T10:00:00+00:00"),
    ]
    result = dispatch_command(_make_ctx_with_store("/agents", store=store))
    assert result.handled is True
    assert "main" in result.message
    assert "5" in result.message


def test_dispatch_agents_empty_store():
    """/agents with an empty store reports no sessions."""
    store = MagicMock()
    store.list_sessions.return_value = []
    result = dispatch_command(_make_ctx_with_store("/agents", store=store))
    assert result.handled is True
    assert "No sessions" in result.message


# ---------------------------------------------------------------------------
# dispatch_command — /switch
# ---------------------------------------------------------------------------

def test_dispatch_switch_known_agent():
    """/switch <known> must return activate_agent_id and a message."""
    sessions = {"main": _make_session(), "researcher": _make_session()}
    agents_cfg = _make_agents_cfg(include_route=True)
    ctx = CommandContext(
        raw="/switch researcher",
        command="/switch",
        args="researcher",
        target_agent_id="main",
        sessions=sessions,
        agents_cfg=agents_cfg,
    )
    result = dispatch_command(ctx)
    assert result.handled is True
    assert result.activate_agent_id == "researcher"


def test_dispatch_switch_unknown_agent_returns_error():
    """/switch <unknown> must report the known agents and NOT set activate_agent_id."""
    ctx = _make_ctx_with_store("/switch", args="ghost")
    result = dispatch_command(ctx)
    assert result.handled is True
    assert result.activate_agent_id is None
    assert "ghost" in result.message
    assert "main" in result.message  # list of known agents shown


def test_dispatch_switch_no_args_targets_current_agent():
    """/switch with no args defaults to the already-active agent."""
    sessions = {"main": _make_session()}
    agents_cfg = _make_agents_cfg()
    ctx = CommandContext(
        raw="/switch",
        command="/switch",
        args="",
        target_agent_id="main",
        sessions=sessions,
        agents_cfg=agents_cfg,
    )
    result = dispatch_command(ctx)
    assert result.handled is True
    assert result.activate_agent_id == "main"


# ---------------------------------------------------------------------------
# dispatch_command — /diagnose
# ---------------------------------------------------------------------------

def test_dispatch_diagnose_shows_provider_info():
    """/diagnose must show provider name, model, and auth status for each agent."""
    result = dispatch_command(_make_ctx("/diagnose"))
    assert result.handled is True
    assert result.message is not None
    # Agent id present
    assert "main" in result.message


def test_dispatch_diagnose_shows_missing_key():
    """/diagnose must flag a missing API key as MISSING."""
    agents_cfg = _make_agents_cfg()
    agents_cfg["main"].provider.api_key = ""
    agents_cfg["main"].provider.api = "openai-completions"
    agents_cfg["main"].provider.name = "openai"
    ctx = _make_ctx("/diagnose", agents_cfg=agents_cfg)
    result = dispatch_command(ctx)
    assert "MISSING" in result.message


def test_dispatch_diagnose_shows_ok_for_local_provider():
    """/diagnose must not flag lmstudio providers as missing a key."""
    agents_cfg = _make_agents_cfg()
    agents_cfg["main"].provider.api = "lmstudio"
    agents_cfg["main"].provider.api_key = ""
    ctx = _make_ctx("/diagnose", agents_cfg=agents_cfg)
    result = dispatch_command(ctx)
    assert "MISSING" not in result.message


# ---------------------------------------------------------------------------
# dispatch_command — /mcp-reload
# ---------------------------------------------------------------------------

def test_dispatch_mcp_reload_no_manager_returns_message():
    """/mcp-reload without an MCP manager must explain what to do."""
    ctx = CommandContext(
        raw="/mcp-reload",
        command="/mcp-reload",
        args="",
        target_agent_id="main",
        sessions={"main": _make_session()},
        agents_cfg=_make_agents_cfg(),
        mcp_manager=None,
    )
    result = dispatch_command(ctx)
    assert result.handled is True
    assert "No MCP manager" in result.message


def test_dispatch_mcp_reload_calls_reconnect_and_refresh():
    """/mcp-reload must call reconnect_all_sync and refresh_mcp_adapters."""
    manager = MagicMock()
    manager.reconnect_all_sync = MagicMock()
    manager.list_statuses = MagicMock(return_value=[
        MagicMock(name="playwright", state="connected", detail="2 tool(s)")
    ])

    session = _make_session()
    session.refresh_mcp_adapters = MagicMock(return_value=2)

    ctx = CommandContext(
        raw="/mcp-reload",
        command="/mcp-reload",
        args="",
        target_agent_id="main",
        sessions={"main": session},
        agents_cfg=_make_agents_cfg(),
        mcp_manager=manager,
    )
    result = dispatch_command(ctx)
    assert result.handled is True
    manager.reconnect_all_sync.assert_called_once()
    session.refresh_mcp_adapters.assert_called_once_with(manager)
    assert "reconnected" in result.message.lower()


# ---------------------------------------------------------------------------
# New commands appear in BUILTIN_COMMANDS
# ---------------------------------------------------------------------------

def test_builtin_commands_includes_new_commands():
    names = {spec.name for spec in BUILTIN_COMMANDS}
    assert "/agents" in names
    assert "/switch" in names
    assert "/diagnose" in names
    assert "/mcp-reload" in names

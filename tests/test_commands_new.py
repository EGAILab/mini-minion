"""Tests for new slash commands: /plan, /auto, /providers, /mcp-list."""

from unittest.mock import MagicMock

from minion_assist.commands import CommandContext, BUILTIN_COMMANDS, dispatch_command
from minion_assist.tools.policy import PermissionPolicy
from minion_assist.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session_with_policy(read_only: bool = False) -> MagicMock:
    """Return a mock AgentSession with a real PermissionPolicy wired in."""
    session = MagicMock()
    registry = ToolRegistry()
    registry.policy = PermissionPolicy(read_only_mode=read_only)
    # registry property returns our ToolRegistry.
    type(session).registry = property(lambda s: registry)
    return session


def _make_agents_cfg(model_id: str = "test-model") -> dict:
    cfg = MagicMock()
    cfg.route_prefix = None
    cfg.model.id = model_id
    cfg.model.context_window = 4096
    cfg.model.max_output_tokens = 512
    cfg.provider.name = "openai"
    cfg.provider.api = "openai"
    cfg.provider.base_url = None
    return {"main": cfg}


def _ctx(command: str, session=None, mcp_manager=None) -> CommandContext:
    if session is None:
        session = _make_session_with_policy()
    return CommandContext(
        raw=command,
        command=command,
        args="",
        target_agent_id="main",
        sessions={"main": session},
        agents_cfg=_make_agents_cfg(),
        mcp_manager=mcp_manager,
    )


# ---------------------------------------------------------------------------
# /plan command
# ---------------------------------------------------------------------------

def test_plan_enables_read_only_mode():
    session = _make_session_with_policy(read_only=False)
    result = dispatch_command(_ctx("/plan", session=session))
    assert result.handled
    assert session.registry.policy.read_only_mode is True


def test_plan_message_mentions_read_only():
    session = _make_session_with_policy()
    result = dispatch_command(_ctx("/plan", session=session))
    assert "read-only" in result.message.lower()


def test_plan_idempotent_when_already_enabled():
    session = _make_session_with_policy(read_only=True)
    result = dispatch_command(_ctx("/plan", session=session))
    assert result.handled
    assert session.registry.policy.read_only_mode is True


def test_plan_unknown_agent_returns_error():
    ctx = CommandContext(
        raw="/plan",
        command="/plan",
        args="",
        target_agent_id="missing",
        sessions={"main": _make_session_with_policy()},
        agents_cfg=_make_agents_cfg(),
    )
    result = dispatch_command(ctx)
    assert result.handled
    assert "unknown" in result.message.lower()


# ---------------------------------------------------------------------------
# /auto command
# ---------------------------------------------------------------------------

def test_auto_disables_read_only_mode():
    session = _make_session_with_policy(read_only=True)
    result = dispatch_command(_ctx("/auto", session=session))
    assert result.handled
    assert session.registry.policy.read_only_mode is False


def test_auto_message_mentions_full_access():
    session = _make_session_with_policy(read_only=True)
    result = dispatch_command(_ctx("/auto", session=session))
    assert "full" in result.message.lower() or "restored" in result.message.lower()


def test_auto_idempotent_when_already_disabled():
    session = _make_session_with_policy(read_only=False)
    result = dispatch_command(_ctx("/auto", session=session))
    assert result.handled
    assert session.registry.policy.read_only_mode is False


# ---------------------------------------------------------------------------
# /plan → /auto round-trip
# ---------------------------------------------------------------------------

def test_plan_then_auto_round_trip():
    session = _make_session_with_policy(read_only=False)
    dispatch_command(_ctx("/plan", session=session))
    assert session.registry.policy.read_only_mode is True
    dispatch_command(_ctx("/auto", session=session))
    assert session.registry.policy.read_only_mode is False


# ---------------------------------------------------------------------------
# /providers command
# ---------------------------------------------------------------------------

def test_providers_returns_handled():
    result = dispatch_command(_ctx("/providers"))
    assert result.handled


def test_providers_lists_agent_ids():
    result = dispatch_command(_ctx("/providers"))
    assert "main" in result.message


def test_providers_shows_model_id():
    result = dispatch_command(_ctx("/providers"))
    assert "test-model" in result.message


def test_providers_shows_context_window():
    result = dispatch_command(_ctx("/providers"))
    assert "4" in result.message  # 4096 context window


# ---------------------------------------------------------------------------
# /mcp-list command
# ---------------------------------------------------------------------------

def test_mcp_list_no_manager():
    result = dispatch_command(_ctx("/mcp-list", mcp_manager=None))
    assert result.handled
    assert "no mcp manager" in result.message.lower() or "not configured" in result.message.lower()


def test_mcp_list_with_manager():
    manager = MagicMock()
    status = MagicMock()
    status.name = "my-server"
    status.state = "connected"
    status.detail = ""
    manager.list_statuses.return_value = [status]
    manager.list_tools.return_value = []
    result = dispatch_command(_ctx("/mcp-list", mcp_manager=manager))
    assert result.handled
    assert "my-server" in result.message


def test_mcp_list_shows_tools():
    manager = MagicMock()
    status = MagicMock()
    status.name = "srv"
    status.state = "connected"
    status.detail = ""
    manager.list_statuses.return_value = [status]
    tool_info = MagicMock()
    tool_info.name = "some_tool"
    manager.list_tools.return_value = [tool_info]
    result = dispatch_command(_ctx("/mcp-list", mcp_manager=manager))
    assert "some_tool" in result.message


# ---------------------------------------------------------------------------
# BUILTIN_COMMANDS catalog completeness
# ---------------------------------------------------------------------------

def test_builtin_commands_includes_new_commands():
    names = {spec.name for spec in BUILTIN_COMMANDS}
    assert "/plan" in names
    assert "/auto" in names
    assert "/providers" in names
    assert "/mcp-list" in names

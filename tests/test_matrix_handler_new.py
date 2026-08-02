"""Tests for new matrix handler features: mention gate, HEARTBEAT_OK suppression,
reaction_level config, and room_cfg threading into _dispatch_and_reply."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from minion_assist.matrix.handler import MatrixMessageHandler
from minion_assist.matrix.config import MatrixConfig, MatrixRoomConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Drive a coroutine to completion without the pytest-asyncio plugin.

    Mirrors tests/matrix/test_handler.py's own helper — the established,
    dependency-free convention every other async Matrix handler test in
    this project already uses, instead of @pytest.mark.asyncio (which
    needs a plugin this project never declared as a dependency).
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_config(user_id="@bot:example.org", groups=None):
    cfg = MagicMock(spec=MatrixConfig)
    cfg.default_agent_id = "main"
    cfg.group_policy = "open"
    cfg.group_allow_from = []
    cfg.ack_reaction = None
    cfg.thread_bindings = MagicMock(enabled=False)
    cfg.groups = groups or {}
    return cfg


def _make_client(user_id="@bot:example.org"):
    client = MagicMock()
    client.user_id = user_id
    return client


def _make_handler(config=None, sessions=None, user_id="@bot:example.org", agents_cfg=None,
                  session_store=None, mcp_manager=None, skills=None, short_term=None):
    config = config or _make_config(user_id=user_id)
    sessions = sessions or {"main": MagicMock()}
    outbound = MagicMock()
    outbound.send_typing = AsyncMock()
    outbound.send_text = AsyncMock()
    outbound.send_reaction = AsyncMock()
    dedupe = MagicMock()
    dedupe.is_seen = AsyncMock(return_value=False)
    bot_loop = MagicMock()
    bot_loop.should_suppress.return_value = False
    thread_mgr = MagicMock()
    return MatrixMessageHandler(
        client=_make_client(user_id),
        config=config,
        sessions=sessions,
        outbound=outbound,
        dedupe=dedupe,
        bot_loop=bot_loop,
        thread_binding_mgr=thread_mgr,
        agents_cfg=agents_cfg,
        session_store=session_store,
        mcp_manager=mcp_manager,
        skills=skills,
        short_term=short_term,
    )


def _make_event(body="hello", sender="@user:example.org", event_id="$evt1"):
    event = MagicMock()
    event.event_id = event_id
    event.sender = sender
    event.body = body
    event.relates_to = None
    return event


def _make_room(room_id="!room:example.org"):
    room = MagicMock()
    room.room_id = room_id
    return room


# ---------------------------------------------------------------------------
# _is_mentioned
# ---------------------------------------------------------------------------

def test_is_mentioned_full_user_id():
    handler = _make_handler(user_id="@bot:example.org")
    assert handler._is_mentioned("hey @bot:example.org can you help?") is True


def test_is_mentioned_localpart():
    handler = _make_handler(user_id="@bot:example.org")
    assert handler._is_mentioned("hey bot, can you help?") is True


def test_is_mentioned_false():
    handler = _make_handler(user_id="@bot:example.org")
    assert handler._is_mentioned("this message doesn't mention anyone") is False


def test_is_mentioned_case_insensitive():
    handler = _make_handler(user_id="@Bot:example.org")
    assert handler._is_mentioned("hey BOT help me") is True


# ---------------------------------------------------------------------------
# Mention gate (require_mention=True)
# ---------------------------------------------------------------------------

def test_mention_gate_blocks_unmentioned_message():
    room_cfg = MatrixRoomConfig(require_mention=True)
    config = _make_config(groups={"!room:example.org": room_cfg})
    session = MagicMock()
    handler = _make_handler(config=config, sessions={"main": session})

    _run(handler.handle_room_message(
        _make_room(),
        _make_event(body="hello everyone", sender="@user:example.org"),
    ))
    session.send.assert_not_called()


def test_mention_gate_allows_mentioned_message():
    room_cfg = MatrixRoomConfig(require_mention=True)
    config = _make_config(groups={"!room:example.org": room_cfg})
    session = MagicMock()
    session.send.return_value = "I'm here!"
    handler = _make_handler(config=config, sessions={"main": session})

    _run(handler.handle_room_message(
        _make_room(),
        _make_event(body="@bot:example.org can you help?", sender="@user:example.org"),
    ))
    session.send.assert_called_once()


def test_no_mention_gate_when_require_mention_false():
    room_cfg = MatrixRoomConfig(require_mention=False)
    config = _make_config(groups={"!room:example.org": room_cfg})
    session = MagicMock()
    session.send.return_value = "response"
    handler = _make_handler(config=config, sessions={"main": session})

    _run(handler.handle_room_message(
        _make_room(),
        _make_event(body="hello", sender="@user:example.org"),
    ))
    session.send.assert_called_once()


# ---------------------------------------------------------------------------
# HEARTBEAT_OK suppression
# ---------------------------------------------------------------------------

def test_heartbeat_ok_reply_is_suppressed():
    session = MagicMock()
    session.send.return_value = "HEARTBEAT_OK"
    handler = _make_handler(sessions={"main": session})

    _run(handler.handle_room_message(
        _make_room(),
        _make_event(body="heartbeat check"),
    ))
    handler._outbound.send_text.assert_not_called()


def test_normal_reply_is_not_suppressed():
    session = MagicMock()
    session.send.return_value = "Hello there!"
    handler = _make_handler(sessions={"main": session})

    _run(handler.handle_room_message(
        _make_room(),
        _make_event(body="hi"),
    ))
    handler._outbound.send_text.assert_called_once()


# ---------------------------------------------------------------------------
# MatrixRoomConfig reaction_level
# ---------------------------------------------------------------------------

def test_room_config_reaction_level_default():
    cfg = MatrixRoomConfig()
    assert cfg.reaction_level == "off"


def test_room_config_reaction_level_from_dict():
    cfg = MatrixRoomConfig.from_dict({"reactionLevel": "all"})
    assert cfg.reaction_level == "all"


def test_room_config_reaction_level_mentions():
    cfg = MatrixRoomConfig.from_dict({"reactionLevel": "mentions"})
    assert cfg.reaction_level == "mentions"


def test_room_config_reaction_level_off_by_default_in_from_dict():
    cfg = MatrixRoomConfig.from_dict({})
    assert cfg.reaction_level == "off"


# ---------------------------------------------------------------------------
# Slash command dispatch
# ---------------------------------------------------------------------------

def _make_agents_cfg():
    """Minimal agents_cfg dict for slash command tests."""
    cfg = MagicMock()
    cfg.route_prefix = None
    return {"main": cfg}


def test_slash_command_new_clears_history():
    session = MagicMock()
    session.send.return_value = "response"
    handler = _make_handler(sessions={"main": session}, agents_cfg=_make_agents_cfg())

    _run(handler.handle_room_message(
        _make_room(),
        _make_event(body="/new"),
    ))
    session.reset.assert_called_once()
    session.send.assert_not_called()


def test_slash_command_reply_sent_to_room():
    session = MagicMock()
    handler = _make_handler(sessions={"main": session}, agents_cfg=_make_agents_cfg())

    _run(handler.handle_room_message(
        _make_room(),
        _make_event(body="/new"),
    ))
    handler._outbound.send_text.assert_called_once()
    call_args = handler._outbound.send_text.call_args
    assert "Cleared" in call_args[0][1]


def test_slash_disallowed_command_quit_blocked():
    session = MagicMock()
    handler = _make_handler(sessions={"main": session}, agents_cfg=_make_agents_cfg())

    _run(handler.handle_room_message(
        _make_room(),
        _make_event(body="/quit"),
    ))
    session.send.assert_not_called()
    handler._outbound.send_text.assert_called_once()
    assert "not available" in handler._outbound.send_text.call_args[0][1]


def test_slash_disallowed_command_export_blocked():
    session = MagicMock()
    handler = _make_handler(sessions={"main": session}, agents_cfg=_make_agents_cfg())

    _run(handler.handle_room_message(
        _make_room(),
        _make_event(body="/export /tmp/out.md"),
    ))
    session.send.assert_not_called()
    handler._outbound.send_text.assert_called_once()


def test_unknown_slash_command_falls_through_to_llm():
    session = MagicMock()
    session.send.return_value = "I don't know that command"
    handler = _make_handler(sessions={"main": session}, agents_cfg=_make_agents_cfg())

    _run(handler.handle_room_message(
        _make_room(),
        _make_event(body="/frobnicate"),
    ))
    session.send.assert_called_once()


def test_bang_prefix_new_clears_history():
    """!new should work identically to /new (Element-safe prefix)."""
    session = MagicMock()
    handler = _make_handler(sessions={"main": session}, agents_cfg=_make_agents_cfg())

    _run(handler.handle_room_message(
        _make_room(),
        _make_event(body="!new"),
    ))
    session.reset.assert_called_once()
    session.send.assert_not_called()


def test_bang_prefix_session_dispatched():
    """!session 2 should dispatch /session with args '2'."""
    session = MagicMock()
    handler = _make_handler(
        sessions={"main": session}, agents_cfg=_make_agents_cfg(), short_term=MagicMock()
    )
    # short_term.list_sessions returns empty list → "No session history found" message
    handler._short_term.list_sessions.return_value = []

    _run(handler.handle_room_message(
        _make_room(),
        _make_event(body="!session 2"),
    ))
    session.send.assert_not_called()
    handler._outbound.send_text.assert_called_once()
    assert "session" in handler._outbound.send_text.call_args[0][1].lower()


def test_bang_quit_blocked():
    """!quit should be blocked just like /quit."""
    session = MagicMock()
    handler = _make_handler(sessions={"main": session}, agents_cfg=_make_agents_cfg())

    _run(handler.handle_room_message(
        _make_room(),
        _make_event(body="!quit"),
    ))
    session.send.assert_not_called()
    assert "not available" in handler._outbound.send_text.call_args[0][1]


def test_bang_plain_message_not_treated_as_command():
    """A lone '!' or '! text' (space after !) is not a command."""
    session = MagicMock()
    session.send.return_value = "response"
    handler = _make_handler(sessions={"main": session}, agents_cfg=_make_agents_cfg())

    _run(handler.handle_room_message(
        _make_room(),
        _make_event(body="! not a command"),
    ))
    session.send.assert_called_once()


def test_slash_command_skipped_when_agents_cfg_none():
    """Without agents_cfg, /new goes to the LLM unchanged (backward compat)."""
    session = MagicMock()
    session.send.return_value = "response"
    handler = _make_handler(sessions={"main": session}, agents_cfg=None)

    _run(handler.handle_room_message(
        _make_room(),
        _make_event(body="/new"),
    ))
    session.send.assert_called_once()
    session.reset.assert_not_called()

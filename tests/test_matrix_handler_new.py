"""Tests for new matrix handler features: mention gate, HEARTBEAT_OK suppression,
reaction_level config, and room_cfg threading into _dispatch_and_reply."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from minion_assist.matrix.handler import MatrixMessageHandler
from minion_assist.matrix.config import MatrixConfig, MatrixRoomConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _make_handler(config=None, sessions=None, user_id="@bot:example.org"):
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

@pytest.mark.asyncio
async def test_mention_gate_blocks_unmentioned_message():
    room_cfg = MatrixRoomConfig(require_mention=True)
    config = _make_config(groups={"!room:example.org": room_cfg})
    session = MagicMock()
    handler = _make_handler(config=config, sessions={"main": session})

    await handler.handle_room_message(
        _make_room(),
        _make_event(body="hello everyone", sender="@user:example.org"),
    )
    session.send.assert_not_called()


@pytest.mark.asyncio
async def test_mention_gate_allows_mentioned_message():
    room_cfg = MatrixRoomConfig(require_mention=True)
    config = _make_config(groups={"!room:example.org": room_cfg})
    session = MagicMock()
    session.send.return_value = "I'm here!"
    handler = _make_handler(config=config, sessions={"main": session})

    await handler.handle_room_message(
        _make_room(),
        _make_event(body="@bot:example.org can you help?", sender="@user:example.org"),
    )
    session.send.assert_called_once()


@pytest.mark.asyncio
async def test_no_mention_gate_when_require_mention_false():
    room_cfg = MatrixRoomConfig(require_mention=False)
    config = _make_config(groups={"!room:example.org": room_cfg})
    session = MagicMock()
    session.send.return_value = "response"
    handler = _make_handler(config=config, sessions={"main": session})

    await handler.handle_room_message(
        _make_room(),
        _make_event(body="hello", sender="@user:example.org"),
    )
    session.send.assert_called_once()


# ---------------------------------------------------------------------------
# HEARTBEAT_OK suppression
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_heartbeat_ok_reply_is_suppressed():
    session = MagicMock()
    session.send.return_value = "HEARTBEAT_OK"
    handler = _make_handler(sessions={"main": session})

    await handler.handle_room_message(
        _make_room(),
        _make_event(body="heartbeat check"),
    )
    handler._outbound.send_text.assert_not_called()


@pytest.mark.asyncio
async def test_normal_reply_is_not_suppressed():
    session = MagicMock()
    session.send.return_value = "Hello there!"
    handler = _make_handler(sessions={"main": session})

    await handler.handle_room_message(
        _make_room(),
        _make_event(body="hi"),
    )
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

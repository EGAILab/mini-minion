"""Tests for matrix/handler.py — MatrixMessageHandler."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from minion_assist.matrix.bot_loop import BotLoopProtection
from minion_assist.matrix.config import MatrixBotLoopConfig, MatrixConfig, MatrixRoomConfig
from minion_assist.matrix.handler import MatrixMessageHandler
from minion_assist.matrix.outbound import MatrixOutbound


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_config(group_policy="open", group_allow_from=None, groups=None, ack_reaction="👀",
                 default_agent_id="main", thread_bindings_enabled=False):
    cfg = MagicMock(spec=MatrixConfig)
    cfg.group_policy = group_policy
    cfg.group_allow_from = group_allow_from or []
    cfg.groups = groups or {}
    cfg.ack_reaction = ack_reaction
    cfg.default_agent_id = default_agent_id
    tb = MagicMock()
    tb.enabled = thread_bindings_enabled
    cfg.thread_bindings = tb
    return cfg


def _make_client(user_id="@bot:ex.org"):
    c = MagicMock()
    c.user_id = user_id
    return c


def _make_dedupe(seen=False):
    d = MagicMock()
    d.is_seen = AsyncMock(return_value=seen)
    return d


def _make_bot_loop(suppress=False):
    bl = MagicMock(spec=BotLoopProtection)
    bl.should_suppress = MagicMock(return_value=suppress)
    return bl


def _make_outbound():
    o = MagicMock(spec=MatrixOutbound)
    o.send_reaction = AsyncMock()
    o.send_text = AsyncMock()
    return o


def _make_thread_mgr():
    t = MagicMock()
    t.get_or_create_session_key = AsyncMock(return_value="matrix-thread-abc")
    return t


def _make_session(response="Agent reply"):
    s = MagicMock()
    s.send = MagicMock(return_value=response)
    return s


def _make_event(event_id="$e1", sender="@alice:ex.org", body="hello", relates_to=None):
    e = MagicMock()
    e.event_id = event_id
    e.sender = sender
    e.body = body
    e.relates_to = relates_to
    return e


def _make_room(room_id="!room:ex.org"):
    r = MagicMock()
    r.room_id = room_id
    return r


def _make_handler(config=None, sessions=None, dedupe=None, bot_loop=None, outbound=None):
    config = config or _make_config()
    sessions = sessions or {"main": _make_session()}
    dedupe = dedupe or _make_dedupe()
    bot_loop = bot_loop or _make_bot_loop()
    outbound = outbound or _make_outbound()
    thread_mgr = _make_thread_mgr()
    return MatrixMessageHandler(
        client=_make_client(),
        config=config,
        sessions=sessions,
        outbound=outbound,
        dedupe=dedupe,
        bot_loop=bot_loop,
        thread_binding_mgr=thread_mgr,
        exec_approval_handler=None,
    )


class TestDeduplication:
    def test_duplicate_event_skipped(self):
        dedupe = _make_dedupe(seen=True)
        outbound = _make_outbound()
        handler = _make_handler(dedupe=dedupe, outbound=outbound)
        _run(handler.handle_room_message(_make_room(), _make_event()))
        outbound.send_text.assert_not_called()


class TestOwnMessageSkip:
    def test_own_message_skipped(self):
        outbound = _make_outbound()
        handler = _make_handler(outbound=outbound)
        handler._client.user_id = "@bot:ex.org"
        event = _make_event(sender="@bot:ex.org")
        _run(handler.handle_room_message(_make_room(), event))
        outbound.send_text.assert_not_called()


class TestBotLoopSuppression:
    def test_suppressed_room_skipped(self):
        bot_loop = _make_bot_loop(suppress=True)
        outbound = _make_outbound()
        handler = _make_handler(bot_loop=bot_loop, outbound=outbound)
        _run(handler.handle_room_message(_make_room(), _make_event()))
        outbound.send_text.assert_not_called()


class TestAllowlist:
    def test_sender_not_on_allowlist_skipped(self):
        cfg = _make_config(group_policy="allowlist", group_allow_from=["@admin:ex.org"])
        outbound = _make_outbound()
        handler = _make_handler(config=cfg, outbound=outbound)
        event = _make_event(sender="@stranger:ex.org")
        _run(handler.handle_room_message(_make_room(), event))
        outbound.send_text.assert_not_called()

    def test_sender_on_allowlist_proceeds(self):
        cfg = _make_config(group_policy="open", group_allow_from=["@alice:ex.org"])
        session = _make_session("hello back")
        outbound = _make_outbound()
        handler = _make_handler(config=cfg, sessions={"main": session}, outbound=outbound)
        event = _make_event(sender="@alice:ex.org", body="hello")
        _run(handler.handle_room_message(_make_room(), event))
        outbound.send_text.assert_called_once()


class TestAgentRouting:
    def test_routes_to_configured_agent(self):
        room_cfg = MatrixRoomConfig(agent="researcher", enabled=True)
        cfg = _make_config(groups={"!room:ex.org": room_cfg})
        researcher_session = _make_session("research result")
        sessions = {
            "main": _make_session("main result"),
            "researcher": researcher_session,
        }
        outbound = _make_outbound()
        handler = _make_handler(config=cfg, sessions=sessions, outbound=outbound)
        _run(handler.handle_room_message(_make_room("!room:ex.org"), _make_event()))
        researcher_session.send.assert_called_once()

    def test_routes_to_default_when_no_room_config(self):
        cfg = _make_config()
        session = _make_session("default reply")
        sessions = {"main": session}
        outbound = _make_outbound()
        handler = _make_handler(config=cfg, sessions=sessions, outbound=outbound)
        _run(handler.handle_room_message(_make_room("!unknown:ex.org"), _make_event()))
        session.send.assert_called_once()


class TestAckReaction:
    def test_ack_reaction_sent(self):
        outbound = _make_outbound()
        handler = _make_handler(outbound=outbound)
        event = _make_event(event_id="$e1")
        _run(handler.handle_room_message(_make_room(), event))
        outbound.send_reaction.assert_called_once_with("!room:ex.org", "$e1", "👀")

    def test_no_ack_reaction_when_not_configured(self):
        cfg = _make_config(ack_reaction=None)
        outbound = _make_outbound()
        handler = _make_handler(config=cfg, outbound=outbound)
        _run(handler.handle_room_message(_make_room(), _make_event()))
        outbound.send_reaction.assert_not_called()


class TestResponseDelivery:
    def test_reply_sent_to_room(self):
        session = _make_session("hello reply")
        outbound = _make_outbound()
        handler = _make_handler(sessions={"main": session}, outbound=outbound)
        _run(handler.handle_room_message(_make_room("!room:ex.org"), _make_event(body="hi")))
        outbound.send_text.assert_called_once()
        call = outbound.send_text.call_args
        assert call.args[0] == "!room:ex.org"
        assert "hello reply" in call.args[1]

    def test_empty_body_not_dispatched(self):
        session = _make_session()
        outbound = _make_outbound()
        handler = _make_handler(sessions={"main": session}, outbound=outbound)
        _run(handler.handle_room_message(_make_room(), _make_event(body="   ")))
        session.send.assert_not_called()

"""Tests for matrix/handler.py — MatrixMessageHandler."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nio import RoomMessageImage
from nio.responses import DownloadError

from minion_assist.matrix.bot_loop import BotLoopProtection
from minion_assist.matrix.config import MatrixBotLoopConfig, MatrixConfig, MatrixRoomConfig
from minion_assist.matrix.handler import MatrixMessageHandler
from minion_assist.matrix.outbound import MatrixOutbound

# A minimal byte blob that satisfies media.py's PNG signature sniff — only
# the first 8 bytes matter for _sniff_image_mime(), so this is not a real
# decodable PNG, just enough to pass the "is this actually image bytes"
# check the same way a real download's first bytes would.
_VALID_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_config(group_policy="open", group_allow_from=None, groups=None, ack_reaction="👀",
                 default_agent_id="main"):
    cfg = MagicMock(spec=MatrixConfig)
    cfg.group_policy = group_policy
    cfg.group_allow_from = group_allow_from or []
    cfg.groups = groups or {}
    cfg.ack_reaction = ack_reaction
    cfg.default_agent_id = default_agent_id
    return cfg


def _make_client(user_id="@bot:ex.org", download_response=None):
    c = MagicMock()
    c.user_id = user_id
    # download() is only ever awaited on the image-attachment path; tests
    # that don't exercise it never call it, so a harmless default is fine.
    c.download = AsyncMock(return_value=download_response)
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


def _make_room_session_mgr():
    m = MagicMock()
    m.get_or_create_session_id = AsyncMock(return_value="room-session-abc")
    return m


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


def _make_image_event(event_id="$img1", sender="@alice:ex.org", body="photo.png",
                       url="mxc://ex.org/abc123", relates_to=None):
    # spec=RoomMessageImage makes isinstance(e, RoomMessageImage) True — the
    # exact check handle_room_message() uses to route this down the
    # image-download path instead of the plain-text path.
    e = MagicMock(spec=RoomMessageImage)
    e.event_id = event_id
    e.sender = sender
    e.body = body
    e.url = url
    e.relates_to = relates_to
    return e


def _make_room(room_id="!room:ex.org"):
    r = MagicMock()
    r.room_id = room_id
    return r


def _make_handler(config=None, sessions=None, dedupe=None, bot_loop=None, outbound=None,
                  session_factories=None, client=None, media_dir=None):
    config = config or _make_config()
    sessions = sessions or {"main": _make_session()}
    dedupe = dedupe or _make_dedupe()
    bot_loop = bot_loop or _make_bot_loop()
    outbound = outbound or _make_outbound()
    client = client or _make_client()
    room_session_mgr = _make_room_session_mgr()
    if session_factories is None:
        # R2-GAP-009: _get_or_build_session() now fails closed when no
        # factory is registered — these tests are about
        # routing/delivery/dedup, not that failure mode, so default to a
        # trivial factory returning the same per-agent session `sessions`
        # already provides (keeps every existing assertion against that
        # object valid).
        session_factories = {aid: (lambda sid, s=s: s) for aid, s in sessions.items()}
    return MatrixMessageHandler(
        client=client,
        config=config,
        sessions=sessions,
        outbound=outbound,
        dedupe=dedupe,
        bot_loop=bot_loop,
        room_session_mgr=room_session_mgr,
        exec_approval_handler=None,
        session_factories=session_factories,
        media_dir=media_dir,
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


class TestImageAttachments:
    """Covers the m.image handling added after images were silently
    dropped in Matrix — see handler.py's module docstring and
    _download_image_attachment()'s docstring for the full story."""

    def test_image_with_caption_is_staged_and_sent_as_attachment(self, tmp_path):
        response = MagicMock()
        response.body = _VALID_PNG_BYTES
        client = _make_client(download_response=response)
        session = _make_session("I see a screenshot")
        outbound = _make_outbound()
        handler = _make_handler(
            sessions={"main": session}, outbound=outbound, client=client, media_dir=tmp_path
        )
        event = _make_image_event(body="what is this?")

        _run(handler.handle_room_message(_make_room(), event))

        session.send.assert_called_once()
        assert session.send.call_args.args[0] == "what is this?"
        attachments = session.send.call_args.kwargs["attachments"]
        assert attachments is not None and len(attachments) == 1
        assert attachments[0].media_type == "image/png"
        outbound.send_text.assert_called_once()

    def test_image_without_caption_uses_placeholder_text(self, tmp_path):
        response = MagicMock()
        response.body = _VALID_PNG_BYTES
        client = _make_client(download_response=response)
        session = _make_session("caption-less reply")
        handler = _make_handler(sessions={"main": session}, client=client, media_dir=tmp_path)
        event = _make_image_event(body="")

        _run(handler.handle_room_message(_make_room(), event))

        assert session.send.call_args.args[0] == "[Image attached]"

    def test_download_failure_replies_in_room_and_does_not_dispatch(self, tmp_path):
        client = _make_client(download_response=DownloadError("not found"))
        session = _make_session()
        outbound = _make_outbound()
        handler = _make_handler(
            sessions={"main": session}, outbound=outbound, client=client, media_dir=tmp_path
        )

        _run(handler.handle_room_message(_make_room(), _make_image_event()))

        session.send.assert_not_called()
        outbound.send_text.assert_called_once()
        assert "Couldn't process that image" in outbound.send_text.call_args.args[1]

    def test_no_media_dir_configured_replies_in_room(self):
        session = _make_session()
        outbound = _make_outbound()
        handler = _make_handler(sessions={"main": session}, outbound=outbound, media_dir=None)

        _run(handler.handle_room_message(_make_room(), _make_image_event()))

        session.send.assert_not_called()
        outbound.send_text.assert_called_once()
        assert "not configured" in outbound.send_text.call_args.args[1]

    def test_invalid_image_bytes_reply_with_staging_error(self, tmp_path):
        response = MagicMock()
        response.body = b"not actually image bytes"
        client = _make_client(download_response=response)
        session = _make_session()
        outbound = _make_outbound()
        handler = _make_handler(
            sessions={"main": session}, outbound=outbound, client=client, media_dir=tmp_path
        )
        event = _make_image_event(body="fake.png")

        _run(handler.handle_room_message(_make_room(), event))

        session.send.assert_not_called()
        outbound.send_text.assert_called_once()
        assert "Couldn't process that image" in outbound.send_text.call_args.args[1]

    def test_text_events_are_unaffected_by_image_isinstance_check(self):
        # A bare (unspec'd) MagicMock event must never satisfy
        # isinstance(event, RoomMessageImage) — this is what keeps every
        # other test in this file (built on plain _make_event()) exercising
        # the ordinary text path with no changes required.
        session = _make_session("plain text reply")
        handler = _make_handler(sessions={"main": session})

        _run(handler.handle_room_message(_make_room(), _make_event(body="hello")))

        session.send.assert_called_once()
        assert session.send.call_args.kwargs["attachments"] is None

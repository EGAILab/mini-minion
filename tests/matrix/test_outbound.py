"""Tests for matrix/outbound.py — MatrixOutbound."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from minion_assist.matrix.config import MatrixConfig
from minion_assist.matrix.outbound import MatrixOutbound


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client():
    client = MagicMock()
    client.room_send = AsyncMock(return_value=MagicMock(event_id="$sent_event"))
    client.room_typing = AsyncMock(return_value=None)
    return client


def _make_outbound(client=None, chunk_limit=4000):
    if client is None:
        client = _make_client()
    cfg = MagicMock(spec=MatrixConfig)
    cfg.text_chunk_limit = chunk_limit
    return MatrixOutbound(client, cfg), client


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# send_text
# ---------------------------------------------------------------------------

def test_send_text_calls_room_send():
    outbound, client = _make_outbound()
    _run(outbound.send_text("!room:ex.org", "hello world"))
    client.room_send.assert_called_once()
    content = client.room_send.call_args.kwargs["content"]
    assert content["msgtype"] == "m.text"
    assert "hello world" in content["body"]


def test_send_text_returns_event_id():
    outbound, _ = _make_outbound()
    event_id = _run(outbound.send_text("!room:ex.org", "hi"))
    assert event_id == "$sent_event"


def test_send_text_markdown_produces_formatted_body():
    outbound, client = _make_outbound()
    _run(outbound.send_text("!room:ex.org", "**bold** and `code`"))
    content = client.room_send.call_args.kwargs["content"]
    assert content["format"] == "org.matrix.custom.html"
    assert "<strong>" in content["formatted_body"]
    assert "<code>" in content["formatted_body"]


def test_send_text_plain_text_no_formatted_body():
    outbound, client = _make_outbound()
    _run(outbound.send_text("!room:ex.org", "just plain text"))
    content = client.room_send.call_args.kwargs["content"]
    # Plain text: no HTML markup needed
    assert "formatted_body" not in content or content.get("format") is None


def test_send_text_chunks_long_text():
    outbound, client = _make_outbound(chunk_limit=10)
    long_text = "a" * 25
    _run(outbound.send_text("!room:ex.org", long_text))
    assert client.room_send.call_count >= 2


def test_send_text_with_thread_id_sets_relates_to():
    outbound, client = _make_outbound()
    _run(outbound.send_text("!room:ex.org", "reply", thread_id="$thread"))
    content = client.room_send.call_args.kwargs["content"]
    assert "m.relates_to" in content
    assert content["m.relates_to"]["event_id"] == "$thread"


# ---------------------------------------------------------------------------
# send_draft / edit_draft / finalise_draft
# ---------------------------------------------------------------------------

def test_send_draft_posts_placeholder():
    outbound, client = _make_outbound()
    event_id = _run(outbound.send_draft("!room:ex.org"))
    client.room_send.assert_called_once()
    content = client.room_send.call_args.kwargs["content"]
    assert content["body"] == "…"
    assert event_id == "$sent_event"


def test_edit_draft_uses_m_new_content():
    outbound, client = _make_outbound()
    _run(outbound.edit_draft("!room:ex.org", "$draft_event", "updated text"))
    content = client.room_send.call_args.kwargs["content"]
    assert "m.new_content" in content
    assert content["m.relates_to"]["rel_type"] == "m.replace"
    assert content["m.relates_to"]["event_id"] == "$draft_event"
    assert content["m.new_content"]["body"] == "updated text"


def test_edit_draft_markdown_renders_html():
    outbound, client = _make_outbound()
    _run(outbound.edit_draft("!room:ex.org", "$draft_event", "**bold**"))
    new_content = client.room_send.call_args.kwargs["content"]["m.new_content"]
    assert new_content.get("format") == "org.matrix.custom.html"
    assert "<strong>" in new_content.get("formatted_body", "")


def test_finalise_draft_calls_edit():
    outbound, client = _make_outbound()
    _run(outbound.finalise_draft("!room:ex.org", "$draft_event", "final text"))
    content = client.room_send.call_args.kwargs["content"]
    assert "m.new_content" in content
    assert content["m.new_content"]["body"] == "final text"


# ---------------------------------------------------------------------------
# send_reaction
# ---------------------------------------------------------------------------

def test_send_reaction():
    outbound, client = _make_outbound()
    _run(outbound.send_reaction("!room:ex.org", "$target", "👍"))
    assert client.room_send.call_count == 1
    call_kwargs = client.room_send.call_args.kwargs
    assert call_kwargs["message_type"] == "m.reaction"
    content = call_kwargs["content"]
    assert content["m.relates_to"]["key"] == "👍"
    assert content["m.relates_to"]["event_id"] == "$target"


# ---------------------------------------------------------------------------
# send_typing
# ---------------------------------------------------------------------------

def test_send_typing_true():
    outbound, client = _make_outbound()
    _run(outbound.send_typing("!room:ex.org", True))
    client.room_typing.assert_called_once_with(
        "!room:ex.org", typing_state=True, timeout=30000
    )


def test_send_typing_false():
    outbound, client = _make_outbound()
    _run(outbound.send_typing("!room:ex.org", False))
    client.room_typing.assert_called_once_with(
        "!room:ex.org", typing_state=False, timeout=30000
    )


def test_send_typing_error_does_not_raise():
    outbound, client = _make_outbound()
    client.room_typing = AsyncMock(side_effect=Exception("network error"))
    # Should swallow the exception silently
    _run(outbound.send_typing("!room:ex.org", True))

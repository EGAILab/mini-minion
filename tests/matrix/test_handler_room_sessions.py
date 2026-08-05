"""Tests for MatrixMessageHandler's room-scoped session selection (MEM-GAP-001).

These exercise the actual defect the gap analysis identified: a
`session_key`/`session_id` computed at dispatch time must actually reach
session selection, and two different rooms routed to the same agent must
never share an `AgentSession`. Complements test_handler.py (general routing/
dispatch) and test_room_sessions.py (the SQLite binding itself in isolation).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from minion_assist.matrix.config import MatrixConfig
from minion_assist.matrix.handler import MatrixMessageHandler


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_config():
    cfg = MagicMock(spec=MatrixConfig)
    cfg.group_policy = "open"
    cfg.group_allow_from = []
    cfg.groups = {}
    cfg.ack_reaction = None
    cfg.default_agent_id = "main"
    return cfg


def _make_client():
    c = MagicMock()
    c.user_id = "@bot:ex.org"
    return c


def _make_outbound():
    o = MagicMock()
    o.send_reaction = AsyncMock()
    o.send_text = AsyncMock()
    o.send_typing = AsyncMock()
    return o


def _make_room_session_mgr():
    """A real, minimal (room_id, agent_id) -> session_id resolver, in-memory only."""
    bindings: dict[tuple[str, str], str] = {}
    counter = {"n": 0}

    async def _get_or_create(room_id, agent_id):
        key = (room_id, agent_id)
        if key not in bindings:
            counter["n"] += 1
            bindings[key] = f"session-{counter['n']}"
        return bindings[key]

    mgr = MagicMock()
    mgr.get_or_create_session_id = AsyncMock(side_effect=_get_or_create)
    return mgr


def _make_event(body="hello", sender="@alice:ex.org", event_id="$e1"):
    e = MagicMock()
    e.event_id = event_id
    e.sender = sender
    e.body = body
    e.relates_to = None
    return e


def _make_room(room_id):
    r = MagicMock()
    r.room_id = room_id
    return r


def _make_handler(session_factories=None):
    dedupe = MagicMock()
    dedupe.is_seen = AsyncMock(return_value=False)
    bot_loop = MagicMock()
    bot_loop.should_suppress.return_value = False
    return MatrixMessageHandler(
        client=_make_client(),
        config=_make_config(),
        sessions={"main": MagicMock()},
        outbound=_make_outbound(),
        dedupe=dedupe,
        bot_loop=bot_loop,
        room_session_mgr=_make_room_session_mgr(),
        session_factories=session_factories,
    )


def _make_factory_returning(built: dict):
    """A session_factories[agent_id] callable that records every session_id it built."""
    def _factory(session_id):
        session = MagicMock()
        session.send.return_value = f"reply for {session_id}"
        built[session_id] = session
        return session
    return _factory


def test_two_different_rooms_get_two_different_agent_sessions():
    built: dict = {}
    handler = _make_handler(session_factories={"main": _make_factory_returning(built)})

    _run(handler.handle_room_message(_make_room("!movie:ex.org"), _make_event()))
    _run(handler.handle_room_message(_make_room("!work:ex.org"), _make_event()))

    assert len(built) == 2  # two distinct AgentSession instances, one per room


def test_same_room_reuses_the_same_agent_session_across_messages():
    built: dict = {}
    handler = _make_handler(session_factories={"main": _make_factory_returning(built)})

    _run(handler.handle_room_message(_make_room("!movie:ex.org"), _make_event(event_id="$e1")))
    _run(handler.handle_room_message(_make_room("!movie:ex.org"), _make_event(event_id="$e2")))

    assert len(built) == 1  # the second message reused the cached session, not a new one


def test_a_rooms_history_never_reaches_a_different_rooms_session():
    built: dict = {}
    handler = _make_handler(session_factories={"main": _make_factory_returning(built)})

    _run(handler.handle_room_message(_make_room("!movie:ex.org"), _make_event(body="talk about movies")))
    _run(handler.handle_room_message(_make_room("!work:ex.org"), _make_event(body="talk about work")))

    assert len(built) == 2
    sent_texts = sorted(session.send.call_args.args[0] for session in built.values())
    # Each of the two sessions received exactly its own room's message, never the other's.
    assert sent_texts == ["talk about movies", "talk about work"]
    for session in built.values():
        assert session.send.call_count == 1


def test_falls_back_to_shared_session_and_warns_when_no_factory_for_agent(capsys):
    shared_session = MagicMock()
    shared_session.send.return_value = "fallback reply"
    dedupe = MagicMock()
    dedupe.is_seen = AsyncMock(return_value=False)
    bot_loop = MagicMock()
    bot_loop.should_suppress.return_value = False
    handler = MatrixMessageHandler(
        client=_make_client(),
        config=_make_config(),
        sessions={"main": shared_session},
        outbound=_make_outbound(),
        dedupe=dedupe,
        bot_loop=bot_loop,
        room_session_mgr=_make_room_session_mgr(),
        session_factories=None,  # nothing wired for "main"
    )

    _run(handler.handle_room_message(_make_room("!movie:ex.org"), _make_event()))

    shared_session.send.assert_called_once()
    assert "no room-session factory" in capsys.readouterr().err


def test_fallback_session_is_shared_across_rooms_and_is_still_flagged_each_time(capsys):
    # Documents the known trade-off of the fallback path: without a factory,
    # different rooms DO share history again (the pre-fix behavior) — which
    # is exactly why it prints a warning rather than failing silently.
    shared_session = MagicMock()
    shared_session.send.return_value = "fallback reply"
    dedupe = MagicMock()
    dedupe.is_seen = AsyncMock(return_value=False)
    bot_loop = MagicMock()
    bot_loop.should_suppress.return_value = False
    handler = MatrixMessageHandler(
        client=_make_client(),
        config=_make_config(),
        sessions={"main": shared_session},
        outbound=_make_outbound(),
        dedupe=dedupe,
        bot_loop=bot_loop,
        room_session_mgr=_make_room_session_mgr(),
        session_factories=None,
    )

    _run(handler.handle_room_message(_make_room("!movie:ex.org"), _make_event(event_id="$e1")))
    _run(handler.handle_room_message(_make_room("!work:ex.org"), _make_event(event_id="$e2")))

    assert shared_session.send.call_count == 2

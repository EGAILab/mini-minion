"""Tests for HeartbeatScheduler."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from minion_assist.heartbeat import HeartbeatScheduler


def _make_config(
    enabled=True,
    interval_seconds=60,
    prompt="HEARTBEAT_OK",
    agent_id="main",
    notification_room_id=None,
):
    cfg = MagicMock()
    cfg.enabled = enabled
    cfg.interval_seconds = interval_seconds
    cfg.prompt = prompt
    cfg.agent_id = agent_id
    cfg.notification_room_id = notification_room_id
    return cfg


def _make_session(response="HEARTBEAT_OK"):
    session = MagicMock()
    session.send.return_value = response
    return session


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------

def test_stop_cancels_timer():
    cfg = _make_config(interval_seconds=3600)
    scheduler = HeartbeatScheduler(cfg, {"main": _make_session()})
    scheduler.start()
    assert scheduler._timer is not None
    scheduler.stop()
    assert scheduler._stopped is True


def test_stop_idempotent():
    cfg = _make_config(interval_seconds=3600)
    scheduler = HeartbeatScheduler(cfg, {"main": _make_session()})
    scheduler.start()
    scheduler.stop()
    scheduler.stop()  # Should not raise


# ---------------------------------------------------------------------------
# _run_heartbeat
# ---------------------------------------------------------------------------

def test_run_heartbeat_calls_session_send():
    session = _make_session("HEARTBEAT_OK")
    cfg = _make_config(prompt="check things")
    scheduler = HeartbeatScheduler(cfg, {"main": session})
    scheduler._run_heartbeat()
    session.send.assert_called_once()
    call_args = session.send.call_args
    assert call_args[0][0] == "check things"


def test_run_heartbeat_injects_heartbeat_respond_tool():
    from minion_assist.tools.heartbeat_respond import HeartbeatRespondTool
    session = _make_session("HEARTBEAT_OK")
    cfg = _make_config()
    scheduler = HeartbeatScheduler(cfg, {"main": session})
    scheduler._run_heartbeat()
    call_kwargs = session.send.call_args[1]
    extra = call_kwargs.get("extra_tools") or []
    assert any(isinstance(t, HeartbeatRespondTool) for t in extra)


def test_run_heartbeat_skips_unknown_agent():
    cfg = _make_config(agent_id="nonexistent")
    scheduler = HeartbeatScheduler(cfg, {"main": _make_session()})
    # Should not raise; just prints a warning and returns
    scheduler._run_heartbeat()


def test_run_heartbeat_ok_suppressed_no_delivery():
    session = _make_session("HEARTBEAT_OK")
    cfg = _make_config()
    delivered = []
    scheduler = HeartbeatScheduler(cfg, {"main": session})
    scheduler._deliver = lambda msg: delivered.append(msg)
    scheduler._run_heartbeat()
    assert delivered == []


def test_run_heartbeat_prose_response_delivered():
    session = _make_session("You have an important email!")
    cfg = _make_config()
    delivered = []
    scheduler = HeartbeatScheduler(cfg, {"main": session})
    scheduler._deliver = lambda msg: delivered.append(msg)
    scheduler._run_heartbeat()
    assert any("important email" in m for m in delivered)


def test_run_heartbeat_respond_tool_messages_delivered():
    """Messages captured by HeartbeatRespondTool are delivered."""
    from minion_assist.tools.heartbeat_respond import HeartbeatRespondTool, HeartbeatResponseCapture

    # Simulate agent calling heartbeat_respond during send()
    def fake_send(prompt, extra_tools=None, stream=False):
        if extra_tools:
            for tool in extra_tools:
                if isinstance(tool, HeartbeatRespondTool):
                    tool.execute(message="Proactive notification!")
        return "HEARTBEAT_OK"

    session = MagicMock()
    session.send.side_effect = fake_send
    cfg = _make_config()
    delivered = []
    scheduler = HeartbeatScheduler(cfg, {"main": session})
    scheduler._deliver = lambda msg: delivered.append(msg)
    scheduler._run_heartbeat()
    assert "Proactive notification!" in delivered


# ---------------------------------------------------------------------------
# _deliver
# ---------------------------------------------------------------------------

def test_deliver_prints_to_terminal_when_no_room(capsys):
    cfg = _make_config(notification_room_id=None)
    scheduler = HeartbeatScheduler(cfg, {})
    scheduler._deliver("Hello from heartbeat")
    out = capsys.readouterr().out
    assert "Hello from heartbeat" in out


def test_deliver_calls_matrix_when_room_configured():
    import asyncio
    cfg = _make_config(notification_room_id="!room:example.org")
    outbound = MagicMock()
    # Make send_text a coroutine
    async def _send_text(room_id, msg):
        pass
    outbound.send_text = _send_text
    loop = asyncio.new_event_loop()
    scheduler = HeartbeatScheduler(cfg, {}, matrix_outbound=outbound, matrix_loop=loop)
    # run_coroutine_threadsafe needs the loop running; use a thread
    result = []
    def run_loop():
        loop.run_forever()
    t = threading.Thread(target=run_loop, daemon=True)
    t.start()
    try:
        scheduler._deliver("Matrix notification")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=2)

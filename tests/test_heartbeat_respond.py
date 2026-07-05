"""Tests for HeartbeatResponseCapture and HeartbeatRespondTool."""

import pytest
from minion_assist.tools.heartbeat_respond import HeartbeatResponseCapture, HeartbeatRespondTool


# ---------------------------------------------------------------------------
# HeartbeatResponseCapture
# ---------------------------------------------------------------------------

def test_capture_starts_empty():
    cap = HeartbeatResponseCapture()
    assert cap.messages == []


def test_capture_messages_are_mutable():
    cap = HeartbeatResponseCapture()
    cap.messages.append("hello")
    assert cap.messages == ["hello"]


# ---------------------------------------------------------------------------
# HeartbeatRespondTool schema
# ---------------------------------------------------------------------------

def test_schema_name():
    tool = HeartbeatRespondTool(HeartbeatResponseCapture())
    assert tool.schema.name == "heartbeat_respond"


def test_schema_is_read_only():
    tool = HeartbeatRespondTool(HeartbeatResponseCapture())
    assert tool.schema.is_read_only is True


def test_schema_has_message_parameter():
    tool = HeartbeatRespondTool(HeartbeatResponseCapture())
    params = tool.schema.parameters
    assert "message" in params["properties"]
    assert "message" in params["required"]


# ---------------------------------------------------------------------------
# HeartbeatRespondTool execute
# ---------------------------------------------------------------------------

def test_execute_appends_message_to_capture():
    cap = HeartbeatResponseCapture()
    tool = HeartbeatRespondTool(cap)
    tool.execute(message="Important update!")
    assert cap.messages == ["Important update!"]


def test_execute_appends_multiple_messages():
    cap = HeartbeatResponseCapture()
    tool = HeartbeatRespondTool(cap)
    tool.execute(message="First")
    tool.execute(message="Second")
    assert cap.messages == ["First", "Second"]


def test_execute_returns_confirmation():
    cap = HeartbeatResponseCapture()
    tool = HeartbeatRespondTool(cap)
    result = tool.execute(message="Test notification")
    assert "queued" in result.lower() or "1 total" in result


def test_execute_empty_message_does_not_append():
    cap = HeartbeatResponseCapture()
    tool = HeartbeatRespondTool(cap)
    result = tool.execute(message="")
    assert cap.messages == []
    assert "Empty" in result or "empty" in result


def test_execute_whitespace_message_does_not_append():
    cap = HeartbeatResponseCapture()
    tool = HeartbeatRespondTool(cap)
    result = tool.execute(message="   ")
    assert cap.messages == []


def test_different_captures_are_independent():
    cap1 = HeartbeatResponseCapture()
    cap2 = HeartbeatResponseCapture()
    HeartbeatRespondTool(cap1).execute(message="for cap1")
    assert cap1.messages == ["for cap1"]
    assert cap2.messages == []

"""Tests for HeartbeatConfig and _resolve_heartbeat config resolver."""

import pytest
from unittest.mock import patch

from minion_assist.config import HeartbeatConfig


# ---------------------------------------------------------------------------
# HeartbeatConfig defaults
# ---------------------------------------------------------------------------

def test_heartbeat_config_disabled_by_default():
    cfg = HeartbeatConfig()
    assert cfg.enabled is False


def test_heartbeat_config_default_interval():
    cfg = HeartbeatConfig()
    assert cfg.interval_seconds == 1800


def test_heartbeat_config_default_agent_id():
    cfg = HeartbeatConfig()
    assert cfg.agent_id == "main"


def test_heartbeat_config_default_notification_room_none():
    cfg = HeartbeatConfig()
    assert cfg.notification_room_id is None


def test_heartbeat_config_is_frozen():
    cfg = HeartbeatConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.enabled = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _resolve_heartbeat
# ---------------------------------------------------------------------------

def test_resolve_heartbeat_defaults_when_absent():
    from minion_assist.config import _resolve_heartbeat
    with patch("minion_assist.config._raw", {}):
        cfg = _resolve_heartbeat()
    assert cfg.enabled is False
    assert cfg.interval_seconds == 1800
    assert cfg.agent_id == "main"
    assert cfg.notification_room_id is None


def test_resolve_heartbeat_enabled():
    from minion_assist.config import _resolve_heartbeat
    with patch("minion_assist.config._raw", {"heartbeat": {"enabled": True}}):
        cfg = _resolve_heartbeat()
    assert cfg.enabled is True


def test_resolve_heartbeat_custom_interval():
    from minion_assist.config import _resolve_heartbeat
    with patch("minion_assist.config._raw", {"heartbeat": {"interval_seconds": 900}}):
        cfg = _resolve_heartbeat()
    assert cfg.interval_seconds == 900


def test_resolve_heartbeat_custom_agent_id():
    from minion_assist.config import _resolve_heartbeat
    with patch("minion_assist.config._raw", {"heartbeat": {"agent_id": "researcher"}}):
        cfg = _resolve_heartbeat()
    assert cfg.agent_id == "researcher"


def test_resolve_heartbeat_notification_room():
    from minion_assist.config import _resolve_heartbeat
    with patch("minion_assist.config._raw", {"heartbeat": {"notification_room_id": "!abc:example.org"}}):
        cfg = _resolve_heartbeat()
    assert cfg.notification_room_id == "!abc:example.org"


def test_resolve_heartbeat_custom_prompt():
    from minion_assist.config import _resolve_heartbeat
    with patch("minion_assist.config._raw", {"heartbeat": {"prompt": "Check email now."}}):
        cfg = _resolve_heartbeat()
    assert cfg.prompt == "Check email now."


def test_resolve_heartbeat_bad_section_returns_defaults():
    from minion_assist.config import _resolve_heartbeat
    with patch("minion_assist.config._raw", {"heartbeat": "not-a-dict"}):
        cfg = _resolve_heartbeat()
    assert cfg.enabled is False

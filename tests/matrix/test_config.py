"""Tests for matrix/config.py — MatrixConfig.from_dict()."""

import pytest

from minion_assist.matrix.config import (
    MatrixBotLoopConfig,
    MatrixConfig,
    MatrixDmConfig,
    MatrixExecApprovalsConfig,
    MatrixRoomConfig,
    MatrixThreadBindingsConfig,
)

_VALID = {
    "homeserver": "https://matrix.example.org",
    "userId": "@bot:example.org",
    "accessToken": "syt_abc123",
}


def test_valid_minimal_config():
    cfg = MatrixConfig.from_dict(_VALID)
    assert cfg.homeserver == "https://matrix.example.org"
    assert cfg.user_id == "@bot:example.org"
    assert cfg.access_token == "syt_abc123"
    assert cfg.password is None
    assert cfg.auto_join == "off"
    assert cfg.default_agent_id == "main"


def test_missing_homeserver_raises():
    raw = {**_VALID}
    del raw["homeserver"]
    with pytest.raises(ValueError, match="homeserver"):
        MatrixConfig.from_dict(raw)


def test_empty_homeserver_raises():
    raw = {**_VALID, "homeserver": "   "}
    with pytest.raises(ValueError, match="homeserver"):
        MatrixConfig.from_dict(raw)


def test_missing_user_id_raises():
    raw = {**_VALID}
    del raw["userId"]
    with pytest.raises(ValueError, match="userId"):
        MatrixConfig.from_dict(raw)


def test_missing_auth_raises():
    raw = {"homeserver": "https://matrix.example.org", "userId": "@bot:example.org"}
    with pytest.raises(ValueError, match="accessToken.*password|password.*accessToken"):
        MatrixConfig.from_dict(raw)


def test_password_auth_accepted():
    raw = {
        "homeserver": "https://matrix.example.org",
        "userId": "@bot:example.org",
        "password": "s3cr3t",
    }
    cfg = MatrixConfig.from_dict(raw)
    assert cfg.password == "s3cr3t"
    assert cfg.access_token is None


def test_optional_fields_defaults():
    cfg = MatrixConfig.from_dict(_VALID)
    assert cfg.text_chunk_limit == 4000
    assert cfg.ack_reaction == "👀"
    assert cfg.group_policy == "open"
    assert isinstance(cfg.dm, MatrixDmConfig)
    assert isinstance(cfg.thread_bindings, MatrixThreadBindingsConfig)
    assert isinstance(cfg.exec_approvals, MatrixExecApprovalsConfig)
    assert isinstance(cfg.bot_loop, MatrixBotLoopConfig)


def test_groups_parsed():
    raw = {
        **_VALID,
        "groups": {
            "!room1:example.org": {"agent": "researcher", "requireMention": True},
        },
    }
    cfg = MatrixConfig.from_dict(raw)
    assert "!room1:example.org" in cfg.groups
    room = cfg.groups["!room1:example.org"]
    assert room.agent == "researcher"
    assert room.require_mention is True


def test_dm_config_parsed():
    raw = {
        **_VALID,
        "dm": {"policy": "allowlist", "allowFrom": ["@alice:example.org"]},
    }
    cfg = MatrixConfig.from_dict(raw)
    assert cfg.dm.policy == "allowlist"
    assert "@alice:example.org" in cfg.dm.allow_from


def test_thread_bindings_parsed():
    raw = {**_VALID, "threadBindings": {"enabled": True, "idleHours": 2, "maxAgeHours": 48}}
    cfg = MatrixConfig.from_dict(raw)
    assert cfg.thread_bindings.idle_hours == 2.0
    assert cfg.thread_bindings.max_age_hours == 48.0


def test_exec_approvals_parsed():
    raw = {
        **_VALID,
        "execApprovals": {"enabled": True, "approvers": ["@admin:example.org"]},
    }
    cfg = MatrixConfig.from_dict(raw)
    assert cfg.exec_approvals.enabled is True
    assert "@admin:example.org" in cfg.exec_approvals.approvers


def test_bot_loop_parsed():
    raw = {
        **_VALID,
        "botLoopProtection": {"enabled": True, "maxEventsPerWindow": 5, "windowSeconds": 30},
    }
    cfg = MatrixConfig.from_dict(raw)
    assert cfg.bot_loop.max_events_per_window == 5
    assert cfg.bot_loop.window_seconds == 30


def test_matrix_room_config_defaults():
    room = MatrixRoomConfig.from_dict({})
    assert room.agent == "main"
    assert room.enabled is True
    assert room.require_mention is False
    assert room.users == []

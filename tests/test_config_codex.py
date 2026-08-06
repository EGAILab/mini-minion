"""Tests for CodexConfig and _resolve_codex config resolver."""

import pytest
from unittest.mock import patch

from minion_assist.config import CodexConfig


# ---------------------------------------------------------------------------
# CodexConfig defaults
# ---------------------------------------------------------------------------

def test_codex_config_allow_all_commands_disabled_by_default():
    cfg = CodexConfig()
    assert cfg.allow_all_commands is False


def test_codex_config_default_auth_refresh_interval():
    cfg = CodexConfig()
    assert cfg.auth_refresh_interval_seconds == 300.0


def test_codex_config_is_frozen():
    cfg = CodexConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.allow_all_commands = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _resolve_codex
# ---------------------------------------------------------------------------

def test_resolve_codex_defaults_when_absent():
    from minion_assist.config import _resolve_codex
    with patch("minion_assist.config._raw", {}):
        cfg = _resolve_codex()
    assert cfg.allow_all_commands is False
    assert cfg.auth_refresh_interval_seconds == 300.0


def test_resolve_codex_allow_all_commands_enabled():
    from minion_assist.config import _resolve_codex
    with patch("minion_assist.config._raw", {"codex": {"allow_all_commands": True}}):
        cfg = _resolve_codex()
    assert cfg.allow_all_commands is True


def test_resolve_codex_custom_auth_refresh_interval():
    from minion_assist.config import _resolve_codex
    with patch("minion_assist.config._raw", {"codex": {"auth_refresh_interval_seconds": 60}}):
        cfg = _resolve_codex()
    assert cfg.auth_refresh_interval_seconds == 60.0


def test_resolve_codex_non_dict_section_falls_back_to_defaults():
    from minion_assist.config import _resolve_codex
    with patch("minion_assist.config._raw", {"codex": "not-a-dict"}):
        cfg = _resolve_codex()
    assert cfg.allow_all_commands is False
    assert cfg.auth_refresh_interval_seconds == 300.0

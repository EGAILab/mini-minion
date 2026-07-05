"""Tests for DreamingConfig resolution in config.py."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch


def _resolve(raw: dict) -> "DreamingConfig":
    """Import DreamingConfig and resolve from a raw dict without touching config.json."""
    # Monkey-patch config._raw then re-run _resolve_dreaming().
    import minion_assist.config as cfg_mod
    from minion_assist.config import DreamingConfig, _resolve_dreaming

    with patch.object(cfg_mod, "_raw", raw):
        return _resolve_dreaming()


class TestDreamingConfigDefaults:
    def test_all_defaults_when_key_absent(self) -> None:
        dc = _resolve({})
        assert dc.enabled is False
        assert dc.hour == 3
        assert dc.minute == 0
        assert dc.timezone == "Australia/Sydney"
        assert dc.lookback_days == 3
        assert dc.agent_id == "main"

    def test_defaults_when_dreaming_is_not_dict(self) -> None:
        dc = _resolve({"dreaming": "true"})
        assert dc.enabled is False
        assert dc.hour == 3

    def test_defaults_when_dreaming_is_empty_dict(self) -> None:
        dc = _resolve({"dreaming": {}})
        assert dc.enabled is False
        assert dc.hour == 3

    def test_enabled_false_is_default(self) -> None:
        dc = _resolve({"dreaming": {"hour": 2}})
        assert dc.enabled is False


class TestDreamingConfigExplicitValues:
    def test_enabled_true(self) -> None:
        dc = _resolve({"dreaming": {"enabled": True}})
        assert dc.enabled is True

    def test_hour_override(self) -> None:
        dc = _resolve({"dreaming": {"hour": 4}})
        assert dc.hour == 4

    def test_minute_override(self) -> None:
        dc = _resolve({"dreaming": {"minute": 30}})
        assert dc.minute == 30

    def test_timezone_override(self) -> None:
        dc = _resolve({"dreaming": {"timezone": "America/New_York"}})
        assert dc.timezone == "America/New_York"

    def test_lookback_days_override(self) -> None:
        dc = _resolve({"dreaming": {"lookback_days": 7}})
        assert dc.lookback_days == 7

    def test_agent_id_override(self) -> None:
        dc = _resolve({"dreaming": {"agent_id": "researcher"}})
        assert dc.agent_id == "researcher"

    def test_all_fields_together(self) -> None:
        dc = _resolve(
            {
                "dreaming": {
                    "enabled": True,
                    "hour": 2,
                    "minute": 45,
                    "timezone": "Europe/London",
                    "lookback_days": 5,
                    "agent_id": "main",
                }
            }
        )
        assert dc.enabled is True
        assert dc.hour == 2
        assert dc.minute == 45
        assert dc.timezone == "Europe/London"
        assert dc.lookback_days == 5
        assert dc.agent_id == "main"


class TestDreamingConfigClamping:
    def test_hour_clamped_below(self) -> None:
        dc = _resolve({"dreaming": {"hour": -5}})
        assert dc.hour == 0

    def test_hour_clamped_above(self) -> None:
        dc = _resolve({"dreaming": {"hour": 99}})
        assert dc.hour == 23

    def test_minute_clamped_below(self) -> None:
        dc = _resolve({"dreaming": {"minute": -1}})
        assert dc.minute == 0

    def test_minute_clamped_above(self) -> None:
        dc = _resolve({"dreaming": {"minute": 100}})
        assert dc.minute == 59

    def test_lookback_days_minimum_one(self) -> None:
        dc = _resolve({"dreaming": {"lookback_days": 0}})
        assert dc.lookback_days == 1


class TestDreamingConfigImmutable:
    def test_frozen_dataclass(self) -> None:
        import dataclasses
        from minion_assist.config import DreamingConfig

        assert dataclasses.is_dataclass(DreamingConfig)
        dc = DreamingConfig()
        try:
            dc.enabled = True  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except dataclasses.FrozenInstanceError:
            pass

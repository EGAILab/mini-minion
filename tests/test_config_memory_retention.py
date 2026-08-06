"""Tests for MemoryRetentionConfig resolution in config.py (MEM-GAP-015)."""

from __future__ import annotations

from unittest.mock import patch


def _resolve(raw: dict) -> "MemoryRetentionConfig":
    """Import MemoryRetentionConfig and resolve from a raw dict without touching config.json."""
    import minion_assist.config as cfg_mod
    from minion_assist.config import _resolve_memory_retention

    with patch.object(cfg_mod, "_raw", raw):
        return _resolve_memory_retention()


class TestMemoryRetentionConfigDefaults:
    def test_all_defaults_when_key_absent(self) -> None:
        mr = _resolve({})
        assert mr.enabled is False
        assert mr.hour == 4
        assert mr.minute == 45
        assert mr.timezone == "Australia/Sydney"
        assert mr.retention_days == 30

    def test_defaults_when_memory_retention_is_not_dict(self) -> None:
        mr = _resolve({"memory_retention": "true"})
        assert mr.enabled is False
        assert mr.retention_days == 30

    def test_defaults_when_memory_retention_is_empty_dict(self) -> None:
        mr = _resolve({"memory_retention": {}})
        assert mr.enabled is False
        assert mr.hour == 4

    def test_enabled_false_is_default(self) -> None:
        mr = _resolve({"memory_retention": {"hour": 2}})
        assert mr.enabled is False


class TestMemoryRetentionConfigExplicitValues:
    def test_enabled_true(self) -> None:
        mr = _resolve({"memory_retention": {"enabled": True}})
        assert mr.enabled is True

    def test_hour_override(self) -> None:
        mr = _resolve({"memory_retention": {"hour": 5}})
        assert mr.hour == 5

    def test_minute_override(self) -> None:
        mr = _resolve({"memory_retention": {"minute": 15}})
        assert mr.minute == 15

    def test_timezone_override(self) -> None:
        mr = _resolve({"memory_retention": {"timezone": "America/New_York"}})
        assert mr.timezone == "America/New_York"

    def test_retention_days_override(self) -> None:
        mr = _resolve({"memory_retention": {"retention_days": 7}})
        assert mr.retention_days == 7

    def test_all_fields_together(self) -> None:
        mr = _resolve(
            {
                "memory_retention": {
                    "enabled": True,
                    "hour": 2,
                    "minute": 10,
                    "timezone": "Europe/London",
                    "retention_days": 90,
                }
            }
        )
        assert mr.enabled is True
        assert mr.hour == 2
        assert mr.minute == 10
        assert mr.timezone == "Europe/London"
        assert mr.retention_days == 90


class TestMemoryRetentionConfigClamping:
    def test_hour_clamped_below(self) -> None:
        mr = _resolve({"memory_retention": {"hour": -5}})
        assert mr.hour == 0

    def test_hour_clamped_above(self) -> None:
        mr = _resolve({"memory_retention": {"hour": 99}})
        assert mr.hour == 23

    def test_minute_clamped_below(self) -> None:
        mr = _resolve({"memory_retention": {"minute": -1}})
        assert mr.minute == 0

    def test_minute_clamped_above(self) -> None:
        mr = _resolve({"memory_retention": {"minute": 100}})
        assert mr.minute == 59

    def test_retention_days_minimum_one(self) -> None:
        mr = _resolve({"memory_retention": {"retention_days": 0}})
        assert mr.retention_days == 1


class TestMemoryRetentionConfigImmutable:
    def test_frozen_dataclass(self) -> None:
        import dataclasses

        from minion_assist.config import MemoryRetentionConfig

        assert dataclasses.is_dataclass(MemoryRetentionConfig)
        mr = MemoryRetentionConfig()
        try:
            mr.enabled = True  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except dataclasses.FrozenInstanceError:
            pass

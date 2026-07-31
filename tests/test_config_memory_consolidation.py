"""Tests for MemoryConsolidationConfig resolution in config.py (Stage One Phase 5, slice D)."""

from __future__ import annotations

from unittest.mock import patch


def _resolve(raw: dict) -> "MemoryConsolidationConfig":
    """Import MemoryConsolidationConfig and resolve from a raw dict without touching config.json."""
    import minion_assist.config as cfg_mod
    from minion_assist.config import _resolve_memory_consolidation

    with patch.object(cfg_mod, "_raw", raw):
        return _resolve_memory_consolidation()


class TestMemoryConsolidationConfigDefaults:
    def test_all_defaults_when_key_absent(self) -> None:
        mc = _resolve({})
        assert mc.enabled is False
        assert mc.hour == 4
        assert mc.minute == 0
        assert mc.timezone == "Australia/Sydney"
        assert mc.agent_id == "main"
        assert mc.top_n == 5

    def test_defaults_when_memory_consolidation_is_not_dict(self) -> None:
        mc = _resolve({"memory_consolidation": "true"})
        assert mc.enabled is False
        assert mc.hour == 4

    def test_defaults_when_memory_consolidation_is_empty_dict(self) -> None:
        mc = _resolve({"memory_consolidation": {}})
        assert mc.enabled is False
        assert mc.hour == 4

    def test_enabled_false_is_default(self) -> None:
        mc = _resolve({"memory_consolidation": {"hour": 2}})
        assert mc.enabled is False


class TestMemoryConsolidationConfigExplicitValues:
    def test_enabled_true(self) -> None:
        mc = _resolve({"memory_consolidation": {"enabled": True}})
        assert mc.enabled is True

    def test_hour_override(self) -> None:
        mc = _resolve({"memory_consolidation": {"hour": 5}})
        assert mc.hour == 5

    def test_minute_override(self) -> None:
        mc = _resolve({"memory_consolidation": {"minute": 30}})
        assert mc.minute == 30

    def test_timezone_override(self) -> None:
        mc = _resolve({"memory_consolidation": {"timezone": "America/New_York"}})
        assert mc.timezone == "America/New_York"

    def test_agent_id_override(self) -> None:
        mc = _resolve({"memory_consolidation": {"agent_id": "researcher"}})
        assert mc.agent_id == "researcher"

    def test_top_n_override(self) -> None:
        mc = _resolve({"memory_consolidation": {"top_n": 10}})
        assert mc.top_n == 10

    def test_all_fields_together(self) -> None:
        mc = _resolve(
            {
                "memory_consolidation": {
                    "enabled": True,
                    "hour": 2,
                    "minute": 45,
                    "timezone": "Europe/London",
                    "agent_id": "main",
                    "top_n": 3,
                }
            }
        )
        assert mc.enabled is True
        assert mc.hour == 2
        assert mc.minute == 45
        assert mc.timezone == "Europe/London"
        assert mc.agent_id == "main"
        assert mc.top_n == 3


class TestMemoryConsolidationConfigClamping:
    def test_hour_clamped_below(self) -> None:
        mc = _resolve({"memory_consolidation": {"hour": -5}})
        assert mc.hour == 0

    def test_hour_clamped_above(self) -> None:
        mc = _resolve({"memory_consolidation": {"hour": 99}})
        assert mc.hour == 23

    def test_minute_clamped_below(self) -> None:
        mc = _resolve({"memory_consolidation": {"minute": -1}})
        assert mc.minute == 0

    def test_minute_clamped_above(self) -> None:
        mc = _resolve({"memory_consolidation": {"minute": 100}})
        assert mc.minute == 59

    def test_top_n_minimum_one(self) -> None:
        mc = _resolve({"memory_consolidation": {"top_n": 0}})
        assert mc.top_n == 1


class TestMemoryConsolidationConfigImmutable:
    def test_frozen_dataclass(self) -> None:
        import dataclasses

        from minion_assist.config import MemoryConsolidationConfig

        assert dataclasses.is_dataclass(MemoryConsolidationConfig)
        mc = MemoryConsolidationConfig()
        try:
            mc.enabled = True  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except dataclasses.FrozenInstanceError:
            pass

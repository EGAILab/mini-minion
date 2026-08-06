"""Tests for MemoryReconciliationConfig resolution in config.py (MEM-GAP-007)."""

from __future__ import annotations

from unittest.mock import patch


def _resolve(raw: dict) -> "MemoryReconciliationConfig":
    """Import MemoryReconciliationConfig and resolve from a raw dict without touching config.json."""
    import minion_assist.config as cfg_mod
    from minion_assist.config import _resolve_memory_reconciliation

    with patch.object(cfg_mod, "_raw", raw):
        return _resolve_memory_reconciliation()


class TestMemoryReconciliationConfigDefaults:
    def test_all_defaults_when_key_absent(self) -> None:
        mr = _resolve({})
        assert mr.interval_seconds == 300
        assert mr.quiet_seconds == 60

    def test_defaults_when_memory_reconciliation_is_not_dict(self) -> None:
        mr = _resolve({"memory_reconciliation": "true"})
        assert mr.interval_seconds == 300

    def test_defaults_when_memory_reconciliation_is_empty_dict(self) -> None:
        mr = _resolve({"memory_reconciliation": {}})
        assert mr.interval_seconds == 300
        assert mr.quiet_seconds == 60


class TestMemoryReconciliationConfigExplicitValues:
    def test_interval_seconds_override(self) -> None:
        mr = _resolve({"memory_reconciliation": {"interval_seconds": 600}})
        assert mr.interval_seconds == 600

    def test_quiet_seconds_override(self) -> None:
        mr = _resolve({"memory_reconciliation": {"quiet_seconds": 120}})
        assert mr.quiet_seconds == 120

    def test_all_fields_together(self) -> None:
        mr = _resolve(
            {"memory_reconciliation": {"interval_seconds": 900, "quiet_seconds": 30}}
        )
        assert mr.interval_seconds == 900
        assert mr.quiet_seconds == 30


class TestMemoryReconciliationConfigClamping:
    def test_interval_seconds_clamped_below(self) -> None:
        mr = _resolve({"memory_reconciliation": {"interval_seconds": 1}})
        assert mr.interval_seconds == 60

    def test_interval_seconds_clamped_above(self) -> None:
        mr = _resolve({"memory_reconciliation": {"interval_seconds": 999999}})
        assert mr.interval_seconds == 3600

    def test_quiet_seconds_minimum_zero(self) -> None:
        mr = _resolve({"memory_reconciliation": {"quiet_seconds": -10}})
        assert mr.quiet_seconds == 0


class TestMemoryReconciliationConfigImmutable:
    def test_frozen_dataclass(self) -> None:
        import dataclasses

        from minion_assist.config import MemoryReconciliationConfig

        assert dataclasses.is_dataclass(MemoryReconciliationConfig)
        mr = MemoryReconciliationConfig()
        try:
            mr.interval_seconds = 999  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except dataclasses.FrozenInstanceError:
            pass

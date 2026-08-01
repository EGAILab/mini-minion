"""Tests for CommitmentsConfig resolution in config.py (Stage One Phase 6, slice B)."""

from __future__ import annotations

from unittest.mock import patch


def _resolve(raw: dict) -> "CommitmentsConfig":
    """Import CommitmentsConfig and resolve from a raw dict without touching config.json."""
    import minion_assist.config as cfg_mod
    from minion_assist.config import _resolve_commitments

    with patch.object(cfg_mod, "_raw", raw):
        return _resolve_commitments()


class TestCommitmentsConfigDefaults:
    def test_disabled_when_key_absent(self) -> None:
        cc = _resolve({})
        assert cc.enabled is False

    def test_defaults_when_commitments_is_not_dict(self) -> None:
        cc = _resolve({"commitments": "true"})
        assert cc.enabled is False

    def test_defaults_when_commitments_is_empty_dict(self) -> None:
        cc = _resolve({"commitments": {}})
        assert cc.enabled is False


class TestCommitmentsConfigExplicitValues:
    def test_enabled_true(self) -> None:
        cc = _resolve({"commitments": {"enabled": True}})
        assert cc.enabled is True

    def test_enabled_false_explicit(self) -> None:
        cc = _resolve({"commitments": {"enabled": False}})
        assert cc.enabled is False


class TestCommitmentsConfigImmutable:
    def test_frozen_dataclass(self) -> None:
        import dataclasses

        from minion_assist.config import CommitmentsConfig

        assert dataclasses.is_dataclass(CommitmentsConfig)
        cc = CommitmentsConfig()
        try:
            cc.enabled = True  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except dataclasses.FrozenInstanceError:
            pass

"""Tests for KnowledgeDigestConfig resolution in config.py (Stage One Phase 7, slice D)."""

from __future__ import annotations

from unittest.mock import patch


def _resolve(raw: dict) -> "KnowledgeDigestConfig":
    """Import KnowledgeDigestConfig and resolve from a raw dict without touching config.json."""
    import minion_assist.config as cfg_mod
    from minion_assist.config import _resolve_knowledge_digest

    with patch.object(cfg_mod, "_raw", raw):
        return _resolve_knowledge_digest()


class TestKnowledgeDigestConfigDefaults:
    def test_all_defaults_when_key_absent(self) -> None:
        kd = _resolve({})
        assert kd.enabled is False
        assert kd.hour == 4
        assert kd.minute == 30
        assert kd.timezone == "Australia/Sydney"
        assert kd.agent_id == "main"
        assert kd.max_chars == 8000

    def test_defaults_when_knowledge_digest_is_not_dict(self) -> None:
        kd = _resolve({"knowledge_digest": "true"})
        assert kd.enabled is False
        assert kd.hour == 4

    def test_defaults_when_knowledge_digest_is_empty_dict(self) -> None:
        kd = _resolve({"knowledge_digest": {}})
        assert kd.enabled is False
        assert kd.hour == 4

    def test_enabled_false_is_default(self) -> None:
        kd = _resolve({"knowledge_digest": {"hour": 2}})
        assert kd.enabled is False


class TestKnowledgeDigestConfigExplicitValues:
    def test_enabled_true(self) -> None:
        kd = _resolve({"knowledge_digest": {"enabled": True}})
        assert kd.enabled is True

    def test_hour_override(self) -> None:
        kd = _resolve({"knowledge_digest": {"hour": 5}})
        assert kd.hour == 5

    def test_minute_override(self) -> None:
        kd = _resolve({"knowledge_digest": {"minute": 15}})
        assert kd.minute == 15

    def test_timezone_override(self) -> None:
        kd = _resolve({"knowledge_digest": {"timezone": "America/New_York"}})
        assert kd.timezone == "America/New_York"

    def test_agent_id_override(self) -> None:
        kd = _resolve({"knowledge_digest": {"agent_id": "researcher"}})
        assert kd.agent_id == "researcher"

    def test_max_chars_override(self) -> None:
        kd = _resolve({"knowledge_digest": {"max_chars": 2000}})
        assert kd.max_chars == 2000

    def test_all_fields_together(self) -> None:
        kd = _resolve(
            {
                "knowledge_digest": {
                    "enabled": True,
                    "hour": 2,
                    "minute": 45,
                    "timezone": "Europe/London",
                    "agent_id": "main",
                    "max_chars": 4000,
                }
            }
        )
        assert kd.enabled is True
        assert kd.hour == 2
        assert kd.minute == 45
        assert kd.timezone == "Europe/London"
        assert kd.agent_id == "main"
        assert kd.max_chars == 4000


class TestKnowledgeDigestConfigClamping:
    def test_hour_clamped_below(self) -> None:
        kd = _resolve({"knowledge_digest": {"hour": -5}})
        assert kd.hour == 0

    def test_hour_clamped_above(self) -> None:
        kd = _resolve({"knowledge_digest": {"hour": 99}})
        assert kd.hour == 23

    def test_minute_clamped_below(self) -> None:
        kd = _resolve({"knowledge_digest": {"minute": -1}})
        assert kd.minute == 0

    def test_minute_clamped_above(self) -> None:
        kd = _resolve({"knowledge_digest": {"minute": 100}})
        assert kd.minute == 59

    def test_max_chars_minimum_one(self) -> None:
        kd = _resolve({"knowledge_digest": {"max_chars": 0}})
        assert kd.max_chars == 1


class TestKnowledgeDigestConfigImmutable:
    def test_frozen_dataclass(self) -> None:
        import dataclasses

        from minion_assist.config import KnowledgeDigestConfig

        assert dataclasses.is_dataclass(KnowledgeDigestConfig)
        kd = KnowledgeDigestConfig()
        try:
            kd.enabled = True  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except dataclasses.FrozenInstanceError:
            pass

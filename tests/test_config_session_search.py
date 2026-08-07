"""Tests for SessionSearchConfig resolution in config.py (R2-GAP-012)."""

from __future__ import annotations

from unittest.mock import patch


def _resolve(raw: dict) -> "SessionSearchConfig":
    """Import SessionSearchConfig and resolve from a raw dict without touching config.json."""
    import minion_assist.config as cfg_mod
    from minion_assist.config import _resolve_session_search

    with patch.object(cfg_mod, "_raw", raw):
        return _resolve_session_search()


class TestSessionSearchConfigDefaults:
    def test_default_when_key_absent(self) -> None:
        ss = _resolve({})
        assert ss.min_similarity == 0.15

    def test_default_when_session_search_is_not_dict(self) -> None:
        ss = _resolve({"session_search": "0.5"})
        assert ss.min_similarity == 0.15

    def test_default_when_session_search_is_empty_dict(self) -> None:
        ss = _resolve({"session_search": {}})
        assert ss.min_similarity == 0.15


class TestSessionSearchConfigExplicitValues:
    def test_min_similarity_override(self) -> None:
        ss = _resolve({"session_search": {"min_similarity": 0.4}})
        assert ss.min_similarity == 0.4

    def test_min_similarity_accepts_zero(self) -> None:
        ss = _resolve({"session_search": {"min_similarity": 0.0}})
        assert ss.min_similarity == 0.0

    def test_min_similarity_accepts_negative(self) -> None:
        # Not clamped -- a negative floor is a legitimate (if unusual) way
        # to widen retrieval rather than narrow it.
        ss = _resolve({"session_search": {"min_similarity": -0.2}})
        assert ss.min_similarity == -0.2


class TestSessionSearchConfigImmutable:
    def test_frozen_dataclass(self) -> None:
        import dataclasses

        from minion_assist.config import SessionSearchConfig

        assert dataclasses.is_dataclass(SessionSearchConfig)
        ss = SessionSearchConfig()
        try:
            ss.min_similarity = 0.9  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except dataclasses.FrozenInstanceError:
            pass

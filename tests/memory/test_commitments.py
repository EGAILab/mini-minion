"""Tests for memory/commitments.py: commitment extraction (Stage One Phase 6, slice B)."""

from __future__ import annotations

import datetime as _dt
import json
from unittest.mock import Mock

import pytest

from minion_assist.memory.commitments import (
    _build_user_message,
    _parse_extraction_output,
    _parse_time_epoch,
    _validate_candidate,
    extract_commitments,
)
from minion_assist.providers.base import LLMResponse

_NOW = 2_000_000_000.0  # a fixed reference "now" for deterministic tests


def _iso(epoch: float) -> str:
    return _dt.datetime.fromtimestamp(epoch).isoformat()


def _raw_candidate(**overrides) -> dict:
    base = {
        "kind": "open_loop",
        "sensitivity": "routine",
        "source": "inferred_user_context",
        "reason": "User mentioned an interview.",
        "suggested_text": "How did the interview go?",
        "dedupe_key": "interview:2026-08-01",
        "confidence": 0.8,
        "due_earliest": _iso(_NOW + 3600),
        "due_latest": _iso(_NOW + 7200),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# _parse_time_epoch
# ---------------------------------------------------------------------------

def test_parse_time_epoch_parses_a_valid_iso_string():
    assert _parse_time_epoch(_iso(_NOW)) == pytest.approx(_NOW, abs=1.0)


def test_parse_time_epoch_returns_none_for_garbage():
    assert _parse_time_epoch("not a date") is None


def test_parse_time_epoch_returns_none_for_a_non_string():
    assert _parse_time_epoch(None) is None


# ---------------------------------------------------------------------------
# _parse_extraction_output
# ---------------------------------------------------------------------------

def test_parse_extraction_output_extracts_candidates():
    text = json.dumps({"candidates": [_raw_candidate()]})

    candidates = _parse_extraction_output(text)

    assert len(candidates) == 1
    assert candidates[0]["dedupe_key"] == "interview:2026-08-01"


def test_parse_extraction_output_returns_empty_list_for_empty_text():
    assert _parse_extraction_output("") == []


def test_parse_extraction_output_returns_empty_list_for_invalid_json():
    assert _parse_extraction_output("not json at all") == []


def test_parse_extraction_output_returns_empty_list_when_candidates_key_missing():
    assert _parse_extraction_output(json.dumps({"other": "stuff"})) == []


def test_parse_extraction_output_returns_empty_list_when_candidates_is_not_a_list():
    assert _parse_extraction_output(json.dumps({"candidates": "oops"})) == []


def test_parse_extraction_output_ignores_non_dict_candidate_entries():
    text = json.dumps({"candidates": [_raw_candidate(), "not a dict", 42]})

    candidates = _parse_extraction_output(text)

    assert len(candidates) == 1


def test_parse_extraction_output_returns_empty_list_for_a_bare_json_array():
    assert _parse_extraction_output(json.dumps([_raw_candidate()])) == []


# ---------------------------------------------------------------------------
# _validate_candidate
# ---------------------------------------------------------------------------

def test_validate_candidate_accepts_a_well_formed_candidate():
    result = _validate_candidate(_raw_candidate(), _NOW, min_due_seconds=60.0)

    assert result is not None
    assert result["kind"] == "open_loop"
    assert result["confidence"] == 0.8


def test_validate_candidate_rejects_an_unknown_kind():
    assert _validate_candidate(_raw_candidate(kind="not_a_kind"), _NOW, 60.0) is None


def test_validate_candidate_rejects_an_unknown_sensitivity():
    assert _validate_candidate(_raw_candidate(sensitivity="extreme"), _NOW, 60.0) is None


def test_validate_candidate_rejects_an_unknown_source():
    assert _validate_candidate(_raw_candidate(source="made_up"), _NOW, 60.0) is None


def test_validate_candidate_rejects_a_missing_reason():
    raw = _raw_candidate()
    del raw["reason"]
    assert _validate_candidate(raw, _NOW, 60.0) is None


def test_validate_candidate_rejects_an_empty_dedupe_key():
    assert _validate_candidate(_raw_candidate(dedupe_key=""), _NOW, 60.0) is None


def test_validate_candidate_rejects_a_non_numeric_confidence():
    assert _validate_candidate(_raw_candidate(confidence="high"), _NOW, 60.0) is None


def test_validate_candidate_rejects_confidence_below_the_routine_threshold():
    assert _validate_candidate(_raw_candidate(confidence=0.5), _NOW, 60.0) is None


def test_validate_candidate_accepts_confidence_at_the_routine_threshold():
    assert _validate_candidate(_raw_candidate(confidence=0.72), _NOW, 60.0) is not None


def test_validate_candidate_rejects_a_care_candidate_below_the_care_threshold():
    raw = _raw_candidate(kind="care_check_in", sensitivity="care", confidence=0.8)
    assert _validate_candidate(raw, _NOW, 60.0) is None


def test_validate_candidate_accepts_a_care_candidate_above_the_care_threshold():
    raw = _raw_candidate(kind="care_check_in", sensitivity="care", confidence=0.9)
    assert _validate_candidate(raw, _NOW, 60.0) is not None


def test_validate_candidate_applies_the_care_threshold_when_sensitivity_is_care_even_if_kind_is_not():
    # Either kind == "care_check_in" OR sensitivity == "care" triggers the
    # higher bar -- matches OpenClaw's own either/or gate.
    raw = _raw_candidate(kind="open_loop", sensitivity="care", confidence=0.8)
    assert _validate_candidate(raw, _NOW, 60.0) is None


def test_validate_candidate_rejects_a_due_earliest_in_the_past():
    raw = _raw_candidate(due_earliest=_iso(_NOW - 3600))
    assert _validate_candidate(raw, _NOW, 60.0) is None


def test_validate_candidate_rejects_an_unparseable_due_earliest():
    raw = _raw_candidate(due_earliest="not a date")
    assert _validate_candidate(raw, _NOW, 60.0) is None


def test_validate_candidate_clamps_due_earliest_to_the_minimum_due_seconds():
    # Model says 10 seconds from now, but min_due_seconds requires 1800.
    raw = _raw_candidate(due_earliest=_iso(_NOW + 10), due_latest=_iso(_NOW + 20))

    result = _validate_candidate(raw, _NOW, min_due_seconds=1800.0)

    assert result["due_earliest"] >= _NOW + 1800.0


def test_validate_candidate_falls_back_to_a_default_window_when_due_latest_missing():
    raw = _raw_candidate(due_earliest=_iso(_NOW + 3600))
    del raw["due_latest"]

    result = _validate_candidate(raw, _NOW, min_due_seconds=60.0)

    assert result["due_latest"] > result["due_earliest"]


def test_validate_candidate_falls_back_to_a_default_window_when_due_latest_is_before_due_earliest():
    raw = _raw_candidate(due_earliest=_iso(_NOW + 3600), due_latest=_iso(_NOW + 1800))

    result = _validate_candidate(raw, _NOW, min_due_seconds=60.0)

    assert result["due_latest"] > result["due_earliest"]


# ---------------------------------------------------------------------------
# _build_user_message
# ---------------------------------------------------------------------------

def test_build_user_message_includes_the_exchange():
    message = _build_user_message(_NOW, "I have an interview tomorrow.", "Good luck!", [])

    assert "I have an interview tomorrow." in message
    assert "Good luck!" in message


def test_build_user_message_lists_existing_pending_commitments():
    existing = [{"dedupe_key": "interview:2026-08-01", "kind": "open_loop", "reason": "..."}]

    message = _build_user_message(_NOW, "user text", "assistant text", existing)

    assert "interview:2026-08-01" in message


def test_build_user_message_shows_none_when_no_existing_pending():
    message = _build_user_message(_NOW, "user text", "assistant text", [])

    assert "(none)" in message


# ---------------------------------------------------------------------------
# extract_commitments
# ---------------------------------------------------------------------------

def _provider_returning(candidates: list[dict]) -> Mock:
    provider = Mock()
    provider.chat = Mock(
        return_value=LLMResponse(
            text=json.dumps({"candidates": candidates}), finish_reason="stop"
        )
    )
    return provider


def test_extract_commitments_returns_validated_candidates():
    provider = _provider_returning([_raw_candidate()])

    result = extract_commitments(provider, "user text", "assistant text", [], _NOW, 60.0)

    assert len(result) == 1
    assert result[0]["dedupe_key"] == "interview:2026-08-01"


def test_extract_commitments_returns_empty_list_when_nothing_qualifies():
    provider = _provider_returning([])

    result = extract_commitments(provider, "hi", "hello", [], _NOW, 60.0)

    assert result == []


def test_extract_commitments_filters_out_invalid_candidates():
    provider = _provider_returning([_raw_candidate(confidence=0.1)])

    result = extract_commitments(provider, "user text", "assistant text", [], _NOW, 60.0)

    assert result == []


def test_extract_commitments_calls_the_provider_with_no_tools():
    provider = _provider_returning([])

    extract_commitments(provider, "user text", "assistant text", [], _NOW, 60.0)

    assert provider.chat.call_args.kwargs["tools"] == []


def test_extract_commitments_uses_the_fixed_extraction_system_prompt():
    from minion_assist.memory.commitments import _EXTRACT_SYSTEM

    provider = _provider_returning([])

    extract_commitments(provider, "user text", "assistant text", [], _NOW, 60.0)

    assert provider.chat.call_args.kwargs["system"] == _EXTRACT_SYSTEM


def test_extract_commitments_tells_the_model_to_skip_explicit_reminders():
    from minion_assist.memory.commitments import _EXTRACT_SYSTEM

    assert "remind me" in _EXTRACT_SYSTEM.lower()
    assert "skip" in _EXTRACT_SYSTEM.lower()

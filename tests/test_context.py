"""Tests for context window management (Compactor).

Tests cover:
- Token estimation
- needs_compaction() True/False
- _select() split point
- _format_head() message rendering
- _prune() tool output truncation
- compact() no-op when under budget
- compact() end-to-end with mock provider
- compact() graceful fallback on provider error
- compact() with too-short history (< 2 messages)
"""

import json

from minion_assist.context import (
    Compactor,
    _estimate_tokens,
)
from minion_assist.providers.base import LLMResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(role: str, content: str, **extra) -> dict:
    """Build a minimal message dict."""
    return {"role": role, "content": content, **extra}


def _tool_result(content: str, call_id: str = "tc1") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


class _MockProvider:
    """Minimal provider that returns a fixed summary text."""

    def __init__(self, summary: str = "Mock summary.") -> None:
        self._summary = summary
        self.calls: list[dict] = []

    def chat(self, system, messages, tools, max_tokens, on_token=None) -> LLMResponse:
        self.calls.append({"system": system, "messages": messages})
        return LLMResponse(text=self._summary)


class _FailingProvider:
    """Provider that raises on every call — tests error fallback."""

    def chat(self, system, messages, tools, max_tokens, on_token=None) -> LLMResponse:
        raise RuntimeError("provider unavailable")


# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------

def test_estimate_tokens_returns_positive():
    msg = _msg("user", "hello")
    assert _estimate_tokens(msg) >= 1


def test_estimate_tokens_grows_with_content():
    short = _msg("user", "hi")
    long = _msg("user", "x" * 400)
    assert _estimate_tokens(long) > _estimate_tokens(short)


def test_estimate_tokens_consistent_with_json_length():
    """When tiktoken is not installed, estimate uses the 4-char heuristic."""
    try:
        import tiktoken  # noqa: F401
        # tiktoken installed — exact counts differ from heuristic, skip check
        import pytest
        pytest.skip("tiktoken installed — heuristic formula test not applicable")
    except ImportError:
        msg = _msg("user", "hello world")
        expected = max(1, len(json.dumps(msg)) // 4)
        assert _estimate_tokens(msg) == expected


# ---------------------------------------------------------------------------
# needs_compaction
# ---------------------------------------------------------------------------

def test_needs_compaction_false_when_small():
    compactor = Compactor(context_window=100_000, preserve_tokens=4_000)
    msgs = [_msg("user", "hi"), _msg("assistant", "hello")]
    assert not compactor.needs_compaction(msgs)


def test_needs_compaction_true_when_large():
    # usable = 200 - 2000 is negative, but clamped: preserve=2000, context=200
    # Actually usable = 200 - 2000 < 0, so any message triggers it.
    # Let's use a realistic tiny context for testing.
    compactor = Compactor(context_window=50, preserve_tokens=2_000)
    # usable = 50 - 2000 < 0, but we clamp preserve to min(2000, 8000)=2000
    # So usable = 50 - 2000 = -1950 → any message overflows. Good for testing.
    msgs = [_msg("user", "x")]
    assert compactor.needs_compaction(msgs)


def test_needs_compaction_false_empty():
    compactor = Compactor(context_window=100_000, preserve_tokens=4_000)
    assert not compactor.needs_compaction([])


def test_needs_compaction_threshold_is_exclusive():
    """Exactly at the usable budget should NOT trigger compaction."""
    compactor = Compactor(context_window=10_000, preserve_tokens=2_000)
    # usable = 8_000 tokens = 32_000 chars of JSON content
    # Build a message that is exactly at the boundary
    usable_chars = compactor._usable_tokens * 4
    # One message with content of usable_chars - json overhead
    json_overhead = len('{"role": "user", "content": ""}')
    content = "x" * max(0, usable_chars - json_overhead)
    msg = _msg("user", content)
    tok = _estimate_tokens(msg)
    # This message's tokens should not exceed usable
    if tok <= compactor._usable_tokens:
        assert not compactor.needs_compaction([msg])


# ---------------------------------------------------------------------------
# _select
# ---------------------------------------------------------------------------

def _make_compactor_with_usable(usable_tokens: int) -> Compactor:
    """Create a Compactor where usable = usable_tokens."""
    # preserve = _MIN_PRESERVE = 2000; context = usable + 2000
    return Compactor(context_window=usable_tokens + 2_000, preserve_tokens=2_000)


def test_select_splits_at_budget_overflow():
    # Each message dict {"role":"user","content":"x"*100} is ~108 chars → ~27 tokens.
    # With usable=50 tokens, first message (27 tok) fits, second (27 tok) makes 54 > 50.
    compactor = _make_compactor_with_usable(50)
    content = "x" * 100  # ~27 tokens each
    msgs = [_msg("user", content), _msg("assistant", content), _msg("user", content)]
    head, tail = compactor._select(msgs)
    # head should have the earliest messages; tail should have the rest
    assert len(head) >= 1
    assert len(tail) >= 1
    assert head + tail == msgs


def test_select_always_keeps_min_one_in_each():
    """Even if the very first message overflows, head=[msg[0]], tail=rest."""
    compactor = _make_compactor_with_usable(1)  # almost nothing fits
    msgs = [_msg("user", "a"), _msg("assistant", "b"), _msg("user", "c")]
    head, tail = compactor._select(msgs)
    assert len(head) >= 1
    assert len(tail) >= 1


def test_select_single_message_returns_empty_head():
    compactor = _make_compactor_with_usable(1)
    head, tail = compactor._select([_msg("user", "only one")])
    assert head == []


def test_select_two_messages_always_splits():
    compactor = _make_compactor_with_usable(1)
    msgs = [_msg("user", "a"), _msg("assistant", "b")]
    head, tail = compactor._select(msgs)
    assert len(head) == 1
    assert len(tail) == 1


def test_select_preserves_message_order():
    compactor = _make_compactor_with_usable(50)
    msgs = [_msg("user", "x" * 80), _msg("assistant", "y" * 80), _msg("user", "z" * 80)]
    head, tail = compactor._select(msgs)
    assert (head + tail) == msgs


# ---------------------------------------------------------------------------
# _format_head
# ---------------------------------------------------------------------------

def test_format_head_includes_user_messages():
    compactor = Compactor(context_window=10_000)
    msgs = [_msg("user", "explain async")]
    result = compactor._format_head(msgs)
    assert "[user]: explain async" in result


def test_format_head_includes_assistant_text():
    compactor = Compactor(context_window=10_000)
    msgs = [_msg("assistant", "async is a pattern")]
    result = compactor._format_head(msgs)
    assert "[assistant]: async is a pattern" in result


def test_format_head_includes_tool_calls():
    compactor = Compactor(context_window=10_000)
    msgs = [{
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "read", "arguments": '{"path": "/foo"}'}}],
    }]
    result = compactor._format_head(msgs)
    assert "[tool call]: read" in result


def test_format_head_truncates_tool_results():
    compactor = Compactor(context_window=10_000)
    msgs = [_tool_result("x" * 1000)]
    result = compactor._format_head(msgs)
    assert "[tool result]:" in result
    # Only first 500 chars of content should appear
    assert "x" * 500 in result
    assert "x" * 501 not in result


def test_format_head_handles_none_content():
    compactor = Compactor(context_window=10_000)
    msgs = [{"role": "assistant", "content": None, "tool_calls": []}]
    # Should not raise
    result = compactor._format_head(msgs)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _prune
# ---------------------------------------------------------------------------

def test_prune_truncates_long_tool_output():
    compactor = Compactor(context_window=10_000)
    long_content = "x" * (compactor._max_tool_output + 100)
    tail = [_tool_result(long_content)]
    pruned = compactor._prune(tail)
    assert len(pruned[0]["content"]) < len(long_content)
    assert "[truncated during compaction]" in pruned[0]["content"]


def test_prune_leaves_short_tool_output_unchanged():
    compactor = Compactor(context_window=10_000)
    short_content = "result"
    tail = [_tool_result(short_content)]
    pruned = compactor._prune(tail)
    assert pruned[0]["content"] == short_content


def test_prune_does_not_mutate_original():
    compactor = Compactor(context_window=10_000)
    long_content = "x" * 5_000
    original = _tool_result(long_content)
    tail = [original]
    compactor._prune(tail)
    # Original dict should be unchanged
    assert original["content"] == long_content


def test_prune_leaves_non_tool_messages_unchanged():
    compactor = Compactor(context_window=10_000)
    user_msg = _msg("user", "x" * 5_000)
    assistant_msg = _msg("assistant", "y" * 5_000)
    pruned = compactor._prune([user_msg, assistant_msg])
    assert pruned[0]["content"] == user_msg["content"]
    assert pruned[1]["content"] == assistant_msg["content"]


def test_prune_preserves_tool_call_id():
    compactor = Compactor(context_window=10_000)
    msg = _tool_result("x" * 5_000, call_id="abc123")
    pruned = compactor._prune([msg])
    assert pruned[0]["tool_call_id"] == "abc123"


# ---------------------------------------------------------------------------
# compact() — end to end
# ---------------------------------------------------------------------------

def test_compact_noop_when_under_budget():
    compactor = Compactor(context_window=100_000, preserve_tokens=2_000)
    msgs = [_msg("user", "hi"), _msg("assistant", "hello")]
    provider = _MockProvider()
    result = compactor.compact(msgs, provider)
    assert result is msgs  # same object — no allocation
    assert provider.calls == []  # no LLM call made


def test_compact_triggers_when_over_budget():
    compactor = _make_compactor_with_usable(1)  # everything overflows
    msgs = [_msg("user", "a" * 100), _msg("assistant", "b" * 100), _msg("user", "c" * 100)]
    provider = _MockProvider("Summary of the conversation.")
    compaction_calls: list[bool] = []
    result = compactor.compact(msgs, provider, on_compaction=lambda: compaction_calls.append(True))

    # First two messages are the summary injection pair (user + assistant ack)
    assert result[0]["role"] == "user"
    assert result[0]["content"].startswith("[Previous conversation summary")
    assert "Summary of the conversation." in result[0]["content"]
    assert result[1]["role"] == "assistant"
    # Provider was called
    assert len(provider.calls) == 1
    # on_compaction was called exactly once
    assert compaction_calls == [True]


def test_compact_returns_original_on_provider_error():
    compactor = _make_compactor_with_usable(1)
    msgs = [_msg("user", "a" * 100), _msg("assistant", "b" * 100)]
    result = compactor.compact(msgs, _FailingProvider())
    # Falls back to original list unchanged
    assert result == msgs


def test_compact_too_short_history_no_compaction():
    compactor = _make_compactor_with_usable(1)
    single = [_msg("user", "just one message")]
    provider = _MockProvider()
    result = compactor.compact(single, provider)
    assert result == single
    assert provider.calls == []


def test_compact_prunes_tool_outputs_in_tail():
    compactor = _make_compactor_with_usable(1)
    long_output = "x" * 5_000
    msgs = [
        _msg("user", "a" * 100),
        _msg("assistant", "b" * 100),
        _tool_result(long_output),  # this lands in the tail
    ]
    provider = _MockProvider("summary")
    result = compactor.compact(msgs, provider)
    # Find the tool message in the result (skip the summary injection at indices 0-1)
    tool_msgs = [m for m in result if m.get("role") == "tool"]
    if tool_msgs:
        assert len(tool_msgs[0]["content"]) <= compactor._max_tool_output + len("\n[truncated during compaction]")


def test_compact_summary_uses_structured_prompt():
    """The summarisation call uses the structured _SUMMARY_PROMPT as system."""
    from minion_assist.context import _SUMMARY_PROMPT
    compactor = _make_compactor_with_usable(1)
    msgs = [_msg("user", "x" * 100), _msg("assistant", "y" * 100)]
    provider = _MockProvider()
    compactor.compact(msgs, provider)
    assert provider.calls[0]["system"] == _SUMMARY_PROMPT


# ---------------------------------------------------------------------------
# config integration
# ---------------------------------------------------------------------------

def test_compaction_config_loaded_from_config_json():
    """CompactionConfig is read from config.json; preserve_tokens is None when omitted (auto-compute)."""
    from minion_assist.config import agents, compaction
    # preserve_tokens is now optional — None means "use model.max_output_tokens" at runtime.
    assert compaction.preserve_tokens is None or compaction.preserve_tokens > 0
    # context_window is now per-agent via ModelConfig, not in CompactionConfig
    for cfg in agents.values():
        assert cfg.model.context_window > 0


def test_compaction_config_preserve_tokens_clamped_low():
    """preserve_tokens below _MIN_PRESERVE is clamped up."""
    compactor = Compactor(context_window=10_000, preserve_tokens=100)
    assert compactor._preserve_tokens == 2_000  # _MIN_PRESERVE


def test_compaction_config_preserve_tokens_clamped_high():
    """preserve_tokens above context_window // 2 is clamped to context_window // 2."""
    compactor = Compactor(context_window=100_000, preserve_tokens=80_000)
    assert compactor._preserve_tokens == 50_000  # context_window // 2


# ---------------------------------------------------------------------------
# _format_head — per-role content caps
# ---------------------------------------------------------------------------

def test_format_head_truncates_long_assistant_message():
    # Use a large context window so _max_head_content > 100 chars (testable truncation).
    compactor = Compactor(context_window=100_000)
    cap = compactor._max_head_content
    long_content = "a" * (cap + 100)
    msgs = [_msg("assistant", long_content)]
    result = compactor._format_head(msgs)
    assert "a" * cap in result
    assert "a" * (cap + 1) not in result


def test_format_head_truncates_long_user_message():
    compactor = Compactor(context_window=100_000)
    cap = compactor._max_head_content
    long_content = "u" * (cap + 100)
    msgs = [_msg("user", long_content)]
    result = compactor._format_head(msgs)
    assert "u" * cap in result
    assert "u" * (cap + 1) not in result


# ---------------------------------------------------------------------------
# _prune — microcompact behaviour
# ---------------------------------------------------------------------------

def test_prune_microcompacts_old_tool_results():
    """Tool results older than the last _tail_keep_full are replaced with one-liners."""
    from minion_assist.context import _TAIL_KEEP_FULL_RESULTS
    compactor = Compactor(context_window=10_000)
    # Build more tool results than the keep-full threshold.
    tail = [_tool_result("x" * 100, call_id=f"tc{i}") for i in range(_TAIL_KEEP_FULL_RESULTS + 2)]
    pruned = compactor._prune(tail)

    # The first two results (oldest) should be microcompacted.
    assert "chars" in pruned[0]["content"]
    assert "re-read" in pruned[0]["content"]
    assert pruned[0]["content"] != "x" * 100

    # The last _TAIL_KEEP_FULL_RESULTS results should be kept in full.
    for msg in pruned[-_TAIL_KEEP_FULL_RESULTS:]:
        assert msg["content"] == "x" * 100


def test_prune_keeps_all_when_fewer_than_threshold(tmp_path=None):
    """When the tail has ≤ _tail_keep_full tool results, all are kept in full."""
    from minion_assist.context import _TAIL_KEEP_FULL_RESULTS
    compactor = Compactor(context_window=10_000)
    tail = [_tool_result("content", call_id=f"tc{i}") for i in range(_TAIL_KEEP_FULL_RESULTS - 1)]
    pruned = compactor._prune(tail)
    for msg in pruned:
        assert msg["content"] == "content"


def test_prune_microcompact_preserves_tool_call_id():
    """Microcompacted messages must retain their tool_call_id."""
    compactor = Compactor(context_window=10_000, tail_keep_full_results=1)
    tail = [
        _tool_result("old result", call_id="old_id"),
        _tool_result("new result", call_id="new_id"),
    ]
    pruned = compactor._prune(tail)
    assert pruned[0]["tool_call_id"] == "old_id"
    assert pruned[1]["tool_call_id"] == "new_id"


def test_prune_microcompact_does_not_mutate_original():
    """Microcompaction must not mutate the original dict."""
    compactor = Compactor(context_window=10_000, tail_keep_full_results=0)
    original = _tool_result("original content", call_id="x")
    compactor._prune([original])
    assert original["content"] == "original content"


def test_prune_non_tool_messages_unchanged_with_microcompact():
    """Non-tool messages must pass through _prune unchanged even with microcompact."""
    compactor = Compactor(context_window=10_000, tail_keep_full_results=0)
    user_msg = _msg("user", "x" * 5_000)
    pruned = compactor._prune([user_msg])
    assert pruned[0]["content"] == user_msg["content"]


def test_tail_keep_full_results_configurable():
    """tail_keep_full_results constructor param must override the default."""
    compactor = Compactor(context_window=10_000, tail_keep_full_results=2)
    tail = [_tool_result(f"result {i}", call_id=f"tc{i}") for i in range(5)]
    pruned = compactor._prune(tail)
    # First 3 (indices 0-2) are microcompacted; last 2 (indices 3-4) are kept full.
    assert "chars" in pruned[0]["content"]
    assert "chars" in pruned[1]["content"]
    assert "chars" in pruned[2]["content"]
    assert pruned[3]["content"] == "result 3"
    assert pruned[4]["content"] == "result 4"


# ---------------------------------------------------------------------------
# compact() — CompactionFailed callback
# ---------------------------------------------------------------------------

def test_compact_calls_on_compaction_failed_on_provider_error():
    """When summarisation fails, on_compaction_failed must be called with the error."""
    compactor = _make_compactor_with_usable(1)
    msgs = [_msg("user", "a" * 100), _msg("assistant", "b" * 100)]
    failures: list[str] = []
    compactor.compact(msgs, _FailingProvider(), on_compaction_failed=failures.append)
    assert len(failures) == 1
    assert "RuntimeError" in failures[0]


def test_compact_returns_original_when_failed_with_callback():
    """compact() must still return original history when on_compaction_failed is set."""
    compactor = _make_compactor_with_usable(1)
    msgs = [_msg("user", "a" * 100), _msg("assistant", "b" * 100)]
    result = compactor.compact(msgs, _FailingProvider(), on_compaction_failed=lambda e: None)
    assert result == msgs


def test_compact_no_failure_callback_when_succeeds():
    """on_compaction_failed must NOT be called on a successful compaction."""
    compactor = _make_compactor_with_usable(1)
    msgs = [_msg("user", "a" * 100), _msg("assistant", "b" * 100), _msg("user", "c")]
    failures: list[str] = []
    compactor.compact(msgs, _MockProvider(), on_compaction_failed=failures.append)
    assert failures == []


# ---------------------------------------------------------------------------
# peek_compaction_head (Stage One Phase 2, slice B)
# ---------------------------------------------------------------------------

def test_peek_compaction_head_none_when_not_needed():
    compactor = Compactor(context_window=100_000, preserve_tokens=4_000)
    msgs = [_msg("user", "hi"), _msg("assistant", "hello")]
    assert compactor.peek_compaction_head(msgs) is None


def test_peek_compaction_head_matches_select_head():
    """peek_compaction_head() must return exactly what _select() would use as head."""
    compactor = _make_compactor_with_usable(50)
    content = "x" * 100
    msgs = [_msg("user", content), _msg("assistant", content), _msg("user", content)]
    expected_head, _expected_tail = compactor._select(msgs)

    assert compactor.peek_compaction_head(msgs) == expected_head


def test_peek_compaction_head_does_not_mutate_messages():
    compactor = _make_compactor_with_usable(50)
    content = "x" * 100
    msgs = [_msg("user", content), _msg("assistant", content), _msg("user", content)]
    original = [dict(m) for m in msgs]

    compactor.peek_compaction_head(msgs)

    assert msgs == original


def test_peek_compaction_head_none_when_history_too_short():
    """A single message can't be split into head+tail — _select returns ([], msgs)."""
    compactor = _make_compactor_with_usable(1)
    msgs = [_msg("user", "a" * 100)]
    assert compactor.peek_compaction_head(msgs) is None

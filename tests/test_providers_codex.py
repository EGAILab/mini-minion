"""Tests for CodexProvider (app-server / OAuth path).

All tests use a stub _CodexRpcClient so no real subprocess is spawned.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from minion_assist.providers.codex import CodexProvider, _CodexRpcClient
from minion_assist.providers.base import LLMResponse


# ---------------------------------------------------------------------------
# Stub RPC client — simulates the Codex binary over stdio
# ---------------------------------------------------------------------------


class _StubRpc:
    """Test double for _CodexRpcClient.

    ``turn/start`` dispatches fake notifications (via a short timer) before
    returning so that the polling loop in CodexProvider.chat() finds a result.

    When ``agent_text`` is set, an ``item/completed`` notification is fired
    first (simulating the real binary's text delivery path), then
    ``turn/completed`` fires.  This lets tests verify that the provider reads
    text from ``item/completed`` rather than ``turn.items``.
    """

    def __init__(
        self,
        turn_items: list[dict] | None = None,
        turn_status: str = "completed",
        thread_id: str = "thread-1",
        rpc_error: Exception | None = None,
        notification_delay: float = 0.01,
        agent_text: str = "",
    ) -> None:
        self.turn_items = turn_items or [{"text": "Hello from Codex"}]
        self.turn_status = turn_status
        self.thread_id = thread_id
        self.rpc_error = rpc_error
        self.notification_delay = notification_delay
        self.agent_text = agent_text
        self._handlers: list[Callable[[dict], None]] = []
        self.calls: list[tuple[str, dict]] = []

    def _fire_notification(self) -> None:
        if self.agent_text:
            item_notification = {
                "method": "item/completed",
                "params": {
                    "item": {
                        "type": "agentMessage",
                        "id": "item-1",
                        "text": self.agent_text,
                    }
                },
            }
            for handler in list(self._handlers):
                try:
                    handler(item_notification)
                except Exception:
                    pass

        notification = {
            "method": "turn/completed",
            "params": {
                "turn": {
                    "id": "turn-1",
                    "threadId": self.thread_id,
                    "status": self.turn_status,
                    "items": self.turn_items,
                }
            },
        }
        for handler in list(self._handlers):
            try:
                handler(notification)
            except Exception:
                pass

    def request(self, method: str, params: dict | None = None, timeout: float = 60.0) -> dict:
        self.calls.append((method, params or {}))
        if self.rpc_error:
            raise self.rpc_error
        if method == "initialize":
            return {"serverInfo": {"name": "codex", "version": "1.0"}, "capabilities": {}}
        if method == "thread/start":
            # thread/start creates the thread only — the real binary does NOT
            # send any turn notifications here (no inference happens until
            # turn/start is called).  Return the thread dict immediately.
            return {"thread": {"id": self.thread_id}, "model": "gpt-5.5"}
        if method == "turn/start":
            # Dispatch the turn completion notification asynchronously so that
            # CodexProvider.chat() has time to register done.wait() first.
            timer = threading.Timer(self.notification_delay, self._fire_notification)
            timer.daemon = True
            timer.start()
            return {"turn": {"id": "turn-1", "threadId": self.thread_id, "status": "idle"}}
        return {}

    def add_handler(self, handler: Callable[[dict], None]) -> None:
        self._handlers.append(handler)

    def remove_handler(self, handler: Callable[[dict], None]) -> None:
        try:
            self._handlers.remove(handler)
        except ValueError:
            pass

    def close(self) -> None:
        pass


def _make_provider(stub: _StubRpc, model: str = "gpt-5.5") -> CodexProvider:
    """Build a CodexProvider with a pre-injected stub RPC client."""
    p = object.__new__(CodexProvider)
    p._codex_bin = "codex"
    p._model = model
    p._turn_timeout = 5.0
    p._rpc = stub
    p._thread_id = None
    p._sent_count = 0
    p._log_dir = None
    p._registry = None
    p._approve_command = None
    return p


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------


def test_extract_text_single_item():
    turn = {"items": [{"text": "hello"}]}
    assert CodexProvider._extract_text(turn) == "hello"


def test_extract_text_multiple_items():
    turn = {"items": [{"text": "part one"}, {"text": "part two"}]}
    assert CodexProvider._extract_text(turn) == "part one\n\npart two"


def test_extract_text_skips_empty():
    turn = {"items": [{"text": ""}, {"text": "real text"}, {"text": ""}]}
    assert CodexProvider._extract_text(turn) == "real text"


def test_extract_text_non_dict_items_skipped():
    turn = {"items": ["bad", None, {"text": "ok"}]}
    assert CodexProvider._extract_text(turn) == "ok"


def test_extract_text_no_items():
    assert CodexProvider._extract_text({}) == ""
    assert CodexProvider._extract_text({"items": []}) == ""


def test_extract_text_items_without_text_field():
    turn = {"items": [{"type": "tool_call", "name": "bash"}, {"text": "done"}]}
    assert CodexProvider._extract_text(turn) == "done"


# ---------------------------------------------------------------------------
# CodexProvider.chat() — first turn (thread/start)
# ---------------------------------------------------------------------------


def test_first_turn_sends_thread_start():
    stub = _StubRpc()
    p = _make_provider(stub)

    result = p.chat(
        system="You are helpful.",
        messages=[{"role": "user", "content": "Hello"}],
        tools=[],
        max_tokens=100,
    )

    # initialize was called via _get_rpc, but since we inject _rpc directly
    # the stub already has the client; check thread/start was called.
    methods = [c[0] for c in stub.calls]
    assert "thread/start" in methods


def test_first_turn_passes_developer_instructions():
    stub = _StubRpc()
    p = _make_provider(stub)

    p.chat(system="Be concise.", messages=[{"role": "user", "content": "hi"}], tools=[], max_tokens=50)

    call = next(c for c in stub.calls if c[0] == "thread/start")
    assert call[1].get("developerInstructions") == "Be concise."


def test_first_turn_passes_user_message_as_input():
    stub = _StubRpc()
    p = _make_provider(stub)

    p.chat(system="", messages=[{"role": "user", "content": "What time is it?"}], tools=[], max_tokens=50)

    # thread/start creates the thread (no input); turn/start carries the message
    thread_call = next(c for c in stub.calls if c[0] == "thread/start")
    assert "input" not in thread_call[1]
    turn_call = next(c for c in stub.calls if c[0] == "turn/start")
    assert turn_call[1].get("input") == [{"type": "text", "text": "What time is it?"}]


def test_first_turn_returns_text_from_turn_items():
    stub = _StubRpc(turn_items=[{"text": "It is noon."}])
    p = _make_provider(stub)

    result = p.chat(system="", messages=[{"role": "user", "content": "hi"}], tools=[], max_tokens=50)

    assert isinstance(result, LLMResponse)
    assert result.text == "It is noon."
    assert result.tool_calls == []
    assert result.finish_reason == "stop"


def test_first_turn_stores_thread_id():
    stub = _StubRpc(thread_id="my-thread")
    p = _make_provider(stub)

    p.chat(system="", messages=[{"role": "user", "content": "hi"}], tools=[], max_tokens=50)

    assert p._thread_id == "my-thread"


def test_first_turn_passes_model():
    stub = _StubRpc()
    p = _make_provider(stub, model="gpt-5.5")

    p.chat(system="", messages=[{"role": "user", "content": "hi"}], tools=[], max_tokens=50)

    call = next(c for c in stub.calls if c[0] == "thread/start")
    assert call[1].get("model") == "gpt-5.5"


def test_empty_model_omits_model_param():
    stub = _StubRpc()
    p = _make_provider(stub, model="")

    p.chat(system="", messages=[{"role": "user", "content": "hi"}], tools=[], max_tokens=50)

    call = next(c for c in stub.calls if c[0] == "thread/start")
    assert "model" not in call[1]


# ---------------------------------------------------------------------------
# CodexProvider.chat() — second turn (turn/start)
# ---------------------------------------------------------------------------


def test_second_turn_uses_turn_start():
    stub = _StubRpc()
    p = _make_provider(stub)

    # First turn
    p.chat(system="", messages=[{"role": "user", "content": "hi"}], tools=[], max_tokens=50)
    # Second turn (extend message list as the runner would)
    p.chat(
        system="",
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "follow up"},
        ],
        tools=[],
        max_tokens=50,
    )

    methods = [c[0] for c in stub.calls]
    assert "thread/start" in methods
    assert "turn/start" in methods


def test_second_turn_sends_only_new_user_message():
    stub = _StubRpc()
    p = _make_provider(stub)

    p.chat(system="", messages=[{"role": "user", "content": "first"}], tools=[], max_tokens=50)
    p.chat(
        system="",
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "second"},
        ],
        tools=[],
        max_tokens=50,
    )

    # Both turns use turn/start; the second one carries "second"
    turn_starts = [c for c in stub.calls if c[0] == "turn/start"]
    assert len(turn_starts) == 2
    inputs = turn_starts[1][1].get("input", [])
    assert inputs == [{"type": "text", "text": "second"}]


def test_second_turn_passes_thread_id():
    stub = _StubRpc(thread_id="abc-123")
    p = _make_provider(stub)

    p.chat(system="", messages=[{"role": "user", "content": "hi"}], tools=[], max_tokens=50)
    p.chat(
        system="",
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "user", "content": "follow"},
        ],
        tools=[],
        max_tokens=50,
    )

    turn_start = next(c for c in stub.calls if c[0] == "turn/start")
    assert turn_start[1].get("threadId") == "abc-123"


# ---------------------------------------------------------------------------
# on_token callback
# ---------------------------------------------------------------------------


def test_on_token_called_with_full_text():
    stub = _StubRpc(turn_items=[{"text": "Full response here"}])
    p = _make_provider(stub)

    tokens: list[str] = []
    result = p.chat(
        system="",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        max_tokens=50,
        on_token=tokens.append,
    )

    assert tokens == ["Full response here"]
    assert result.text == "Full response here"


def test_on_token_not_called_for_empty_response():
    stub = _StubRpc(turn_items=[{"text": ""}])
    p = _make_provider(stub)

    tokens: list[str] = []
    p.chat(system="", messages=[{"role": "user", "content": "hi"}], tools=[], max_tokens=50, on_token=tokens.append)

    assert tokens == []


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------


def test_chat_raises_on_turn_timeout():
    """If no completion notification arrives, chat() raises TimeoutError."""

    class _HangingStub(_StubRpc):
        def request(self, method, params=None, timeout=60.0):
            self.calls.append((method, params or {}))
            # Deliberately never fire the notification
            if method in ("thread/start", "turn/start"):
                return {"thread": {"id": "t1"}, "model": "gpt-5.5"}
            return {}

    stub = _HangingStub()
    p = _make_provider(stub)
    p._turn_timeout = 0.05  # Very short timeout for the test

    with pytest.raises(TimeoutError, match="timed out"):
        p.chat(system="", messages=[{"role": "user", "content": "hi"}], tools=[], max_tokens=50)


# ---------------------------------------------------------------------------
# Message slicing — sent_count tracks what has been sent
# ---------------------------------------------------------------------------


def test_sent_count_advances_after_each_call():
    stub = _StubRpc()
    p = _make_provider(stub)

    assert p._sent_count == 0
    p.chat(system="", messages=[{"role": "user", "content": "one"}], tools=[], max_tokens=50)
    assert p._sent_count == 1

    p.chat(
        system="",
        messages=[{"role": "user", "content": "one"}, {"role": "user", "content": "two"}],
        tools=[],
        max_tokens=50,
    )
    assert p._sent_count == 2


def test_no_new_user_message_sends_empty_input():
    """If the new slice has no user message (e.g., only tool results), input is empty."""
    stub = _StubRpc()
    p = _make_provider(stub)
    p._thread_id = "existing"
    p._sent_count = 1

    p.chat(
        system="",
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "c1", "content": "tool result"},
        ],
        tools=[],
        max_tokens=50,
    )

    turn_start = next(c for c in stub.calls if c[0] == "turn/start")
    # No user message in new slice → fallback to joining content
    inputs = turn_start[1].get("input", [])
    assert inputs == [{"type": "text", "text": "tool result"}]


# ---------------------------------------------------------------------------
# Notification matching — various terminal statuses are accepted
# ---------------------------------------------------------------------------


def test_canceled_turn_status_still_returns():
    stub = _StubRpc(turn_status="canceled", turn_items=[{"text": ""}])
    p = _make_provider(stub)

    result = p.chat(system="", messages=[{"role": "user", "content": "hi"}], tools=[], max_tokens=50)
    assert isinstance(result, LLMResponse)


def test_error_turn_status_still_returns():
    stub = _StubRpc(turn_status="error", turn_items=[{"text": "something went wrong"}])
    p = _make_provider(stub)

    result = p.chat(system="", messages=[{"role": "user", "content": "hi"}], tools=[], max_tokens=50)
    assert result.text == "something went wrong"


# ---------------------------------------------------------------------------
# _CodexRpcClient — _extract_text unit (edge cases)
# ---------------------------------------------------------------------------


def test_extract_text_concatenates_with_double_newline():
    turn = {"items": [{"text": "A"}, {"text": "B"}, {"text": "C"}]}
    result = CodexProvider._extract_text(turn)
    assert result == "A\n\nB\n\nC"


# ---------------------------------------------------------------------------
# Protocol: thread/start vs turn/start split
# ---------------------------------------------------------------------------


def test_first_turn_does_not_pass_input_to_thread_start():
    """thread/start must not carry input — the binary ignores it and stays idle."""
    stub = _StubRpc()
    p = _make_provider(stub)
    p.chat(system="", messages=[{"role": "user", "content": "hello"}], tools=[], max_tokens=50)
    call = next(c for c in stub.calls if c[0] == "thread/start")
    assert "input" not in call[1]


def test_first_turn_calls_turn_start():
    """turn/start must be called on every turn, including the first."""
    stub = _StubRpc()
    p = _make_provider(stub)
    p.chat(system="", messages=[{"role": "user", "content": "hello"}], tools=[], max_tokens=50)
    methods = [c[0] for c in stub.calls]
    assert "turn/start" in methods


# ---------------------------------------------------------------------------
# item/completed as the text source
# ---------------------------------------------------------------------------


def test_text_collected_from_item_completed_notification():
    """Response text comes from item/completed, not turn.items."""
    stub = _StubRpc(turn_items=[], agent_text="hello from item")
    p = _make_provider(stub)
    result = p.chat(system="", messages=[{"role": "user", "content": "hi"}], tools=[], max_tokens=50)
    assert result.text == "hello from item"


def test_item_completed_overrides_turn_items_text():
    """When both item/completed and turn.items have text, item/completed wins."""
    stub = _StubRpc(turn_items=[{"text": "from turn.items"}], agent_text="from item/completed")
    p = _make_provider(stub)
    result = p.chat(system="", messages=[{"role": "user", "content": "hi"}], tools=[], max_tokens=50)
    assert result.text == "from item/completed"

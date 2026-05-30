"""Tests for REPL exception handling and history persistence (minion.py)."""

import pytest
from unittest.mock import Mock, patch

from mini_minion.memory.short_term import ShortTermMemory


def _run_main(tmp_path, inputs, run_turn_effect=None):
    """Run main() with a controlled workspace, input sequence, and run_turn behaviour.

    Args:
        tmp_path: Pytest temporary directory used as the workspace root.
        inputs: Sequence of strings returned by successive ``input()`` calls.
        run_turn_effect: Passed as ``side_effect`` to the ``run_turn`` mock.
            Pass an exception instance to make it raise, a callable to replace
            the implementation, or ``None`` for a no-op mock.

    Returns:
        The ``run_turn`` Mock so callers can inspect call counts / args.
    """
    import mini_minion.minion as minion_mod

    rt_mock = Mock(side_effect=run_turn_effect) if run_turn_effect is not None else Mock()

    with (
        patch("mini_minion.minion.workspace", tmp_path),
        patch("mini_minion.minion.run_turn", rt_mock),
        patch("mini_minion.minion.create_provider", return_value=Mock()),
        patch("builtins.input", side_effect=iter(inputs)),
    ):
        minion_mod.main()

    return rt_mock


def test_repl_survives_provider_exception(tmp_path, capsys):
    """REPL loop continues after run_turn raises; user sees a friendly error line."""
    _run_main(tmp_path, ["hello", "quit"], run_turn_effect=RuntimeError("connection refused"))

    captured = capsys.readouterr()
    assert "[Error]" in captured.err


def test_provider_exception_records_error_in_history(tmp_path):
    """After a provider exception, history has the user message and an assistant error entry."""
    _run_main(tmp_path, ["hello", "quit"], run_turn_effect=RuntimeError("timeout"))

    history = ShortTermMemory(tmp_path / "sessions").load("main")
    assert any(m["role"] == "user" and m["content"] == "hello" for m in history)
    assert any(
        m["role"] == "assistant" and "Provider error" in m["content"]
        for m in history
    )


def test_provider_exception_rolls_back_partial_messages(tmp_path):
    """Partial messages appended by run_turn before a crash are stripped from saved history."""

    def _crash_after_partial(provider, name, soul, max_tokens, tools, messages, **kwargs):
        # Simulate run_turn appending a partial assistant message before crashing.
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        })
        raise RuntimeError("mid-turn crash")

    _run_main(tmp_path, ["hello", "quit"], run_turn_effect=_crash_after_partial)

    history = ShortTermMemory(tmp_path / "sessions").load("main")
    # The partial assistant+tool_calls message must not appear in saved history.
    assert not any("tool_calls" in m for m in history)
    # Only the user message and the assistant error message should remain.
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_user_message_persisted_on_exception(tmp_path):
    """User message is on disk after a provider crash (early save before run_turn)."""
    _run_main(tmp_path, ["my question", "quit"], run_turn_effect=RuntimeError("boom"))

    history = ShortTermMemory(tmp_path / "sessions").load("main")
    assert history[0] == {"role": "user", "content": "my question"}


def test_successful_turn_persists_history(tmp_path):
    """On a successful turn, history is saved and contains the user message."""
    _run_main(tmp_path, ["hello", "quit"])

    history = ShortTermMemory(tmp_path / "sessions").load("main")
    assert any(m["role"] == "user" and m["content"] == "hello" for m in history)


def test_main_exits_on_agent_identity_mismatch(tmp_path):
    """main() must raise SystemExit when AGENTS and config.json agent keys differ."""
    import mini_minion.minion as minion_mod
    from mini_minion.agents.definitions import AgentConfig

    # Replace AGENTS with a key that doesn't exist in config.json.
    wrong_agents = {"unknown_agent": AgentConfig(name="X", soul="x")}

    with (
        patch("mini_minion.minion.workspace", tmp_path),
        patch("mini_minion.minion.AGENTS", wrong_agents),
        patch("builtins.input", side_effect=iter([])),
    ):
        with pytest.raises(SystemExit):
            minion_mod.main()


def test_compaction_receives_user_message(tmp_path):
    """compact() must be called after the user message is appended to the list."""
    import mini_minion.minion as minion_mod
    from mini_minion.context import Compactor

    seen: list[list[dict]] = []

    def spy_compact(self, messages, provider):
        seen.append(list(messages))
        return messages  # pass-through, no actual compaction

    with (
        patch("mini_minion.minion.workspace", tmp_path),
        patch("mini_minion.minion.run_turn", Mock()),
        patch("mini_minion.minion.create_provider", return_value=Mock()),
        patch("builtins.input", side_effect=iter(["hello", "quit"])),
        patch.object(Compactor, "compact", spy_compact),
    ):
        minion_mod.main()

    # Every compaction call must have seen the user message already appended.
    user_msgs_seen = [
        m for call in seen for m in call
        if m.get("role") == "user" and m.get("content") == "hello"
    ]
    assert user_msgs_seen, "compact() was called before the user message was appended"

"""Tests for REPL exception handling and history persistence (minion.py)."""

import pytest
from types import SimpleNamespace
from unittest.mock import Mock, patch, call

from minion_assist.memory.short_term import ShortTermMemory


def _run_main(tmp_path, inputs, run_turn_effect=None):
    """Run main() with a controlled workspace, input sequence, and run_turn behaviour.

    After the refactor, main() uses AgentSession which calls run_turn internally.
    We patch minion_assist.agents.session.run_turn (where AgentSession imports it)
    rather than minion_assist.minion.run_turn (which no longer exists there).

    Args:
        tmp_path: Pytest temporary directory used as the workspace root.
        inputs: Sequence of strings returned by successive ``input()`` calls.
        run_turn_effect: Passed as ``side_effect`` to the ``run_turn`` mock.
            Pass an exception instance to make it raise, a callable to replace
            the implementation, or ``None`` for a no-op mock.

    Returns:
        The ``run_turn`` Mock so callers can inspect call counts / args.
    """
    import minion_assist.minion as minion_mod

    rt_mock = Mock(side_effect=run_turn_effect) if run_turn_effect is not None else Mock()

    with (
        patch("minion_assist.minion.workspace", tmp_path),
        patch("minion_assist.minion.mcp_cfg", SimpleNamespace(servers=())),
        patch("minion_assist.minion.channels_cfg", SimpleNamespace(matrix=None)),
        patch("minion_assist.agents.session.run_turn", rt_mock),
        patch("minion_assist.minion.create_provider", return_value=Mock()),
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


def test_route_prefix_message_is_not_treated_as_unknown_command(tmp_path, capsys):
    """A routed chat message like '/research hello' must reach the routed agent."""
    rt_mock = _run_main(tmp_path, ["/research how to prepare", "quit"])

    assert rt_mock.call_count == 1
    assert rt_mock.call_args.args[1] == "Elizabeth"

    history = ShortTermMemory(tmp_path / "sessions").load("researcher")
    assert any(
        m["role"] == "user" and m["content"] == "how to prepare"
        for m in history
    )
    assert "Unknown command '/research'" not in capsys.readouterr().out


def test_main_exits_on_agent_identity_mismatch(tmp_path):
    """main() must raise SystemExit when AGENTS and config.json agent keys differ."""
    import minion_assist.minion as minion_mod
    from minion_assist.agents.definitions import AgentConfig

    # Replace AGENTS with a key that doesn't exist in config.json.
    wrong_agents = {"unknown_agent": AgentConfig(name="X", soul="x")}

    with (
        patch("minion_assist.minion.workspace", tmp_path),
        patch("minion_assist.minion.AGENTS", wrong_agents),
        patch("builtins.input", side_effect=iter([])),
    ):
        with pytest.raises(SystemExit):
            minion_mod.main()


def test_compaction_receives_user_message(tmp_path):
    """compact() must be called after the user message is appended to the list."""
    import minion_assist.minion as minion_mod
    from minion_assist.context import Compactor

    seen: list[list[dict]] = []

    def spy_compact(self, messages, provider, on_compaction=None, on_compaction_failed=None):
        seen.append(list(messages))
        return messages  # pass-through, no actual compaction

    with (
        patch("minion_assist.minion.workspace", tmp_path),
        patch("minion_assist.agents.session.run_turn", Mock()),
        patch("minion_assist.minion.create_provider", return_value=Mock()),
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


def test_streaming_response_printed_once(tmp_path, capsys):
    """Streamed response text must appear exactly once — not duplicated by FinalAnswer handler."""
    import minion_assist.minion as minion_mod
    from minion_assist.agents.events import FinalAnswer, StreamingStarted, TokenStreamed

    def _emit_streaming_events(provider, name, soul, max_tokens, tools, messages, on_event=None, **kwargs):
        # Simulate the runner emitting streaming events then FinalAnswer.
        if on_event:
            on_event(StreamingStarted(agent_name=name))
            on_event(TokenStreamed(token="Hello"))
            on_event(TokenStreamed(token=" world"))
            on_event(FinalAnswer(agent_name=name, text="Hello world"))

    with (
        patch("minion_assist.minion.workspace", tmp_path),
        patch("minion_assist.agents.session.run_turn", side_effect=_emit_streaming_events),
        patch("minion_assist.minion.create_provider", return_value=Mock()),
        patch("minion_assist.minion.streaming") as mock_streaming,
        patch("builtins.input", side_effect=iter(["hi", "quit"])),
    ):
        mock_streaming.chat_mode = True
        minion_mod.main()

    out = capsys.readouterr().out
    # "Hello world" should appear exactly once in the output, not twice.
    assert out.count("Hello world") == 1


# ---------------------------------------------------------------------------
# IMP-15: Graceful shutdown
# ---------------------------------------------------------------------------


def test_keyboard_interrupt_at_prompt_exits_cleanly(tmp_path, capsys):
    """KeyboardInterrupt at the input() prompt causes a clean exit, not a traceback."""
    import minion_assist.minion as minion_mod
    with (
        patch("minion_assist.minion.workspace", tmp_path),
        patch("minion_assist.agents.session.run_turn", Mock()),
        patch("minion_assist.minion.create_provider", return_value=Mock()),
        patch("builtins.input", side_effect=KeyboardInterrupt),
    ):
        minion_mod.main()  # must return without raising

    out = capsys.readouterr().out
    assert "Goodbye" in out


def test_keyboard_interrupt_during_turn_continues_repl(tmp_path, capsys):
    """KeyboardInterrupt mid-turn prints a message and continues the REPL."""
    import minion_assist.minion as minion_mod
    inputs = iter(["hello", "quit"])
    with (
        patch("minion_assist.minion.workspace", tmp_path),
        patch("minion_assist.agents.session.run_turn", side_effect=KeyboardInterrupt),
        patch("minion_assist.minion.create_provider", return_value=Mock()),
        patch("builtins.input", side_effect=inputs),
    ):
        minion_mod.main()  # must return without raising

    out = capsys.readouterr().out
    assert "interrupted" in out.lower()


# ---------------------------------------------------------------------------
# ephemeral_history
# ---------------------------------------------------------------------------


def test_ephemeral_history_clears_history_before_session_start(tmp_path):
    """When ephemeral_history=True, pre-existing JSONL is wiped at startup."""
    import minion_assist.minion as minion_mod
    from minion_assist.config import AgentModelConfig, ProviderConfig, ModelConfig

    # Pre-populate the researcher's JSONL with stale history.
    stm = ShortTermMemory(tmp_path / "sessions")
    stm.append("researcher", {"role": "user", "content": "old question"})
    stm.append("researcher", {"role": "assistant", "content": "old answer"})
    assert len(stm.load("researcher")) == 2

    _provider = SimpleNamespace(
        base_url="", api="lmstudio", api_key="", name="lmstudio",
    )
    _model = SimpleNamespace(id="test", context_window=8192, max_output_tokens=512)
    _ephemeral_cfg = SimpleNamespace(
        provider=_provider, model=_model, route_prefix="/research", ephemeral_history=True,
    )
    _main_provider = SimpleNamespace(
        base_url="", api="lmstudio", api_key="", name="lmstudio",
    )
    _main_cfg = SimpleNamespace(
        provider=_main_provider, model=_model, route_prefix=None, ephemeral_history=False,
    )
    fake_agents_cfg = {"main": _main_cfg, "researcher": _ephemeral_cfg}

    with (
        patch("minion_assist.minion.workspace", tmp_path),
        patch("minion_assist.minion.agents_cfg", fake_agents_cfg),
        patch("minion_assist.minion.mcp_cfg", SimpleNamespace(servers=())),
        patch("minion_assist.minion.channels_cfg", SimpleNamespace(matrix=None)),
        patch("minion_assist.agents.session.run_turn", Mock()),
        patch("minion_assist.minion.create_provider", return_value=Mock()),
        patch("builtins.input", side_effect=iter(["quit"])),
    ):
        minion_mod.main()

    # After startup with ephemeral_history=True, the JSONL must be empty.
    assert stm.load("researcher") == []


def test_non_ephemeral_history_preserves_history_at_startup(tmp_path):
    """When ephemeral_history=False (default), pre-existing JSONL is kept."""
    import minion_assist.minion as minion_mod

    stm = ShortTermMemory(tmp_path / "sessions")
    stm.append("researcher", {"role": "user", "content": "old question"})
    stm.append("researcher", {"role": "assistant", "content": "old answer"})

    _provider = SimpleNamespace(base_url="", api="lmstudio", api_key="", name="lmstudio")
    _model = SimpleNamespace(id="test", context_window=8192, max_output_tokens=512)
    _persistent_cfg = SimpleNamespace(
        provider=_provider, model=_model, route_prefix="/research", ephemeral_history=False,
    )
    _main_cfg = SimpleNamespace(
        provider=_provider, model=_model, route_prefix=None, ephemeral_history=False,
    )
    fake_agents_cfg = {"main": _main_cfg, "researcher": _persistent_cfg}

    with (
        patch("minion_assist.minion.workspace", tmp_path),
        patch("minion_assist.minion.agents_cfg", fake_agents_cfg),
        patch("minion_assist.minion.mcp_cfg", SimpleNamespace(servers=())),
        patch("minion_assist.minion.channels_cfg", SimpleNamespace(matrix=None)),
        patch("minion_assist.agents.session.run_turn", Mock()),
        patch("minion_assist.minion.create_provider", return_value=Mock()),
        patch("builtins.input", side_effect=iter(["quit"])),
    ):
        minion_mod.main()

    # With ephemeral_history=False, the JSONL must still have the old messages.
    history = stm.load("researcher")
    assert any(m["content"] == "old question" for m in history)

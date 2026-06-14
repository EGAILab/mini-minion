"""Tests for P1 additions to AgentSession — verify_fn, fork, export (IMP-17, NEW-04)."""

from unittest.mock import Mock

from minion_assistant.agents.definitions import AgentConfig
from minion_assistant.agents.session import AgentSession, _export_html, _export_md, _had_write_call
from minion_assistant.context import Compactor
from minion_assistant.memory.short_term import ShortTermMemory
from minion_assistant.providers.base import LLMResponse, ToolCall
from minion_assistant.session import SessionStore
from minion_assistant.tools import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_provider(text="ok", tool_calls=None, finish_reason="stop"):
    provider = Mock()
    provider.chat = Mock(return_value=LLMResponse(
        text=text,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
    ))
    return provider


def _make_session(tmp_path, provider=None, agent_id="main"):
    if provider is None:
        provider = _mock_provider()
    short_term = ShortTermMemory(tmp_path / "sessions")
    session_store = SessionStore(tmp_path / "sessions.json")
    compactor = Compactor(context_window=100_000, preserve_tokens=2_000)
    return AgentSession(
        agent_id=agent_id,
        agent=AgentConfig(name="Ada", soul="You are Ada."),
        provider=provider,
        max_output_tokens=512,
        tools=ToolRegistry(),
        compactor=compactor,
        short_term=short_term,
        session_store=session_store,
    )


# ---------------------------------------------------------------------------
# _had_write_call helper
# ---------------------------------------------------------------------------

def test_had_write_call_detects_write_tool():
    history = [{"role": "assistant", "tool_calls": [
        {"function": {"name": "write", "arguments": "{}"}}
    ]}]
    assert _had_write_call(history)


def test_had_write_call_detects_edit_tool():
    history = [{"role": "assistant", "tool_calls": [
        {"function": {"name": "edit", "arguments": "{}"}}
    ]}]
    assert _had_write_call(history)


def test_had_write_call_detects_bash_tool():
    history = [{"role": "assistant", "tool_calls": [
        {"function": {"name": "bash", "arguments": "{}"}}
    ]}]
    assert _had_write_call(history)


def test_had_write_call_false_for_read_only():
    history = [{"role": "assistant", "tool_calls": [
        {"function": {"name": "read", "arguments": "{}"}}
    ]}]
    assert not _had_write_call(history)


def test_had_write_call_false_for_empty_history():
    assert not _had_write_call([])


def test_had_write_call_ignores_non_assistant_messages():
    history = [{"role": "tool", "name": "write", "content": "done"}]
    assert not _had_write_call(history)


# ---------------------------------------------------------------------------
# IMP-17: verify_fn
# ---------------------------------------------------------------------------

def test_send_calls_verify_fn_after_write_tool(tmp_path):
    """verify_fn must be called when the turn included a write-type tool call."""
    # Provider returns a tool_call for "write" on first call, then a final text.
    call_count = [0]

    def _side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: model requests the "write" tool.
            return LLMResponse(
                text="",
                tool_calls=[ToolCall(id="tc1", name="write", arguments={"path": "/f", "content": "x"})],
                finish_reason="tool_calls",
            )
        # Second call: model sends a final answer.
        return LLMResponse(text="Done.", finish_reason="stop")

    provider = Mock()
    provider.chat = Mock(side_effect=_side_effect)
    session = _make_session(tmp_path, provider=provider)
    # Register a dummy write tool that always succeeds.
    from minion_assistant.tools.base import Tool, ToolSchema
    class DummyWrite(Tool):
        @property
        def schema(self):
            return ToolSchema(name="write", description="", parameters={"type": "object", "properties": {}})
        def execute(self, **kw):
            return "ok"
    session.registry.register(DummyWrite())

    verify_calls = []
    session.send("write something", verify_fn=lambda: verify_calls.append(1) or "tests pass")

    assert len(verify_calls) == 1


def test_send_does_not_call_verify_fn_when_no_write_tool(tmp_path):
    """verify_fn must NOT be called when the turn only used read tools."""
    provider = _mock_provider(text="Here is the info.")
    session = _make_session(tmp_path, provider=provider)

    called = []
    session.send("just read", verify_fn=lambda: called.append(1) or "ok")

    assert called == []  # no write tools → verify_fn not called


def test_send_verify_fn_result_injected_into_history(tmp_path):
    """The verify_fn return value must appear in history as a user message."""
    call_count = [0]

    def _side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return LLMResponse(
                text="",
                tool_calls=[ToolCall(id="tc1", name="edit", arguments={"path": "/f"})],
                finish_reason="tool_calls",
            )
        return LLMResponse(text="Done.", finish_reason="stop")

    provider = Mock()
    provider.chat = Mock(side_effect=_side_effect)
    session = _make_session(tmp_path, provider=provider)

    from minion_assistant.tools.base import Tool, ToolSchema
    class DummyEdit(Tool):
        @property
        def schema(self):
            return ToolSchema(name="edit", description="", parameters={"type": "object", "properties": {}})
        def execute(self, **kw):
            return "ok"
    session.registry.register(DummyEdit())

    session.send("edit something", verify_fn=lambda: "linter: OK")

    # History must include the verification message.
    history = session.history
    verification_msgs = [
        m for m in history
        if m.get("role") == "user" and "[verification]" in str(m.get("content", ""))
    ]
    assert verification_msgs
    assert "linter: OK" in verification_msgs[0]["content"]


# ---------------------------------------------------------------------------
# NEW-04: fork
# ---------------------------------------------------------------------------

def test_fork_creates_new_history_file(tmp_path):
    """fork() must save a copy of history under the new agent_id."""
    provider = _mock_provider(text="Hello!")
    session = _make_session(tmp_path, provider=provider)
    session.send("hi")

    session.fork("forked")

    short_term = ShortTermMemory(tmp_path / "sessions")
    forked_history = short_term.load("forked")
    assert len(forked_history) >= 2  # user + assistant


def test_fork_creates_session_record(tmp_path):
    """fork() must create a SessionInfo record with parent_id set."""
    session = _make_session(tmp_path)
    session.send("hello")

    session.fork("child")

    store = SessionStore(tmp_path / "sessions.json")
    records = store.list_sessions()
    child = next((r for r in records if r.agent_id == "child"), None)
    assert child is not None
    assert child.parent_id == "main"


def test_fork_does_not_modify_original_history(tmp_path):
    """Forking must not alter the original session's history."""
    provider = _mock_provider(text="original")
    session = _make_session(tmp_path, provider=provider)
    session.send("original")
    original_len = len(session.history)

    session.fork("copy")

    assert len(session.history) == original_len


# ---------------------------------------------------------------------------
# NEW-04: export
# ---------------------------------------------------------------------------

def test_export_md_basic(tmp_path):
    provider = _mock_provider(text="I am Ada.")
    session = _make_session(tmp_path, provider=provider)
    session.send("who are you?")

    result = session.export(format="md")

    assert "**User:**" in result
    assert "who are you?" in result
    assert "**Ada:**" in result
    assert "I am Ada." in result


def test_export_html_basic(tmp_path):
    provider = _mock_provider(text="Hello.")
    session = _make_session(tmp_path, provider=provider)
    session.send("hi")

    result = session.export(format="html")

    assert "<!DOCTYPE html>" in result
    assert "<b>User:</b>" in result
    assert "<b>Ada:</b>" in result


def test_export_md_default_format(tmp_path):
    provider = _mock_provider(text="Sure.")
    session = _make_session(tmp_path, provider=provider)
    session.send("yes")

    result = session.export()  # default should be "md"

    assert "**User:**" in result


# ---------------------------------------------------------------------------
# _export_md and _export_html helpers (unit tests)
# ---------------------------------------------------------------------------

def test_export_md_omits_tool_calls():
    history = [
        {"role": "user", "content": "run something"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "bash"}}]},
        {"role": "tool", "content": "output"},
        {"role": "assistant", "content": "Done."},
    ]
    result = _export_md(history, "TestAgent")
    assert "bash" not in result
    assert "output" not in result
    assert "Done." in result


def test_export_html_escapes_special_chars():
    history = [
        {"role": "user", "content": "<script>alert('xss')</script>"},
        {"role": "assistant", "content": "Safe <response> & more"},
    ]
    result = _export_html(history, "Agent")
    assert "<script>" not in result
    assert "&lt;script&gt;" in result
    assert "&amp;" in result


# ---------------------------------------------------------------------------
# provider property
# ---------------------------------------------------------------------------

def test_provider_property_returns_provider(tmp_path):
    provider = _mock_provider()
    session = _make_session(tmp_path, provider=provider)
    assert session.provider is provider

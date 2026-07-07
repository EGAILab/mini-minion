"""Tests for AgentSession.send() extra_tools and system_suffix parameters."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from minion_assist.agents.session import AgentSession
from minion_assist.tools.base import Tool, ToolSchema


# ---------------------------------------------------------------------------
# Test helpers (from test_agents_session.py pattern)
# ---------------------------------------------------------------------------

def _make_session(response="test response", workspace=None):
    """Build a minimal AgentSession backed by a stub provider."""
    from minion_assist.context import Compactor
    from minion_assist.memory.short_term import ShortTermMemory
    from minion_assist.session import SessionStore
    from minion_assist.tools import default_registry
    from minion_assist.agents.definitions import AgentConfig
    import tempfile

    tmp = tempfile.mkdtemp()

    provider = MagicMock()
    provider.chat.return_value = MagicMock(
        text=response, tool_calls=[], usage=None, finish_reason="stop"
    )

    registry = default_registry(root=Path(tmp))
    compactor = Compactor(context_window=8192, preserve_tokens=1024)
    short_term = ShortTermMemory(Path(tmp) / "sessions")
    store = SessionStore(Path(tmp) / "sessions.json")

    agent = AgentConfig(
        name="TestAgent",
        soul="You are a test agent.",
        max_tool_rounds=1,
    )

    return AgentSession(
        agent_id="test",
        session_id="test-session",
        agent=agent,
        provider=provider,
        max_output_tokens=512,
        tools=registry,
        compactor=compactor,
        short_term=short_term,
        session_store=store,
    )


class _CaptureTool(Tool):
    """A test tool that records its execute calls."""

    def __init__(self):
        self.calls = []

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="capture_tool",
            description="A test tool.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    def execute(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return "captured"


# ---------------------------------------------------------------------------
# extra_tools
# ---------------------------------------------------------------------------

def test_send_extra_tools_registered_for_turn(tmp_path):
    """extra_tools are included in the registry used for that turn."""
    session = _make_session()
    capture_tool = _CaptureTool()

    # The provider doesn't call the tool — just verify it was available in
    # the definitions list passed to provider.chat().
    # chat(system, messages, tools, max_tokens) — tools is positional arg index 2.
    session.send("hello", extra_tools=[capture_tool])

    call_args = session._provider.chat.call_args
    # Positional arg 2 is `tools` (list of tool definitions).
    tools_defs = call_args[0][2]
    tool_names = [t["function"]["name"] for t in tools_defs]
    assert "capture_tool" in tool_names


def test_send_extra_tools_do_not_persist(tmp_path):
    """extra_tools from one call don't appear in the next call's registry."""
    session = _make_session()
    capture_tool = _CaptureTool()

    session.send("first", extra_tools=[capture_tool])
    session.send("second")  # no extra_tools

    second_call_args = session._provider.chat.call_args
    tools_defs = second_call_args[0][2]  # positional arg 2 = tools
    tool_names = [t["function"]["name"] for t in tools_defs]
    assert "capture_tool" not in tool_names


# ---------------------------------------------------------------------------
# system_suffix
# ---------------------------------------------------------------------------

def test_send_system_suffix_included_in_prompt(tmp_path):
    """system_suffix is appended to the system prompt for that turn."""
    session = _make_session()
    session.send("hello", system_suffix="SPECIAL GROUP CONTEXT")

    call_args = session._provider.chat.call_args
    system = call_args[0][0]  # positional arg 0 = system
    assert "SPECIAL GROUP CONTEXT" in system


def test_send_system_suffix_does_not_persist(tmp_path):
    """system_suffix from one call doesn't appear in the next call's system prompt."""
    session = _make_session()
    session.send("first", system_suffix="ONE TIME CONTEXT")
    session.send("second")  # no system_suffix

    second_call_args = session._provider.chat.call_args
    system = second_call_args[0][0]  # positional arg 0 = system
    assert "ONE TIME CONTEXT" not in system


def test_send_without_extra_tools_uses_main_registry(tmp_path):
    """When no extra_tools, the permanent registry is passed unchanged."""
    session = _make_session()
    session.send("hello")
    call_args = session._provider.chat.call_args
    tools_defs = call_args[0][2]  # positional arg 2 = tools
    # Should have permanent tools (read, write, etc.)
    tool_names = [t["function"]["name"] for t in tools_defs]
    assert "read" in tool_names

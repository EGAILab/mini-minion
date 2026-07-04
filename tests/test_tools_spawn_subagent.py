"""Tests for SpawnSubagentTool and spawn_registry helpers.

All tests use mock spawn_fn / mock SessionStore — no real LLM calls.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from minion_assist.spawn_registry import (
    MAX_CHILDREN_PER_AGENT,
    MAX_SPAWN_DEPTH,
    count_active_children,
    get_spawn_depth,
)
from minion_assist.tools.spawn_subagent import SpawnSubagentTool, _make_subagent_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(*sessions):
    """Build a duck-typed store returning the given list of session SimpleNamespace objects."""
    return SimpleNamespace(list_sessions=lambda: list(sessions))


def _session(agent_id, parent_id=None):
    return SimpleNamespace(agent_id=agent_id, parent_id=parent_id)


# ---------------------------------------------------------------------------
# spawn_registry — get_spawn_depth
# ---------------------------------------------------------------------------

class TestGetSpawnDepth:
    def test_root_agent_has_depth_zero(self):
        store = _make_store(_session("main"))
        assert get_spawn_depth("main", store) == 0

    def test_unknown_session_has_depth_zero(self):
        store = _make_store()
        assert get_spawn_depth("nonexistent", store) == 0

    def test_child_has_depth_one(self):
        store = _make_store(
            _session("main"),
            _session("sub-main-researcher-abc", parent_id="main"),
        )
        assert get_spawn_depth("sub-main-researcher-abc", store) == 1

    def test_grandchild_has_depth_two(self):
        store = _make_store(
            _session("main"),
            _session("child", parent_id="main"),
            _session("grandchild", parent_id="child"),
        )
        assert get_spawn_depth("grandchild", store) == 2

    def test_cycle_breaks_at_cap(self):
        """Cycles in parent_id chain terminate at MAX_SPAWN_DEPTH + 2."""
        # Create a cycle: A → B → A
        store = _make_store(
            _session("a", parent_id="b"),
            _session("b", parent_id="a"),
        )
        # Should not raise; returns some depth ≤ MAX_SPAWN_DEPTH + 2
        depth = get_spawn_depth("a", store)
        assert depth <= MAX_SPAWN_DEPTH + 2


# ---------------------------------------------------------------------------
# spawn_registry — count_active_children
# ---------------------------------------------------------------------------

class TestCountActiveChildren:
    def test_no_children(self):
        store = _make_store(_session("main"))
        assert count_active_children("main", store) == 0

    def test_one_child(self):
        store = _make_store(
            _session("main"),
            _session("child1", parent_id="main"),
        )
        assert count_active_children("main", store) == 1

    def test_multiple_children(self):
        store = _make_store(
            _session("main"),
            _session("c1", parent_id="main"),
            _session("c2", parent_id="main"),
            _session("c3", parent_id="main"),
        )
        assert count_active_children("main", store) == 3

    def test_only_counts_direct_children(self):
        """Grandchildren are not counted as direct children of root."""
        store = _make_store(
            _session("main"),
            _session("child", parent_id="main"),
            _session("grandchild", parent_id="child"),
        )
        assert count_active_children("main", store) == 1


# ---------------------------------------------------------------------------
# SpawnSubagentTool — schema
# ---------------------------------------------------------------------------

class TestSpawnSubagentToolSchema:
    def test_tool_name_is_spawn_subagent(self):
        tool = SpawnSubagentTool(spawn_fn=lambda *a: "")
        assert tool.schema.name == "spawn_subagent"

    def test_schema_requires_task(self):
        tool = SpawnSubagentTool(spawn_fn=lambda *a: "")
        assert "task" in tool.schema.parameters["required"]

    def test_schema_agent_id_optional(self):
        tool = SpawnSubagentTool(spawn_fn=lambda *a: "")
        assert "agent_id" not in tool.schema.parameters.get("required", [])

    def test_schema_timeout_optional(self):
        tool = SpawnSubagentTool(spawn_fn=lambda *a: "")
        assert "timeout_seconds" not in tool.schema.parameters.get("required", [])


# ---------------------------------------------------------------------------
# SpawnSubagentTool — execute
# ---------------------------------------------------------------------------

class TestSpawnSubagentToolExecute:
    def test_returns_spawn_fn_result(self):
        def _spawn(task, agent_id, timeout, relay_fn):
            return f"done:{task}"
        tool = SpawnSubagentTool(spawn_fn=_spawn)
        result = tool.execute(task="summarise the logs")
        assert result == "done:summarise the logs"

    def test_default_agent_id_is_researcher(self):
        calls = []
        def _spawn(task, agent_id, timeout, relay_fn):
            calls.append(agent_id)
            return "ok"
        tool = SpawnSubagentTool(spawn_fn=_spawn)
        tool.execute(task="do something")
        assert calls == ["researcher"]

    def test_custom_agent_id_passed_through(self):
        calls = []
        def _spawn(task, agent_id, timeout, relay_fn):
            calls.append(agent_id)
            return "ok"
        tool = SpawnSubagentTool(spawn_fn=_spawn)
        tool.execute(task="do something", agent_id="main")
        assert calls == ["main"]

    def test_default_timeout_is_120(self):
        calls = []
        def _spawn(task, agent_id, timeout, relay_fn):
            calls.append(timeout)
            return "ok"
        tool = SpawnSubagentTool(spawn_fn=_spawn)
        tool.execute(task="x")
        assert calls == [120]

    def test_custom_timeout_passed_through(self):
        calls = []
        def _spawn(task, agent_id, timeout, relay_fn):
            calls.append(timeout)
            return "ok"
        tool = SpawnSubagentTool(spawn_fn=_spawn)
        tool.execute(task="x", timeout_seconds=60)
        assert calls == [60]

    def test_relay_fn_passed_to_spawn(self):
        relay_received = []
        def _spawn(task, agent_id, timeout, relay_fn):
            relay_received.append(relay_fn)
            return "ok"
        relay = lambda e: None
        tool = SpawnSubagentTool(spawn_fn=_spawn, relay_fn=relay)
        tool.execute(task="x")
        assert relay_received[0] is relay

    def test_empty_task_returns_error(self):
        tool = SpawnSubagentTool(spawn_fn=lambda *a: "should not reach")
        result = tool.execute(task="")
        assert "Error" in result or "error" in result.lower()

    def test_whitespace_task_returns_error(self):
        tool = SpawnSubagentTool(spawn_fn=lambda *a: "should not reach")
        result = tool.execute(task="   ")
        assert "Error" in result or "error" in result.lower()

    def test_spawn_fn_error_string_returned(self):
        """When spawn_fn returns an error message, it propagates unchanged."""
        def _spawn(task, agent_id, timeout, relay_fn):
            return "[spawn_subagent] Depth limit reached (4 ≥ 4)."
        tool = SpawnSubagentTool(spawn_fn=_spawn)
        result = tool.execute(task="do something")
        assert "Depth limit" in result


# ---------------------------------------------------------------------------
# _make_subagent_registry
# ---------------------------------------------------------------------------

class TestMakeSubagentRegistry:
    def test_returns_tool_registry(self):
        from minion_assist.tools import ToolRegistry
        registry = _make_subagent_registry()
        assert isinstance(registry, ToolRegistry)

    def test_includes_read_tool(self):
        registry = _make_subagent_registry()
        assert "read" in registry._tools

    def test_includes_glob_tool(self):
        registry = _make_subagent_registry()
        assert "glob" in registry._tools

    def test_includes_grep_tool(self):
        registry = _make_subagent_registry()
        assert "grep" in registry._tools

    def test_includes_web_search(self):
        registry = _make_subagent_registry()
        assert "web_search" in registry._tools

    def test_includes_web_fetch(self):
        registry = _make_subagent_registry()
        assert "web_fetch" in registry._tools

    def test_excludes_write_tool(self):
        registry = _make_subagent_registry()
        assert "write" not in registry._tools

    def test_excludes_bash_tool(self):
        registry = _make_subagent_registry()
        assert "bash" not in registry._tools

    def test_excludes_edit_tool(self):
        registry = _make_subagent_registry()
        assert "edit" not in registry._tools

    def test_with_root_path(self, tmp_path):
        registry = _make_subagent_registry(root=tmp_path)
        assert "read" in registry._tools

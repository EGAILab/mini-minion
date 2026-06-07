"""Tests for MCP tool adapters in tools/mcp.py.

Covers McpToolAdapter, McpStatusTool, ListMcpResourcesTool, ReadMcpResourceTool:
- Adapter schema: correct name (mcp__server__tool), description, parameters,
  is_read_only=False (MCP tools default to non-read-only for safety).
- Adapter execute(): forwards kwargs to manager.call_tool_sync() using the
  ORIGINAL tool name (not the mcp__... wrapper) and returns the result string.
- Management tools (status, list, read): schema names, is_read_only=True,
  correct delegation to manager methods.

WHY MOCK THE MANAGER?
---------------------
McpClientManager owns a background asyncio thread and live MCP sessions.
In unit tests we don't want to start a real process. Using MagicMock lets
us control exactly what call_tool_sync() returns and assert that the adapter
forwards arguments correctly — without any subprocess overhead.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mini_minion.tools.mcp import (
    ListMcpResourcesTool,
    McpStatusTool,
    McpToolAdapter,
    ReadMcpResourceTool,
)
from mini_minion.mcp.types import McpConnectionStatus, McpResourceInfo, McpToolInfo


# ---------------------------------------------------------------------------
# Helpers — build fake tool/resource info and managers
# ---------------------------------------------------------------------------

def _make_tool_info(server="srv", name="do_thing", description="Does a thing", schema=None):
    return McpToolInfo(
        server_name=server,
        name=name,
        description=description,
        input_schema=schema or {"type": "object", "properties": {}},
    )


def _make_manager(tool_result="tool output", resource_result="resource content"):
    mgr = MagicMock()
    mgr.call_tool_sync.return_value = tool_result
    mgr.read_resource_sync.return_value = resource_result
    mgr.list_statuses.return_value = []
    mgr.list_tools.return_value = []
    mgr.list_resources.return_value = []
    return mgr


# ---------------------------------------------------------------------------
# McpToolAdapter — schema
# ---------------------------------------------------------------------------

class TestMcpToolAdapterSchema:
    def test_schema_name_uses_mcp_prefix(self):
        info = _make_tool_info(server="context7", name="search")
        adapter = McpToolAdapter(info, _make_manager())
        assert adapter.schema.name == "mcp__context7__search"

    def test_schema_description_from_info(self):
        info = _make_tool_info(description="Searches the web")
        adapter = McpToolAdapter(info, _make_manager())
        assert adapter.schema.description == "Searches the web"

    def test_schema_description_fallback_when_empty(self):
        info = _make_tool_info(description="")
        adapter = McpToolAdapter(info, _make_manager())
        assert "srv" in adapter.schema.description  # server name in fallback

    def test_schema_parameters_from_info(self):
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        info = _make_tool_info(schema=schema)
        adapter = McpToolAdapter(info, _make_manager())
        assert adapter.schema.parameters == schema

    def test_schema_is_not_read_only(self):
        # MCP tools may mutate state — default to NOT read-only
        info = _make_tool_info()
        adapter = McpToolAdapter(info, _make_manager())
        assert adapter.schema.is_read_only is False

    def test_schema_name_dot_in_server_sanitized(self):
        info = _make_tool_info(server="my.server", name="do_thing")
        adapter = McpToolAdapter(info, _make_manager())
        assert "." not in adapter.schema.name


# ---------------------------------------------------------------------------
# McpToolAdapter — execute
# ---------------------------------------------------------------------------

class TestMcpToolAdapterExecute:
    def test_execute_delegates_to_manager(self):
        mgr = _make_manager(tool_result="result string")
        info = _make_tool_info(server="srv", name="do_thing")
        adapter = McpToolAdapter(info, mgr)
        result = adapter.execute(query="hello")
        mgr.call_tool_sync.assert_called_once_with(
            server_name="srv",
            mcp_tool_name_str="do_thing",  # ORIGINAL name, not mcp__... wrapper
            arguments={"query": "hello"},
        )
        assert result == "result string"

    def test_execute_returns_error_string_on_failure(self):
        mgr = _make_manager()
        mgr.call_tool_sync.return_value = "[MCP error: server 'srv' is failed]"
        info = _make_tool_info()
        adapter = McpToolAdapter(info, mgr)
        result = adapter.execute()
        assert "[MCP error" in result

    def test_execute_passes_all_kwargs(self):
        mgr = _make_manager()
        info = _make_tool_info()
        adapter = McpToolAdapter(info, mgr)
        adapter.execute(a=1, b="two", c=True)
        call_args = mgr.call_tool_sync.call_args
        assert call_args.kwargs["arguments"] == {"a": 1, "b": "two", "c": True}


# ---------------------------------------------------------------------------
# McpStatusTool
# ---------------------------------------------------------------------------

class TestMcpStatusTool:
    def test_schema_is_read_only(self):
        tool = McpStatusTool(_make_manager())
        assert tool.schema.is_read_only is True

    def test_schema_name(self):
        tool = McpStatusTool(_make_manager())
        assert tool.schema.name == "mcp_status"

    def test_execute_shows_all_servers_by_default(self):
        mgr = _make_manager()
        mgr.list_statuses.return_value = [
            McpConnectionStatus(name="s1", state="connected", transport="stdio", detail="2 tool(s)"),
            McpConnectionStatus(name="s2", state="failed", transport="sse", detail="ConnectError: ..."),
        ]
        tool = McpStatusTool(mgr)
        result = tool.execute()
        assert "s1" in result
        assert "s2" in result
        assert "CONNECTED" in result
        assert "FAILED" in result

    def test_execute_filters_by_server(self):
        mgr = _make_manager()
        mgr.list_statuses.return_value = [
            McpConnectionStatus(name="s1", state="connected", transport="stdio"),
            McpConnectionStatus(name="s2", state="connected", transport="stdio"),
        ]
        tool = McpStatusTool(mgr)
        result = tool.execute(server="s1")
        assert "s1" in result
        assert "s2" not in result

    def test_execute_shows_tool_names(self):
        tool_info = McpToolInfo(server_name="s", name="search", description="", input_schema={})
        mgr = _make_manager()
        mgr.list_statuses.return_value = [
            McpConnectionStatus(
                name="s", state="connected", transport="stdio", tools=[tool_info]
            )
        ]
        tool = McpStatusTool(mgr)
        result = tool.execute()
        assert "search" in result


# ---------------------------------------------------------------------------
# ListMcpResourcesTool
# ---------------------------------------------------------------------------

class TestListMcpResourcesTool:
    def test_schema_is_read_only(self):
        tool = ListMcpResourcesTool(_make_manager())
        assert tool.schema.is_read_only is True

    def test_schema_name(self):
        assert ListMcpResourcesTool(_make_manager()).schema.name == "list_mcp_resources"

    def test_execute_no_resources(self):
        mgr = _make_manager()
        mgr.list_resources.return_value = []
        tool = ListMcpResourcesTool(mgr)
        result = tool.execute()
        assert "No MCP resources" in result

    def test_execute_lists_resources(self):
        mgr = _make_manager()
        mgr.list_resources.return_value = [
            McpResourceInfo(server_name="s", uri="fake://test", name="test", description="A test", mime_type="text/plain")
        ]
        tool = ListMcpResourcesTool(mgr)
        result = tool.execute()
        assert "fake://test" in result
        assert "s" in result

    def test_execute_passes_server_filter(self):
        mgr = _make_manager()
        mgr.list_resources.return_value = []
        tool = ListMcpResourcesTool(mgr)
        tool.execute(server="myserver")
        mgr.list_resources.assert_called_once_with(server_name="myserver")


# ---------------------------------------------------------------------------
# ReadMcpResourceTool
# ---------------------------------------------------------------------------

class TestReadMcpResourceTool:
    def test_schema_is_read_only(self):
        tool = ReadMcpResourceTool(_make_manager())
        assert tool.schema.is_read_only is True

    def test_schema_name(self):
        assert ReadMcpResourceTool(_make_manager()).schema.name == "read_mcp_resource"

    def test_execute_calls_manager(self):
        mgr = _make_manager(resource_result="file content")
        tool = ReadMcpResourceTool(mgr)
        result = tool.execute(server="mysrv", uri="fake://test")
        mgr.read_resource_sync.assert_called_once_with("mysrv", "fake://test")
        assert result == "file content"

    def test_execute_missing_server_returns_error(self):
        mgr = _make_manager()
        tool = ReadMcpResourceTool(mgr)
        result = tool.execute(uri="fake://test")
        assert "Error" in result
        mgr.read_resource_sync.assert_not_called()

    def test_execute_missing_uri_returns_error(self):
        mgr = _make_manager()
        tool = ReadMcpResourceTool(mgr)
        result = tool.execute(server="mysrv")
        assert "Error" in result
        mgr.read_resource_sync.assert_not_called()


# ---------------------------------------------------------------------------
# default_registry integration — mcp_manager wires up tools
# ---------------------------------------------------------------------------

class TestDefaultRegistryMcpIntegration:
    def test_mcp_tools_registered_when_manager_provided(self):
        from mini_minion.tools import default_registry
        from mini_minion.mcp.types import McpToolInfo

        mgr = _make_manager()
        mgr.list_tools.return_value = [
            McpToolInfo(
                server_name="srv",
                name="do_thing",
                description="Does things",
                input_schema={"type": "object", "properties": {}},
            )
        ]
        reg = default_registry(mcp_manager=mgr)
        names = {d["function"]["name"] for d in reg.definitions}
        assert "mcp_status" in names
        assert "list_mcp_resources" in names
        assert "read_mcp_resource" in names
        assert "mcp__srv__do_thing" in names

    def test_no_mcp_tools_without_manager(self):
        from mini_minion.tools import default_registry
        reg = default_registry()
        names = {d["function"]["name"] for d in reg.definitions}
        assert "mcp_status" not in names
        assert "mcp__" not in "".join(names)

    def test_existing_tools_preserved_with_mcp_manager(self):
        from mini_minion.tools import default_registry
        mgr = _make_manager()
        reg = default_registry(mcp_manager=mgr)
        names = {d["function"]["name"] for d in reg.definitions}
        # Core tools must still be present
        assert "read" in names
        assert "write" in names
        assert "bash" in names

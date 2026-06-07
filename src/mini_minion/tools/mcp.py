"""MCP tool adapters for mini-minion's ToolRegistry.

Each MCP server tool is wrapped as an McpToolAdapter so it appears in the
registry under a provider-safe name like mcp__server__tool_name.

The three management tools (McpStatusTool, ListMcpResourcesTool,
ReadMcpResourceTool) are always registered when an MCP manager is present,
giving the agent visibility into what MCP servers are available.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from .base import Tool, ToolSchema
from ..mcp.schema import mcp_tool_name

if TYPE_CHECKING:
    from ..mcp.client import McpClientManager
    from ..mcp.types import McpToolInfo


class McpToolAdapter(Tool):
    """Wraps one MCP server tool as a mini-minion Tool.

    The provider sees:
      name: "mcp__server__tool" — provider-safe, visually namespaced
      description: from MCP server's tool metadata
      parameters: normalized JSON Schema from MCP server

    Internally, the adapter stores the ORIGINAL MCP tool name so the call
    to McpClientManager.call_tool_sync() uses the right name.
    """

    def __init__(self, info: "McpToolInfo", manager: "McpClientManager") -> None:
        self._info = info
        self._manager = manager
        # Pre-compute the provider-safe name so schema() doesn't recompute it
        self._safe_name = mcp_tool_name(info.server_name, info.name)

    @property
    def schema(self) -> ToolSchema:
        desc = self._info.description or f"MCP tool from server '{self._info.server_name}'."
        return ToolSchema(
            name=self._safe_name,
            description=desc,
            parameters=self._info.input_schema or {"type": "object", "properties": {}},
            # MCP tools may mutate state — default to NOT read-only for safety.
            # The user can opt in to parallelism by knowing which tools are safe.
            is_read_only=False,
        )

    def execute(self, **kwargs: object) -> str:
        """Call the MCP tool and return its output as a string.

        Delegates to McpClientManager.call_tool_sync() which handles the
        async-to-sync bridge, retries, and timeout.
        """
        return self._manager.call_tool_sync(
            server_name=self._info.server_name,
            mcp_tool_name_str=self._info.name,  # ORIGINAL MCP name, not the mcp__... wrapper
            arguments=dict(kwargs),
        )


class McpStatusTool(Tool):
    """Shows the connection status of all configured MCP servers.

    Useful for debugging — the agent can call this to see which servers are
    connected and what tools are available from each.
    """

    def __init__(self, manager: "McpClientManager") -> None:
        self._manager = manager

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="mcp_status",
            description=(
                "Show the connection status of all configured MCP servers. "
                "Returns server name, state (connected/failed/pending), "
                "available tools, and error details for failed servers."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "Filter to one server name (optional).",
                    }
                },
            },
            is_read_only=True,
        )

    def execute(self, **kwargs: object) -> str:
        server_filter = str(kwargs.get("server", "")) or None
        lines = ["MCP server status:"]
        for status in self._manager.list_statuses():
            if server_filter and status.name != server_filter:
                continue
            lines.append(f"\n  [{status.state.upper()}] {status.name} ({status.transport})")
            if status.detail:
                lines.append(f"    {status.detail}")
            if status.tools:
                lines.append(f"    Tools: {', '.join(t.name for t in status.tools)}")
        return "\n".join(lines)


class ListMcpResourcesTool(Tool):
    """Lists all resources exposed by connected MCP servers."""

    def __init__(self, manager: "McpClientManager") -> None:
        self._manager = manager

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="list_mcp_resources",
            description="List resources available from connected MCP servers.",
            parameters={
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "Filter to one server name (optional).",
                    }
                },
            },
            is_read_only=True,
        )

    def execute(self, **kwargs: object) -> str:
        server_filter = str(kwargs.get("server", "")) or None
        resources = self._manager.list_resources(server_name=server_filter)
        if not resources:
            return "No MCP resources available."
        lines = [f"MCP resources ({len(resources)}):"]
        for r in resources:
            lines.append(f"  {r.server_name}  {r.uri}  {r.description or r.name}")
        return "\n".join(lines)


class ReadMcpResourceTool(Tool):
    """Reads the content of one MCP resource by URI."""

    def __init__(self, manager: "McpClientManager") -> None:
        self._manager = manager

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="read_mcp_resource",
            description="Read the content of an MCP resource by server name and URI.",
            parameters={
                "type": "object",
                "properties": {
                    "server": {"type": "string", "description": "MCP server name."},
                    "uri": {"type": "string", "description": "Resource URI."},
                },
                "required": ["server", "uri"],
            },
            is_read_only=True,
        )

    def execute(self, **kwargs: object) -> str:
        server = str(kwargs.get("server", "")).strip()
        uri = str(kwargs.get("uri", "")).strip()
        if not server or not uri:
            return "Error: 'server' and 'uri' are required."
        return self._manager.read_resource_sync(server, uri)

"""Runtime types for MCP client state.

Separates config (what the user wrote in config.json) from live status
(what happened when we tried to connect).

Note: McpServerConfig lives in config.py alongside the other config types
(ProviderConfig, AgentModelConfig, etc.) so it can be validated by _validate()
at import time. It is re-exported here for convenience.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

# Re-export McpServerConfig from config to keep the MCP package self-contained
# while having one source of truth for the config dataclass.
from ..config import McpServerConfig

__all__ = ["McpServerConfig", "McpToolInfo", "McpResourceInfo", "McpConnectionStatus", "McpServerNotConnectedError"]


@dataclass
class McpToolInfo:
    """Metadata for one tool discovered from an MCP server."""
    server_name: str
    name: str          # original MCP tool name (not the mcp__server__tool wrapper)
    description: str
    input_schema: dict[str, Any]


@dataclass
class McpResourceInfo:
    """Metadata for one resource discovered from an MCP server."""
    server_name: str
    uri: str
    name: str
    description: str
    mime_type: str


@dataclass
class McpConnectionStatus:
    """Live connection state for one MCP server.

    Updated by McpClientManager.connect_all_sync() and stored in manager._statuses.
    State transitions:
      "pending"   → initial state before connect_all_sync() completes
      "connected" → session initialized and tools/resources discovered
      "failed"    → connection or initialization raised an exception
      "disabled"  → not yet used (reserved for future config-level enable/disable)
    """
    name: str
    state: str          # "pending", "connected", "failed", "disabled"
    transport: str
    detail: str = ""    # human-readable status line / redacted error message
    tools: list[McpToolInfo] = field(default_factory=list)
    resources: list[McpResourceInfo] = field(default_factory=list)


class McpServerNotConnectedError(Exception):
    """Raised when a tool call targets a server that is not connected."""

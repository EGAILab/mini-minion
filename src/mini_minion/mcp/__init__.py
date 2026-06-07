"""MCP (Model Context Protocol) client package for mini-minion.

What is MCP?
------------
MCP is an open standard for connecting AI agents to external "tool servers".
Instead of hard-coding every tool into mini-minion, you can run a separate
MCP server process (e.g. a filesystem server, a database server, a web
scraper) and mini-minion connects to it and uses its tools automatically.

Think of it like a plugin system: MCP servers expose tools with JSON Schema
definitions. mini-minion discovers those tools at startup and registers them
in its ToolRegistry so any agent can call them during the TAO loop.

How the connection works
------------------------
  config.json  →  McpServerConfig  →  McpClientManager  →  MCP server
       ↑                 ↑                    ↑
  user writes    config.py parses      client.py manages
  one entry       & validates          the live session

Package contents
----------------
  client.py   — McpClientManager: owns a background asyncio loop, connects
                to all configured servers, calls tools, reads resources.
  types.py    — runtime dataclasses (McpToolInfo, McpConnectionStatus, etc.)
  schema.py   — name sanitization, JSON Schema normalization, secret redaction.

Typical usage (from minion.py)
-------------------------------
    from mini_minion.mcp import McpClientManager
    manager = McpClientManager(list(mcp_cfg.servers))
    manager.connect_all_sync()   # blocks until all servers are tried
    # ...REPL runs...
    manager.close_sync()         # clean shutdown on exit
"""
from .client import McpClientManager
from .types import McpServerConfig

__all__ = ["McpClientManager", "McpServerConfig"]

"""Fake MCP server for integration tests.

What this file is
-----------------
A minimal MCP server that the integration test in test_mcp_client.py
launches as a child process to test the full stdio transport roundtrip
without needing a real external MCP server.

How it works
------------
1. test_mcp_client.py starts this script as a subprocess using the stdio
   transport: `command=sys.executable, args=(str(fake_server_path),)`.
2. McpClientManager spawns the process and connects to it over stdin/stdout.
3. The integration test calls the "hello" tool and verifies the response.
4. On close_sync(), McpClientManager terminates the process.

FastMCP
-------
FastMCP is a helper library included in the `mcp` package that lets you
define an MCP server with simple Python decorators:
  - @app.tool()    — exposes a Python function as an MCP tool
  - @app.resource("uri") — exposes a Python function as an MCP resource

You can run this file directly to manually test your MCP client:
    python tests/fixtures/fake_mcp_server.py

Exposed capabilities:
    tool     "hello"     — returns "Hello, {name}!"
    resource "fake://test" — returns a static text string
"""
from mcp.server.fastmcp import FastMCP

# FastMCP("name") creates a new MCP server identified by the given name.
# The name is returned to clients during the MCP initialize handshake.
app = FastMCP("fake-test-server")


@app.tool()
def hello(name: str = "world") -> str:
    """Say hello to `name`. Used by integration test to verify tool dispatch."""
    return f"Hello, {name}!"


@app.resource("fake://test")
def test_resource() -> str:
    """A static test resource. Used by integration test to verify resource reads."""
    return "Test resource content"


if __name__ == "__main__":
    # app.run() starts the MCP server and listens on stdin/stdout (stdio transport).
    # The process blocks here until the parent closes the connection.
    app.run()

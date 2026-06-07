"""McpClientManager — manages connections to configured MCP servers.

THE ASYNC/SYNC BRIDGING PROBLEM
--------------------------------
mini-minion's Tool.execute() contract is synchronous (plain Python function,
returns a string). The MCP Python SDK is fully async (all operations are
coroutines that must be awaited inside an asyncio event loop).

We can't just call `asyncio.run(some_mcp_coroutine())` from inside
Tool.execute() because:
  - asyncio.run() creates a new event loop each time — expensive and slow.
  - It would close and recreate the MCP session on every tool call, which
    breaks stdio transport (each run spawns a new subprocess).

THE SOLUTION: ONE PERMANENT BACKGROUND LOOP
--------------------------------------------
McpClientManager creates a single asyncio event loop in a daemon background
thread at construction time. The loop runs forever (run_forever()) waiting
for work. Synchronous callers submit coroutines to it via:

    asyncio.run_coroutine_threadsafe(coro, self._loop)

This returns a concurrent.futures.Future. Calling .result(timeout=N) on that
future BLOCKS the calling thread until the coroutine completes. This is the
bridge: synchronous code → submits coroutine → blocks until done.

Visual:
  Main thread (sync)             Background thread (async loop)
  ─────────────────              ─────────────────────────────
  Tool.execute()                 loop.run_forever()
    └─ call_tool_sync()              ↑ waiting for work
        └─ run_coroutine_threadsafe(coro, loop)  ──→  schedules coro
           future.result(timeout=30)  ←─────────────  returns result

ONE LOOP, ALL SERVERS
---------------------
All MCP servers and both agents (Ada and Elizabeth) share the same background
loop. This avoids opening duplicate sessions when both agents use the same
server. Per-server asyncio.Locks prevent concurrent calls on a single session.

CONNECTION ISOLATION
--------------------
Each server gets its own AsyncExitStack. If one server fails to connect,
its exception is caught and recorded as state="failed". The other servers
continue connecting normally. A broken server never blocks startup.
"""
from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from .schema import (
    mcp_tool_name,
    normalize_schema_for_provider,
    normalize_windows_command,
    redact_secret_text,
)
from .types import McpConnectionStatus, McpResourceInfo, McpServerConfig, McpToolInfo

# MCP SDK — installed via pyproject.toml as "mcp>=1.0.0".
# The try/except allows the rest of mini-minion to import cleanly even if
# someone runs without MCP installed (they just can't use MCP features).
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    try:
        from mcp.client.sse import sse_client
        _HAS_SSE = True
    except ImportError:
        _HAS_SSE = False
    try:
        from mcp.client.streamable_http import streamablehttp_client
        _HAS_HTTP = True
    except ImportError:
        _HAS_HTTP = False
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    ClientSession = None  # type: ignore[assignment,misc]

# Maximum chars to return from a single MCP tool call result.
# Prevents a single noisy tool from flooding the context window.
_MAX_OUTPUT_CHARS = 8_000


class McpClientManager:
    """Manages all MCP server connections for one mini-minion process.

    Lifecycle:
        manager = McpClientManager(server_configs, output_dir=Path("~/.mini-minion/playwright-output"))
        manager.connect_all_sync()   # called once at startup; blocks until all servers tried
        # ... REPL runs, agents call tools via McpToolAdapter.execute() ...
        manager.close_sync()         # called in finally/SIGTERM; stops background loop

    Thread safety:
        Synchronous methods (call_tool_sync, read_resource_sync, etc.) are safe
        to call from any thread. They schedule coroutines onto the background
        loop and block until the coroutine completes (with a timeout).

    output_dir:
        When provided, any tool result that returns image content (e.g. a
        browser screenshot from Playwright MCP) is saved as a PNG file in this
        directory. The agent receives the file path so it can reference the
        screenshot in its response. When None (default), image content is
        described as "[Screenshot: N KB]" without saving.
    """

    def __init__(
        self,
        servers: list[McpServerConfig],
        output_dir: Path | None = None,
    ) -> None:
        self._servers = servers

        # Directory for saving image content (screenshots) from MCP tool results.
        # When set, images returned by tools like browser_take_screenshot are saved
        # to disk as PNG files and the tool result includes the saved file path.
        # When None, image content is described as "[Screenshot: N KB]" only.
        self._output_dir = output_dir

        # Live connection state — one entry per server, keyed by server name.
        self._statuses: dict[str, McpConnectionStatus] = {
            s.name: McpConnectionStatus(name=s.name, state="pending", transport=s.transport)
            for s in servers
        }

        # MCP SDK session objects — populated on successful connection.
        self._sessions: dict[str, ClientSession] = {}

        # One AsyncExitStack per server — manages the transport context manager
        # so each server can be set up and torn down independently.
        self._stacks: dict[str, AsyncExitStack] = {}

        # Per-server lock prevents concurrent tool calls on the same session.
        # MCP sessions are not thread-safe for concurrent calls.
        self._locks: dict[str, asyncio.Lock] = {}

        # Background asyncio event loop running in a daemon thread.
        # daemon=True means the thread is killed automatically when the main
        # thread exits, preventing mini-minion from hanging at shutdown.
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="mcp-event-loop",
            daemon=True,
        )
        self._thread.start()

    def _run_loop(self) -> None:
        """Entry point for the background thread — runs the asyncio event loop forever."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_sync(self, coro: Any, timeout: float = 60.0) -> Any:
        """Schedule a coroutine on the background loop and block for the result.

        This is THE core bridge between synchronous callers (Tool.execute) and
        the async MCP SDK. Every public sync method (call_tool_sync, etc.) goes
        through here.

        How asyncio.run_coroutine_threadsafe works:
        - It is the ONLY thread-safe way to submit a coroutine to an event loop
          that is running in a different thread.
        - It returns a concurrent.futures.Future (NOT an asyncio.Future).
        - future.result(timeout=N) blocks the CALLING thread until the coroutine
          finishes on the background loop thread, then returns the result (or
          re-raises any exception the coroutine raised).

        Why timeout?
        - A slow or broken MCP server should not hang mini-minion indefinitely.
        - Default is 60 s for connect operations; per-server tool_timeout
          (default 30 s) is passed explicitly for tool call coroutines.

        Args:
            coro:    An awaitable coroutine to run on the background loop.
            timeout: Maximum seconds to wait before raising TimeoutError.

        Raises:
            TimeoutError: If the coroutine does not complete within `timeout`.
            Any exception raised inside the coroutine is re-raised here.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # -------------------------------------------------------------------------
    # Connection lifecycle
    # -------------------------------------------------------------------------

    def connect_all_sync(self) -> None:
        """Connect to all configured MCP servers (blocking until all done).

        A failed server does not prevent other servers from connecting.
        Status for each server is updated in self._statuses.
        """
        if not _MCP_AVAILABLE:
            for name in self._statuses:
                self._statuses[name].state = "failed"
                self._statuses[name].detail = "MCP SDK not installed (run: uv add mcp)"
            return

        # Schedule all connections concurrently on the background loop.
        # 120s total timeout gives slow npm packages time to download.
        self._run_sync(self._connect_all_async(), timeout=120.0)

    async def _connect_all_async(self) -> None:
        """Async implementation — connects each server with isolated error handling."""
        tasks = [self._connect_one_async(server) for server in self._servers]
        # return_exceptions=True ensures one server failure doesn't cancel others
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _connect_one_async(self, server: McpServerConfig) -> None:
        """Connect to one MCP server and discover its tools and resources."""
        status = self._statuses[server.name]
        status.state = "pending"

        try:
            stack = AsyncExitStack()
            await stack.__aenter__()

            session = await self._build_session(server, stack)

            # MCP handshake — exchanges client/server capability announcements
            await session.initialize()

            # Discover tools, filtering by the server's enabled_tools allowlist
            tools_response = await session.list_tools()
            tools = [
                McpToolInfo(
                    server_name=server.name,
                    name=t.name,
                    description=t.description or "",
                    input_schema=normalize_schema_for_provider(
                        t.inputSchema if hasattr(t, "inputSchema") else {}
                    ),
                )
                for t in (tools_response.tools if tools_response else [])
                if _tool_allowed(t.name, server)
            ]

            # Discover resources — optional, some servers don't implement this
            resources: list[McpResourceInfo] = []
            try:
                res_response = await session.list_resources()
                resources = [
                    McpResourceInfo(
                        server_name=server.name,
                        uri=r.uri,
                        name=r.name or r.uri,
                        description=r.description or "",
                        mime_type=r.mimeType or "",
                    )
                    for r in (res_response.resources if res_response else [])
                ]
            except Exception:
                pass  # resources/list is optional in the MCP spec

            self._sessions[server.name] = session
            self._stacks[server.name] = stack
            # Lock must be created in the event loop that will use it
            self._locks[server.name] = asyncio.Lock()

            status.state = "connected"
            status.tools = tools
            status.resources = resources
            status.detail = f"{len(tools)} tool(s), {len(resources)} resource(s)"

        except Exception as exc:
            status.state = "failed"
            # Redact any secrets that might appear in the error message
            status.detail = redact_secret_text(f"{type(exc).__name__}: {exc}")

    async def _build_session(
        self, server: McpServerConfig, stack: AsyncExitStack
    ) -> "ClientSession":
        """Build and enter the appropriate transport for this server config."""
        transport = server.transport

        if transport == "stdio":
            cmd, args = normalize_windows_command(server.command, list(server.args))
            params = StdioServerParameters(
                command=cmd,
                args=args,
                env=server.env or None,
            )
            read, write = await stack.enter_async_context(stdio_client(params))

        elif transport == "sse":
            if not _HAS_SSE:
                raise RuntimeError("mcp.client.sse not available in this MCP SDK version")
            read, write = await stack.enter_async_context(
                sse_client(server.url, headers=server.headers or None)
            )

        elif transport == "streamableHttp":
            if not _HAS_HTTP:
                raise RuntimeError("mcp.client.streamable_http not available in this MCP SDK version")
            read, write = await stack.enter_async_context(
                streamablehttp_client(server.url, headers=server.headers or None)
            )
        else:
            raise ValueError(f"Unknown transport: {transport!r}")

        session = await stack.enter_async_context(ClientSession(read, write))
        return session

    # -------------------------------------------------------------------------
    # Tool calls
    # -------------------------------------------------------------------------

    def call_tool_sync(
        self,
        server_name: str,
        mcp_tool_name_str: str,
        arguments: dict,
        timeout: float | None = None,
    ) -> str:
        """Call an MCP tool synchronously and return the result as a string.

        Retries once on transient transport failures (broken pipe, connection reset).

        Args:
            server_name:       The MCP server key (matches McpServerConfig.name).
            mcp_tool_name_str: The ORIGINAL MCP tool name (not the mcp__... wrapper).
            arguments:         Tool input arguments dict.
            timeout:           Seconds to wait (default: server's tool_timeout).

        Returns:
            Formatted string result, or an error string if the call failed.
        """
        status = self._statuses.get(server_name)
        if status is None or status.state != "connected":
            state = status.state if status else "unknown"
            return f"[MCP error: server '{server_name}' is {state}]"

        # Resolve timeout from per-server config when not explicitly provided
        if timeout is None:
            cfg = next((s for s in self._servers if s.name == server_name), None)
            timeout = float(cfg.tool_timeout if cfg else 30)

        try:
            return self._run_sync(
                self._call_tool_async(server_name, mcp_tool_name_str, arguments),
                timeout=timeout,
            )
        except TimeoutError:
            return f"[MCP error: tool '{mcp_tool_name_str}' timed out after {timeout:.0f}s]"
        except Exception as exc:
            # Retry once on transient transport errors (broken pipe, EOF, etc.)
            _transient = ("BrokenPipeError", "ConnectionResetError", "EOFError")
            if type(exc).__name__ in _transient:
                try:
                    return self._run_sync(
                        self._call_tool_async(server_name, mcp_tool_name_str, arguments),
                        timeout=timeout,
                    )
                except Exception as exc2:
                    return f"[MCP error (retry): {type(exc2).__name__}: {exc2}]"
            return f"[MCP error: {type(exc).__name__}: {exc}]"

    async def _call_tool_async(
        self, server_name: str, tool_name: str, arguments: dict
    ) -> str:
        """Async implementation — calls the MCP session and formats the result."""
        session = self._sessions[server_name]
        lock = self._locks[server_name]

        # Per-server lock ensures only one in-flight call at a time.
        # MCP sessions use a single request/response stream and don't support
        # concurrent calls without multiplexing.
        async with lock:
            result = await session.call_tool(tool_name, arguments=arguments)

        # Pass output_dir so image content (screenshots) can be saved to disk.
        return _format_tool_result(result, output_dir=self._output_dir)

    # -------------------------------------------------------------------------
    # Resource reads
    # -------------------------------------------------------------------------

    def read_resource_sync(self, server_name: str, uri: str, timeout: float = 30.0) -> str:
        """Read an MCP resource synchronously and return content as a string."""
        status = self._statuses.get(server_name)
        if status is None or status.state != "connected":
            state = status.state if status else "unknown"
            return f"[MCP error: server '{server_name}' is {state}]"

        try:
            return self._run_sync(
                self._read_resource_async(server_name, uri),
                timeout=timeout,
            )
        except Exception as exc:
            return f"[MCP error reading resource: {type(exc).__name__}: {exc}]"

    async def _read_resource_async(self, server_name: str, uri: str) -> str:
        session = self._sessions[server_name]
        lock = self._locks[server_name]
        async with lock:
            result = await session.read_resource(uri)
        return _format_resource_result(result)

    # -------------------------------------------------------------------------
    # Status and discovery
    # -------------------------------------------------------------------------

    def list_statuses(self) -> list[McpConnectionStatus]:
        """Return a snapshot of all server connection statuses."""
        return list(self._statuses.values())

    def list_tools(self) -> list[McpToolInfo]:
        """Return all tools from all connected servers."""
        tools = []
        for status in self._statuses.values():
            if status.state == "connected":
                tools.extend(status.tools)
        return tools

    def list_resources(self, server_name: str | None = None) -> list[McpResourceInfo]:
        """Return resources, optionally filtered to one server."""
        resources = []
        for status in self._statuses.values():
            if status.state == "connected":
                if server_name is None or status.name == server_name:
                    resources.extend(status.resources)
        return resources

    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------

    def close_sync(self, timeout: float = 10.0) -> None:
        """Close all MCP sessions and stop the background event loop.

        Called in the finally block of minion.py's main() so sessions are
        cleaned up on both normal exit and KeyboardInterrupt.
        """
        if self._loop.is_closed():
            return
        try:
            self._run_sync(self._close_all_async(), timeout=timeout)
        except Exception:
            pass  # don't let cleanup errors hide user history persistence

        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)

    async def _close_all_async(self) -> None:
        """Close all AsyncExitStacks (which closes sessions and transports)."""
        for stack in list(self._stacks.values()):
            try:
                await stack.__aexit__(None, None, None)
            except Exception:
                pass
        self._stacks.clear()
        self._sessions.clear()


# ─── Result formatting helpers ────────────────────────────────────────────────

def _format_tool_result(result: Any, output_dir: Path | None = None) -> str:
    """Convert an MCP CallToolResult to a plain string for the agent to read.

    WHY output_dir THREADS THROUGH SO MANY LAYERS
    -----------------------------------------------
    McpClientManager.__init__ receives output_dir from minion.py.
    _call_tool_async passes it to this function.
    This function passes it to _handle_image_content.

    This threading is necessary because:
      - _format_tool_result is a module-level function (not a method), so it
        can't access self._output_dir directly.
      - We want to keep _format_tool_result and _handle_image_content testable
        in isolation — the tests can pass any output_dir (including tmp_path)
        without constructing a full McpClientManager.

    CONTENT BLOCK TYPES
    --------------------
    The MCP SDK returns tool results as a list of typed content blocks:
      - TextContent  (type="text"):  plain string output — most common
      - ImageContent (type="image"): base64-encoded PNG/JPEG — from screenshot tools
      - Other types (type="resource", etc.): show a compact descriptor

    Args:
        result:     The MCP SDK CallToolResult object (or None if call failed).
        output_dir: If set, image blocks are saved to disk at this path.
    """
    if result is None:
        return "(no output)"

    parts: list[str] = []

    # result.content is a list of content blocks (TextContent, ImageContent, etc.)
    content = getattr(result, "content", None) or []
    for block in content:
        block_type = getattr(block, "type", "")

        if block_type == "text":
            # Plain text — add directly to output.
            parts.append(getattr(block, "text", ""))

        elif block_type == "image":
            # Image content (e.g. a browser screenshot from Playwright MCP).
            # The block contains base64-encoded bytes and a MIME type.
            parts.append(_handle_image_content(block, output_dir))

        else:
            # Unknown content type — show a compact descriptor so the agent
            # at least knows something non-text was returned.
            parts.append(f"[MCP content: type={block_type}]")

    # If no content blocks, try structuredContent (newer MCP SDK versions)
    if not parts:
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            try:
                parts.append(json.dumps(structured, indent=2))
            except Exception:
                parts.append(str(structured))

    output = "\n".join(p for p in parts if p) or "(no output)"

    # Cap output to avoid flooding the context window.
    if len(output) > _MAX_OUTPUT_CHARS:
        output = output[:_MAX_OUTPUT_CHARS] + "\n[... MCP output truncated]"

    return output


def _handle_image_content(block: Any, output_dir: Path | None) -> str:
    """Handle an MCP ImageContent block, optionally saving to disk.

    WHAT IS AN MCP IMAGECONTENT BLOCK?
    ------------------------------------
    When a tool like browser_take_screenshot returns an image, the MCP SDK
    wraps it in an ImageContent object with three fields:
      - type:     "image"  (always "image" for this handler)
      - mimeType: e.g. "image/png", "image/jpeg"
      - data:     the raw image bytes, base64-encoded as a string

    WHY BASE64?
    -----------
    MCP uses JSON to carry all messages. JSON can't embed raw binary data, so
    the image bytes are encoded as a base64 ASCII string. We decode them here
    to get the original bytes back (needed to compute file size and save to disk).

    WHY SAVE TO DISK?
    -----------------
    If we included the raw base64 string in the tool result, it could easily
    be 50-100+ KB of ASCII text, which would:
      1. Flood the context window (the agent can't process an image as text)
      2. Slow down rendering in the terminal
    Instead, we save the image to a stable file path and tell the agent where
    it was saved. The agent can then reference the path in its response.

    WHY A MILLISECOND TIMESTAMP IN THE FILENAME?
    ---------------------------------------------
    Multiple screenshots can be taken in quick succession (e.g. the agent
    clicks and immediately takes another screenshot). Using milliseconds
    instead of whole seconds ensures each screenshot gets a unique filename
    even within the same second.

    Args:
        block:      An MCP ImageContent object with .mimeType and .data fields.
        output_dir: Where to save screenshots. Created automatically if needed.
                    Pass None when you don't need files saved (e.g. in tests).

    Returns:
        A human-readable string for the agent:
          - With output_dir: "[Screenshot saved: /path/to/file.png (image/png, 45 KB)]"
          - Without: "[Screenshot: image/png, 45 KB — configure output_dir ...]"
    """
    # We use getattr() throughout this function instead of direct attribute access
    # (e.g. block.mimeType) because:
    #   1. The MCP SDK version might change field names
    #   2. Our test mocks use MagicMock which behaves differently than SDK objects
    #   3. getattr(..., default) gives us a safe fallback if the field is absent
    mime = getattr(block, "mimeType", "") or "image/png"
    raw_data = getattr(block, "data", "") or ""

    # Decode the base64 string back to raw bytes.
    # We wrap this in try/except because malformed base64 (missing padding,
    # invalid characters) raises binascii.Error — we'd rather show a 0 KB
    # descriptor than crash the entire tool call.
    try:
        img_bytes = base64.b64decode(raw_data) if raw_data else b""
    except Exception:
        img_bytes = b""

    size_kb = len(img_bytes) / 1024

    if output_dir is not None and img_bytes:
        # Save the image to disk so the agent can reference the file path.
        # mkdir(parents=True, exist_ok=True) creates the full path including
        # any missing parent directories, and does nothing if it already exists.
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract the file extension from the MIME type.
        # "image/png" → "png",  "image/jpeg" → "jpeg"
        ext = mime.split("/")[-1] if "/" in mime else "png"

        # Millisecond timestamp for a unique, sortable filename.
        timestamp = int(time.time() * 1000)
        filename = f"screenshot-{timestamp}.{ext}"
        dest = output_dir / filename

        try:
            dest.write_bytes(img_bytes)
            return (
                f"[Screenshot saved: {dest}  ({mime}, {size_kb:.0f} KB)]\n"
                f"The screenshot shows the current browser state. "
                f"Use browser_snapshot for text-based page analysis."
            )
        except OSError as e:
            # File write can fail if the disk is full, the directory is read-only,
            # or the path is too long (Windows has a 260-char limit by default).
            # Returning a string (not raising) preserves the "execute() never raises"
            # contract that the rest of mini-minion's tool infrastructure relies on.
            return f"[Screenshot: {mime}, {size_kb:.0f} KB — save failed: {e}]"

    # No output_dir was configured, OR the image data was empty/invalid.
    # Return a descriptor so the agent still knows something non-text was produced.
    return (
        f"[Screenshot: {mime}, {size_kb:.0f} KB — "
        f"configure output_dir in McpClientManager to save screenshots to disk]"
    )


def _format_resource_result(result: Any) -> str:
    """Convert an MCP ReadResourceResult to a plain string."""
    if result is None:
        return "(no content)"
    contents = getattr(result, "contents", None) or []
    parts = []
    for item in contents:
        if hasattr(item, "text"):
            parts.append(item.text)
        elif hasattr(item, "blob"):
            blob = item.blob
            size = len(blob) if blob else 0
            parts.append(f"[Binary resource: {size} bytes]")
        else:
            parts.append(str(item))
    return "\n".join(p for p in parts if p) or "(no content)"


def _tool_allowed(tool_name: str, server: McpServerConfig) -> bool:
    """Check if a tool name is in the server's enabled_tools allowlist.

    Accepts both the original MCP tool name and the mcp__server__tool wrapper.
    "*" in enabled_tools means all tools are allowed.
    """
    if "*" in server.enabled_tools:
        return True
    return (
        tool_name in server.enabled_tools
        or mcp_tool_name(server.name, tool_name) in server.enabled_tools
    )

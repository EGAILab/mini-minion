"""Tests for McpClientManager — connection lifecycle and tool dispatch.

HOW THESE TESTS WORK WITHOUT REAL MCP SERVERS
----------------------------------------------
Most tests here don't spawn real MCP processes. Instead they:

1. Create a McpClientManager with configs pointing at non-existent commands.
2. Verify that failures are isolated (bad server → "failed", not crash).
3. Verify that disconnected-server calls return error strings, not exceptions.
4. Test the result-formatting helpers (_format_tool_result, etc.) with
   MagicMock objects that mimic the MCP SDK's result types.

This means tests run in milliseconds and don't require network access.

INTEGRATION TEST
----------------
The last test (test_integration_fake_stdio_server) is the real deal:
it spawns tests/fixtures/fake_mcp_server.py as a subprocess and tests
the full stdio transport roundtrip. It is skipped when the `mcp` package
is not installed, so CI always passes even without MCP support.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from minion_assistant.mcp.client import (
    McpClientManager,
    _MCP_AVAILABLE,
    _format_tool_result,
    _format_resource_result,
    _handle_image_content,
    _tool_allowed,
)
from minion_assistant.mcp.types import McpConnectionStatus, McpServerConfig


# ---------------------------------------------------------------------------
# Helpers — build a minimal McpServerConfig
# ---------------------------------------------------------------------------

def _stdio_config(name="test", command="echo", args=()):
    return McpServerConfig(name=name, transport="stdio", command=command, args=args)


def _sse_config(name="test", url="http://localhost:9999/sse"):
    return McpServerConfig(name=name, transport="sse", url=url)


# ---------------------------------------------------------------------------
# McpClientManager — initialization
# ---------------------------------------------------------------------------

class TestMcpClientManagerInit:
    def test_creates_pending_status_for_each_server(self):
        cfg = [_stdio_config("s1"), _stdio_config("s2")]
        mgr = McpClientManager(cfg)
        try:
            assert set(mgr._statuses) == {"s1", "s2"}
            assert all(s.state == "pending" for s in mgr._statuses.values())
        finally:
            mgr.close_sync()

    def test_background_thread_started(self):
        mgr = McpClientManager([_stdio_config()])
        try:
            assert mgr._thread.is_alive()
        finally:
            mgr.close_sync()

    def test_empty_servers_list(self):
        mgr = McpClientManager([])
        try:
            assert mgr.list_tools() == []
            assert mgr.list_statuses() == []
        finally:
            mgr.close_sync()


# ---------------------------------------------------------------------------
# McpClientManager — failure isolation
# ---------------------------------------------------------------------------

class TestFailureIsolation:
    def test_one_bad_server_does_not_stop_others(self):
        """A server that fails to connect must not prevent others from connecting."""
        bad = _stdio_config("bad", command="this-command-does-not-exist-xyz")
        good = _stdio_config("good", command="this-also-does-not-exist-abc")

        mgr = McpClientManager([bad, good])
        try:
            mgr.connect_all_sync()
            # Both should end up in "failed" state (commands don't exist),
            # but neither should still be "pending"
            statuses = {s.name: s.state for s in mgr.list_statuses()}
            assert statuses["bad"] in ("failed", "connected")
            assert statuses["good"] in ("failed", "connected")
            # The critical property: both reached a terminal state
            assert "pending" not in statuses.values()
        finally:
            mgr.close_sync()


# ---------------------------------------------------------------------------
# McpClientManager — call_tool_sync on disconnected server
# ---------------------------------------------------------------------------

class TestCallToolOnDisconnectedServer:
    def test_unknown_server_returns_error_string(self):
        mgr = McpClientManager([])
        try:
            result = mgr.call_tool_sync("nonexistent", "some_tool", {})
            assert "error" in result.lower() or "unknown" in result.lower()
        finally:
            mgr.close_sync()

    def test_pending_server_returns_error_string(self):
        cfg = _stdio_config("pending_srv")
        mgr = McpClientManager([cfg])
        try:
            # Don't call connect_all_sync — server stays "pending"
            result = mgr.call_tool_sync("pending_srv", "tool", {})
            assert "[MCP error" in result
            assert "pending" in result
        finally:
            mgr.close_sync()

    def test_failed_server_returns_error_string(self):
        cfg = _stdio_config("failsrv", command="this-command-xyz-does-not-exist")
        mgr = McpClientManager([cfg])
        try:
            mgr.connect_all_sync()
            # After failed connection, call_tool must return an error string
            result = mgr.call_tool_sync("failsrv", "any_tool", {})
            assert "[MCP error" in result
        finally:
            mgr.close_sync()


# ---------------------------------------------------------------------------
# McpClientManager — read_resource on disconnected server
# ---------------------------------------------------------------------------

class TestReadResourceOnDisconnectedServer:
    def test_unknown_server_returns_error_string(self):
        mgr = McpClientManager([])
        try:
            result = mgr.read_resource_sync("nonexistent", "fake://uri")
            assert "[MCP error" in result
        finally:
            mgr.close_sync()


# ---------------------------------------------------------------------------
# McpClientManager — close_sync idempotency
# ---------------------------------------------------------------------------

class TestCloseSyncIdempotency:
    def test_close_twice_does_not_raise(self):
        mgr = McpClientManager([])
        mgr.close_sync()
        mgr.close_sync()  # second call must not raise

    def test_close_empty_manager_does_not_raise(self):
        mgr = McpClientManager([])
        mgr.close_sync()


# ---------------------------------------------------------------------------
# McpClientManager — list_tools and list_resources
# ---------------------------------------------------------------------------

class TestListToolsAndResources:
    def test_list_tools_empty_when_no_connected_servers(self):
        mgr = McpClientManager([_stdio_config()])
        try:
            tools = mgr.list_tools()
            assert tools == []
        finally:
            mgr.close_sync()

    def test_list_resources_with_server_filter(self):
        mgr = McpClientManager([])
        try:
            result = mgr.list_resources(server_name="nonexistent")
            assert result == []
        finally:
            mgr.close_sync()


# ---------------------------------------------------------------------------
# _format_tool_result
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _handle_image_content — screenshot saving
#
# WHY THESE TESTS EXIST
# ----------------------
# When Playwright MCP takes a screenshot, the MCP SDK returns an ImageContent
# block — NOT a text string. The block looks like this in Python:
#
#   block.type     = "image"
#   block.mimeType = "image/png"
#   block.data     = "<very long base64 string>"   ← the actual PNG bytes, encoded
#
# Without special handling, the agent would receive "[MCP content: type=image]"
# which tells it nothing useful. _handle_image_content fixes this by:
#   1. Decoding the base64 data back to raw bytes
#   2. Saving the PNG to disk (if output_dir is configured)
#   3. Returning the saved file path so the agent can mention it in its response
#
# HOW THE MOCK WORKS
# ------------------
# We can't use a real Playwright MCP server in unit tests — it would require
# a browser and network access. Instead, _make_block() builds a MagicMock
# that has the same fields as a real ImageContent object (mimeType, data).
# We set block.data to base64-encoded bytes so the decoding code exercises
# the same path it would with a real screenshot.
# ---------------------------------------------------------------------------

class TestHandleImageContent:
    def _make_block(self, mime="image/png", data_bytes=b"") -> "MagicMock":
        """Build a fake MCP ImageContent block for testing.

        A real MCP ImageContent object has these fields:
          - type:     "image"  (set by the dispatch code, not needed here)
          - mimeType: e.g. "image/png"
          - data:     base64-encoded image bytes as a string

        We base64-encode the data_bytes here exactly as the MCP SDK would,
        so _handle_image_content sees the same input format as with a real
        Playwright screenshot.
        """
        import base64
        block = MagicMock()
        block.mimeType = mime
        # base64.b64encode returns bytes; .decode() converts to str
        # because the MCP SDK stores data as a string, not bytes
        block.data = base64.b64encode(data_bytes).decode() if data_bytes else ""
        return block

    def test_no_output_dir_returns_descriptor_without_path(self):
        """Without output_dir, image content must describe size but not save.

        This covers the case where minion-assistant is used without a workspace
        or when image saving is explicitly disabled. The agent still receives
        useful information (mime type, size) instead of a silent empty string.
        """
        block = self._make_block(mime="image/png", data_bytes=b"fake-png-bytes")
        result = _handle_image_content(block, output_dir=None)
        assert "Screenshot" in result      # agent knows something visual happened
        assert "image/png" in result       # agent knows the mime type
        assert "saved" not in result.lower()  # no false claim about saving

    def test_with_output_dir_saves_file(self, tmp_path):
        """With output_dir, the PNG must be written to disk and path returned.

        The agent uses this file path in its response ("I took a screenshot,
        saved at /path/to/screenshot-1234.png"). A file that doesn't actually
        exist on disk would make that response misleading.

        We verify both:
          1. The result string contains evidence of saving ("saved" OR the path).
          2. A real .png file exists at that directory.
        """
        block = self._make_block(mime="image/png", data_bytes=b"\x89PNG-fake-data")
        result = _handle_image_content(block, output_dir=tmp_path)
        # Both assertion forms are acceptable — the function may phrase it either way
        assert "saved" in result.lower() or str(tmp_path) in result
        # The file must exist on disk — not just mentioned in the string
        saved_files = list(tmp_path.glob("*.png"))
        assert len(saved_files) == 1

    def test_with_output_dir_creates_directory(self, tmp_path):
        """output_dir must be auto-created — callers should not need to mkdir first.

        minion.py passes `workspace / "playwright-output"` which doesn't exist
        on first run. If _handle_image_content doesn't create it, the first
        screenshot would always fail with FileNotFoundError.
        """
        nested = tmp_path / "screenshots" / "nested"
        # The nested directory does NOT exist yet — the function must create it
        block = self._make_block(data_bytes=b"\x89PNG")
        _handle_image_content(block, output_dir=nested)
        assert nested.exists()

    def test_empty_data_no_file_written(self, tmp_path):
        """Empty image data must not write a zero-byte PNG file to disk.

        A zero-byte PNG is invalid and would confuse any image viewer. The
        function should skip saving entirely when it has nothing to write,
        returning a descriptor instead.
        """
        block = self._make_block(data_bytes=b"")
        result = _handle_image_content(block, output_dir=tmp_path)
        # No files at all — not even a zero-byte one
        assert len(list(tmp_path.glob("*"))) == 0


class TestFormatToolResult:
    def test_none_returns_no_output(self):
        assert _format_tool_result(None) == "(no output)"

    def test_text_content_extracted(self):
        block = MagicMock()
        block.type = "text"
        block.text = "hello world"
        result_obj = MagicMock()
        result_obj.content = [block]
        result_obj.structuredContent = None
        assert _format_tool_result(result_obj) == "hello world"

    def test_image_content_shows_screenshot_descriptor(self):
        """Image content (browser_take_screenshot) must return a Screenshot descriptor.

        Before this improvement, image content returned "[MCP content: type=image]"
        which is useless — the agent has no idea what was captured or where.
        Now it returns "[Screenshot: image/png, 0 KB — ...]" which is informative.

        We set block.mimeType explicitly (not leaving it as a raw MagicMock attribute)
        because _handle_image_content uses getattr(block, "mimeType", "") — if
        mimeType is a MagicMock object the string would contain "mock" not "image/png".
        Setting it to a real string ensures the assertion checks meaningful content.
        """
        block = MagicMock()
        block.type = "image"
        block.mimeType = "image/png"   # must be a real string, not a MagicMock
        block.data = ""                # empty data → 0 KB, no file write attempt
        result_obj = MagicMock()
        result_obj.content = [block]
        result_obj.structuredContent = None
        result = _format_tool_result(result_obj)
        # New behavior: always returns "Screenshot:" descriptor for image blocks
        assert "Screenshot" in result
        assert "image/png" in result

    def test_output_truncated_at_max(self):
        block = MagicMock()
        block.type = "text"
        block.text = "x" * 10_000
        result_obj = MagicMock()
        result_obj.content = [block]
        result_obj.structuredContent = None
        result = _format_tool_result(result_obj)
        assert "truncated" in result
        assert len(result) < 10_000

    def test_empty_content_falls_back_to_structured(self):
        result_obj = MagicMock()
        result_obj.content = []
        result_obj.structuredContent = {"key": "value"}
        result = _format_tool_result(result_obj)
        assert "key" in result


# ---------------------------------------------------------------------------
# _format_resource_result
# ---------------------------------------------------------------------------

class TestFormatResourceResult:
    def test_none_returns_no_content(self):
        assert _format_resource_result(None) == "(no content)"

    def test_text_item_extracted(self):
        item = MagicMock(spec=["text"])
        item.text = "resource content"
        result_obj = MagicMock()
        result_obj.contents = [item]
        assert _format_resource_result(result_obj) == "resource content"

    def test_binary_item_shows_size(self):
        item = MagicMock(spec=["blob"])
        item.blob = b"abc"
        result_obj = MagicMock()
        result_obj.contents = [item]
        result = _format_resource_result(result_obj)
        assert "Binary" in result or "bytes" in result


# ---------------------------------------------------------------------------
# _tool_allowed
# ---------------------------------------------------------------------------

class TestToolAllowed:
    def test_star_allows_all(self):
        cfg = McpServerConfig(name="s", transport="stdio", enabled_tools=("*",))
        assert _tool_allowed("any_tool", cfg) is True

    def test_explicit_allowlist(self):
        cfg = McpServerConfig(name="s", transport="stdio", enabled_tools=("search",))
        assert _tool_allowed("search", cfg) is True
        assert _tool_allowed("other", cfg) is False

    def test_mcp_wrapper_name_allowed(self):
        # Tools can be allowed by either original name or mcp__server__tool name
        cfg = McpServerConfig(name="srv", transport="stdio", enabled_tools=("mcp__srv__search",))
        assert _tool_allowed("search", cfg) is True


# ---------------------------------------------------------------------------
# Integration test — requires mcp package + real fake_mcp_server.py
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _MCP_AVAILABLE, reason="mcp package not installed")
def test_integration_fake_stdio_server():
    """Full stdio roundtrip test: McpClientManager ↔ fake_mcp_server.py.

    This test verifies the whole async-to-sync bridge works end to end:
      1. McpClientManager spawns fake_mcp_server.py as a child process.
      2. They exchange the MCP initialize handshake over stdin/stdout.
      3. list_tools() discovers the "hello" tool.
      4. call_tool_sync("fake", "hello", {"name": "test"}) returns "Hello, test!".
      5. close_sync() terminates the child process cleanly.

    If this test fails it means either the MCP SDK API changed, the async
    event loop bridge is broken, or the fake server fixture has a bug.
    """
    fake_server_path = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"
    if not fake_server_path.exists():
        pytest.skip("fake_mcp_server.py fixture not found")

    config = McpServerConfig(
        name="fake",
        transport="stdio",
        command=sys.executable,
        args=(str(fake_server_path),),
    )
    manager = McpClientManager([config])
    try:
        manager.connect_all_sync()
        status = manager._statuses["fake"]
        assert status.state == "connected", f"Expected connected, got {status.state}: {status.detail}"
        assert any(t.name == "hello" for t in status.tools), f"hello tool not found: {[t.name for t in status.tools]}"

        result = manager.call_tool_sync("fake", "hello", {"name": "test"})
        assert "Hello" in result, f"Unexpected result: {result}"
    finally:
        manager.close_sync()

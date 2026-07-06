"""Tests for CodexProvider dynamic tool bridge and approval handling.

These tests exercise the two server-request paths in _CodexRpcClient without
starting a real Codex process:

1. item/tool/call → dynamic tool execution via ToolRegistry
2. item/commandExecution/requestApproval (and variants) → approval callback

The approach: instantiate _CodexRpcClient with a MagicMock subprocess, call
_handle_server_request directly, and assert on what was written to stdin.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from minion_assist.providers.codex import (
    _CodexRpcClient,
    _build_dynamic_tool_specs,
    _codex_tool_name,
    _DYNAMIC_TOOL_NAMESPACE,
)
from minion_assist.tools.registry import ToolRegistry
from minion_assist.tools.base import Tool, ToolSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _EchoTool(Tool):
    """Minimal tool that echoes its 'text' argument."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="echo",
            description="Echo text back.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        return str(kwargs.get("text", ""))


class _ErrorTool(Tool):
    """Tool that always raises an exception."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="boom",
            description="Always fails.",
            parameters={"type": "object", "properties": {}},
        )

    def execute(self, **kwargs: object) -> str:
        raise RuntimeError("tool exploded")


class _McpEchoTool(Tool):
    """Simulates an MCP tool (name has mcp__ prefix)."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="mcp__playwright__browser_close",
            description="Close the browser.",
            parameters={"type": "object", "properties": {}},
        )

    def execute(self, **kwargs: object) -> str:
        return "browser closed"


def _make_client(registry=None, approve_command=None) -> tuple[_CodexRpcClient, list[str]]:
    """Build a _CodexRpcClient with a fake subprocess; return (client, written_lines)."""
    written: list[str] = []

    # Fake stdin that records writes.
    fake_stdin = MagicMock()
    fake_stdin.write = lambda line: written.append(line.rstrip("\n"))
    fake_stdin.flush = lambda: None

    # Fake stdout that yields nothing (reader thread will exit immediately).
    fake_stdout = iter([])

    fake_proc = MagicMock()
    fake_proc.stdin = fake_stdin
    fake_proc.stdout = fake_stdout

    with patch("subprocess.Popen", return_value=fake_proc):
        client = _CodexRpcClient(
            ["codex", "app-server"],
            registry=registry,
            approve_command=approve_command,
        )

    # Stop the reader thread (it exits immediately since stdout is exhausted).
    client._reader.join(timeout=1.0)

    return client, written


# ---------------------------------------------------------------------------
# _build_dynamic_tool_specs
# ---------------------------------------------------------------------------

class TestBuildDynamicToolSpecs:
    def test_empty_registry_returns_empty_list(self):
        registry = ToolRegistry()
        specs, name_map = _build_dynamic_tool_specs(registry)
        assert specs == []
        assert name_map == {}

    def test_single_tool_produces_namespace_spec(self):
        registry = ToolRegistry()
        registry.register(_EchoTool())
        specs, _ = _build_dynamic_tool_specs(registry)

        assert len(specs) == 1
        ns = specs[0]
        assert ns["type"] == "namespace"
        assert ns["name"] == _DYNAMIC_TOOL_NAMESPACE
        assert len(ns["tools"]) == 1

    def test_tool_spec_fields(self):
        registry = ToolRegistry()
        registry.register(_EchoTool())
        specs, _ = _build_dynamic_tool_specs(registry)
        tool_spec = specs[0]["tools"][0]

        assert tool_spec["type"] == "function"
        assert tool_spec["name"] == "echo"
        assert tool_spec["description"] == "Echo text back."
        # inputSchema is the parameters JSON Schema object.
        assert tool_spec["inputSchema"]["type"] == "object"
        # deferLoading tells Codex to load the schema lazily.
        assert tool_spec["deferLoading"] is True

    def test_multiple_tools_all_in_namespace(self):
        registry = ToolRegistry()
        registry.register(_EchoTool())
        registry.register(_ErrorTool())
        specs, _ = _build_dynamic_tool_specs(registry)
        tool_names = [t["name"] for t in specs[0]["tools"]]
        assert "echo" in tool_names
        assert "boom" in tool_names

    def test_mcp_prefix_stripped_for_codex(self):
        """mcp__* names are stripped to avoid Codex's reserved mcp__ namespace."""
        assert _codex_tool_name("mcp__playwright__browser_close") == "playwright__browser_close"
        assert _codex_tool_name("mcp__search__web_search") == "search__web_search"

    def test_non_mcp_names_pass_through(self):
        assert _codex_tool_name("web_search") == "web_search"
        assert _codex_tool_name("bash") == "bash"

    def test_name_map_maps_codex_name_to_registry_name(self):
        """name_map returned by _build_dynamic_tool_specs allows reverse lookup."""
        registry = ToolRegistry()
        registry.register(_EchoTool())  # name = "echo" (no mcp__ prefix)
        _, name_map = _build_dynamic_tool_specs(registry)
        assert name_map["echo"] == "echo"


# ---------------------------------------------------------------------------
# item/tool/call dispatch
# ---------------------------------------------------------------------------

class TestHandleToolCall:
    def test_known_tool_executes_and_returns_content_items(self):
        registry = ToolRegistry()
        registry.register(_EchoTool())
        client, written = _make_client(registry=registry)

        client._handle_server_request(req_id=1, method="item/tool/call", _params={
            "tool": "echo",
            "arguments": {"text": "hello"},
        })

        assert written, "expected a response to be written"
        resp = json.loads(written[-1])
        assert resp["id"] == 1
        result = resp["result"]
        assert result["success"] is True
        assert result["contentItems"][0]["type"] == "inputText"
        assert result["contentItems"][0]["text"] == "hello"

    def test_unknown_tool_returns_failure(self):
        registry = ToolRegistry()
        client, written = _make_client(registry=registry)

        client._handle_server_request(req_id=2, method="item/tool/call", _params={
            "tool": "nonexistent",
            "arguments": {},
        })

        resp = json.loads(written[-1])
        result = resp["result"]
        assert result["success"] is False
        assert "nonexistent" in result["contentItems"][0]["text"]

    def test_tool_exception_text_in_content(self):
        # ToolRegistry.execute() catches tool exceptions and returns them as error
        # strings rather than raising — so the protocol response is success=True
        # with the error message in contentItems.  Codex reads the text to determine
        # what happened.
        registry = ToolRegistry()
        registry.register(_ErrorTool())
        client, written = _make_client(registry=registry)

        client._handle_server_request(req_id=3, method="item/tool/call", _params={
            "tool": "boom",
            "arguments": {},
        })

        resp = json.loads(written[-1])
        # Registry catches exceptions and returns error strings; success=True because
        # execution completed — the error content tells Codex what went wrong.
        assert resp["result"]["success"] is True
        assert "exploded" in resp["result"]["contentItems"][0]["text"]

    def test_no_registry_returns_failure(self):
        client, written = _make_client(registry=None)

        client._handle_server_request(req_id=4, method="item/tool/call", _params={
            "tool": "echo",
            "arguments": {},
        })

        resp = json.loads(written[-1])
        assert resp["result"]["success"] is False

    def test_response_shape_matches_codex_protocol(self):
        """Verify the response has the exact fields Codex expects."""
        registry = ToolRegistry()
        registry.register(_EchoTool())
        client, written = _make_client(registry=registry)

        client._handle_server_request(req_id=5, method="item/tool/call", _params={
            "tool": "echo",
            "arguments": {"text": "x"},
        })

        resp = json.loads(written[-1])
        # Codex protocol: {"jsonrpc": "2.0", "id": ..., "result": {"contentItems": [...], "success": bool}}
        assert resp["jsonrpc"] == "2.0"
        assert "result" in resp
        assert "contentItems" in resp["result"]
        assert "success" in resp["result"]

    def test_mcp_tool_dispatched_via_name_map(self):
        """Codex sends the stripped name; name_map translates back to the mcp__ registry name."""
        registry = ToolRegistry()
        registry.register(_McpEchoTool())  # registered as "mcp__playwright__browser_close"
        client, written = _make_client(registry=registry)

        # Simulate what CodexProvider.chat() does: set the name_map after building specs.
        _, name_map = _build_dynamic_tool_specs(registry)
        client._tool_name_map = name_map

        # Codex calls the tool using the stripped name ("playwright__browser_close").
        client._handle_server_request(req_id=6, method="item/tool/call", _params={
            "tool": "playwright__browser_close",
            "arguments": {},
        })

        resp = json.loads(written[-1])
        assert resp["result"]["success"] is True
        assert "browser closed" in resp["result"]["contentItems"][0]["text"]


# ---------------------------------------------------------------------------
# Approval requests
# ---------------------------------------------------------------------------

class TestHandleApproval:
    def test_approve_maps_to_acceptForSession_when_available(self):
        approve = lambda method, params: "approve"
        client, written = _make_client(approve_command=approve)

        client._handle_server_request(
            req_id=10,
            method="item/commandExecution/requestApproval",
            _params={"available": ["accept", "acceptForSession", "decline", "cancel"]},
        )

        resp = json.loads(written[-1])
        # Prefer "acceptForSession" to persist approval across the session.
        assert resp["result"]["decision"] == "acceptForSession"

    def test_approve_falls_back_to_accept_when_acceptForSession_unavailable(self):
        approve = lambda method, params: "approve"
        client, written = _make_client(approve_command=approve)

        client._handle_server_request(
            req_id=11,
            method="item/commandExecution/requestApproval",
            _params={"available": ["accept", "decline"]},
        )

        resp = json.loads(written[-1])
        assert resp["result"]["decision"] == "accept"

    def test_deny_maps_to_decline_when_available(self):
        deny = lambda method, params: "deny"
        client, written = _make_client(approve_command=deny)

        client._handle_server_request(
            req_id=12,
            method="item/commandExecution/requestApproval",
            _params={"available": ["accept", "acceptForSession", "decline", "cancel"]},
        )

        resp = json.loads(written[-1])
        assert resp["result"]["decision"] == "decline"

    def test_no_callback_auto_denies(self):
        client, written = _make_client(approve_command=None)

        client._handle_server_request(
            req_id=13,
            method="item/commandExecution/requestApproval",
            _params={"available": ["accept", "decline"]},
        )

        resp = json.loads(written[-1])
        assert resp["result"]["decision"] == "decline"

    def test_file_approval_uses_decision_format(self):
        approve = lambda method, params: "approve"
        client, written = _make_client(approve_command=approve)

        client._handle_server_request(
            req_id=14,
            method="item/fileChange/requestApproval",
            _params={"available": ["accept", "acceptForSession", "decline"]},
        )

        resp = json.loads(written[-1])
        assert "decision" in resp["result"]

    def test_permissions_approval_returns_permissions_and_scope(self):
        approve = lambda method, params: "approve"
        client, written = _make_client(approve_command=approve)

        client._handle_server_request(
            req_id=15,
            method="item/permissions/requestApproval",
            _params={"permissions": {"network": True}},
        )

        resp = json.loads(written[-1])
        result = resp["result"]
        # Different response shape: {permissions, scope} not {decision}.
        assert "permissions" in result
        assert result["scope"] == "session"

    def test_permissions_denial_returns_empty_permissions_turn_scope(self):
        deny = lambda method, params: "deny"
        client, written = _make_client(approve_command=deny)

        client._handle_server_request(
            req_id=16,
            method="item/permissions/requestApproval",
            _params={"permissions": {"network": True}},
        )

        resp = json.loads(written[-1])
        result = resp["result"]
        assert result["permissions"] == {}
        assert result["scope"] == "turn"

    def test_unknown_method_sends_no_response(self):
        """Unknown server-request methods are silently ignored."""
        client, written = _make_client()
        before = len(written)

        client._handle_server_request(
            req_id=99,
            method="item/unknown/serverRequest",
            _params={},
        )

        assert len(written) == before, "no response expected for unknown methods"

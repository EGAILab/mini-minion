"""Tests for mcp/schema.py — naming, normalization, and redaction utilities."""
import sys

import pytest

from mini_minion.mcp.schema import (
    mcp_tool_name,
    normalize_schema_for_provider,
    normalize_windows_command,
    redact_secret_text,
    sanitize_tool_segment,
)


# ---------------------------------------------------------------------------
# sanitize_tool_segment
# ---------------------------------------------------------------------------

class TestSanitizeToolSegment:
    def test_alphanumeric_preserved(self):
        assert sanitize_tool_segment("hello123") == "hello123"

    def test_hyphen_preserved(self):
        assert sanitize_tool_segment("my-tool") == "my-tool"

    def test_underscore_preserved(self):
        assert sanitize_tool_segment("my_tool") == "my_tool"

    def test_dots_replaced(self):
        assert sanitize_tool_segment("my.server") == "my_server"

    def test_spaces_replaced(self):
        assert sanitize_tool_segment("my server") == "my_server"

    def test_leading_underscore_gets_prefix(self):
        # Provider names must start with alphanumeric
        result = sanitize_tool_segment("_bad")
        assert result[0].isalnum()

    def test_leading_hyphen_gets_prefix(self):
        result = sanitize_tool_segment("-bad")
        assert result[0].isalnum()

    def test_empty_string_returns_unknown(self):
        assert sanitize_tool_segment("") == "unknown"

    def test_mixed_special_chars(self):
        result = sanitize_tool_segment("a/b:c")
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for c in result)


# ---------------------------------------------------------------------------
# mcp_tool_name
# ---------------------------------------------------------------------------

class TestMcpToolName:
    def test_simple_name(self):
        assert mcp_tool_name("context7", "search") == "mcp__context7__search"

    def test_server_with_dot(self):
        # Dot in server name becomes underscore; hyphen in tool name is preserved
        # (hyphens are valid provider tool name characters per OpenAI spec)
        name = mcp_tool_name("my.server", "list-files")
        assert name == "mcp__my_server__list-files"

    def test_double_underscore_separator(self):
        # The separator makes it visually obvious this is an MCP tool
        name = mcp_tool_name("srv", "tool")
        assert name.startswith("mcp__")
        assert "__" in name[4:]  # after "mcp_"

    def test_format_structure(self):
        name = mcp_tool_name("server", "tool")
        parts = name.split("__")
        assert parts[0] == "mcp"
        assert parts[1] == "server"
        assert parts[2] == "tool"


# ---------------------------------------------------------------------------
# normalize_schema_for_provider
# ---------------------------------------------------------------------------

class TestNormalizeSchemaForProvider:
    def test_plain_object_schema_unchanged(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        result = normalize_schema_for_provider(schema)
        assert result["type"] == "object"

    def test_non_dict_returns_fallback(self):
        result = normalize_schema_for_provider(None)
        assert result == {"type": "object", "properties": {}}

    def test_type_list_with_null_normalized(self):
        schema = {"type": ["string", "null"]}
        result = normalize_schema_for_provider(schema)
        assert result["type"] == "string"
        assert result["nullable"] is True

    def test_type_list_without_null_unchanged(self):
        # Multiple non-null types — leave as-is (rare, can't simplify)
        schema = {"type": ["string", "integer"]}
        result = normalize_schema_for_provider(schema)
        # Both non-null types remain; no nullable added
        assert "nullable" not in result

    def test_anyof_with_null_normalized(self):
        schema = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        result = normalize_schema_for_provider(schema)
        assert result.get("type") == "string"
        assert result.get("nullable") is True
        assert "anyOf" not in result

    def test_oneof_with_null_normalized(self):
        schema = {"oneOf": [{"type": "integer"}, {"type": "null"}]}
        result = normalize_schema_for_provider(schema)
        assert result.get("type") == "integer"
        assert result.get("nullable") is True
        assert "oneOf" not in result

    def test_nested_properties_normalized(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
            },
        }
        result = normalize_schema_for_provider(schema)
        assert result["properties"]["name"]["type"] == "string"
        assert result["properties"]["name"]["nullable"] is True

    def test_pure_null_type_removes_type(self):
        schema = {"type": ["null"]}
        result = normalize_schema_for_provider(schema)
        assert "type" not in result


# ---------------------------------------------------------------------------
# redact_secret_text
# ---------------------------------------------------------------------------

class TestRedactSecretText:
    def test_empty_string(self):
        assert redact_secret_text("") == ""

    def test_token_redacted(self):
        # "authorization" key followed by colon: "Authorization: Bearer" → [REDACTED]
        # The redactor matches "key: value" so "Bearer" is the redacted value here
        text = "authorization: abc123xyz"
        result = redact_secret_text(text)
        assert "abc123xyz" not in result
        assert "[REDACTED]" in result

    def test_api_key_equals_redacted(self):
        text = "api_key=sk-1234567890abcdef"
        result = redact_secret_text(text)
        assert "sk-1234567890" not in result
        assert "[REDACTED]" in result

    def test_password_colon_redacted(self):
        text = "password: supersecret"
        result = redact_secret_text(text)
        assert "supersecret" not in result

    def test_case_insensitive(self):
        text = "TOKEN=abc"
        result = redact_secret_text(text)
        assert "abc" not in result

    def test_non_secret_key_not_redacted(self):
        text = "model=gpt-4"
        result = redact_secret_text(text)
        assert "gpt-4" in result


# ---------------------------------------------------------------------------
# normalize_windows_command
# ---------------------------------------------------------------------------

class TestNormalizeWindowsCommand:
    def test_non_windows_passthrough(self):
        if sys.platform == "win32":
            pytest.skip("Windows-specific test")
        cmd, args = normalize_windows_command("npx", ["-y", "some-pkg"])
        assert cmd == "npx"
        assert args == ["-y", "some-pkg"]

    def test_windows_npx_wrapped(self):
        if sys.platform != "win32":
            pytest.skip("Windows-specific test")
        cmd, args = normalize_windows_command("npx", ["-y", "some-pkg"])
        assert cmd == "cmd.exe"
        assert args[:3] == ["/d", "/c", "npx"]
        assert "-y" in args
        assert "some-pkg" in args

    def test_windows_npm_wrapped(self):
        if sys.platform != "win32":
            pytest.skip("Windows-specific test")
        cmd, args = normalize_windows_command("npm", ["install"])
        assert cmd == "cmd.exe"

    def test_windows_cmd_extension_wrapped(self):
        if sys.platform != "win32":
            pytest.skip("Windows-specific test")
        cmd, args = normalize_windows_command("my-script.cmd", ["arg1"])
        assert cmd == "cmd.exe"
        assert "my-script.cmd" in args

    def test_windows_python_not_wrapped(self):
        if sys.platform != "win32":
            pytest.skip("Windows-specific test")
        # python.exe is already an .exe — no wrapping needed
        cmd, args = normalize_windows_command("python", ["script.py"])
        assert cmd == "python"
        assert args == ["script.py"]

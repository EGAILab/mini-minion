"""Tests for MCP configuration parsing and validation in config.py."""
import json
import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helper to parse an MCP config dict through the full _validate + _resolve pipeline.
# We can't import config directly (it runs at import time against the real config.json),
# so we test the helpers in isolation.
# ---------------------------------------------------------------------------

from mini_minion.config import (
    McpConfig,
    McpServerConfig,
    _DANGEROUS_ENV_KEYS,
    _SERVER_NAME_RE,
    _TRANSPORT_ALIASES,
    _VALID_TRANSPORTS,
    _expand_env_vars,
    _resolve_mcp,
)


# ---------------------------------------------------------------------------
# Constants / regex
# ---------------------------------------------------------------------------

class TestServerNameRegex:
    def test_valid_simple(self):
        assert _SERVER_NAME_RE.match("context7")

    def test_valid_with_hyphen(self):
        assert _SERVER_NAME_RE.match("my-server")

    def test_valid_with_underscore(self):
        assert _SERVER_NAME_RE.match("my_server")

    def test_invalid_starts_with_underscore(self):
        assert not _SERVER_NAME_RE.match("_bad")

    def test_invalid_starts_with_hyphen(self):
        assert not _SERVER_NAME_RE.match("-bad")

    def test_invalid_too_long(self):
        # Max 63 chars after the first = 64 total
        assert not _SERVER_NAME_RE.match("a" * 65)

    def test_valid_max_length(self):
        assert _SERVER_NAME_RE.match("a" * 64)


class TestTransportAliases:
    def test_http_alias(self):
        assert _TRANSPORT_ALIASES["http"] == "streamableHttp"

    def test_streamable_http_alias(self):
        assert _TRANSPORT_ALIASES["streamable-http"] == "streamableHttp"


class TestDangerousEnvKeys:
    def test_node_options_dangerous(self):
        assert "NODE_OPTIONS" in _DANGEROUS_ENV_KEYS

    def test_pythonpath_dangerous(self):
        assert "PYTHONPATH" in _DANGEROUS_ENV_KEYS

    def test_ld_preload_dangerous(self):
        assert "LD_PRELOAD" in _DANGEROUS_ENV_KEYS


# ---------------------------------------------------------------------------
# _expand_env_vars
# ---------------------------------------------------------------------------

class TestExpandEnvVars:
    def test_no_vars(self):
        assert _expand_env_vars("hello world") == "hello world"

    def test_existing_var(self, monkeypatch):
        monkeypatch.setenv("TEST_MY_TOKEN", "secret123")
        result = _expand_env_vars("Bearer ${TEST_MY_TOKEN}")
        assert result == "Bearer secret123"

    def test_missing_var_left_as_is(self):
        # Unset vars are left as ${VAR} rather than becoming empty strings
        key = "DEFINITELY_NOT_SET_XYZ_ABC_123"
        os.environ.pop(key, None)
        result = _expand_env_vars(f"${{{key}}}")
        assert result == f"${{{key}}}"

    def test_dollar_without_braces_not_expanded(self):
        # $VAR without braces is not expanded — avoids accidents in URLs/commands
        result = _expand_env_vars("$HOME/path")
        assert result == "$HOME/path"


# ---------------------------------------------------------------------------
# _resolve_mcp — tests using monkeypatching of the _raw global
# ---------------------------------------------------------------------------

class TestResolveMcp:
    """These tests monkeypatch mini_minion.config._raw to inject fake config."""

    def _patch_raw(self, monkeypatch, mcp_section: dict):
        import mini_minion.config as cfg_module
        fake_raw = dict(cfg_module._raw)  # shallow copy of real _raw
        fake_raw["mcp"] = mcp_section
        monkeypatch.setattr(cfg_module, "_raw", fake_raw)

    def test_absent_mcp_section_returns_empty(self, monkeypatch):
        import mini_minion.config as cfg_module
        fake_raw = {k: v for k, v in cfg_module._raw.items() if k != "mcp"}
        monkeypatch.setattr(cfg_module, "_raw", fake_raw)
        result = _resolve_mcp()
        assert isinstance(result, McpConfig)
        assert result.servers == ()

    def test_valid_stdio_server(self, monkeypatch):
        self._patch_raw(monkeypatch, {
            "servers": {
                "myserver": {
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "some-pkg"],
                }
            }
        })
        result = _resolve_mcp()
        assert len(result.servers) == 1
        s = result.servers[0]
        assert s.name == "myserver"
        assert s.transport == "stdio"
        assert s.command == "npx"
        assert s.args == ("-y", "some-pkg")

    def test_valid_sse_server(self, monkeypatch):
        self._patch_raw(monkeypatch, {
            "servers": {
                "sseserver": {
                    "transport": "sse",
                    "url": "http://localhost:8080/sse",
                }
            }
        })
        result = _resolve_mcp()
        assert result.servers[0].transport == "sse"
        assert result.servers[0].url == "http://localhost:8080/sse"

    def test_http_alias_canonicalized(self, monkeypatch):
        """'http' transport alias must be normalized to 'streamableHttp'."""
        self._patch_raw(monkeypatch, {
            "servers": {
                "httpsrv": {
                    "transport": "http",
                    "url": "http://localhost:9000",
                }
            }
        })
        result = _resolve_mcp()
        assert result.servers[0].transport == "streamableHttp"

    def test_streamable_http_alias_canonicalized(self, monkeypatch):
        self._patch_raw(monkeypatch, {
            "servers": {
                "httpsrv": {
                    "transport": "streamable-http",
                    "url": "http://localhost:9000",
                }
            }
        })
        result = _resolve_mcp()
        assert result.servers[0].transport == "streamableHttp"

    def test_tool_timeout_clamped_low(self, monkeypatch):
        """tool_timeout below 5 must be clamped to 5."""
        self._patch_raw(monkeypatch, {
            "servers": {
                "fast": {
                    "transport": "stdio",
                    "command": "echo",
                    "tool_timeout": 1,
                }
            }
        })
        result = _resolve_mcp()
        assert result.servers[0].tool_timeout == 5

    def test_tool_timeout_clamped_high(self, monkeypatch):
        """tool_timeout above 600 must be clamped to 600."""
        self._patch_raw(monkeypatch, {
            "servers": {
                "slow": {
                    "transport": "stdio",
                    "command": "echo",
                    "tool_timeout": 9999,
                }
            }
        })
        result = _resolve_mcp()
        assert result.servers[0].tool_timeout == 600

    def test_tool_timeout_in_range_preserved(self, monkeypatch):
        self._patch_raw(monkeypatch, {
            "servers": {
                "ok": {
                    "transport": "stdio",
                    "command": "echo",
                    "tool_timeout": 45,
                }
            }
        })
        result = _resolve_mcp()
        assert result.servers[0].tool_timeout == 45

    def test_env_var_expansion(self, monkeypatch):
        monkeypatch.setenv("MY_MCP_TOKEN", "tok-xyz")
        self._patch_raw(monkeypatch, {
            "servers": {
                "networked": {
                    "transport": "sse",
                    "url": "http://api.example.com/sse",
                    "headers": {"Authorization": "Bearer ${MY_MCP_TOKEN}"},
                }
            }
        })
        result = _resolve_mcp()
        assert result.servers[0].headers["Authorization"] == "Bearer tok-xyz"

    def test_enabled_tools_default_is_star(self, monkeypatch):
        self._patch_raw(monkeypatch, {
            "servers": {
                "alltools": {
                    "transport": "stdio",
                    "command": "echo",
                }
            }
        })
        result = _resolve_mcp()
        assert result.servers[0].enabled_tools == ("*",)

    def test_enabled_tools_explicit(self, monkeypatch):
        self._patch_raw(monkeypatch, {
            "servers": {
                "limited": {
                    "transport": "stdio",
                    "command": "echo",
                    "enabled_tools": ["search", "fetch"],
                }
            }
        })
        result = _resolve_mcp()
        assert result.servers[0].enabled_tools == ("search", "fetch")

    def test_multiple_servers(self, monkeypatch):
        self._patch_raw(monkeypatch, {
            "servers": {
                "a": {"transport": "stdio", "command": "cmd_a"},
                "b": {"transport": "stdio", "command": "cmd_b"},
            }
        })
        result = _resolve_mcp()
        assert len(result.servers) == 2
        names = {s.name for s in result.servers}
        assert names == {"a", "b"}


# ---------------------------------------------------------------------------
# _validate() MCP section — tested through the config module's _validate helper
# ---------------------------------------------------------------------------

class TestMcpValidation:
    """Test that _validate() catches bad MCP configs."""

    def _validate_mcp(self, mcp_section: dict):
        """Run _validate with only the mcp section patched in."""
        from mini_minion.config import _validate, _raw
        raw = dict(_raw)
        raw["mcp"] = mcp_section
        return _validate(raw)

    def test_valid_stdio_no_issues(self):
        issues = self._validate_mcp({
            "servers": {
                "good": {"transport": "stdio", "command": "npx"}
            }
        })
        mcp_issues = [i for i in issues if i.path.startswith("mcp")]
        assert mcp_issues == []

    def test_invalid_server_name_rejected(self):
        issues = self._validate_mcp({
            "servers": {
                "_bad-name": {"transport": "stdio", "command": "cmd"}
            }
        })
        paths = [i.path for i in issues]
        assert any("mcp.servers._bad-name" in p for p in paths)

    def test_stdio_missing_command_rejected(self):
        issues = self._validate_mcp({
            "servers": {
                "nocmd": {"transport": "stdio"}
            }
        })
        paths = [i.path for i in issues]
        assert any("command" in p for p in paths)

    def test_sse_missing_url_rejected(self):
        issues = self._validate_mcp({
            "servers": {
                "nourl": {"transport": "sse"}
            }
        })
        paths = [i.path for i in issues]
        assert any("url" in p for p in paths)

    def test_dangerous_env_key_rejected(self):
        issues = self._validate_mcp({
            "servers": {
                "evil": {
                    "transport": "stdio",
                    "command": "cmd",
                    "env": {"NODE_OPTIONS": "--inspect"},
                }
            }
        })
        paths = [i.path for i in issues]
        assert any("NODE_OPTIONS" in p for p in paths)

    def test_invalid_transport_rejected(self):
        issues = self._validate_mcp({
            "servers": {
                "badtransport": {"transport": "ftp", "command": "x"}
            }
        })
        paths = [i.path for i in issues]
        assert any("transport" in p for p in paths)

    def test_env_non_string_value_rejected(self):
        issues = self._validate_mcp({
            "servers": {
                "nonstr": {
                    "transport": "stdio",
                    "command": "cmd",
                    "env": {"MY_VAR": 123},
                }
            }
        })
        paths = [i.path for i in issues]
        assert any("MY_VAR" in p for p in paths)

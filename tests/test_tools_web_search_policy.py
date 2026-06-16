"""Tests for WebSearchTool SSRF policy integration.

The SSRF check blocks queries that contain markers like "169.254.169.254"
(AWS metadata), "localhost", etc. — hardened against prompt-injection attacks
that try to use the search tool to surface internal network addresses.
"""

from unittest.mock import MagicMock

import minion_assist.tools.web_search as ws_module
from minion_assist.tools.policy import PermissionPolicy
from minion_assist.tools.web_search import WebSearchTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _policy_with_ssrf_markers(*markers: str) -> PermissionPolicy:
    """Build a PermissionPolicy whose ssrf_markers include the given values."""
    policy = PermissionPolicy.default()
    policy.ssrf_markers = frozenset(markers)
    return policy


def _patch_ddgs(monkeypatch, results=None):
    """Make WebSearchTool believe duckduckgo-search is installed and return results.

    DDGS is used as a context manager (with DDGS(...) as ddgs:), so the mock
    must implement __enter__ and __exit__ in addition to .text().
    """
    monkeypatch.setattr(ws_module, "_DDGS_AVAILABLE", True)
    fake_results = results or [{"title": "T", "href": "https://example.com", "body": "B"}]

    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=False)
    mock_instance.text = MagicMock(return_value=iter(fake_results))

    monkeypatch.setattr(ws_module, "DDGS", lambda **kw: mock_instance)


# ---------------------------------------------------------------------------
# Tests — SSRF blocking
# ---------------------------------------------------------------------------

def test_ssrf_marker_blocked(monkeypatch):
    """A query containing a known SSRF marker is rejected before the network call."""
    _patch_ddgs(monkeypatch)
    policy = _policy_with_ssrf_markers("169.254.169.254")
    tool = WebSearchTool(policy=policy)
    result = tool.execute(query="info 169.254.169.254 token")
    assert "Error" in result
    assert "169.254.169.254" in result or "blocked" in result.lower()


def test_ssrf_localhost_blocked(monkeypatch):
    """Queries containing 'localhost' are rejected when it's in ssrf_markers."""
    _patch_ddgs(monkeypatch)
    policy = _policy_with_ssrf_markers("localhost")
    tool = WebSearchTool(policy=policy)
    result = tool.execute(query="localhost admin panel")
    assert "Error" in result or "blocked" in result.lower()


def test_ssrf_safe_query_passes(monkeypatch):
    """A normal query without any SSRF marker proceeds normally."""
    _patch_ddgs(monkeypatch)
    policy = _policy_with_ssrf_markers("169.254.169.254", "localhost")
    tool = WebSearchTool(policy=policy)
    result = tool.execute(query="Python asyncio tutorial")
    assert "Error" not in result
    assert "example.com" in result


def test_no_policy_passes(monkeypatch):
    """Without a policy, all queries proceed (no SSRF check)."""
    _patch_ddgs(monkeypatch)
    tool = WebSearchTool()
    result = tool.execute(query="some search query")
    assert "Error" not in result


def test_ssrf_check_is_case_sensitive(monkeypatch):
    """SSRF marker matching respects case — '169.254.169.254' won't match 'LOCALHOST'."""
    _patch_ddgs(monkeypatch)
    policy = _policy_with_ssrf_markers("localhost")
    tool = WebSearchTool(policy=policy)
    # "LOCALHOST" should NOT be blocked because the check is case-sensitive.
    result = tool.execute(query="LOCALHOST admin")
    # If not blocked, the fake search result should appear.
    assert "example.com" in result or "Error" not in result


def test_ssrf_empty_markers_passes(monkeypatch):
    """An empty ssrf_markers list means no SSRF check — all queries pass."""
    _patch_ddgs(monkeypatch)
    policy = PermissionPolicy.default()
    policy.ssrf_markers = frozenset()
    tool = WebSearchTool(policy=policy)
    result = tool.execute(query="169.254.169.254 metadata")
    # No markers → not blocked.
    assert "example.com" in result

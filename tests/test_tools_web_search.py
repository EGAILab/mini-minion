"""Tests for WebSearchTool (DuckDuckGo web search)."""

import pytest
from unittest.mock import MagicMock, patch

import mini_minion.tools.web_search as ws_module
from mini_minion.tools.web_search import (
    WebSearchTool,
    _format_results,
    _DEFAULT_MAX_RESULTS,
    _MAX_RESULTS_HARD_CAP,
    _OUTPUT_MAX_CHARS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_results(n: int = 3) -> list[dict]:
    """Return n fake DuckDuckGo result dicts."""
    return [
        {
            "title": f"Result {i}",
            "href": f"https://example.com/page/{i}",
            "body": f"Snippet for result {i}.",
        }
        for i in range(1, n + 1)
    ]


def _mock_ddgs(results: list[dict]):
    """Context-manager mock that returns ``results`` from .text()."""
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    mock.text = MagicMock(return_value=iter(results))
    return mock


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_schema_name():
    assert WebSearchTool().schema.name == "web_search"


def test_schema_is_read_only():
    """web_search must be read-only so the runner may batch it concurrently."""
    assert WebSearchTool().schema.is_read_only is True


def test_schema_requires_query():
    params = WebSearchTool().schema.parameters
    assert "query" in params["required"]


def test_schema_max_results_bounds():
    props = WebSearchTool().schema.parameters["properties"]
    assert props["max_results"]["minimum"] == 1
    assert props["max_results"]["maximum"] == _MAX_RESULTS_HARD_CAP


# ---------------------------------------------------------------------------
# execute() — normal path
# ---------------------------------------------------------------------------

def test_execute_returns_formatted_string(monkeypatch):
    """execute() must return a non-empty formatted string on success."""
    mock_ddgs = _mock_ddgs(_make_results(3))
    monkeypatch.setattr(ws_module, "_DDGS_AVAILABLE", True)
    monkeypatch.setattr(ws_module, "DDGS", lambda **kw: mock_ddgs)

    result = WebSearchTool().execute(query="python tutorial")

    assert "python tutorial" in result
    assert "Result 1" in result
    assert "https://example.com/page/1" in result
    assert "Snippet for result 1." in result


def test_execute_numbers_results(monkeypatch):
    """Results must be numbered [1], [2], [3]..."""
    mock_ddgs = _mock_ddgs(_make_results(3))
    monkeypatch.setattr(ws_module, "_DDGS_AVAILABLE", True)
    monkeypatch.setattr(ws_module, "DDGS", lambda **kw: mock_ddgs)

    result = WebSearchTool().execute(query="test")

    assert "[1]" in result
    assert "[2]" in result
    assert "[3]" in result


def test_execute_passes_max_results_to_ddgs(monkeypatch):
    """The max_results kwarg must be forwarded to DDGS.text()."""
    mock_ddgs = _mock_ddgs(_make_results(2))
    monkeypatch.setattr(ws_module, "_DDGS_AVAILABLE", True)
    monkeypatch.setattr(ws_module, "DDGS", lambda **kw: mock_ddgs)

    WebSearchTool().execute(query="test", max_results=2)

    call_kwargs = mock_ddgs.text.call_args.kwargs
    assert call_kwargs["max_results"] == 2


def test_execute_passes_region_to_ddgs(monkeypatch):
    """The region kwarg must be forwarded to DDGS.text()."""
    mock_ddgs = _mock_ddgs(_make_results(1))
    monkeypatch.setattr(ws_module, "_DDGS_AVAILABLE", True)
    monkeypatch.setattr(ws_module, "DDGS", lambda **kw: mock_ddgs)

    WebSearchTool().execute(query="news", region="uk-en")

    call_kwargs = mock_ddgs.text.call_args.kwargs
    assert call_kwargs["region"] == "uk-en"


def test_execute_defaults_region_to_worldwide(monkeypatch):
    """When region is omitted, DDGS.text() should receive 'wt-wt'."""
    mock_ddgs = _mock_ddgs(_make_results(1))
    monkeypatch.setattr(ws_module, "_DDGS_AVAILABLE", True)
    monkeypatch.setattr(ws_module, "DDGS", lambda **kw: mock_ddgs)

    WebSearchTool().execute(query="test")

    call_kwargs = mock_ddgs.text.call_args.kwargs
    assert call_kwargs["region"] == "wt-wt"


def test_execute_clamps_max_results_above_cap(monkeypatch):
    """max_results > hard cap must be silently clamped to _MAX_RESULTS_HARD_CAP."""
    mock_ddgs = _mock_ddgs(_make_results(1))
    monkeypatch.setattr(ws_module, "_DDGS_AVAILABLE", True)
    monkeypatch.setattr(ws_module, "DDGS", lambda **kw: mock_ddgs)

    WebSearchTool().execute(query="test", max_results=99)

    call_kwargs = mock_ddgs.text.call_args.kwargs
    assert call_kwargs["max_results"] == _MAX_RESULTS_HARD_CAP


def test_execute_clamps_max_results_below_one(monkeypatch):
    """max_results < 1 must be clamped to 1."""
    mock_ddgs = _mock_ddgs(_make_results(1))
    monkeypatch.setattr(ws_module, "_DDGS_AVAILABLE", True)
    monkeypatch.setattr(ws_module, "DDGS", lambda **kw: mock_ddgs)

    WebSearchTool().execute(query="test", max_results=0)

    call_kwargs = mock_ddgs.text.call_args.kwargs
    assert call_kwargs["max_results"] == 1


# ---------------------------------------------------------------------------
# execute() — edge cases
# ---------------------------------------------------------------------------

def test_execute_no_results_returns_not_found_message(monkeypatch):
    """Empty result list must produce a 'no results' message, not an error."""
    mock_ddgs = _mock_ddgs([])
    monkeypatch.setattr(ws_module, "_DDGS_AVAILABLE", True)
    monkeypatch.setattr(ws_module, "DDGS", lambda **kw: mock_ddgs)

    result = WebSearchTool().execute(query="xyzzy123notfound")

    assert "No results" in result
    assert "xyzzy123notfound" in result


def test_execute_network_error_returns_error_string(monkeypatch):
    """Network exceptions must be caught and returned as a descriptive string."""
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    mock.text = MagicMock(side_effect=ConnectionError("connection refused"))

    monkeypatch.setattr(ws_module, "_DDGS_AVAILABLE", True)
    monkeypatch.setattr(ws_module, "DDGS", lambda **kw: mock)

    result = WebSearchTool().execute(query="test")

    assert "Search failed" in result
    assert "connection refused" in result


def test_execute_missing_package_returns_install_hint(monkeypatch):
    """When duckduckgo-search is not installed, return a clear install hint."""
    monkeypatch.setattr(ws_module, "_DDGS_AVAILABLE", False)

    result = WebSearchTool().execute(query="anything")

    assert "duckduckgo-search" in result
    assert "uv add" in result


def test_execute_empty_query_returns_error(monkeypatch):
    """An empty query string must return an error without calling DDGS."""
    monkeypatch.setattr(ws_module, "_DDGS_AVAILABLE", True)
    fake_ddgs = MagicMock()
    monkeypatch.setattr(ws_module, "DDGS", lambda **kw: fake_ddgs)

    result = WebSearchTool().execute(query="   ")

    assert "Error" in result
    fake_ddgs.__enter__.assert_not_called()


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------

def test_output_truncated_when_too_long():
    """_format_results must truncate output exceeding _OUTPUT_MAX_CHARS."""
    # Build a result with a very long snippet to force truncation.
    huge_results = [{"title": "Title", "href": "https://x.com", "body": "x" * _OUTPUT_MAX_CHARS}]
    output = _format_results("query", huge_results)

    assert len(output) <= _OUTPUT_MAX_CHARS + len("\n[... output truncated]")
    assert "[... output truncated]" in output


def test_short_output_not_truncated():
    """Short output must not contain the truncation notice."""
    output = _format_results("query", _make_results(2))
    assert "[... output truncated]" not in output


# ---------------------------------------------------------------------------
# Constructor behaviour
# ---------------------------------------------------------------------------

def test_constructor_max_results_clamped():
    """max_results > hard cap in the constructor must be clamped."""
    tool = WebSearchTool(max_results=100)
    assert tool._max_results == _MAX_RESULTS_HARD_CAP


def test_constructor_default_max_results():
    assert WebSearchTool()._max_results == _DEFAULT_MAX_RESULTS


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_web_search_registered_in_default_registry():
    """WebSearchTool must appear in the default registry."""
    from mini_minion.tools import default_registry
    registry = default_registry()
    assert "web_search" in {t.schema.name for t in registry._tools.values()}


def test_web_search_is_read_only_in_registry():
    """The registry must report web_search as read-only."""
    from mini_minion.tools import default_registry
    registry = default_registry()
    assert registry.is_read_only("web_search") is True

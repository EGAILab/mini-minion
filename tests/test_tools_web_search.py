"""Tests for WebSearchTool (DuckDuckGo web search).

WHY THESE TESTS NEVER HIT THE REAL INTERNET
--------------------------------------------
All tests use pytest's `monkeypatch` fixture to replace the real `DDGS` class
(and the `_DDGS_AVAILABLE` flag) with fast, deterministic fakes.  This means:

  - Tests run in milliseconds, not seconds.
  - Tests pass even when DuckDuckGo is down or rate-limiting.
  - Tests are deterministic — no flaky failures from real network results.
  - Tests can simulate rare conditions (no results, network errors, missing
    package) that are hard to trigger against the live service.

HOW MONKEYPATCHING WORKS HERE
------------------------------
`execute()` reads `_DDGS_AVAILABLE` and calls `DDGS(...)` from the module's
own namespace (`web_search._DDGS_AVAILABLE`, `web_search.DDGS`).  We patch at
the *module* level — not the import level — so every code path in `execute()`
sees the fake regardless of how the import was written.

  monkeypatch.setattr(ws_module, "_DDGS_AVAILABLE", True)   # pretend installed
  monkeypatch.setattr(ws_module, "DDGS", lambda **kw: mock) # inject fake client
"""

import pytest
from unittest.mock import MagicMock, patch

import minion_assistant.tools.web_search as ws_module
from minion_assistant.tools.web_search import (
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
    """Return n fake DuckDuckGo result dicts.

    The real duckduckgo-search library returns a list of dicts with these keys:
      - "title" : page title
      - "href"  : full URL
      - "body"  : snippet / description text

    This helper produces the same shape so tests don't need to repeat the dict
    structure every time.
    """
    return [
        {
            "title": f"Result {i}",
            "href": f"https://example.com/page/{i}",
            "body": f"Snippet for result {i}.",
        }
        for i in range(1, n + 1)
    ]


def _mock_ddgs(results: list[dict]):
    """Build a fake DDGS client that behaves like the real one.

    The real DDGS is used as a context manager in execute():

        with DDGS(timeout=...) as ddgs:
            raw_results = list(ddgs.text(...))

    To fake this we need to mock three things:
      1. __enter__ — called when `with DDGS(...)` starts; must return the mock
         itself so `ddgs.text(...)` is callable.
      2. __exit__  — called when the `with` block ends; returning False means
         "don't suppress exceptions" (the normal behaviour).
      3. text      — the method that returns search result dicts; we wrap our
         fixed list in iter() to match the real generator interface.
    """
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)   # `as ddgs` → the mock itself
    mock.__exit__ = MagicMock(return_value=False)   # don't suppress exceptions
    mock.text = MagicMock(return_value=iter(results))
    return mock


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# Schema tests verify the *contract* exposed to the LLM: the tool name, which
# parameters are required, and what values the JSON Schema allows.  These tests
# catch regressions where a refactor accidentally renames the tool or changes a
# required field.

def test_schema_name():
    assert WebSearchTool().schema.name == "web_search"


def test_schema_is_read_only():
    """web_search must be read-only so the runner may batch it concurrently.

    `is_read_only=True` tells the TAO-loop runner (runner.py) that this tool
    never mutates files or memory, so multiple web_search calls in the same
    model response can be executed in parallel via ThreadPoolExecutor rather
    than one-at-a-time.
    """
    assert WebSearchTool().schema.is_read_only is True


def test_schema_requires_query():
    # "query" is the only field the LLM must always provide.
    params = WebSearchTool().schema.parameters
    assert "query" in params["required"]


def test_schema_max_results_bounds():
    # The JSON Schema bounds must match the hard-coded constants so the LLM
    # receives accurate documentation about what values are accepted.
    props = WebSearchTool().schema.parameters["properties"]
    assert props["max_results"]["minimum"] == 1
    assert props["max_results"]["maximum"] == _MAX_RESULTS_HARD_CAP


# ---------------------------------------------------------------------------
# execute() — normal path
# ---------------------------------------------------------------------------
# These tests verify what the agent sees when a search succeeds.  The fake DDGS
# always returns the results built by _make_results(), so assertions can be
# exact strings rather than regexes.

def test_execute_returns_formatted_string(monkeypatch):
    """execute() must return a non-empty formatted string on success."""
    mock_ddgs = _mock_ddgs(_make_results(3))
    monkeypatch.setattr(ws_module, "_DDGS_AVAILABLE", True)
    # lambda **kw: mock_ddgs — the constructor call `DDGS(timeout=...)` passes
    # keyword args we don't care about, so ** absorbs them and we return the
    # pre-built mock regardless.
    monkeypatch.setattr(ws_module, "DDGS", lambda **kw: mock_ddgs)

    result = WebSearchTool().execute(query="python tutorial")

    # The query itself should appear in the header line.
    assert "python tutorial" in result
    # Each result's title, URL, and snippet should be present.
    assert "Result 1" in result
    assert "https://example.com/page/1" in result
    assert "Snippet for result 1." in result


def test_execute_numbers_results(monkeypatch):
    """Results must be numbered [1], [2], [3]...

    Numbered results let the agent and the user easily refer to a specific
    result ("see result [2]") rather than quoting a full URL.
    """
    mock_ddgs = _mock_ddgs(_make_results(3))
    monkeypatch.setattr(ws_module, "_DDGS_AVAILABLE", True)
    monkeypatch.setattr(ws_module, "DDGS", lambda **kw: mock_ddgs)

    result = WebSearchTool().execute(query="test")

    assert "[1]" in result
    assert "[2]" in result
    assert "[3]" in result


def test_execute_passes_max_results_to_ddgs(monkeypatch):
    """The max_results kwarg must be forwarded to DDGS.text().

    We verify by inspecting `mock_ddgs.text.call_args.kwargs` — pytest's
    MagicMock records every call including its arguments, so we can assert
    exactly what execute() passed without executing any real code.
    """
    mock_ddgs = _mock_ddgs(_make_results(2))
    monkeypatch.setattr(ws_module, "_DDGS_AVAILABLE", True)
    monkeypatch.setattr(ws_module, "DDGS", lambda **kw: mock_ddgs)

    WebSearchTool().execute(query="test", max_results=2)

    # .call_args holds the most recent call; .kwargs is the keyword arguments dict.
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
    """When region is omitted, DDGS.text() should receive 'wt-wt'.

    'wt-wt' is DuckDuckGo's worldwide region code — it returns global results
    instead of localising to a specific country.  Defaulting to worldwide is
    the safest choice since we don't know the agent's location.
    """
    mock_ddgs = _mock_ddgs(_make_results(1))
    monkeypatch.setattr(ws_module, "_DDGS_AVAILABLE", True)
    monkeypatch.setattr(ws_module, "DDGS", lambda **kw: mock_ddgs)

    WebSearchTool().execute(query="test")

    call_kwargs = mock_ddgs.text.call_args.kwargs
    assert call_kwargs["region"] == "wt-wt"


def test_execute_clamps_max_results_above_cap(monkeypatch):
    """max_results > hard cap must be silently clamped to _MAX_RESULTS_HARD_CAP.

    DuckDuckGo becomes unreliable above ~10 results per request.  Rather than
    raising an error (which would crash the agent's tool call), we clamp
    silently so the agent still gets a useful answer.
    """
    mock_ddgs = _mock_ddgs(_make_results(1))
    monkeypatch.setattr(ws_module, "_DDGS_AVAILABLE", True)
    monkeypatch.setattr(ws_module, "DDGS", lambda **kw: mock_ddgs)

    WebSearchTool().execute(query="test", max_results=99)

    call_kwargs = mock_ddgs.text.call_args.kwargs
    assert call_kwargs["max_results"] == _MAX_RESULTS_HARD_CAP


def test_execute_clamps_max_results_below_one(monkeypatch):
    """max_results < 1 must be clamped to 1.

    Requesting 0 or negative results is nonsensical.  Clamping to 1 ensures
    we always request at least one result rather than sending an invalid API
    parameter to DuckDuckGo.
    """
    mock_ddgs = _mock_ddgs(_make_results(1))
    monkeypatch.setattr(ws_module, "_DDGS_AVAILABLE", True)
    monkeypatch.setattr(ws_module, "DDGS", lambda **kw: mock_ddgs)

    WebSearchTool().execute(query="test", max_results=0)

    call_kwargs = mock_ddgs.text.call_args.kwargs
    assert call_kwargs["max_results"] == 1


# ---------------------------------------------------------------------------
# execute() — edge cases
# ---------------------------------------------------------------------------
# These tests cover degraded or invalid inputs.  The key design rule is that
# execute() NEVER raises — it always returns a string the agent can read and
# react to, even when something goes wrong.

def test_execute_no_results_returns_not_found_message(monkeypatch):
    """Empty result list must produce a 'no results' message, not an error.

    DuckDuckGo can legitimately return zero results for obscure queries.  The
    agent should be told there were no results (and what query was tried) so it
    can rephrase its query rather than receiving a confusing empty string.
    """
    mock_ddgs = _mock_ddgs([])   # empty list → no search hits
    monkeypatch.setattr(ws_module, "_DDGS_AVAILABLE", True)
    monkeypatch.setattr(ws_module, "DDGS", lambda **kw: mock_ddgs)

    result = WebSearchTool().execute(query="xyzzy123notfound")

    assert "No results" in result
    assert "xyzzy123notfound" in result   # include the query so the agent knows what failed


def test_execute_network_error_returns_error_string(monkeypatch):
    """Network exceptions must be caught and returned as a descriptive string.

    The `except Exception` block in execute() catches all errors — network
    timeouts, ConnectionError, rate-limit exceptions, and DuckDuckGo HTML
    changes.  We simulate a ConnectionError here to verify the except clause
    fires and produces a readable message rather than crashing the TAO loop.
    """
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    # side_effect tells MagicMock to raise the exception when .text() is called,
    # instead of returning a value.
    mock.text = MagicMock(side_effect=ConnectionError("connection refused"))

    monkeypatch.setattr(ws_module, "_DDGS_AVAILABLE", True)
    monkeypatch.setattr(ws_module, "DDGS", lambda **kw: mock)

    result = WebSearchTool().execute(query="test")

    # The error string includes both the exception type and message so the
    # agent can decide whether to retry.
    assert "Search failed" in result
    assert "connection refused" in result


def test_execute_missing_package_returns_install_hint(monkeypatch):
    """When ddgs is not installed, return a clear install hint.

    Setting _DDGS_AVAILABLE=False simulates an ImportError at module load time.
    execute() checks this flag first and returns a human-readable message with
    the exact install command, rather than an AttributeError on `None` (which
    is what DDGS is set to when the import fails).
    """
    monkeypatch.setattr(ws_module, "_DDGS_AVAILABLE", False)

    result = WebSearchTool().execute(query="anything")

    assert "ddgs" in result
    assert "uv add" in result


def test_execute_empty_query_returns_error(monkeypatch):
    """An empty query string must return an error without calling DDGS.

    An all-whitespace query is not meaningful and would result in a confusing
    DuckDuckGo response.  We validate the query early and return immediately
    so the real DDGS client is never called (verified by asserting __enter__
    was never invoked on the fake context manager).
    """
    monkeypatch.setattr(ws_module, "_DDGS_AVAILABLE", True)
    fake_ddgs = MagicMock()
    monkeypatch.setattr(ws_module, "DDGS", lambda **kw: fake_ddgs)

    result = WebSearchTool().execute(query="   ")   # only whitespace

    assert "Error" in result
    # If DDGS was never entered we know the early-return guard worked.
    fake_ddgs.__enter__.assert_not_called()


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------
# The output cap (_OUTPUT_MAX_CHARS = 6 000) prevents execute() from flooding
# the model's context window.  These tests verify the truncation logic in
# _format_results() directly — no need to go through the full execute() path.

def test_output_truncated_when_too_long():
    """_format_results must truncate output exceeding _OUTPUT_MAX_CHARS.

    We craft a single result whose "body" snippet is already longer than the
    cap.  After formatting, the total output length should be at most
    _OUTPUT_MAX_CHARS + the length of the truncation notice string, and the
    notice "[... output truncated]" must appear at the end.
    """
    huge_results = [{"title": "Title", "href": "https://x.com", "body": "x" * _OUTPUT_MAX_CHARS}]
    output = _format_results("query", huge_results)

    assert len(output) <= _OUTPUT_MAX_CHARS + len("\n[... output truncated]")
    assert "[... output truncated]" in output


def test_short_output_not_truncated():
    """Short output must not contain the truncation notice.

    Two normal-sized results will be well under the 6 000-char cap, so no
    truncation notice should appear.  This confirms the truncation code is
    gated on the actual length rather than always appending the notice.
    """
    output = _format_results("query", _make_results(2))
    assert "[... output truncated]" not in output


# ---------------------------------------------------------------------------
# Constructor behaviour
# ---------------------------------------------------------------------------
# The constructor clamps max_results at construction time so a badly configured
# WebSearchTool(max_results=999) never silently passes 999 to DuckDuckGo later.

def test_constructor_max_results_clamped():
    """max_results > hard cap in the constructor must be clamped."""
    tool = WebSearchTool(max_results=100)
    assert tool._max_results == _MAX_RESULTS_HARD_CAP


def test_constructor_default_max_results():
    # Verify the default is the module constant (5), not a magic number.
    assert WebSearchTool()._max_results == _DEFAULT_MAX_RESULTS


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------
# These tests verify that the tool is wired into the application correctly —
# not just that the class works in isolation.  default_registry() is what
# minion.py calls at startup, so a missing registration would break the whole
# application even if all the unit tests above pass.

def test_web_search_registered_in_default_registry():
    """WebSearchTool must appear in the default registry.

    default_registry() builds the ToolRegistry that every agent uses.  If
    WebSearchTool is not registered here, the agent's system prompt won't list
    the tool and the LLM will never know it can search the web.
    """
    from minion_assistant.tools import default_registry
    registry = default_registry()
    assert "web_search" in {t.schema.name for t in registry._tools.values()}


def test_web_search_is_read_only_in_registry():
    """The registry must report web_search as read-only.

    ToolRegistry.is_read_only() is what runner.py calls to decide whether a
    batch of tool calls can run in parallel.  If this returns False for
    web_search, the runner will serialize search calls that could be concurrent.
    """
    from minion_assistant.tools import default_registry
    registry = default_registry()
    assert registry.is_read_only("web_search") is True

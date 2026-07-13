"""Tests for BrowserTool.

All Playwright calls are mocked — no real browser is launched. The module-level
globals (_playwright, _browser, _page) are reset before and after every test via
the autouse fixture so tests stay independent.

Test strategy
-------------
- For start() tests: patch sys.modules to inject a mock playwright package.
- For all other actions: set browser_module._page directly to a MagicMock,
  bypassing the start() flow entirely. This is simpler and tests the action
  logic in isolation.
"""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest

import minion_assist.tools.browser as browser_module
from minion_assist.tools.browser import BrowserTool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_browser_state():
    """Reset module-level browser globals before and after every test."""
    browser_module._playwright = None
    browser_module._browser = None
    browser_module._page = None
    yield
    browser_module._playwright = None
    browser_module._browser = None
    browser_module._page = None


@pytest.fixture()
def mock_playwright(monkeypatch):
    """Inject a mock playwright package into sys.modules.

    Returns a dict with keys 'pw', 'browser', 'page' pointing to MagicMocks
    that mirror the real Playwright object hierarchy.
    """
    mock_pw = MagicMock()
    mock_browser = MagicMock()
    mock_page = MagicMock()
    mock_page.title.return_value = "Mock Page"
    mock_browser.new_page.return_value = mock_page
    mock_pw.chromium.launch.return_value = mock_browser

    # sync_playwright() returns an object whose .start() returns the Playwright instance.
    mock_spw_cm = MagicMock()
    mock_spw_cm.start.return_value = mock_pw
    mock_sync_playwright_fn = MagicMock(return_value=mock_spw_cm)

    mock_sync_api = MagicMock()
    mock_sync_api.sync_playwright = mock_sync_playwright_fn

    monkeypatch.setitem(sys.modules, "playwright", MagicMock())
    monkeypatch.setitem(sys.modules, "playwright.sync_api", mock_sync_api)

    return {"pw": mock_pw, "browser": mock_browser, "page": mock_page}


@pytest.fixture()
def live_page():
    """Set a mock page so actions that need a running browser work immediately."""
    mock_page = MagicMock()
    mock_page.title.return_value = "Test Page"
    browser_module._page = mock_page
    return mock_page


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_schema_name():
    assert BrowserTool().schema.name == "browser"


def test_schema_required():
    assert BrowserTool().schema.parameters["required"] == ["action"]


def test_schema_all_actions():
    actions = set(BrowserTool().schema.parameters["properties"]["action"]["enum"])
    assert actions == {"start", "navigate", "evaluate", "screenshot", "pick", "cookies", "stop"}


def test_schema_is_not_read_only():
    # browser mutates page state, so is_read_only must be False
    assert BrowserTool().schema.is_read_only is False


# ---------------------------------------------------------------------------
# Unknown action
# ---------------------------------------------------------------------------

def test_unknown_action_returns_helpful_message(live_page):
    result = BrowserTool().execute(action="hover")
    assert "Unknown action" in result
    assert "hover" in result


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------

def test_start_headed_launches_headed_browser(mock_playwright):
    BrowserTool().execute(action="start", headless=False)
    mock_playwright["pw"].chromium.launch.assert_called_once_with(headless=False)


def test_start_headless_passes_headless_true(mock_playwright):
    BrowserTool().execute(action="start", headless=True)
    mock_playwright["pw"].chromium.launch.assert_called_once_with(headless=True)


def test_start_result_mentions_mode_headed(mock_playwright):
    result = BrowserTool().execute(action="start", headless=False)
    assert "headed" in result


def test_start_result_mentions_mode_headless(mock_playwright):
    result = BrowserTool().execute(action="start", headless=True)
    assert "headless" in result


def test_start_stores_browser_global(mock_playwright):
    BrowserTool().execute(action="start")
    assert browser_module._browser is not None


def test_start_stores_page_global(mock_playwright):
    BrowserTool().execute(action="start")
    assert browser_module._page is not None


def test_start_already_running_returns_error():
    browser_module._browser = MagicMock()
    result = BrowserTool().execute(action="start")
    assert "already running" in result.lower()


def test_start_playwright_not_installed(monkeypatch):
    # Simulate ImportError by removing playwright from sys.modules.
    monkeypatch.setitem(sys.modules, "playwright", None)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    result = BrowserTool().execute(action="start")
    assert "not installed" in result.lower()


def test_start_connect_to_port_uses_cdp(mock_playwright):
    mock_pw = mock_playwright["pw"]
    mock_cdp_browser = MagicMock()
    mock_cdp_page = MagicMock()
    mock_cdp_page.url = "https://already-open.com"
    mock_context = MagicMock()
    mock_context.pages = [mock_cdp_page]
    mock_cdp_browser.contexts = [mock_context]
    mock_pw.chromium.connect_over_cdp.return_value = mock_cdp_browser

    result = BrowserTool().execute(action="start", connect_to_port=9222)
    mock_pw.chromium.connect_over_cdp.assert_called_once_with("http://localhost:9222")
    assert "9222" in result


def test_start_connect_to_port_uses_existing_page(mock_playwright):
    mock_pw = mock_playwright["pw"]
    mock_cdp_browser = MagicMock()
    mock_cdp_page = MagicMock()
    mock_cdp_page.url = "https://existing.com"
    mock_context = MagicMock()
    mock_context.pages = [mock_cdp_page]
    mock_cdp_browser.contexts = [mock_context]
    mock_pw.chromium.connect_over_cdp.return_value = mock_cdp_browser

    BrowserTool().execute(action="start", connect_to_port=9222)
    # Should use the existing page, not create a new one.
    assert browser_module._page is mock_cdp_page


# ---------------------------------------------------------------------------
# navigate
# ---------------------------------------------------------------------------

def test_navigate_calls_goto(live_page):
    BrowserTool().execute(action="navigate", url="https://example.com")
    live_page.goto.assert_called_once_with(
        "https://example.com", wait_until="domcontentloaded"
    )


def test_navigate_includes_url_in_result(live_page):
    result = BrowserTool().execute(action="navigate", url="https://example.com")
    assert "example.com" in result


def test_navigate_includes_page_title(live_page):
    live_page.title.return_value = "Example Domain"
    result = BrowserTool().execute(action="navigate", url="https://example.com")
    assert "Example Domain" in result


def test_navigate_missing_url_returns_error(live_page):
    result = BrowserTool().execute(action="navigate")
    assert "Error" in result
    assert "url" in result.lower()


def test_navigate_no_browser_returns_error():
    result = BrowserTool().execute(action="navigate", url="https://example.com")
    assert "Error" in result
    assert "start" in result.lower()


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def test_evaluate_dict_result_is_json(live_page):
    live_page.evaluate.return_value = {"title": "hello"}
    result = BrowserTool().execute(action="evaluate", script="({title: document.title})")
    assert json.loads(result) == {"title": "hello"}


def test_evaluate_list_result(live_page):
    live_page.evaluate.return_value = [1, 2, 3]
    result = BrowserTool().execute(action="evaluate", script="[1,2,3]")
    assert json.loads(result) == [1, 2, 3]


def test_evaluate_primitive_int(live_page):
    live_page.evaluate.return_value = 42
    result = BrowserTool().execute(action="evaluate", script="40 + 2")
    assert json.loads(result) == 42


def test_evaluate_null_returns_string_null(live_page):
    live_page.evaluate.return_value = None
    result = BrowserTool().execute(action="evaluate", script="void 0")
    assert result == "null"


def test_evaluate_missing_script_returns_error(live_page):
    result = BrowserTool().execute(action="evaluate")
    assert "Error" in result
    assert "script" in result.lower()


def test_evaluate_no_browser_returns_error():
    result = BrowserTool().execute(action="evaluate", script="1+1")
    assert "Error" in result
    assert "start" in result.lower()


# ---------------------------------------------------------------------------
# screenshot
# ---------------------------------------------------------------------------

def test_screenshot_returns_png_path(live_page):
    result = BrowserTool().execute(action="screenshot")
    assert result.endswith(".png")


def test_screenshot_passes_path_to_playwright(live_page):
    result = BrowserTool().execute(action="screenshot")
    call_kwargs = live_page.screenshot.call_args.kwargs
    assert call_kwargs["path"] == result


def test_screenshot_no_browser_returns_error():
    result = BrowserTool().execute(action="screenshot")
    assert "Error" in result


# ---------------------------------------------------------------------------
# pick
# ---------------------------------------------------------------------------

def test_pick_returns_json_array(live_page):
    live_page.evaluate.side_effect = [
        None,  # _PICK_SETUP_JS — no return value needed
        [{"tag": "button", "id": "ok", "class": "btn primary",
          "text": "OK", "html": "<button id='ok'>OK</button>",
          "parents": ["div.modal", "div#root"]}],
    ]
    result = BrowserTool().execute(action="pick", timeout=30)
    data = json.loads(result)
    assert data[0]["tag"] == "button"
    assert data[0]["id"] == "ok"
    assert "div.modal" in data[0]["parents"]


def test_pick_no_elements_selected(live_page):
    live_page.evaluate.side_effect = [None, []]
    result = BrowserTool().execute(action="pick", timeout=30)
    assert "No elements" in result


def test_pick_timeout_returns_error(live_page):
    live_page.evaluate.return_value = None
    live_page.wait_for_function.side_effect = Exception("Timeout exceeded")
    result = BrowserTool().execute(action="pick", timeout=5)
    assert "timed out" in result.lower()
    assert "5" in result


def test_pick_default_timeout_is_120s(live_page):
    live_page.evaluate.side_effect = [None, []]
    BrowserTool().execute(action="pick")
    live_page.wait_for_function.assert_called_once_with(
        "window.__pick_done__ === true", timeout=120_000
    )


def test_pick_custom_timeout_passed_correctly(live_page):
    live_page.evaluate.side_effect = [None, []]
    BrowserTool().execute(action="pick", timeout=60)
    live_page.wait_for_function.assert_called_once_with(
        "window.__pick_done__ === true", timeout=60_000
    )


def test_pick_no_browser_returns_error():
    result = BrowserTool().execute(action="pick")
    assert "Error" in result


# ---------------------------------------------------------------------------
# cookies
# ---------------------------------------------------------------------------

def test_cookies_returns_json_array(live_page):
    live_page.context.cookies.return_value = [
        {"name": "session", "value": "tok123",
         "domain": "example.com", "httpOnly": True},
        {"name": "theme", "value": "dark",
         "domain": "example.com", "httpOnly": False},
    ]
    result = BrowserTool().execute(action="cookies")
    data = json.loads(result)
    assert len(data) == 2
    assert data[0]["name"] == "session"
    assert data[0]["httpOnly"] is True


def test_cookies_empty_returns_empty_array(live_page):
    live_page.context.cookies.return_value = []
    result = BrowserTool().execute(action="cookies")
    assert json.loads(result) == []


def test_cookies_no_browser_returns_error():
    result = BrowserTool().execute(action="cookies")
    assert "Error" in result


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------

def test_stop_calls_browser_close():
    mock_browser = MagicMock()
    browser_module._browser = mock_browser
    browser_module._playwright = MagicMock()
    browser_module._page = MagicMock()

    BrowserTool().execute(action="stop")
    # Save the reference before stop() clears the global.
    mock_browser.close.assert_called_once()


def test_stop_calls_playwright_stop():
    mock_browser = MagicMock()
    mock_pw = MagicMock()
    browser_module._browser = mock_browser
    browser_module._playwright = mock_pw
    browser_module._page = MagicMock()

    BrowserTool().execute(action="stop")
    mock_pw.stop.assert_called_once()


def test_stop_result_says_stopped():
    browser_module._browser = MagicMock()
    browser_module._playwright = MagicMock()
    browser_module._page = MagicMock()

    result = BrowserTool().execute(action="stop")
    assert "stopped" in result.lower()


def test_stop_clears_all_globals():
    browser_module._browser = MagicMock()
    browser_module._playwright = MagicMock()
    browser_module._page = MagicMock()

    BrowserTool().execute(action="stop")
    assert browser_module._browser is None
    assert browser_module._page is None
    assert browser_module._playwright is None


def test_stop_no_browser_running():
    result = BrowserTool().execute(action="stop")
    assert "No browser" in result


def test_stop_clears_globals_even_if_close_raises():
    """Globals must be cleared via finally even when browser.close() throws."""
    browser_module._browser = MagicMock()
    browser_module._browser.close.side_effect = Exception("process already gone")
    browser_module._playwright = MagicMock()
    browser_module._page = MagicMock()

    # execute() catches the exception and returns an error string
    BrowserTool().execute(action="stop")

    # globals must be cleared regardless
    assert browser_module._browser is None
    assert browser_module._page is None
    assert browser_module._playwright is None

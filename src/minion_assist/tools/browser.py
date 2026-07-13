"""BrowserTool — lightweight Playwright wrapper for browser automation.

Inspired by https://mariozechner.at/posts/2025-11-02-what-if-you-dont-need-mcp/

The core insight: the LLM already knows the DOM API. Rather than wrapping every
DOM operation in a dedicated MCP tool (13k-18k token overhead), expose a thin
evaluate() action and let the agent write plain JavaScript. The total schema
description stays under 300 tokens — far cheaper than any MCP browser server.

Seven actions cover the full browser workflow:

  start      — launch Playwright Chromium or attach to an existing Chrome via CDP
  navigate   — go to a URL, wait for DOMContentLoaded
  evaluate   — run arbitrary JavaScript in the page context, returns JSON
  screenshot — capture the viewport as a PNG, return the file path
  pick       — inject an interactive DOM element picker, return selected elements
  cookies    — dump all cookies for the current page as JSON
  stop       — close the browser and release Playwright resources

State (playwright instance, browser, active page) is stored in three module-level
variables so the browser survives across tool calls within a single session. Tool
calls in minion-assist are sequential, so no locking is needed.

Talks to
--------
- ``tools/__init__.py`` — imported and registered unconditionally in default_registry().
- ``playwright`` (optional install) — uv add playwright && uv run playwright install chromium.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .base import Tool, ToolSchema

# Module-level browser state shared across all BrowserTool instances in a process.
# Set by _start(), cleared by _stop(). The page object is the primary interface;
# browser and playwright are kept only to close them cleanly on stop.
_playwright: Any = None
_browser: Any = None
_page: Any = None


def _require_page() -> Any:
    """Return the active Playwright Page object, or raise if none is started.

    Raises:
        RuntimeError: When action='start' has not been called yet this session.
    """
    if _page is None:
        raise RuntimeError("No browser running. Call action='start' first.")
    return _page


# ---------------------------------------------------------------------------
# Pick overlay JavaScript
# ---------------------------------------------------------------------------
# This JS is injected into the live page by the pick action. It adds a visual
# hover/click system so the user can select DOM elements with the mouse, then
# double-clicks to confirm. The results are stored in window.__picked__ and a
# flag window.__pick_done__ is set to true so Playwright's wait_for_function
# knows when to stop polling.

_PICK_SETUP_JS = r"""
(function () {
    // Cancel any leftover pick session from a previous call.
    if (typeof window.__pick_cleanup__ === 'function') {
        window.__pick_cleanup__();
    }

    window.__picked__ = [];
    window.__pick_done__ = false;

    // --- status banner -------------------------------------------------------
    const banner = document.createElement('div');
    banner.id = '__pick_banner__';
    Object.assign(banner.style, {
        position: 'fixed', top: '0', left: '0', right: '0',
        zIndex: '2147483647', background: 'rgba(0,0,0,0.82)',
        color: '#fff', padding: '9px 14px',
        font: 'bold 14px monospace', textAlign: 'center',
        pointerEvents: 'none',
    });
    banner.textContent =
        'PICK MODE — click elements to select (blue outline) • double-click to confirm';
    document.body.appendChild(banner);

    // --- event handlers ------------------------------------------------------
    function onOver(e) {
        if (e.target === banner) return;
        if (!e.target.__isPicked) {
            e.target.style.outline = '2px solid #f44';
        }
    }
    function onOut(e) {
        if (!e.target.__isPicked) {
            e.target.style.outline = e.target.__origOutline || '';
        }
    }
    function onClick(e) {
        if (e.target === banner) return;
        e.preventDefault();
        e.stopPropagation();
        if (e.target.__isPicked) {
            // deselect
            e.target.__isPicked = false;
            e.target.style.outline = e.target.__origOutline || '';
            window.__picked__ = window.__picked__.filter(function(el) {
                return el !== e.target;
            });
        } else {
            // select — store original outline so we can restore it on deselect
            e.target.__origOutline = e.target.style.outline;
            e.target.__isPicked = true;
            e.target.style.outline = '3px solid #4af';
            window.__picked__.push(e.target);
        }
        banner.textContent =
            'PICK MODE — ' + window.__picked__.length +
            ' selected • double-click to confirm';
    }
    function onDbl(e) {
        e.preventDefault();
        e.stopPropagation();
        cleanup();
        banner.textContent =
            '✓ Done — ' + window.__picked__.length + ' element(s) captured';
        setTimeout(function() {
            if (banner.parentNode) banner.parentNode.removeChild(banner);
        }, 2000);
        window.__pick_done__ = true;
    }

    function cleanup() {
        document.removeEventListener('mouseover', onOver, true);
        document.removeEventListener('mouseout',  onOut,  true);
        document.removeEventListener('click',     onClick, true);
        document.removeEventListener('dblclick',  onDbl,  true);
        window.__pick_cleanup__ = null;
    }

    window.__pick_cleanup__ = cleanup;
    document.addEventListener('mouseover', onOver,  true);
    document.addEventListener('mouseout',  onOut,   true);
    document.addEventListener('click',     onClick, true);
    document.addEventListener('dblclick',  onDbl,   true);
})();
"""

# After the user confirms (double-click), this JS reads window.__picked__ and
# returns structured metadata for each selected element.
_PICK_COLLECT_JS = r"""
window.__picked__.map(function(el) {
    var parents = [];
    var p = el.parentElement;
    for (var i = 0; i < 3 && p; i++, p = p.parentElement) {
        parents.push(
            p.tagName.toLowerCase() +
            (p.id        ? '#' + p.id                               : '') +
            (p.className ? '.' + p.className.trim().split(/\s+/)[0] : '')
        );
    }
    return {
        tag:     el.tagName.toLowerCase(),
        id:      el.id      || null,
        class:   el.className || null,
        text:    (el.innerText  || '').slice(0, 300),
        html:    el.outerHTML.slice(0, 800),
        parents: parents,
    };
});
"""


# ---------------------------------------------------------------------------
# BrowserTool
# ---------------------------------------------------------------------------

class BrowserTool(Tool):
    """Control a Playwright Chromium browser with a minimal, low-token interface.

    The agent can run arbitrary JavaScript via evaluate() rather than relying on
    pre-wrapped DOM helpers — it already knows the DOM API. The pick() action
    gives the user direct mouse-driven element selection without requiring the
    agent to construct complex CSS selectors in advance.
    """

    @property
    def schema(self) -> ToolSchema:
        """Describe the browser tool to the LLM.

        The description is intentionally terse — the model already knows
        DOM APIs, so we don't repeat them here.
        """
        return ToolSchema(
            name="browser",
            description=(
                "Control a Playwright Chromium browser. Actions:\n"
                "  start      — launch browser (headless=false shows a window; "
                "connect_to_port=N attaches to existing Chrome on that port)\n"
                "  navigate   — go to url, wait for DOMContentLoaded\n"
                "  evaluate   — run any JavaScript in the page, returns JSON "
                "(use your full DOM API knowledge here — no wrapping needed)\n"
                "  screenshot — capture viewport as PNG, returns file path for vision\n"
                "  pick       — inject interactive picker; user clicks to select elements, "
                "double-clicks to confirm; returns selected elements as JSON\n"
                "  cookies    — return all cookies for the page context as JSON\n"
                "  stop       — close browser and free resources"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["start", "navigate", "evaluate", "screenshot",
                                 "pick", "cookies", "stop"],
                        "description": "Which browser action to perform.",
                    },
                    "headless": {
                        "type": "boolean",
                        "description": "start only — run without a visible window. Default: false.",
                    },
                    "connect_to_port": {
                        "type": "integer",
                        "description": (
                            "start only — attach to an existing Chrome running with "
                            "--remote-debugging-port=N (e.g. 9222)."
                        ),
                    },
                    "url": {
                        "type": "string",
                        "description": "navigate only — the URL to load.",
                    },
                    "script": {
                        "type": "string",
                        "description": (
                            "evaluate only — JavaScript to execute in the page context. "
                            "The return value of the last expression is captured as JSON."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (
                            "pick only — seconds to wait for the user to double-click "
                            "and confirm the selection. Default: 120."
                        ),
                    },
                },
                "required": ["action"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        """Dispatch to the requested browser action.

        Args:
            action (str): One of start / navigate / evaluate / screenshot /
                          pick / cookies / stop.
            headless (bool): start only — run headless. Default False.
            connect_to_port (int): start only — CDP port of existing Chrome.
            url (str): navigate only — URL to load.
            script (str): evaluate only — JavaScript to run.
            timeout (int): pick only — seconds before giving up. Default 120.

        Returns:
            str: Result or JSON string. Error messages start with "Error:".
        """
        action = str(kwargs.get("action", ""))
        try:
            if action == "start":
                return self._start(
                    headless=bool(kwargs.get("headless", False)),
                    connect_to_port=kwargs.get("connect_to_port"),
                )
            if action == "navigate":
                return self._navigate(str(kwargs.get("url", "")))
            if action == "evaluate":
                return self._evaluate(str(kwargs.get("script", "")))
            if action == "screenshot":
                return self._screenshot()
            if action == "pick":
                return self._pick(int(kwargs.get("timeout") or 120))
            if action == "cookies":
                return self._cookies()
            if action == "stop":
                return self._stop()
            return (
                f"Unknown action {action!r}. "
                "Use: start, navigate, evaluate, screenshot, pick, cookies, stop"
            )
        except RuntimeError as exc:
            # _require_page() raises RuntimeError when no browser is running.
            return f"Error: {exc}"
        except Exception as exc:
            return f"Browser error ({type(exc).__name__}): {exc}"

    # -----------------------------------------------------------------------
    # Individual action handlers
    # -----------------------------------------------------------------------

    def _start(self, headless: bool, connect_to_port: object) -> str:
        """Launch Playwright Chromium or connect to an existing Chrome via CDP.

        Args:
            headless: True runs without a visible window (useful in CI).
            connect_to_port: If set, connects to Chrome at http://localhost:<port>
                             over the Chrome DevTools Protocol instead of launching.

        Returns:
            str: Confirmation message or error if playwright is missing.
        """
        global _playwright, _browser, _page  # noqa: PLW0603

        # Check first so "already running" always wins over "not installed".
        if _browser is not None:
            return "Browser already running. Call action='stop' first to restart."

        try:
            # Lazy import — playwright is an optional dependency.
            from playwright.sync_api import sync_playwright
        except ImportError:
            return (
                "Error: playwright not installed. "
                "Run: uv add playwright && uv run playwright install chromium"
            )

        pw = sync_playwright().start()
        _playwright = pw

        if connect_to_port is not None:
            port = int(connect_to_port)
            _browser = pw.chromium.connect_over_cdp(f"http://localhost:{port}")
            # Attach to the first existing page in the first context.
            contexts = _browser.contexts
            if contexts and contexts[0].pages:
                _page = contexts[0].pages[0]
            elif contexts:
                _page = contexts[0].new_page()
            else:
                _page = _browser.new_context().new_page()
            return f"Connected to Chrome on port {port} — active page: {_page.url!r}"

        _browser = pw.chromium.launch(headless=headless)
        _page = _browser.new_page()
        mode = "headless" if headless else "headed"
        return f"Browser started ({mode})."

    def _navigate(self, url: str) -> str:
        """Go to url and wait for the DOM to be loaded.

        Args:
            url: The URL to navigate to.

        Returns:
            str: Confirmation with the page title, or an error string.
        """
        if not url:
            return "Error: 'url' is required for navigate."
        page = _require_page()
        page.goto(url, wait_until="domcontentloaded")
        return f"Navigated to {url!r} — title: {page.title()!r}"

    def _evaluate(self, script: str) -> str:
        """Execute JavaScript in the page and return the result as JSON.

        Playwright captures the return value of the last expression in the script.
        Primitives, objects, and arrays are all JSON-serialisable; DOM nodes return
        empty objects — use .outerHTML or .textContent to get meaningful data.

        Args:
            script: JavaScript string to evaluate.

        Returns:
            str: JSON-encoded result, or "null" if the script returns nothing.
        """
        if not script:
            return "Error: 'script' is required for evaluate."
        page = _require_page()
        result = page.evaluate(script)
        if result is None:
            return "null"
        return json.dumps(result, indent=2, default=str)

    def _screenshot(self) -> str:
        """Capture the current viewport as a PNG and return its file path.

        The file is written to the system temp directory. Pass the returned
        path to an image-capable model to analyse the screenshot visually.

        Returns:
            str: Absolute path to the PNG file.
        """
        page = _require_page()
        path = Path(tempfile.mktemp(suffix=".png", prefix="browser_shot_"))
        page.screenshot(path=str(path))
        return str(path)

    def _pick(self, timeout_s: int) -> str:
        """Inject the interactive DOM picker and wait for the user to confirm.

        Highlights elements on hover (red outline) and toggles blue selection on
        click. The user double-clicks anywhere to confirm the selection. Once
        confirmed, returns metadata (tag, id, class, text, html, parent chain)
        for each selected element as a JSON array.

        Args:
            timeout_s: Maximum seconds to wait before giving up.

        Returns:
            str: JSON array of selected element metadata, or an error string.
        """
        page = _require_page()

        # Inject the overlay and reset window.__picked__/__pick_done__.
        page.evaluate(_PICK_SETUP_JS)

        # Block until the user double-clicks to confirm, or the timeout fires.
        try:
            page.wait_for_function(
                "window.__pick_done__ === true",
                timeout=timeout_s * 1000,
            )
        except Exception:
            return f"Error: pick timed out after {timeout_s}s — no confirmation received."

        # Collect metadata from each element the user selected.
        result = page.evaluate(_PICK_COLLECT_JS)
        if not result:
            return "No elements selected."
        return json.dumps(result, indent=2, default=str)

    def _cookies(self) -> str:
        """Dump all cookies from the current page's browser context as JSON.

        Returns HTTP-only cookies too (since we're inside the browser process,
        not an HTTP client). Useful for passing session tokens to a scraper.

        Returns:
            str: JSON array of cookie objects.
        """
        page = _require_page()
        cookies = page.context.cookies()
        return json.dumps(cookies, indent=2, default=str)

    def _stop(self) -> str:
        """Close the browser and stop the Playwright background thread.

        Globals are always cleared in the finally block even if browser.close()
        raises (e.g. the Chrome process was already killed externally).

        Returns:
            str: Confirmation message.
        """
        global _playwright, _browser, _page  # noqa: PLW0603

        if _browser is None:
            return "No browser running."

        try:
            _browser.close()
        finally:
            # Always clean up so a subsequent start() can succeed.
            if _playwright is not None:
                _playwright.stop()
            _browser = None
            _page = None
            _playwright = None

        return "Browser stopped."

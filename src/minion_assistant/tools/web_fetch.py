"""WebFetchTool — fetch a URL and return its readable text content.

Why this tool matters
---------------------
:class:`WebSearchTool` returns a list of search results with short snippets.
That is useful for finding *which* page has the answer, but the snippet is
often too short to give the agent the actual content it needs.

:class:`WebFetchTool` fetches a specific URL and returns the full page text,
so the agent can read an article, documentation page, or API reference in one
tool call.

HTML stripping
--------------
Most web pages are HTML.  Raw HTML is noisy and wastes context window tokens
on ``<div>``, ``<script>``, ``<style>`` tags that carry no semantic content.

This tool uses Python's stdlib :mod:`html.parser` to strip tags and decode
HTML entities (``&amp;`` → ``&``, ``&lt;`` → ``<``, etc.).  The result is
plain readable text — usually close to what you'd see if you copied the page
text from a browser.

The parser skips ``<script>`` and ``<style>`` blocks entirely so JavaScript
and CSS never appear in the output.

SSRF protection
---------------
URL safety is checked via :class:`PermissionPolicy` before the HTTP request is
made.  Any URL that matches a cloud metadata endpoint marker is rejected.

Truncation
----------
Web pages can be very large.  The ``max_chars`` parameter (default 8 000)
caps the returned text.  The agent can request more with a higher value if
the page is long.

Talks to
--------
- ``policy.py`` — :class:`PermissionPolicy` for URL validation.
- ``base.py`` — extends :class:`Tool`, returns :class:`ToolSchema`.
- ``__init__.py`` — registered via ``default_registry()``.
- ``httpx`` — HTTP client already in minion-assistant's dependencies.
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

from .base import Tool, ToolSchema
from .policy import PermissionPolicy

_DEFAULT_MAX_CHARS = 8_000
_DEFAULT_TIMEOUT = 30  # seconds


class _TextExtractor(HTMLParser):
    """Minimal HTML parser that extracts visible text and skips scripts/styles.

    ``HTMLParser`` is Python's stdlib pull-parser.  We subclass it and override
    the event methods to collect only the text nodes we care about:

    - ``handle_data``: called for raw text between tags.  We buffer it when
      we're not inside a ``<script>`` or ``<style>`` block.
    - ``handle_entityref`` / ``handle_charref``: decode HTML entities like
      ``&amp;`` and ``&#160;`` into their unicode equivalents.
    - ``handle_starttag`` / ``handle_endtag``: track when we enter/exit
      ``<script>`` and ``<style>`` blocks so we can skip their content.
    """

    # Tags whose entire content (including nested text) should be ignored.
    _SKIP_TAGS = frozenset({"script", "style", "noscript", "head"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        # Counter instead of bool so nested skip tags work correctly.
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def get_text(self) -> str:
        """Return all collected text joined by single newlines."""
        return "\n".join(self._parts)


def _extract_text(html: str) -> str:
    """Strip HTML markup and return plain readable text."""
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()


class WebFetchTool(Tool):
    """Tool for fetching a URL and returning its text content.

    Strips HTML tags from web pages and truncates the output to ``max_chars``
    to avoid flooding the context window with a very large page.
    """

    def __init__(self, policy: PermissionPolicy | None = None) -> None:
        self._policy = policy or PermissionPolicy()

    @property
    def schema(self) -> ToolSchema:
        """Describe the web fetch tool to the LLM."""
        return ToolSchema(
            name="web_fetch",
            description=(
                "Fetch a web page and return its readable text content. "
                "Use this after web_search to read the full content of a specific URL. "
                "HTML tags are stripped; the result is plain text. "
                "Large pages are truncated to max_chars."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": (
                            f"Maximum characters to return. "
                            f"Default {_DEFAULT_MAX_CHARS}. "
                            "Increase for long documents."
                        ),
                    },
                },
                "required": ["url"],
            },
            is_read_only=True,
        )

    def execute(self, **kwargs: object) -> str:
        """Fetch a URL and return text content.

        Args:
            url (str): The URL to fetch.
            max_chars (int, optional): Character cap on output. Default 8000.

        Returns:
            str: Page text (HTML stripped), truncated to max_chars.
                Returns an error string on network failure or SSRF denial.
        """
        url = str(kwargs["url"])
        max_chars = int(kwargs.get("max_chars") or _DEFAULT_MAX_CHARS)

        # URL safety check first — before making any network request.
        error = self._policy.check_url(url)
        if error:
            return error

        try:
            import httpx
        except ImportError:
            return "Error: httpx not installed. Run: uv add httpx"

        try:
            response = httpx.get(
                url,
                follow_redirects=True,
                timeout=_DEFAULT_TIMEOUT,
                headers={"User-Agent": "minion-assistant/1.0"},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return f"Error: HTTP {exc.response.status_code} for {url}"
        except httpx.TimeoutException:
            return f"Error: request timed out after {_DEFAULT_TIMEOUT}s for {url}"
        except httpx.RequestError as exc:
            return f"Error fetching {url}: {type(exc).__name__}: {exc}"
        except Exception as exc:
            return f"Error: {exc}"

        content_type = response.headers.get("content-type", "").lower()
        text = response.text

        # Strip HTML for web pages to get readable text.
        if "html" in content_type:
            text = _extract_text(text)

        # Collapse runs of blank lines — HTML stripping often produces many
        # consecutive empty lines where block elements used to be.
        import re
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n... [truncated at {max_chars} chars; use max_chars to increase]"

        return text or "(empty page)"

"""WebSearchTool — search the web using DuckDuckGo via the duckduckgo-search library.

No API key is required.  DuckDuckGo's HTML endpoint is public and rate-limit
friendly for reasonable query volumes.

Why DuckDuckGo?
---------------
- Zero configuration — no API key, no account, no billing.
- Privacy-respecting: does not track users across searches.
- Reliable HTML endpoint that the ``duckduckgo-search`` Python library wraps.
  (See openclaw ``extensions/duckduckgo/src/ddg-client.ts`` for a reference
  implementation in TypeScript that hits the same endpoint directly.)

Why is_read_only=True?
----------------------
Web searches never mutate local state (no files written, no memory saved).
Marking the tool read-only allows the runner to batch multiple simultaneous
web search calls concurrently when the model requests them in the same turn.

Talks to
--------
- ``tools/__init__.py`` — registers :class:`WebSearchTool` in ``default_registry()``.
- ``duckduckgo_search`` (external) — provides the ``DDGS`` client class.
"""

from __future__ import annotations

from .base import Tool, ToolSchema

# Attempt a module-level import so tests can monkeypatch the symbol directly
# (``monkeypatch.setattr("mini_minion.tools.web_search.DDGS", ...)``) without
# needing to actually install the library.  If the package is absent the tool
# still loads; execute() returns an installation hint instead of crashing.
try:
    from duckduckgo_search import DDGS
    _DDGS_AVAILABLE = True
except ImportError:
    DDGS = None  # type: ignore[assignment,misc]
    _DDGS_AVAILABLE = False

# Default number of results returned when the caller omits max_results.
# Mirrors openclaw ddg-client.ts default (count = 5).
_DEFAULT_MAX_RESULTS = 5

# Hard ceiling on results per call.  DuckDuckGo becomes unreliable above ~10
# results per request.  This also prevents the model from flooding its own
# context window with search output.
_MAX_RESULTS_HARD_CAP = 10

# Character cap on the total formatted output string.
# 6 000 chars ≈ 1 500 tokens — enough for 5–8 results with full snippets while
# staying well within any reasonable preserve_tokens budget.
_OUTPUT_MAX_CHARS = 6_000

# DuckDuckGo safe-search codes (matches openclaw DDG_SAFE_SEARCH_PARAM).
#   strict   →  "1"   — family-safe only
#   moderate → "-1"   — default; filters explicit content
#   off      → "-2"   — unfiltered
# We always use moderate to align with openclaw's DEFAULT_DDG_SAFE_SEARCH.
_SAFE_SEARCH = "moderate"


class WebSearchTool(Tool):
    """Search the web using DuckDuckGo; returns titles, URLs, and snippets.

    Requires the ``duckduckgo-search`` package (``uv add duckduckgo-search``).
    The tool degrades gracefully — if the package is missing it returns a clear
    installation message rather than raising an exception, so the agent can
    inform the user.

    Args:
        max_results: Default number of results when the caller omits
                     ``max_results``.  Clamped to [1, 10].
        timeout:     HTTP timeout in seconds for the DuckDuckGo request.
                     Matches openclaw's DEFAULT_TIMEOUT_SECONDS (20 s).
    """

    def __init__(
        self,
        max_results: int = _DEFAULT_MAX_RESULTS,
        timeout: int = 20,
    ) -> None:
        # Clamp max_results so an invalid constructor argument doesn't silently
        # cause bad behaviour later.
        self._max_results = max(1, min(_MAX_RESULTS_HARD_CAP, max_results))
        self._timeout = timeout

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="web_search",
            description=(
                "Search the web using DuckDuckGo. Returns titles, URLs, and snippets. "
                "No API key required. Use for current events, documentation, recent "
                "releases, or any information not in training data."
            ),
            # Read-only: searches never write files or mutate memory, so the
            # runner may execute multiple web_search calls concurrently.
            is_read_only=True,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": (
                            f"Number of results to return (1–{_MAX_RESULTS_HARD_CAP}, "
                            f"default {_DEFAULT_MAX_RESULTS})."
                        ),
                        "minimum": 1,
                        "maximum": _MAX_RESULTS_HARD_CAP,
                    },
                    "region": {
                        "type": "string",
                        "description": (
                            "DuckDuckGo region code for localised results, e.g. "
                            "'us-en', 'uk-en', 'au-en', 'de-de'. "
                            "Omit for global results."
                        ),
                    },
                },
                "required": ["query"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        """Run a DuckDuckGo search and return formatted results.

        Args:
            query:       The search query (required).
            max_results: How many results to return (optional, default 5).
            region:      DuckDuckGo region code (optional, default worldwide).

        Returns:
            Formatted string with numbered results (title, URL, snippet).
            On error returns a descriptive error string — never raises.
        """
        if not _DDGS_AVAILABLE:
            # Give the user a clear, actionable message rather than a confusing
            # ImportError traceback buried inside a tool result.
            return (
                "web_search is unavailable: the 'duckduckgo-search' package is not "
                "installed.\nRun: uv add duckduckgo-search"
            )

        query = str(kwargs["query"]).strip()
        if not query:
            return "Error: 'query' must be a non-empty string."

        # Clamp user-supplied max_results to the allowed range.
        raw_max = kwargs.get("max_results")
        max_results = (
            max(1, min(_MAX_RESULTS_HARD_CAP, int(raw_max)))
            if raw_max is not None
            else self._max_results
        )

        # 'wt-wt' is DuckDuckGo's worldwide region code — returns global results
        # when the caller doesn't specify a region.
        region = str(kwargs.get("region") or "wt-wt")

        try:
            # DDGS is used as a context manager to ensure the underlying HTTP
            # session is properly closed after the request completes.
            with DDGS(timeout=self._timeout) as ddgs:
                # ddgs.text() returns a generator; wrapping in list() forces
                # evaluation so the session can close before we format output.
                raw_results = list(
                    ddgs.text(
                        keywords=query,
                        region=region,
                        safesearch=_SAFE_SEARCH,
                        max_results=max_results,
                    )
                )
        except Exception as exc:
            # Network errors, rate limiting, and DuckDuckGo HTML changes all
            # land here.  We surface the error as a string so the model can
            # decide whether to retry or inform the user.
            return f"Search failed: {type(exc).__name__}: {exc}"

        if not raw_results:
            return f"No results found for: {query!r}"

        return _format_results(query, raw_results)


def _format_results(query: str, results: list[dict]) -> str:
    """Format a list of DuckDuckGo result dicts into a readable string.

    Each result dict from duckduckgo-search contains:
        title  — page title
        href   — URL
        body   — snippet / description text

    The output is capped at ``_OUTPUT_MAX_CHARS`` to avoid flooding the
    context window when snippets are very long.

    Args:
        query:   The original search query (shown in the header).
        results: List of result dicts from ``DDGS.text()``.

    Returns:
        Multi-line formatted string, capped at ``_OUTPUT_MAX_CHARS`` chars.
    """
    lines: list[str] = [f'Web search results for: "{query}"', ""]

    for i, result in enumerate(results, start=1):
        title = result.get("title") or "(no title)"
        url = result.get("href") or ""
        snippet = (result.get("body") or "").strip()

        lines.append(f"[{i}] {title}")
        if url:
            lines.append(f"    {url}")
        if snippet:
            lines.append(f"    {snippet}")
        lines.append("")  # blank line between results

    output = "\n".join(lines).rstrip()

    # Truncate if the total output exceeds the character cap.
    # We add a notice so the model knows results were cut rather than thinking
    # the search simply returned less data.
    if len(output) > _OUTPUT_MAX_CHARS:
        output = output[:_OUTPUT_MAX_CHARS] + "\n[... output truncated]"

    return output

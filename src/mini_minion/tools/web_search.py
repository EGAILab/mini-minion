"""WebSearchTool — search the web using DuckDuckGo via the ``ddgs`` library.

No API key is required.  DuckDuckGo's HTML endpoint is public and rate-limit
friendly for reasonable query volumes.

Why DuckDuckGo?
---------------
- Zero configuration — no API key, no account, no billing.
- Privacy-respecting: does not track users across searches.
- Reliable HTML endpoint that the ``ddgs`` Python library wraps.
  (See openclaw ``extensions/duckduckgo/src/ddg-client.ts`` for a reference
  implementation in TypeScript that hits the same endpoint directly.)

Note on package naming: the library was originally called ``duckduckgo-search``
(import name ``duckduckgo_search``).  It was renamed to ``ddgs`` (import name
``ddgs``) starting with version 9.  This file imports from ``ddgs``.

Why is_read_only=True?
----------------------
Web searches never mutate local state (no files written, no memory saved).
Marking the tool read-only allows the runner to batch multiple simultaneous
web search calls concurrently when the model requests them in the same turn.

Talks to
--------
- ``tools/__init__.py`` — registers :class:`WebSearchTool` in ``default_registry()``.
- ``ddgs`` (external) — provides the ``DDGS`` client class.
"""

from __future__ import annotations

from .base import Tool, ToolSchema

# ── Conditional import of the ddgs library ───────────────────────────────────
#
# WHY module-level (not inside execute)?
#   If we imported inside execute(), pytest's monkeypatch could not replace
#   the DDGS symbol at test time — the local name would shadow the patch.
#   A module-level import lets tests do:
#       monkeypatch.setattr("mini_minion.tools.web_search.DDGS", MockDDGS)
#   which works because the attribute lives on the module object.
#
# WHY try/except instead of a hard import?
#   The ddgs package is listed in pyproject.toml dependencies, so it will
#   normally be installed.  The try/except is a safety net: if someone sets
#   up the repo without running `uv sync`, the tool still loads (execute()
#   returns a helpful error string) instead of crashing at import time and
#   preventing mini-minion from starting at all.
try:
    from ddgs import DDGS
    _DDGS_AVAILABLE = True
except ImportError:
    # DDGS is None when the package is missing.  execute() checks
    # _DDGS_AVAILABLE before using DDGS so it never calls None().
    DDGS = None  # type: ignore[assignment,misc]
    _DDGS_AVAILABLE = False

# ── Module-level constants ────────────────────────────────────────────────────

# Default number of search results when the caller omits max_results.
# Mirrors openclaw ddg-client.ts (count = 5) — enough to give the model
# useful context without consuming too many tokens.
_DEFAULT_MAX_RESULTS = 5

# Absolute maximum results per call.
# DuckDuckGo's HTML endpoint becomes unreliable above ~10 results per request,
# and returning more would risk large tool outputs flooding the context window.
_MAX_RESULTS_HARD_CAP = 10

# Character limit on the total formatted output string.
# 6 000 chars ≈ 1 500 tokens at 4 chars/token — enough for 5–8 results with
# full snippets while staying well within the compactor's _max_tool_output
# budget (which also caps at 16 000 chars for large-context models).
_OUTPUT_MAX_CHARS = 6_000

# DuckDuckGo safe-search level string.
# Possible values (matching openclaw's DDG_SAFE_SEARCH_PARAM):
#   "strict"   → family-safe only (kp="1")
#   "moderate" → filters explicit content (kp="-1")  ← our default
#   "off"      → no filtering (kp="-2")
# We hard-code moderate to match openclaw's DEFAULT_DDG_SAFE_SEARCH and
# keep the tool safe for general use without burdening the model with an
# extra parameter to decide.
_SAFE_SEARCH = "moderate"


class WebSearchTool(Tool):
    """Search the web using DuckDuckGo; returns titles, URLs, and snippets.

    Requires the ``ddgs`` package (``uv add ddgs``).
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
        # Clamp here (not just in execute) so that a badly-configured tool
        # instance cannot silently request more results than DuckDuckGo handles
        # reliably.  max(1, min(CAP, x)) is a compact "clamp to [1, CAP]".
        self._max_results = max(1, min(_MAX_RESULTS_HARD_CAP, max_results))
        self._timeout = timeout  # HTTP socket timeout in seconds

    @property
    def schema(self) -> ToolSchema:
        # The `parameters` dict is a JSON Schema object.  It is sent to the
        # LLM as part of the tool definition so the model knows what arguments
        # to pass when it calls this tool.  The format is the same as an
        # OpenAPI request body schema:
        #   "type": "object"        — the arguments are a dict, not a list or string
        #   "properties": { ... }   — one entry per supported argument
        #   "required": [...]       — which arguments the model MUST supply
        #
        # "minimum"/"maximum" are JSON Schema keywords that the LLM can read to
        # understand valid value ranges (they are not enforced server-side by
        # mini-minion — we clamp manually in execute() for safety).
        return ToolSchema(
            name="web_search",
            description=(
                "Search the web using DuckDuckGo. Returns titles, URLs, and snippets. "
                "No API key required. Use for current events, documentation, recent "
                "releases, or any information not in training data."
            ),
            # is_read_only=True tells the runner that this tool never changes
            # the filesystem or application state, so it is safe to run
            # multiple web_search calls at the same time (parallel threads)
            # when the model requests them in a single turn.
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
                        # These hints help the model stay in the valid range.
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
                "web_search is unavailable: the 'ddgs' package is not installed.\n"
                "Run: uv add ddgs"
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
                # ddgs v9+: query is the first positional argument.
                # The old duckduckgo_search library used keywords=query instead.
                raw_results = list(
                    ddgs.text(
                        query,
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

    Each result dict from ddgs contains:
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

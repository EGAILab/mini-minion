"""Markdown → Matrix HTML formatting.

Converts markdown text to the ``org.matrix.custom.html`` format that Matrix
clients (Element, Beeper, etc.) render natively.  Matches openclaw's approach:
markdown-it with linkify, strikethrough, and compact list rendering.
"""

from __future__ import annotations

import re


def _get_md():
    """Lazy-load markdown-it to avoid import cost at startup."""
    from markdown_it import MarkdownIt  # noqa: PLC0415
    # html=False: don't allow raw HTML in input (security: user messages shouldn't
    #             inject arbitrary HTML into Matrix rooms).
    # linkify=True: enables auto-linking of bare URLs like https://example.com
    #               (requires the separate linkify-it-py package).
    # breaks=True:  convert single newlines to <br> like GitHub/Slack do.
    md = MarkdownIt(options_update={"html": False, "linkify": True, "breaks": True})
    # "strikethrough" adds ~~text~~ → <s>text</s> support (not on by default).
    md.enable("strikethrough")
    # "linkify" must also be enabled as a plugin rule, not just in options.
    md.enable("linkify")
    return md


def _compact_list_items(html: str) -> str:
    """Remove <p> wrappers inside <li> elements.

    Without this, Element renders unwanted top/bottom margins between list
    items when each item contains a single paragraph.  Matches openclaw's
    ``compactLooseListTokens`` behaviour.

    Turns:  <li><p>foo</p></li>
    Into:   <li>foo</li>
    """
    return re.sub(r"<li>\s*<p>(.*?)</p>\s*</li>", r"<li>\1</li>", html, flags=re.DOTALL)


def to_matrix_html(markdown: str) -> str | None:
    """Convert ``markdown`` to Matrix-compatible HTML.

    Returns ``None`` when the rendered HTML equals the plain text (i.e. no
    markup was applied), signalling that ``formatted_body`` should be omitted.

    Args:
        markdown: The markdown string to convert.

    Returns:
        HTML string, or ``None`` if no formatting was applied.
    """
    md = _get_md()
    html = md.render(markdown).strip()
    html = _compact_list_items(html)
    # markdown-it always wraps plain text in <p>...</p>.  Strip that wrapper
    # and compare to the original; if they match exactly and contain no HTML
    # tags, the text is truly plain and we can skip formatted_body entirely.
    # This avoids sending redundant HTML for simple one-liners.
    stripped = re.sub(r"^<p>(.*)</p>$", r"\1", html, flags=re.DOTALL).strip()
    if stripped == markdown.strip() and "<" not in stripped:
        return None
    return html


def _chunk_at_paragraph(text: str, limit: int) -> list[str]:
    """Split markdown at paragraph boundaries up to ``limit`` chars.

    Falls back to sentence → newline → hard split when no paragraph boundary fits.
    Splitting at semantic boundaries (paragraphs, sentences) keeps each chunk
    readable rather than cutting mid-word or mid-sentence.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        # Priority 1: split at a double newline (paragraph break) — best for
        # preserving markdown structure like separate bullet lists or headings.
        split_at = remaining.rfind("\n\n", 0, limit)
        if split_at == -1:
            # Priority 2: split at a single newline (line break).
            split_at = remaining.rfind("\n", 0, limit)
        if split_at == -1:
            # Priority 3: split after a sentence-ending ". " so the break is
            # at a natural reading pause.
            split_at = remaining.rfind(". ", 0, limit)
            if split_at != -1:
                split_at += 1  # include the period in the first chunk
        if split_at == -1:
            # Last resort: hard cut at the character limit.
            split_at = limit
        chunks.append(remaining[:split_at].rstrip())
        # Strip leading newlines from the next chunk so it doesn't start blank.
        remaining = remaining[split_at:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


def build_content(text: str, chunk_limit: int) -> list[dict]:
    """Build a list of Matrix ``m.room.message`` content dicts for ``text``.

    Each dict has ``msgtype``, ``body``, and optionally ``format`` +
    ``formatted_body`` when the text contains markdown formatting.

    Args:
        text:        Message body (may contain markdown).
        chunk_limit: Maximum characters per message chunk.

    Returns:
        List of content dicts ready to pass to ``room_send``.
    """
    chunks = _chunk_at_paragraph(text, chunk_limit)
    result: list[dict] = []
    for chunk in chunks:
        html = to_matrix_html(chunk)
        content: dict = {"msgtype": "m.text", "body": chunk}
        if html:
            content["format"] = "org.matrix.custom.html"
            content["formatted_body"] = html
        result.append(content)
    return result

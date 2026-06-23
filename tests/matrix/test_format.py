"""Tests for matrix/format.py — markdown → Matrix HTML conversion."""

from minion_assist.matrix.format import (
    _compact_list_items,
    build_content,
    to_matrix_html,
)


# ---------------------------------------------------------------------------
# to_matrix_html
# ---------------------------------------------------------------------------

def test_bold_renders_strong():
    html = to_matrix_html("**bold**")
    assert html is not None
    assert "<strong>bold</strong>" in html


def test_italic_renders_em():
    html = to_matrix_html("*italic*")
    assert html is not None
    assert "<em>italic</em>" in html


def test_inline_code_renders_code():
    html = to_matrix_html("`code`")
    assert html is not None
    assert "<code>code</code>" in html


def test_fenced_code_block():
    md = "```python\nprint('hi')\n```"
    html = to_matrix_html(md)
    assert html is not None
    assert "<pre>" in html
    assert "<code" in html  # may have class="language-python"


def test_heading_renders_h_tag():
    html = to_matrix_html("# Heading")
    assert html is not None
    assert "<h1>" in html


def test_bullet_list():
    html = to_matrix_html("- item one\n- item two")
    assert html is not None
    assert "<ul>" in html
    assert "<li>" in html


def test_numbered_list():
    html = to_matrix_html("1. first\n2. second")
    assert html is not None
    assert "<ol>" in html


def test_plain_text_returns_none():
    # Pure plain text with no markdown markers → no formatting → None
    result = to_matrix_html("just plain text")
    assert result is None


def test_strikethrough():
    html = to_matrix_html("~~strike~~")
    assert html is not None
    assert "<s>" in html or "strikethrough" in html.lower() or "del" in html


def test_linkify():
    html = to_matrix_html("visit https://example.com now")
    assert html is not None
    assert "<a href=" in html


# ---------------------------------------------------------------------------
# _compact_list_items
# ---------------------------------------------------------------------------

def test_compact_removes_p_in_li():
    html = "<ul><li><p>foo</p></li></ul>"
    result = _compact_list_items(html)
    assert "<li>foo</li>" in result
    assert "<p>" not in result


def test_compact_multiline_content():
    html = "<ul><li><p>line one\nline two</p></li></ul>"
    result = _compact_list_items(html)
    assert "<p>" not in result


def test_compact_leaves_non_list_p_alone():
    html = "<p>standalone paragraph</p>"
    result = _compact_list_items(html)
    assert "<p>standalone paragraph</p>" in result


# ---------------------------------------------------------------------------
# build_content
# ---------------------------------------------------------------------------

def test_build_content_markdown_has_formatted_body():
    contents = build_content("**bold** text", chunk_limit=4000)
    assert len(contents) == 1
    c = contents[0]
    assert c["format"] == "org.matrix.custom.html"
    assert "<strong>" in c["formatted_body"]
    assert c["body"] == "**bold** text"


def test_build_content_plain_no_formatted_body():
    contents = build_content("just plain text", chunk_limit=4000)
    assert len(contents) == 1
    assert "formatted_body" not in contents[0]


def test_build_content_splits_long_text():
    long_text = "word " * 200  # 1000 chars
    contents = build_content(long_text, chunk_limit=100)
    assert len(contents) > 1
    for c in contents:
        assert len(c["body"]) <= 100


def test_build_content_splits_at_paragraph():
    text = "paragraph one\n\nparagraph two"
    contents = build_content(text, chunk_limit=20)
    # Should split at the double newline
    assert len(contents) == 2
    assert "paragraph one" in contents[0]["body"]
    assert "paragraph two" in contents[1]["body"]

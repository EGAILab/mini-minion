"""Tests for memory/boundaries.py: action-sensitive memory boundary metadata
(Stage One Phase 6, slice A).
"""

from __future__ import annotations

import time

from minion_assist.memory.boundaries import (
    format_boundary_prefix,
    is_boundary_active,
    parse_frontmatter,
)

# ---------------------------------------------------------------------------
# parse_frontmatter
# ---------------------------------------------------------------------------

def test_parse_frontmatter_returns_empty_for_content_with_no_frontmatter():
    metadata, body, offset = parse_frontmatter("Just a normal note.\nSecond line.")

    assert metadata == {}
    assert body == "Just a normal note.\nSecond line."
    assert offset == 0


def test_parse_frontmatter_extracts_known_fields():
    content = (
        "---\n"
        "owner: main\n"
        "required_approval: user\n"
        "---\n"
        "Body text here."
    )

    metadata, body, offset = parse_frontmatter(content)

    assert metadata == {"owner": "main", "required_approval": "user"}
    assert body == "Body text here."
    assert offset == 4


def test_parse_frontmatter_ignores_unknown_fields():
    content = "---\nowner: main\nsome_random_field: whatever\n---\nBody."

    metadata, _, _ = parse_frontmatter(content)

    assert metadata == {"owner": "main"}


def test_parse_frontmatter_strips_quotes_from_values():
    content = '---\napplies_when: "deploying to production"\n---\nBody.'

    metadata, _, _ = parse_frontmatter(content)

    assert metadata == {"applies_when": "deploying to production"}


def test_parse_frontmatter_treats_a_bare_leading_dash_line_without_a_close_as_no_frontmatter():
    content = "---\nowner: main\nno closing delimiter here"

    metadata, body, offset = parse_frontmatter(content)

    assert metadata == {}
    assert body == content
    assert offset == 0


def test_parse_frontmatter_requires_the_dashes_to_be_the_literal_first_line():
    content = "Some intro text.\n---\nowner: main\n---\nBody."

    metadata, body, offset = parse_frontmatter(content)

    assert metadata == {}
    assert body == content
    assert offset == 0


def test_parse_frontmatter_with_empty_block_returns_empty_metadata():
    content = "---\n---\nBody."

    metadata, body, offset = parse_frontmatter(content)

    assert metadata == {}
    assert body == "Body."
    assert offset == 2


def test_parse_frontmatter_line_offset_lets_a_caller_recover_original_line_numbers():
    # Original file: ["---", "owner: main", "---", "Line A", "Line B"].
    # Body's line 1 ("Line A") is the original file's line 4 -- i.e.
    # body_line + offset == original_line.
    content = "---\nowner: main\n---\nLine A\nLine B"

    metadata, body, offset = parse_frontmatter(content)

    assert offset == 3
    assert body.splitlines()[0] == "Line A"
    assert 1 + offset == 4


def test_parse_frontmatter_ignores_lines_without_a_colon():
    content = "---\nowner: main\njust some text with no colon\n---\nBody."

    metadata, _, _ = parse_frontmatter(content)

    assert metadata == {"owner": "main"}


def test_parse_frontmatter_ignores_a_field_with_an_empty_value():
    content = "---\nowner: main\napplies_when: \n---\nBody."

    metadata, _, _ = parse_frontmatter(content)

    assert metadata == {"owner": "main"}


def test_parse_frontmatter_with_no_content_at_all():
    metadata, body, offset = parse_frontmatter("")

    assert metadata == {}
    assert body == ""
    assert offset == 0


# ---------------------------------------------------------------------------
# is_boundary_active
# ---------------------------------------------------------------------------

def test_is_boundary_active_with_no_metadata_is_always_true():
    assert is_boundary_active({}) is True


def test_is_boundary_active_with_no_time_fields_is_true():
    assert is_boundary_active({"owner": "main"}) is True


def test_is_boundary_active_before_safe_after_is_false():
    future = time.time() + 3600
    metadata = {"safe_after": _iso(future)}

    assert is_boundary_active(metadata, now=time.time()) is False


def test_is_boundary_active_after_safe_after_is_true():
    past = time.time() - 3600
    metadata = {"safe_after": _iso(past)}

    assert is_boundary_active(metadata, now=time.time()) is True


def test_is_boundary_active_before_expires_at_is_true():
    future = time.time() + 3600
    metadata = {"expires_at": _iso(future)}

    assert is_boundary_active(metadata, now=time.time()) is True


def test_is_boundary_active_after_expires_at_is_false():
    past = time.time() - 3600
    metadata = {"expires_at": _iso(past)}

    assert is_boundary_active(metadata, now=time.time()) is False


def test_is_boundary_active_within_both_bounds_is_true():
    now = time.time()
    metadata = {"safe_after": _iso(now - 3600), "expires_at": _iso(now + 3600)}

    assert is_boundary_active(metadata, now=now) is True


def test_is_boundary_active_with_an_unparseable_date_is_not_a_constraint():
    metadata = {"expires_at": "not a real date"}

    assert is_boundary_active(metadata, now=time.time()) is True


def test_is_boundary_active_defaults_now_to_the_current_time():
    metadata = {"expires_at": _iso(time.time() + 3600)}

    assert is_boundary_active(metadata) is True


def _iso(epoch: float) -> str:
    import datetime as _dt

    return _dt.datetime.fromtimestamp(epoch).isoformat()


# ---------------------------------------------------------------------------
# format_boundary_prefix
# ---------------------------------------------------------------------------

def test_format_boundary_prefix_with_empty_metadata_returns_empty_string():
    assert format_boundary_prefix({}) == ""


def test_format_boundary_prefix_includes_every_present_field():
    metadata = {"owner": "main", "required_approval": "user"}

    result = format_boundary_prefix(metadata)

    assert "Owner: main" in result
    assert "Requires approval from: user" in result


def test_format_boundary_prefix_is_explicitly_advisory():
    result = format_boundary_prefix({"owner": "main"})

    assert "advisory only" in result.lower()
    assert "does not itself grant permission" in result.lower()


def test_format_boundary_prefix_orders_fields_consistently():
    metadata = {"required_approval": "user", "owner": "main", "expires_at": "2026-12-01"}

    result = format_boundary_prefix(metadata)

    # owner appears before expires_at, which appears before required_approval,
    # regardless of dict insertion order.
    assert result.index("Owner") < result.index("Expires") < result.index("Requires approval")

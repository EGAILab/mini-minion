"""Tests for memory/knowledge.py: claim marker parsing (Stage One Phase 7, slice A)."""

from __future__ import annotations

from minion_assist.memory.knowledge import parse_claims, parse_time_epoch

# ---------------------------------------------------------------------------
# parse_claims — basics
# ---------------------------------------------------------------------------

def test_parse_claims_returns_empty_for_content_with_no_markers():
    assert parse_claims("Just a normal note.\nSecond line.") == []


def test_parse_claims_extracts_a_single_claim():
    content = (
        "- User's dog is named Biscuit.\n"
        "  <!-- claim:c-a1b2c3d4 status=supported confidence=0.9 -->"
    )

    claims = parse_claims(content)

    assert len(claims) == 1
    assert claims[0].id == "c-a1b2c3d4"
    assert claims[0].status == "supported"
    assert claims[0].confidence == 0.9


def test_parse_claims_extracts_the_claim_text_and_strips_the_list_bullet():
    content = "- User's dog is named Biscuit.\n  <!-- claim:c-1 status=supported -->"

    [claim] = parse_claims(content)

    assert claim.text == "User's dog is named Biscuit."


def test_parse_claims_handles_a_plain_paragraph_without_a_list_bullet():
    content = "User's dog is named Biscuit.\n<!-- claim:c-1 status=supported -->"

    [claim] = parse_claims(content)

    assert claim.text == "User's dog is named Biscuit."


def test_parse_claims_defaults_status_to_unknown_when_absent():
    content = "- Some claim.\n  <!-- claim:c-1 -->"

    [claim] = parse_claims(content)

    assert claim.status == "unknown"


def test_parse_claims_defaults_status_to_unknown_for_an_unrecognized_value():
    content = "- Some claim.\n  <!-- claim:c-1 status=maybe -->"

    [claim] = parse_claims(content)

    assert claim.status == "unknown"


def test_parse_claims_accepts_every_known_status():
    for status in ("supported", "contested", "superseded", "unknown"):
        content = f"- Some claim.\n  <!-- claim:c-1 status={status} -->"
        [claim] = parse_claims(content)
        assert claim.status == status


def test_parse_claims_confidence_defaults_to_none_when_absent():
    content = "- Some claim.\n  <!-- claim:c-1 status=supported -->"

    [claim] = parse_claims(content)

    assert claim.confidence is None


def test_parse_claims_confidence_is_none_for_an_unparseable_value():
    content = "- Some claim.\n  <!-- claim:c-1 confidence=not-a-number -->"

    [claim] = parse_claims(content)

    assert claim.confidence is None


def test_parse_claims_records_the_marker_line_number():
    content = "line one\nline two\n- Some claim.\n  <!-- claim:c-1 -->"

    [claim] = parse_claims(content)

    assert claim.line == 4


# ---------------------------------------------------------------------------
# parse_claims — multiple claims / block boundaries
# ---------------------------------------------------------------------------

def test_parse_claims_handles_two_claims_separated_by_a_blank_line():
    content = (
        "- First claim.\n"
        "  <!-- claim:c-1 status=supported -->\n"
        "\n"
        "- Second claim.\n"
        "  <!-- claim:c-2 status=supported -->"
    )

    claims = parse_claims(content)

    assert [c.text for c in claims] == ["First claim.", "Second claim."]


def test_parse_claims_handles_two_list_items_with_no_blank_line_between_them():
    # The exact scenario this parser must get right: consecutive bullets,
    # no blank line separating them.
    content = (
        "- User's dog is named Biscuit.\n"
        "  <!-- claim:c-a1b2c3d4 status=supported\n"
        "       confidence=0.9 observed=2026-06-01\n"
        "       evidence=proposal:42 -->\n"
        "- Biscuit is a golden retriever.\n"
        "  <!-- claim:c-f9e8d7c6 status=supported\n"
        "       confidence=0.85 observed=2026-07-10\n"
        "       evidence=proposal:51 -->"
    )

    claims = parse_claims(content)

    assert len(claims) == 2
    assert claims[0].id == "c-a1b2c3d4"
    assert claims[0].text == "User's dog is named Biscuit."
    assert claims[1].id == "c-f9e8d7c6"
    assert claims[1].text == "Biscuit is a golden retriever."
    # Neither claim's text bleeds into the other's.
    assert "Biscuit is a golden retriever" not in claims[0].text
    assert "User's dog is named Biscuit" not in claims[1].text


def test_parse_claims_preserves_document_order():
    content = (
        "- A.\n  <!-- claim:c-a -->\n"
        "- B.\n  <!-- claim:c-b -->\n"
        "- C.\n  <!-- claim:c-c -->"
    )

    claims = parse_claims(content)

    assert [c.id for c in claims] == ["c-a", "c-b", "c-c"]


# ---------------------------------------------------------------------------
# parse_claims — optional fields
# ---------------------------------------------------------------------------

def test_parse_claims_extracts_observed_and_valid_time_fields():
    content = (
        "- Some claim.\n"
        "  <!-- claim:c-1 observed=2026-06-01 valid_from=2026-01-01 valid_to=2026-12-31 -->"
    )

    [claim] = parse_claims(content)

    assert claim.observed == "2026-06-01"
    assert claim.valid_from == "2026-01-01"
    assert claim.valid_to == "2026-12-31"


def test_parse_claims_privacy_defaults_to_empty_string():
    content = "- Some claim.\n  <!-- claim:c-1 -->"

    [claim] = parse_claims(content)

    assert claim.privacy == ""


def test_parse_claims_extracts_privacy():
    content = "- Some claim.\n  <!-- claim:c-1 privacy=private -->"

    [claim] = parse_claims(content)

    assert claim.privacy == "private"


def test_parse_claims_entity_defaults_to_none():
    content = "- Some claim.\n  <!-- claim:c-1 -->"

    [claim] = parse_claims(content)

    assert claim.entity is None


def test_parse_claims_extracts_entity():
    content = "- Some claim.\n  <!-- claim:c-1 entity=Biscuit -->"

    [claim] = parse_claims(content)

    assert claim.entity == "Biscuit"


def test_parse_claims_evidence_defaults_to_empty_list():
    content = "- Some claim.\n  <!-- claim:c-1 -->"

    [claim] = parse_claims(content)

    assert claim.evidence == []


def test_parse_claims_extracts_a_single_evidence_ref():
    content = "- Some claim.\n  <!-- claim:c-1 evidence=proposal:42 -->"

    [claim] = parse_claims(content)

    assert claim.evidence == [("proposal", "42")]


def test_parse_claims_extracts_multiple_comma_separated_evidence_refs():
    content = "- Some claim.\n  <!-- claim:c-1 evidence=proposal:42,message:1189 -->"

    [claim] = parse_claims(content)

    assert claim.evidence == [("proposal", "42"), ("message", "1189")]


def test_parse_claims_ignores_a_malformed_evidence_entry():
    content = "- Some claim.\n  <!-- claim:c-1 evidence=proposal:42,garbage,message:1189 -->"

    [claim] = parse_claims(content)

    assert claim.evidence == [("proposal", "42"), ("message", "1189")]


def test_parse_claims_multiline_marker():
    content = (
        "- User's dog is named Biscuit.\n"
        "  <!-- claim:c-a1b2c3d4 status=supported\n"
        "       confidence=0.9 observed=2026-06-01\n"
        "       evidence=proposal:42 -->"
    )

    [claim] = parse_claims(content)

    assert claim.id == "c-a1b2c3d4"
    assert claim.status == "supported"
    assert claim.confidence == 0.9
    assert claim.observed == "2026-06-01"
    assert claim.evidence == [("proposal", "42")]


def test_parse_claims_with_empty_content():
    assert parse_claims("") == []


def test_parse_claims_never_invents_a_claim_from_unmarked_prose():
    content = "This is just a plain note with no claim markers at all.\nAnother sentence."

    assert parse_claims(content) == []


# ---------------------------------------------------------------------------
# parse_time_epoch
# ---------------------------------------------------------------------------

def test_parse_time_epoch_parses_a_valid_date():
    import datetime as _dt

    epoch = parse_time_epoch("2026-06-01")

    assert epoch == _dt.datetime.fromisoformat("2026-06-01").timestamp()


def test_parse_time_epoch_returns_none_for_none():
    assert parse_time_epoch(None) is None


def test_parse_time_epoch_returns_none_for_empty_string():
    assert parse_time_epoch("") is None


def test_parse_time_epoch_returns_none_for_garbage():
    assert parse_time_epoch("not a date") is None

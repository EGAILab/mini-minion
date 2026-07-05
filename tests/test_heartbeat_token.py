"""Tests for heartbeat_token — HEARTBEAT_OK detection and stripping."""

from minion_assist.heartbeat_token import is_heartbeat_ok, strip_heartbeat_token


# ---------------------------------------------------------------------------
# is_heartbeat_ok
# ---------------------------------------------------------------------------

def test_is_heartbeat_ok_exact():
    assert is_heartbeat_ok("HEARTBEAT_OK") is True


def test_is_heartbeat_ok_with_whitespace():
    assert is_heartbeat_ok("  HEARTBEAT_OK  ") is True


def test_is_heartbeat_ok_with_newline():
    assert is_heartbeat_ok("HEARTBEAT_OK\n") is True


def test_is_heartbeat_ok_false_for_empty():
    assert is_heartbeat_ok("") is False


def test_is_heartbeat_ok_false_for_none():
    assert is_heartbeat_ok(None) is False


def test_is_heartbeat_ok_false_for_prose():
    assert is_heartbeat_ok("I found something important!") is False


def test_is_heartbeat_ok_false_for_partial_match():
    # "HEARTBEAT_OK" as part of a longer sentence should also match (startswith)
    assert is_heartbeat_ok("HEARTBEAT_OK — and some extra note") is True


def test_is_heartbeat_ok_case_sensitive():
    # Must be uppercase exactly
    assert is_heartbeat_ok("heartbeat_ok") is False


# ---------------------------------------------------------------------------
# strip_heartbeat_token
# ---------------------------------------------------------------------------

def test_strip_heartbeat_token_removes_standalone_line():
    text = "Some prose\nHEARTBEAT_OK\nMore prose"
    result = strip_heartbeat_token(text)
    assert "HEARTBEAT_OK" not in result
    assert "Some prose" in result
    assert "More prose" in result


def test_strip_heartbeat_token_full_response():
    result = strip_heartbeat_token("HEARTBEAT_OK")
    assert result == ""


def test_strip_heartbeat_token_preserves_prose():
    text = "Nothing to report today."
    result = strip_heartbeat_token(text)
    assert result == text


def test_strip_heartbeat_token_multiline():
    text = "HEARTBEAT_OK\nI checked emails\nHEARTBEAT_OK"
    result = strip_heartbeat_token(text)
    assert "HEARTBEAT_OK" not in result
    assert "I checked emails" in result


def test_strip_heartbeat_token_strips_result():
    text = "\n\nHEARTBEAT_OK\n\n"
    result = strip_heartbeat_token(text)
    assert result == ""

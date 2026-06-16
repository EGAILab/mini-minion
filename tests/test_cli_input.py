"""Tests for cli_input.py — PromptReader, SafeFileHistory, _sanitize_surrogates."""

import pytest
from pathlib import Path

from minion_assist.cli_input import _sanitize_surrogates, SafeFileHistory, PromptReader


# ---------------------------------------------------------------------------
# _sanitize_surrogates
# ---------------------------------------------------------------------------

def test_sanitize_surrogates_preserves_normal_text():
    """Normal ASCII text must pass through unchanged."""
    assert _sanitize_surrogates("hello world") == "hello world"


def test_sanitize_surrogates_preserves_emoji():
    """Valid multi-byte characters (emoji) must survive the round-trip."""
    assert _sanitize_surrogates("hello 🎉") == "hello 🎉"


def test_sanitize_surrogates_replaces_lone_surrogate():
    """A lone high surrogate (invalid UTF-8) must be replaced so it is encodable.

    Python's str.encode('utf-8', errors='replace') replaces unencodable code
    points with b'?' (ASCII 0x3F), so the resulting string contains '?' not
    U+FFFD. The key guarantee is that the surrogate is GONE — not that a
    specific replacement character is present.
    """
    lone = "\ud800"  # lone high surrogate — not valid UTF-8
    result = _sanitize_surrogates(lone)
    assert "\ud800" not in result
    # errors='replace' on encode() produces b'?' for unencodable surrogates.
    assert result == "?"


def test_sanitize_surrogates_replaces_lone_low_surrogate():
    """A lone low surrogate must also be replaced so it no longer appears."""
    lone = "\udc00"
    result = _sanitize_surrogates(lone)
    assert "\udc00" not in result
    assert result == "?"


def test_sanitize_surrogates_preserves_text_around_surrogate():
    """Text around the surrogate must be preserved; only the surrogate is replaced."""
    result = _sanitize_surrogates("before\ud800after")
    assert result == "before?after"


# ---------------------------------------------------------------------------
# SafeFileHistory
# ---------------------------------------------------------------------------

def test_safe_file_history_stores_sanitized_text(tmp_path):
    """Storing normal text must not raise and must create the history file."""
    history_file = tmp_path / "history.txt"
    h = SafeFileHistory(str(history_file))
    h.store_string("hello world")  # must not raise
    assert history_file.exists()


def test_safe_file_history_stores_lone_surrogate_without_raising(tmp_path):
    """Storing a lone surrogate must not raise even though it is invalid UTF-8.

    This is the key Windows-specific bug this class guards against: raw arrow-key
    sequences can produce lone surrogates, which crash FileHistory's UTF-8 write.
    """
    history_file = tmp_path / "history.txt"
    h = SafeFileHistory(str(history_file))
    h.store_string("\ud800hello")  # must not raise even with surrogate
    assert history_file.exists()


def test_safe_file_history_sanitizes_stored_content(tmp_path):
    """The file content written must not contain the raw lone surrogate."""
    history_file = tmp_path / "history.txt"
    h = SafeFileHistory(str(history_file))
    h.store_string("\ud800hello")
    content = history_file.read_text(encoding="utf-8")
    assert "\ud800" not in content
    assert "hello" in content


# ---------------------------------------------------------------------------
# PromptReader
# ---------------------------------------------------------------------------

def test_prompt_reader_creates_parent_directory(tmp_path):
    """PromptReader must create nested parent directories on construction."""
    history_path = tmp_path / "nested" / "dir" / "prompt_history.txt"
    # Parent directories don't exist yet — PromptReader must create them.
    reader = PromptReader(history_path)
    assert history_path.parent.exists()


def test_prompt_reader_stores_history_path(tmp_path):
    """PromptReader.history_path must reflect the path it was given."""
    history_path = tmp_path / "prompt_history.txt"
    reader = PromptReader(history_path)
    assert reader.history_path == history_path


def test_prompt_reader_falls_back_to_input_in_non_tty(tmp_path, monkeypatch):
    """In a non-TTY environment (e.g. tests), PromptReader must use plain input().

    sys.stdin.isatty() returns False in pytest, so PromptReader.read() should
    call input("\nYou: ") rather than the prompt_toolkit session.
    """
    history_path = tmp_path / "prompt_history.txt"
    reader = PromptReader(history_path)

    # Patch builtins.input to capture the call and return a controlled value.
    called_with: list[str] = []

    def fake_input(prompt: str = "") -> str:
        called_with.append(prompt)
        return "test message"

    monkeypatch.setattr("builtins.input", fake_input)
    result = reader.read()

    assert result == "test message"
    assert called_with == ["\nYou: "]

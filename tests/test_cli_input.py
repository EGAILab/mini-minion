"""Tests for cli_input.py -- PromptReader, SafeFileHistory, and completions."""

import pytest

from minion_assist.cli_input import (
    PromptReader,
    SafeFileHistory,
    SlashCompleter,
    _sanitize_surrogates,
)


# ---------------------------------------------------------------------------
# _sanitize_surrogates
# ---------------------------------------------------------------------------

def test_sanitize_surrogates_preserves_normal_text():
    """Normal ASCII text must pass through unchanged."""
    assert _sanitize_surrogates("hello world") == "hello world"


def test_sanitize_surrogates_preserves_emoji():
    """Valid multi-byte characters must survive the round-trip."""
    assert _sanitize_surrogates("hello 🎉") == "hello 🎉"


def test_sanitize_surrogates_replaces_lone_surrogate():
    """A lone high surrogate must be replaced so it is encodable."""
    lone = "\ud800"
    result = _sanitize_surrogates(lone)
    assert "\ud800" not in result
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
    h.store_string("hello world")
    assert history_file.exists()


def test_safe_file_history_stores_lone_surrogate_without_raising(tmp_path):
    """Storing a lone surrogate must not raise even though it is invalid UTF-8."""
    history_file = tmp_path / "history.txt"
    h = SafeFileHistory(str(history_file))
    h.store_string("\ud800hello")
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
    reader = PromptReader(history_path)
    assert history_path.parent.exists()
    assert reader.history_path == history_path


def test_prompt_reader_stores_history_path(tmp_path):
    """PromptReader.history_path must reflect the path it was given."""
    history_path = tmp_path / "prompt_history.txt"
    reader = PromptReader(history_path)
    assert reader.history_path == history_path


def test_prompt_reader_falls_back_to_input_in_non_tty(tmp_path, monkeypatch):
    """In a non-TTY environment, PromptReader must use plain input()."""
    history_path = tmp_path / "prompt_history.txt"
    reader = PromptReader(history_path)

    called_with: list[str] = []

    def fake_input(prompt: str = "") -> str:
        called_with.append(prompt)
        return "test message"

    monkeypatch.setattr("builtins.input", fake_input)
    result = reader.read()

    assert result == "test message"
    assert called_with == ["\nYou: "]


# ---------------------------------------------------------------------------
# SlashCompleter
# ---------------------------------------------------------------------------

def test_slash_completer_shows_items_for_bare_slash():
    prompt_toolkit = pytest.importorskip("prompt_toolkit")
    document = prompt_toolkit.document.Document("/")
    completer = SlashCompleter([
        ("/help", "Show help"),
        ("/research", "route to researcher"),
        ("/skills interview", "Interview preparation"),
    ])

    completions = list(completer.get_completions(document, None))
    values = {c.text for c in completions}

    assert "/help" in values
    assert "/research" in values
    assert "/skills interview" in values


def test_slash_completer_filters_by_prefix():
    prompt_toolkit = pytest.importorskip("prompt_toolkit")
    document = prompt_toolkit.document.Document("/he")
    completer = SlashCompleter([
        ("/help", "Show help"),
        ("/research", "route to researcher"),
    ])

    completions = list(completer.get_completions(document, None))

    assert [c.text for c in completions] == ["/help"]

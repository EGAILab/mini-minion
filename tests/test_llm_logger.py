"""Tests for the LLM request/response logger (llm_logger.py)."""

import json
import re
from datetime import date
from pathlib import Path

import pytest

from minion_assist.llm_logger import _log_file, _now, log_request, log_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
_REQUEST_PREFIX = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\[DEBUG\] Received request: POST to ")
_RESPONSE_PREFIX = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\[INFO\]\[")


def _read_log(log_dir: Path) -> str:
    """Return the full content of today's log file."""
    return _log_file(log_dir).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# _now()
# ---------------------------------------------------------------------------

def test_now_format():
    ts = _now()
    assert _TS_RE.match(ts), f"unexpected timestamp format: {ts!r}"


# ---------------------------------------------------------------------------
# _log_file()
# ---------------------------------------------------------------------------

def test_log_file_path(tmp_path):
    lf = _log_file(tmp_path)
    assert lf.parent == tmp_path
    assert lf.name == f"{date.today().isoformat()}.log"


# ---------------------------------------------------------------------------
# log_request()
# ---------------------------------------------------------------------------

def test_log_request_creates_dir(tmp_path):
    log_dir = tmp_path / "logs"
    assert not log_dir.exists()
    log_request(log_dir, "http://127.0.0.1:1234/v1/chat/completions", {"model": "m"})
    assert log_dir.exists()


def test_log_request_creates_file(tmp_path):
    log_request(tmp_path, "http://example.com/chat", {"model": "m"})
    assert _log_file(tmp_path).exists()


def test_log_request_line_format(tmp_path):
    log_request(tmp_path, "http://example.com/v1/chat/completions", {"model": "gpt-4"})
    content = _read_log(tmp_path)
    assert _REQUEST_PREFIX.match(content)


def test_log_request_endpoint_in_line(tmp_path):
    endpoint = "http://127.0.0.1:1234/v1/chat/completions"
    log_request(tmp_path, endpoint, {"x": 1})
    assert endpoint in _read_log(tmp_path)


def test_log_request_body_is_pretty_json(tmp_path):
    body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
    log_request(tmp_path, "http://x/chat", body)
    content = _read_log(tmp_path)
    # Extract the JSON portion after "with body "
    idx = content.index("with body ") + len("with body ")
    parsed = json.loads(content[idx:].rstrip())
    assert parsed == body


def test_log_request_appends_multiple(tmp_path):
    log_request(tmp_path, "http://x/chat", {"turn": 1})
    log_request(tmp_path, "http://x/chat", {"turn": 2})
    content = _read_log(tmp_path)
    # Each entry starts with a timestamp header line; count them.
    assert content.count("[DEBUG] Received request:") == 2


def test_log_request_body_contains_messages(tmp_path):
    body = {"messages": [{"role": "system", "content": "You are helpful."}], "model": "m"}
    log_request(tmp_path, "http://x/chat", body)
    content = _read_log(tmp_path)
    assert "You are helpful." in content


def test_log_request_no_crash_on_bad_dir(tmp_path):
    # Read-only parent should not raise — errors are swallowed.
    bad_dir = tmp_path / "a" / "b" / "c"
    # We can't easily make a dir read-only on Windows, but we can pass a
    # path inside a file (not a directory) which will fail silently.
    fake_parent = tmp_path / "file.txt"
    fake_parent.write_text("x")
    log_request(fake_parent / "logs", "http://x", {})  # should not raise


def test_log_request_handles_non_ascii(tmp_path):
    log_request(tmp_path, "http://x", {"msg": "こんにちは"})
    content = _read_log(tmp_path)
    assert "こんにちは" in content


def test_log_request_default_serializes_unserializable(tmp_path):
    # Non-serializable values should not crash (default=str is used).
    from pathlib import PurePosixPath
    body = {"path": PurePosixPath("/some/path")}
    log_request(tmp_path, "http://x", body)
    content = _read_log(tmp_path)
    assert "/some/path" in content


# ---------------------------------------------------------------------------
# log_response()
# ---------------------------------------------------------------------------

def test_log_response_creates_file(tmp_path):
    log_response(tmp_path, "gpt-4", {"choices": []})
    assert _log_file(tmp_path).exists()


def test_log_response_line_format(tmp_path):
    log_response(tmp_path, "qwen3.5-9b", {"choices": []})
    content = _read_log(tmp_path)
    assert _RESPONSE_PREFIX.match(content)


def test_log_response_model_in_line(tmp_path):
    log_response(tmp_path, "my-model-v2", {"output": "hello"})
    content = _read_log(tmp_path)
    assert "[my-model-v2]" in content


def test_log_response_body_is_pretty_json(tmp_path):
    resp = {"id": "chatcmpl-123", "model": "gpt-4", "choices": [{"finish_reason": "stop"}]}
    log_response(tmp_path, "gpt-4", resp)
    content = _read_log(tmp_path)
    idx = content.index("Generated prediction: ") + len("Generated prediction: ")
    parsed = json.loads(content[idx:].rstrip())
    assert parsed == resp


def test_log_response_appends_after_request(tmp_path):
    log_request(tmp_path, "http://x/chat", {"turn": 1})
    log_response(tmp_path, "model", {"out": "ok"})
    content = _read_log(tmp_path)
    assert "[DEBUG]" in content
    assert "[INFO]" in content
    assert content.index("[DEBUG]") < content.index("[INFO]")


def test_log_response_no_crash_on_bad_dir(tmp_path):
    fake_parent = tmp_path / "file.txt"
    fake_parent.write_text("x")
    log_response(fake_parent / "logs", "model", {})  # should not raise


def test_log_response_handles_non_ascii(tmp_path):
    log_response(tmp_path, "model", {"content": "日本語テスト"})
    content = _read_log(tmp_path)
    assert "日本語テスト" in content


# ---------------------------------------------------------------------------
# End-to-end: request then response in same file
# ---------------------------------------------------------------------------

def test_full_turn_interleaving(tmp_path):
    body = {"model": "m", "messages": [{"role": "user", "content": "hello"}]}
    resp = {"id": "c-1", "model": "m", "choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    log_request(tmp_path, "http://127.0.0.1:1234/v1/chat/completions", body)
    log_response(tmp_path, "m", resp)
    content = _read_log(tmp_path)
    assert "Received request:" in content
    assert "Generated prediction:" in content
    assert "hello" in content
    assert "hi" in content

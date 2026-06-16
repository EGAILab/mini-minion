"""Tests for WebFetchTool."""

from unittest.mock import MagicMock, patch

import pytest

from minion_assist.tools.policy import PermissionPolicy
from minion_assist.tools.web_fetch import WebFetchTool, _extract_text


# ---------------------------------------------------------------------------
# _extract_text helper
# ---------------------------------------------------------------------------

def test_extract_text_strips_tags():
    html = "<html><body><p>Hello <b>world</b></p></body></html>"
    assert "Hello" in _extract_text(html)
    assert "<" not in _extract_text(html)


def test_extract_text_skips_scripts():
    html = "<html><body><p>Content</p><script>var x=1;</script></body></html>"
    text = _extract_text(html)
    assert "Content" in text
    assert "var x" not in text


def test_extract_text_skips_style():
    html = "<html><head><style>body{color:red}</style></head><body>Text</body></html>"
    text = _extract_text(html)
    assert "Text" in text
    assert "color:red" not in text


def test_extract_text_decodes_entities():
    html = "<p>a &amp; b &lt;c&gt;</p>"
    text = _extract_text(html)
    assert "a & b" in text


# ---------------------------------------------------------------------------
# SSRF blocking
# ---------------------------------------------------------------------------

def test_web_fetch_blocks_metadata_endpoint():
    tool = WebFetchTool()
    result = tool.execute(url="http://169.254.169.254/latest/meta-data/")
    assert "Error" in result
    assert "blocked" in result.lower()


def test_web_fetch_blocks_gcp_metadata():
    tool = WebFetchTool()
    result = tool.execute(url="http://metadata.google.internal/")
    assert "Error" in result


# ---------------------------------------------------------------------------
# Successful fetch (mocked)
# ---------------------------------------------------------------------------

def _mock_response(text: str, content_type: str = "text/html") -> MagicMock:
    resp = MagicMock()
    resp.text = text
    resp.headers = {"content-type": content_type}
    resp.raise_for_status = MagicMock()
    return resp


def test_web_fetch_returns_text_for_html(monkeypatch):
    mock_get = MagicMock(return_value=_mock_response("<p>Hello from web</p>"))
    monkeypatch.setattr("httpx.get", mock_get)
    result = WebFetchTool().execute(url="https://example.com")
    assert "Hello from web" in result


def test_web_fetch_truncates_at_max_chars(monkeypatch):
    long_html = "<p>" + "a" * 20_000 + "</p>"
    monkeypatch.setattr("httpx.get", MagicMock(return_value=_mock_response(long_html)))
    result = WebFetchTool().execute(url="https://example.com", max_chars=100)
    assert "truncated" in result.lower()
    assert len(result) < 300  # well under original size


def test_web_fetch_plain_text_not_stripped(monkeypatch):
    resp = _mock_response("raw text content", content_type="text/plain")
    monkeypatch.setattr("httpx.get", MagicMock(return_value=resp))
    result = WebFetchTool().execute(url="https://example.com/file.txt")
    assert "raw text content" in result


def test_web_fetch_http_error_returns_error_string(monkeypatch):
    import httpx

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    exc = httpx.HTTPStatusError("not found", request=MagicMock(), response=mock_resp)
    mock_get = MagicMock(side_effect=exc)
    monkeypatch.setattr("httpx.get", mock_get)
    result = WebFetchTool().execute(url="https://example.com/missing")
    assert "Error" in result
    assert "404" in result


def test_web_fetch_request_error_returns_error_string(monkeypatch):
    import httpx

    monkeypatch.setattr(
        "httpx.get",
        MagicMock(side_effect=httpx.RequestError("connection refused")),
    )
    result = WebFetchTool().execute(url="https://unreachable.example.com")
    assert "Error" in result


# ---------------------------------------------------------------------------
# Custom policy
# ---------------------------------------------------------------------------

def test_web_fetch_custom_policy_blocks_custom_marker(monkeypatch):
    policy = PermissionPolicy(ssrf_markers=frozenset({"internal.corp"}))
    tool = WebFetchTool(policy=policy)
    result = tool.execute(url="https://internal.corp/secret")
    assert "Error" in result
    assert "blocked" in result.lower()

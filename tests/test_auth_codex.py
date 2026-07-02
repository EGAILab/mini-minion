"""Tests for the Codex device-code OAuth module (no real HTTP calls)."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import minion_assist.auth.codex_auth as _mod
from minion_assist.auth.codex_auth import (
    _decode_jwt_payload,
    _extract_identity,
    _token_expiry_from_jwt,
    load_raw,
    load_token,
    login,
    save_token,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jwt(payload: dict) -> str:
    """Build a fake (unsigned) JWT with the given payload for testing."""
    header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"{header}.{body}.fakesignature"


_FUTURE_EXPIRY = int(time.time()) + 7200   # 2 hours from now


# ---------------------------------------------------------------------------
# JWT decoding
# ---------------------------------------------------------------------------


def test_decode_jwt_payload_valid():
    payload = {"sub": "u-1", "exp": 9_999_999_999}
    token = _make_jwt(payload)
    result = _decode_jwt_payload(token)
    assert result["sub"] == "u-1"
    assert result["exp"] == 9_999_999_999


def test_decode_jwt_payload_invalid_part_count():
    assert _decode_jwt_payload("only_two.parts") == {}
    assert _decode_jwt_payload("no-dots") == {}
    assert _decode_jwt_payload("") == {}


def test_decode_jwt_payload_bad_base64():
    assert _decode_jwt_payload("h.!!!.s") == {}


def test_decode_jwt_payload_non_object():
    # Body encodes a JSON string, not an object
    body = base64.urlsafe_b64encode(b'"just a string"').rstrip(b"=").decode()
    token = f"h.{body}.s"
    assert _decode_jwt_payload(token) == {}


# ---------------------------------------------------------------------------
# Identity extraction
# ---------------------------------------------------------------------------


def test_extract_identity_full_claims():
    payload = {
        "https://api.openai.com/auth": {"chatgpt_account_id": "acct-abc"},
        "https://api.openai.com/profile": {"email": "user@example.com"},
    }
    account_id, email = _extract_identity(_make_jwt(payload))
    assert account_id == "acct-abc"
    assert email == "user@example.com"


def test_extract_identity_no_account_id_falls_back_to_iss_sub():
    payload = {
        "iss": "https://auth.openai.com",
        "sub": "user-xyz",
    }
    account_id, email = _extract_identity(_make_jwt(payload))
    assert account_id == "https://auth.openai.com|user-xyz"
    assert email is None


def test_extract_identity_sub_only():
    account_id, email = _extract_identity(_make_jwt({"sub": "plain-sub"}))
    assert account_id == "plain-sub"
    assert email is None


def test_extract_identity_empty_token():
    account_id, email = _extract_identity("not.a.jwt")
    assert account_id == ""
    assert email is None


def test_extract_identity_strips_whitespace():
    payload = {
        "https://api.openai.com/auth": {"chatgpt_account_id": "  acct-1  "},
        "https://api.openai.com/profile": {"email": "  hi@x.com  "},
    }
    account_id, email = _extract_identity(_make_jwt(payload))
    assert account_id == "acct-1"
    assert email == "hi@x.com"


# ---------------------------------------------------------------------------
# Expiry extraction
# ---------------------------------------------------------------------------


def test_token_expiry_from_jwt_present():
    token = _make_jwt({"exp": 1_800_000_000})
    assert _token_expiry_from_jwt(token) == 1_800_000_000.0


def test_token_expiry_from_jwt_missing():
    assert _token_expiry_from_jwt(_make_jwt({})) is None


def test_token_expiry_from_jwt_negative():
    assert _token_expiry_from_jwt(_make_jwt({"exp": -1})) is None


# ---------------------------------------------------------------------------
# save_token / load_raw
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("MINION_ASSIST_HOME", str(tmp_path))
    save_token("acc", "ref", 9_999.0, "acct-1", "user@x.com")
    raw = load_raw()
    assert raw == {
        "access_token": "acc",
        "refresh_token": "ref",
        "expires_at": 9_999.0,
        "account_id": "acct-1",
        "email": "user@x.com",
    }


def test_save_token_omits_email_when_none(tmp_path, monkeypatch):
    monkeypatch.setenv("MINION_ASSIST_HOME", str(tmp_path))
    save_token("acc", "ref", 9_999.0, "acct-1")
    raw = load_raw()
    assert "email" not in raw


def test_load_raw_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("MINION_ASSIST_HOME", str(tmp_path))
    assert load_raw() is None


def test_load_raw_returns_none_on_corrupted_json(tmp_path, monkeypatch):
    monkeypatch.setenv("MINION_ASSIST_HOME", str(tmp_path))
    (tmp_path / "codex-auth.json").write_text("not json", encoding="utf-8")
    assert load_raw() is None


def test_token_path_creates_directory(tmp_path, monkeypatch):
    nested = tmp_path / "a" / "b" / "c"
    monkeypatch.setenv("MINION_ASSIST_HOME", str(nested))
    save_token("a", "b", 0.0, "x")
    assert (nested / "codex-auth.json").exists()


# ---------------------------------------------------------------------------
# load_token — auto-refresh
# ---------------------------------------------------------------------------


def test_load_token_valid_not_near_expiry(tmp_path, monkeypatch):
    monkeypatch.setenv("MINION_ASSIST_HOME", str(tmp_path))
    save_token("acc", "ref", float(_FUTURE_EXPIRY), "acct-1")
    token = load_token()
    assert token is not None
    assert token["access_token"] == "acc"


def test_load_token_returns_none_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MINION_ASSIST_HOME", str(tmp_path))
    assert load_token() is None


def test_load_token_refreshes_when_near_expiry(tmp_path, monkeypatch):
    monkeypatch.setenv("MINION_ASSIST_HOME", str(tmp_path))
    # Expires in 2 min → below 5-min refresh buffer
    save_token("old-acc", "old-ref", time.time() + 120, "acct-1", "u@x.com")

    new_jwt = _make_jwt({
        "https://api.openai.com/auth": {"chatgpt_account_id": "acct-1"},
        "https://api.openai.com/profile": {"email": "u@x.com"},
        "exp": _FUTURE_EXPIRY,
    })
    refresh_resp = {"access_token": new_jwt, "refresh_token": "new-ref", "expires_in": 3600}

    with patch("minion_assist.auth.codex_auth._post_form", return_value=refresh_resp):
        token = load_token()

    assert token is not None
    assert token["access_token"] == new_jwt
    assert token["refresh_token"] == "new-ref"


def test_load_token_returns_old_when_refresh_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("MINION_ASSIST_HOME", str(tmp_path))
    save_token("old-acc", "old-ref", time.time() + 120, "acct-1")

    with patch("minion_assist.auth.codex_auth._post_form", side_effect=OSError("network")):
        token = load_token()

    assert token is not None
    assert token["access_token"] == "old-acc"


def test_load_token_refresh_empty_response_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("MINION_ASSIST_HOME", str(tmp_path))
    save_token("old", "ref", time.time() + 60, "acct-1")

    with patch("minion_assist.auth.codex_auth._post_form", return_value={}):
        token = load_token()

    assert token["access_token"] == "old"


# ---------------------------------------------------------------------------
# login() — happy path
# ---------------------------------------------------------------------------

def _stub_login_mocks(user_code: str = "WXYZ-1234", device_auth_id: str = "dev-id-1") -> tuple:
    """Return (post_json_mock, post_form_mock) for a successful two-poll login."""
    jwt = _make_jwt({
        "https://api.openai.com/auth": {"chatgpt_account_id": "acct-99"},
        "https://api.openai.com/profile": {"email": "test@example.com"},
        "exp": _FUTURE_EXPIRY,
    })
    # _post_json calls: usercode, pending poll (empty), authorized poll
    post_json = MagicMock(side_effect=[
        {"device_auth_id": device_auth_id, "user_code": user_code, "interval": 0},
        {},
        {"authorization_code": "auth-code", "code_verifier": "verifier"},
    ])
    # _post_form call: token exchange
    post_form = MagicMock(return_value={
        "access_token": jwt,
        "refresh_token": "refresh-1",
        "expires_in": 3600,
    })
    return post_json, post_form, jwt


def test_login_success(tmp_path, monkeypatch):
    monkeypatch.setenv("MINION_ASSIST_HOME", str(tmp_path))
    post_json, post_form, jwt = _stub_login_mocks()

    with (
        patch("minion_assist.auth.codex_auth._post_json", post_json),
        patch("minion_assist.auth.codex_auth._post_form", post_form),
        patch("minion_assist.auth.codex_auth.time.sleep"),
        patch("minion_assist.auth.codex_auth.webbrowser.open"),
    ):
        result = login(open_browser=False)

    assert result["access_token"] == jwt
    assert result["account_id"] == "acct-99"
    assert result.get("email") == "test@example.com"
    assert result["refresh_token"] == "refresh-1"


def test_login_token_persisted(tmp_path, monkeypatch):
    monkeypatch.setenv("MINION_ASSIST_HOME", str(tmp_path))
    post_json, post_form, _ = _stub_login_mocks()

    with (
        patch("minion_assist.auth.codex_auth._post_json", post_json),
        patch("minion_assist.auth.codex_auth._post_form", post_form),
        patch("minion_assist.auth.codex_auth.time.sleep"),
        patch("minion_assist.auth.codex_auth.webbrowser.open"),
    ):
        login(open_browser=False)

    # Token file should now exist
    assert (tmp_path / "codex-auth.json").exists()
    raw = load_raw()
    assert raw is not None
    assert raw["account_id"] == "acct-99"


def test_login_opens_browser(tmp_path, monkeypatch):
    monkeypatch.setenv("MINION_ASSIST_HOME", str(tmp_path))
    post_json, post_form, _ = _stub_login_mocks()
    browser = MagicMock()

    with (
        patch("minion_assist.auth.codex_auth._post_json", post_json),
        patch("minion_assist.auth.codex_auth._post_form", post_form),
        patch("minion_assist.auth.codex_auth.time.sleep"),
        patch("minion_assist.auth.codex_auth.webbrowser.open", browser),
    ):
        login(open_browser=True)

    browser.assert_called_once()
    args = browser.call_args[0][0]
    assert "codex/device" in args


def test_login_skips_browser_when_false(tmp_path, monkeypatch):
    monkeypatch.setenv("MINION_ASSIST_HOME", str(tmp_path))
    post_json, post_form, _ = _stub_login_mocks()
    browser = MagicMock()

    with (
        patch("minion_assist.auth.codex_auth._post_json", post_json),
        patch("minion_assist.auth.codex_auth._post_form", post_form),
        patch("minion_assist.auth.codex_auth.time.sleep"),
        patch("minion_assist.auth.codex_auth.webbrowser.open", browser),
    ):
        login(open_browser=False)

    browser.assert_not_called()


def test_login_multiple_pending_polls_before_success(tmp_path, monkeypatch):
    monkeypatch.setenv("MINION_ASSIST_HOME", str(tmp_path))
    jwt = _make_jwt({"exp": _FUTURE_EXPIRY})
    post_json = MagicMock(side_effect=[
        {"device_auth_id": "d", "user_code": "ABCD", "interval": 0},
        {},
        {},
        {},
        {"authorization_code": "code", "code_verifier": "ver"},
    ])
    post_form = MagicMock(return_value={
        "access_token": jwt, "refresh_token": "ref", "expires_in": 3600,
    })

    with (
        patch("minion_assist.auth.codex_auth._post_json", post_json),
        patch("minion_assist.auth.codex_auth._post_form", post_form),
        patch("minion_assist.auth.codex_auth.time.sleep"),
        patch("minion_assist.auth.codex_auth.webbrowser.open"),
    ):
        result = login(open_browser=False)

    assert result["access_token"] == jwt


# ---------------------------------------------------------------------------
# login() — error paths
# ---------------------------------------------------------------------------


def test_login_raises_on_missing_device_code(tmp_path, monkeypatch):
    monkeypatch.setenv("MINION_ASSIST_HOME", str(tmp_path))
    with (
        patch("minion_assist.auth.codex_auth._post_json", return_value={}),
        patch("minion_assist.auth.codex_auth.time.sleep"),
    ):
        with pytest.raises(RuntimeError, match="Unexpected device code response"):
            login(open_browser=False)


def test_login_raises_when_token_exchange_returns_no_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("MINION_ASSIST_HOME", str(tmp_path))
    post_json = MagicMock(side_effect=[
        {"device_auth_id": "d", "user_code": "ABCD", "interval": 0},
        {"authorization_code": "code", "code_verifier": "ver"},
    ])
    with (
        patch("minion_assist.auth.codex_auth._post_json", post_json),
        patch("minion_assist.auth.codex_auth._post_form", return_value={}),
        patch("minion_assist.auth.codex_auth.time.sleep"),
        patch("minion_assist.auth.codex_auth.webbrowser.open"),
    ):
        with pytest.raises(RuntimeError, match="Token exchange failed"):
            login(open_browser=False)


def test_login_timeout(tmp_path, monkeypatch):
    """When deadline expires before the user approves, TimeoutError is raised."""
    monkeypatch.setenv("MINION_ASSIST_HOME", str(tmp_path))
    post_json = MagicMock(side_effect=[
        {"device_auth_id": "d", "user_code": "ABCD", "interval": 0},
        # Poll responses would follow, but deadline expires before any are reached.
        *[{} for _ in range(50)],
    ])

    # Make monotonic() jump past the deadline on the while-condition check.
    # Call sequence: (1) deadline = monotonic() + timeout → returns 0 → deadline = timeout
    #                (2) while monotonic() < deadline → returns timeout+1 → False → exit
    monotonic_vals = iter([0.0, float(_mod._DEVICE_TIMEOUT_S) + 1.0])

    with (
        patch("minion_assist.auth.codex_auth._post_json", post_json),
        patch("minion_assist.auth.codex_auth.time") as mock_time,
        patch("minion_assist.auth.codex_auth.webbrowser.open"),
    ):
        mock_time.sleep = MagicMock()
        mock_time.monotonic = MagicMock(side_effect=monotonic_vals)
        mock_time.time = time.time

        with pytest.raises(TimeoutError, match="timed out"):
            login(open_browser=False)


# ---------------------------------------------------------------------------
# CodexProvider._inject_auth — integration
# ---------------------------------------------------------------------------


def _make_stub_rpc() -> MagicMock:
    stub = MagicMock()
    stub.request = MagicMock(return_value={})
    return stub


def test_inject_auth_calls_login_start_when_token_present(tmp_path, monkeypatch):
    monkeypatch.setenv("MINION_ASSIST_HOME", str(tmp_path))
    save_token("acc-tok", "ref-tok", float(_FUTURE_EXPIRY), "acct-1", "u@x.com")

    from minion_assist.providers.codex import CodexProvider
    p = object.__new__(CodexProvider)
    rpc = _make_stub_rpc()
    p._inject_auth(rpc)

    rpc.request.assert_called_once()
    call_args = rpc.request.call_args
    assert call_args[0][0] == "account/login/start"
    params = call_args[0][1]
    assert params["type"] == "chatgptAuthTokens"
    assert params["accessToken"] == "acc-tok"
    assert params["chatgptAccountId"] == "acct-1"
    assert params["chatgptPlanType"] is None


def test_inject_auth_skips_when_no_token(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MINION_ASSIST_HOME", str(tmp_path))

    from minion_assist.providers.codex import CodexProvider
    p = object.__new__(CodexProvider)
    rpc = _make_stub_rpc()
    p._inject_auth(rpc)

    rpc.request.assert_not_called()
    captured = capsys.readouterr()
    assert "codex-login" in captured.err


def test_inject_auth_continues_on_rpc_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MINION_ASSIST_HOME", str(tmp_path))
    save_token("acc", "ref", float(_FUTURE_EXPIRY), "acct-1")

    from minion_assist.providers.codex import CodexProvider
    p = object.__new__(CodexProvider)
    rpc = _make_stub_rpc()
    rpc.request.side_effect = RuntimeError("method not found")
    p._inject_auth(rpc)  # must not raise

    captured = capsys.readouterr()
    assert "account/login/start failed" in captured.err

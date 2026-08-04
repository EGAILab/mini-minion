"""Tests for matrix/auth.py — resolve_matrix_auth().

matrix-nio is mocked entirely (injected into sys.modules) so no live server
or nio installation is required.
"""

import asyncio
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import pytest

from minion_assist.matrix.config import MatrixConfig


# ---------------------------------------------------------------------------
# Helpers to inject a fake 'nio' module before the lazy import fires
# ---------------------------------------------------------------------------

def _make_mock_nio(login_success=True):
    """Return a fake 'nio' module with AsyncClient."""
    nio = ModuleType("nio")

    mock_client = MagicMock()
    mock_client.close = AsyncMock()
    mock_client.access_token = None

    if login_success:
        login_resp = MagicMock()
        login_resp.access_token = "syt_fresh"
        mock_client.login = AsyncMock(return_value=login_resp)
    else:
        login_resp = MagicMock(spec=[])
        login_resp.message = "M_FORBIDDEN"
        mock_client.login = AsyncMock(return_value=login_resp)

    nio.AsyncClient = MagicMock(return_value=mock_client)
    return nio, mock_client


def _config(access_token=None, password=None, user_id="@bot:ex.org", homeserver="https://ex.org"):
    raw = {"homeserver": homeserver, "userId": user_id}
    if access_token:
        raw["accessToken"] = access_token
    if password:
        raw["password"] = password
    return MatrixConfig.from_dict(raw)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_access_token_skips_login():
    """When an access token is present, login() should NOT be called."""
    nio, mock_client = _make_mock_nio()
    sys.modules["nio"] = nio

    try:
        # Force re-import of auth so the patched nio is picked up
        if "minion_assist.matrix.auth" in sys.modules:
            del sys.modules["minion_assist.matrix.auth"]
        from minion_assist.matrix.auth import resolve_matrix_auth

        cfg = _config(access_token="syt_abc123")
        result = _run(resolve_matrix_auth(cfg))

        mock_client.login.assert_not_called()
        assert result.access_token == "syt_abc123"
    finally:
        del sys.modules["nio"]
        sys.modules.pop("minion_assist.matrix.auth", None)


def test_password_calls_login():
    """When only a password is provided, login() should be called."""
    nio, mock_client = _make_mock_nio(login_success=True)
    sys.modules["nio"] = nio

    try:
        sys.modules.pop("minion_assist.matrix.auth", None)
        from minion_assist.matrix.auth import resolve_matrix_auth

        cfg = _config(password="s3cr3t")
        _run(resolve_matrix_auth(cfg))
        mock_client.login.assert_called_once()
    finally:
        del sys.modules["nio"]
        sys.modules.pop("minion_assist.matrix.auth", None)


def test_password_login_failure_raises():
    """Failed password login should raise RuntimeError."""
    nio, mock_client = _make_mock_nio(login_success=False)
    sys.modules["nio"] = nio

    try:
        sys.modules.pop("minion_assist.matrix.auth", None)
        from minion_assist.matrix.auth import resolve_matrix_auth

        cfg = _config(password="wrongpassword")
        with pytest.raises(RuntimeError, match="login failed"):
            _run(resolve_matrix_auth(cfg))
    finally:
        del sys.modules["nio"]
        sys.modules.pop("minion_assist.matrix.auth", None)


def test_no_auth_method_raises_at_config():
    """If neither access token nor password is set, from_dict should raise ValueError."""
    with pytest.raises(ValueError, match="accessToken|password"):
        _config()  # no credentials


def test_matrix_nio_not_installed_raises():
    """If matrix-nio is not installed, resolve_matrix_auth should raise RuntimeError."""
    from minion_assist.matrix.auth import resolve_matrix_auth

    cfg = _config(access_token="syt_abc")

    # Block the import by setting the module sentinel to None.
    # This causes `import nio` inside resolve_matrix_auth to raise ImportError
    # regardless of whether the package is installed on disk.
    original_nio = sys.modules.get("nio", ...)
    sys.modules["nio"] = None  # type: ignore[assignment]
    try:
        with pytest.raises((RuntimeError, ImportError)):
            _run(resolve_matrix_auth(cfg))
    finally:
        if original_nio is ...:
            sys.modules.pop("nio", None)
        else:
            sys.modules["nio"] = original_nio

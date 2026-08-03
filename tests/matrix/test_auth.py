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

def _make_mock_nio(login_success=True, encryption_import_warning=False):
    """Return a fake 'nio' module with AsyncClient and AsyncClientConfig.

    Args:
        login_success: Whether the mocked password login() succeeds.
        encryption_import_warning: When True, AsyncClientConfig raises
            ImportWarning whenever encryption_enabled=True is requested —
            simulating python-olm/libolm not being installed, the same way
            nio's real AsyncClientConfig.__post_init__ behaves.
    """
    nio = ModuleType("nio")

    mock_client = MagicMock()
    mock_client.close = AsyncMock()
    mock_client.access_token = None
    mock_client.olm = None
    mock_client.whoami = AsyncMock()

    if login_success:
        login_resp = MagicMock()
        login_resp.access_token = "syt_fresh"
        mock_client.login = AsyncMock(return_value=login_resp)
    else:
        login_resp = MagicMock(spec=[])
        login_resp.message = "M_FORBIDDEN"
        mock_client.login = AsyncMock(return_value=login_resp)

    nio.AsyncClient = MagicMock(return_value=mock_client)

    def _make_client_config(*, encryption_enabled=False, **kwargs):
        if encryption_enabled and encryption_import_warning:
            raise ImportWarning("libolm not installed")
        cfg = MagicMock()
        cfg.encryption_enabled = encryption_enabled
        return cfg

    nio.AsyncClientConfig = MagicMock(side_effect=_make_client_config)
    return nio, mock_client


def _config(
    access_token=None,
    password=None,
    user_id="@bot:ex.org",
    homeserver="https://ex.org",
    device_id=None,
    encryption=False,
):
    raw = {"homeserver": homeserver, "userId": user_id}
    if access_token:
        raw["accessToken"] = access_token
    if password:
        raw["password"] = password
    if device_id:
        raw["deviceId"] = device_id
    if encryption:
        raw["encryption"] = True
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


def test_encryption_with_configured_device_id_calls_restore_login(tmp_path):
    """Token auth + encryption + a configured deviceId should call
    client.restore_login() directly (the call that actually wires up
    matrix-nio's crypto store/client.olm) without needing to look up the
    device_id via whoami().
    """
    nio, mock_client = _make_mock_nio()
    sys.modules["nio"] = nio

    try:
        sys.modules.pop("minion_assist.matrix.auth", None)
        from minion_assist.matrix.auth import resolve_matrix_auth

        cfg = _config(access_token="syt_abc123", device_id="DEVICEXYZ", encryption=True)
        _run(resolve_matrix_auth(cfg, crypto_store_dir=tmp_path / "crypto"))

        mock_client.whoami.assert_not_called()
        mock_client.restore_login.assert_called_once_with(
            user_id="@bot:ex.org", device_id="DEVICEXYZ", access_token="syt_abc123"
        )
        # The crypto store directory must exist before AsyncClient is constructed.
        assert (tmp_path / "crypto").is_dir()
        # store_name must have the ':' stripped so the crypto DB doesn't collide
        # with Windows NTFS alternate-data-stream syntax (see auth.py comment).
        nio.AsyncClientConfig.assert_called_with(
            encryption_enabled=True, store_sync_tokens=True, store_name="@bot_ex.org.db"
        )
    finally:
        del sys.modules["nio"]
        sys.modules.pop("minion_assist.matrix.auth", None)


def test_encryption_without_device_id_discovers_via_whoami(tmp_path):
    """Token auth + encryption with no deviceId configured should call
    whoami() to discover the real device_id the token was issued for, then
    pass that (not a blank/invented one) to restore_login().
    """
    nio, mock_client = _make_mock_nio()
    sys.modules["nio"] = nio

    try:
        sys.modules.pop("minion_assist.matrix.auth", None)
        from minion_assist.matrix.auth import resolve_matrix_auth

        whoami_resp = MagicMock()
        whoami_resp.device_id = "SERVERDEVICE"
        whoami_resp.user_id = "@bot:ex.org"
        mock_client.whoami = AsyncMock(return_value=whoami_resp)

        cfg = _config(access_token="syt_abc123", encryption=True)
        _run(resolve_matrix_auth(cfg, crypto_store_dir=tmp_path / "crypto"))

        mock_client.whoami.assert_called_once()
        mock_client.restore_login.assert_called_once_with(
            user_id="@bot:ex.org", device_id="SERVERDEVICE", access_token="syt_abc123"
        )
    finally:
        del sys.modules["nio"]
        sys.modules.pop("minion_assist.matrix.auth", None)


def test_encryption_requested_but_libolm_missing_falls_back(tmp_path):
    """If AsyncClientConfig(encryption_enabled=True) raises ImportWarning
    (libolm/python-olm missing), auth should fall back to the pre-existing
    unencrypted attribute-assignment path instead of crashing.
    """
    nio, mock_client = _make_mock_nio(encryption_import_warning=True)
    sys.modules["nio"] = nio

    try:
        sys.modules.pop("minion_assist.matrix.auth", None)
        from minion_assist.matrix.auth import resolve_matrix_auth

        cfg = _config(access_token="syt_abc123", device_id="DEVICEXYZ", encryption=True)
        result = _run(resolve_matrix_auth(cfg, crypto_store_dir=tmp_path / "crypto"))

        mock_client.restore_login.assert_not_called()
        assert result.access_token == "syt_abc123"
        assert result.user_id == "@bot:ex.org"
    finally:
        del sys.modules["nio"]
        sys.modules.pop("minion_assist.matrix.auth", None)


def test_encryption_without_device_id_and_whoami_fails_falls_back(tmp_path):
    """If whoami() can't produce a device_id (e.g. it errors or returns an
    unexpected response), encryption should degrade gracefully rather than
    calling restore_login() with a device_id of None.
    """
    nio, mock_client = _make_mock_nio()
    sys.modules["nio"] = nio

    try:
        sys.modules.pop("minion_assist.matrix.auth", None)
        from minion_assist.matrix.auth import resolve_matrix_auth

        # spec=[] means getattr(whoami, "device_id", None) returns None, same
        # as a real WhoamiError response would.
        mock_client.whoami = AsyncMock(return_value=MagicMock(spec=[]))

        cfg = _config(access_token="syt_abc123", encryption=True)
        result = _run(resolve_matrix_auth(cfg, crypto_store_dir=tmp_path / "crypto"))

        mock_client.restore_login.assert_not_called()
        assert result.access_token == "syt_abc123"
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

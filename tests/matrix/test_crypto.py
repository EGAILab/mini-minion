"""Tests for matrix/crypto.py — setup_crypto().

setup_crypto() no longer builds the crypto store itself (that now happens in
auth.resolve_matrix_auth, which is the only place matrix-nio will actually
let it happen — see that module's docstring). It just checks whether
encryption activated (client.olm is set) and uploads device keys if so.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from minion_assist.matrix.crypto import setup_crypto


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_no_olm_returns_false_and_skips_upload():
    """If client.olm is unset, encryption never activated during auth —
    setup_crypto should report that and not attempt a key upload."""
    client = MagicMock()
    client.olm = None
    client.keys_upload = AsyncMock()

    result = _run(setup_crypto(client))

    assert result is False
    client.keys_upload.assert_not_called()


def test_olm_present_uploads_keys():
    """If client.olm is set (encryption activated during auth), upload
    device keys and report success."""
    client = MagicMock()
    client.olm = MagicMock()  # any truthy Olm instance
    client.keys_upload = AsyncMock()

    result = _run(setup_crypto(client))

    assert result is True
    client.keys_upload.assert_called_once()


def test_keys_upload_failure_is_non_fatal():
    """A failed key upload (e.g. already uploaded from a previous run)
    shouldn't raise — the bot should keep running."""
    client = MagicMock()
    client.olm = MagicMock()
    client.keys_upload = AsyncMock(side_effect=RuntimeError("already uploaded"))

    result = _run(setup_crypto(client))

    assert result is True

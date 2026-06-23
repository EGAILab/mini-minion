"""E2E encryption setup for the Matrix channel.

Initialises matrix-nio's ``SqliteCryptoStore`` so the bot can participate in
encrypted rooms.  Requires ``libolm`` to be installed on the host system.

When ``libolm`` is not available a clear error is printed and the function
returns without enabling encryption so the bot can still function in
unencrypted rooms.
"""

from __future__ import annotations

import sys
from pathlib import Path


async def setup_crypto(client, store_path: Path) -> bool:
    """Configure E2E encryption on ``client`` using a SQLite crypto store.

    Creates the store file at ``store_path`` on first call and uploads device
    keys to the homeserver.  Subsequent calls reuse the existing store.

    Args:
        client:     An authenticated matrix-nio ``AsyncClient``.
        store_path: Path to the SQLite crypto store file.

    Returns:
        True if encryption was successfully set up, False if libolm is missing.
    """
    try:
        # SqliteCryptoStore is only available when matrix-nio[e2e] is installed
        # AND the native libolm C library is present on the system.
        from nio import SqliteCryptoStore  # noqa: PLC0415
    except ImportError:
        # Gracefully degrade: warn the user but keep the bot running without
        # E2E encryption rather than crashing.
        print(
            "[matrix] ERROR: encryption=true is set but 'libolm' is not installed. "
            "Install it via your system package manager (e.g. 'libolm-dev' on Debian/Ubuntu, "
            "'libolm' via Homebrew on macOS) and reinstall matrix-nio[e2e]. "
            "Continuing WITHOUT encryption.",
            file=sys.stderr,
        )
        return False

    store_path.parent.mkdir(parents=True, exist_ok=True)
    # SqliteCryptoStore persists Olm session keys and Megolm group session keys
    # across restarts.  Without it the bot would lose decryption ability on restart.
    store = SqliteCryptoStore(str(store_path))
    client.crypto = store
    try:
        # Upload the device's Ed25519 identity key and Curve25519 one-time keys
        # to the homeserver so other users can encrypt messages to this device.
        await client.keys_upload()
    except Exception as exc:
        # Keys upload failure is non-fatal — the bot may already have uploaded
        # keys from a previous run.
        print(f"[matrix] Warning: failed to upload device keys: {exc}", file=sys.stderr)
    return True

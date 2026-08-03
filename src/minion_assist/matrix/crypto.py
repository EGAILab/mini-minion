"""E2E encryption follow-up for the Matrix channel.

The actual crypto store and ``client.olm`` (matrix-nio's encrypt/decrypt
engine) are now wired up during authentication — see
:func:`minion_assist.matrix.auth.resolve_matrix_auth` — because matrix-nio
only builds them from ``store_path`` + an encryption-enabled config that must
be present *before* login/restore_login runs. This module just uploads
device keys once that's done.
"""

from __future__ import annotations

import sys


async def setup_crypto(client) -> bool:
    """Upload E2E device keys if encryption was wired up during auth.

    Args:
        client: An authenticated matrix-nio ``AsyncClient`` returned by
                :func:`~minion_assist.matrix.auth.resolve_matrix_auth`.

    Returns:
        True if ``client.olm`` is set (encryption is active) and a key-upload
        was attempted, False if encryption isn't active for this client (e.g.
        libolm was missing, so ``resolve_matrix_auth`` already degraded and
        printed its own warning).
    """
    # client.olm is only set by matrix-nio's load_store(), which runs inside
    # restore_login()/login() — and only when encryption_enabled was True at
    # AsyncClient construction time. If it's still None here, encryption never
    # activated (resolve_matrix_auth already explained why and printed a
    # warning), so there's nothing to upload.
    if not getattr(client, "olm", None):
        return False

    try:
        # Upload the device's Ed25519 identity key and Curve25519 one-time keys
        # to the homeserver so other users can encrypt messages to this device.
        await client.keys_upload()
    except Exception as exc:
        # Keys upload failure is non-fatal — the bot may already have uploaded
        # keys from a previous run.
        print(f"[matrix] Warning: failed to upload device keys: {exc}", file=sys.stderr)
    return True

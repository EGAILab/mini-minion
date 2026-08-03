"""Matrix authentication helpers.

Resolves an authenticated ``matrix-nio`` ``AsyncClient`` from the three
supported auth methods in priority order:

1. **Access token** — set directly on the client; no login call needed.
2. **Password** — call ``client.login(password)`` to obtain a token.
3. **SSO** — print the SSO URL (or open the browser) and poll for completion.

Requires ``matrix-nio`` (installed with the ``matrix`` optional dependency).

E2E encryption setup also happens here rather than after auth, because
matrix-nio only builds its on-disk crypto store *and* the actual ``Olm``
encryption machine (``client.olm``) inside ``load_store()``, which runs
automatically from ``client.restore_login()`` (token auth) or after
``client.login()`` (password auth) — but only if ``store_path`` and an
encryption-enabled ``AsyncClientConfig`` were already passed to the
``AsyncClient`` constructor. Wiring encryption on *after* construction (the
previous approach) can't work because that construction-time step is
skipped.

Raises:
    RuntimeError: If neither ``accessToken`` nor ``password`` is configured.
    RuntimeError: If password login fails (server returns an error).
"""

from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

from .config import MatrixConfig


async def resolve_matrix_auth(config: MatrixConfig, crypto_store_dir: Path | None = None):
    """Return an authenticated matrix-nio ``AsyncClient`` for ``config``.

    Args:
        config: The :class:`~minion_assist.matrix.config.MatrixConfig` with
                homeserver URL, user ID, and credentials.
        crypto_store_dir: Directory for matrix-nio's on-disk E2E key store.
                Required for encryption to actually activate when
                ``config.encryption`` is True; ignored otherwise.

    Returns:
        An ``AsyncClient`` instance ready for use.  The caller is responsible
        for calling ``client.close()`` when done.  When encryption was
        requested and could be enabled, ``client.olm`` will be set — check it
        (or call :func:`minion_assist.matrix.crypto.setup_crypto`) to confirm.

    Raises:
        RuntimeError: If no usable auth method is available or login fails.
    """
    try:
        # matrix-nio is the Python Matrix SDK.  It's an optional dependency —
        # only installed when the user adds the [matrix] extra group.
        from nio import AsyncClient, AsyncClientConfig  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "matrix-nio is not installed. "
            "Run: uv add --optional matrix 'matrix-nio[e2e]' aiosqlite"
        ) from exc

    # AsyncClientConfig.encryption_enabled defaults to whatever matrix-nio's
    # own internal flag resolved to at import time (True only if python-olm
    # imported successfully). Passing encryption_enabled=True explicitly when
    # that dependency is missing makes nio's own __post_init__ raise
    # ImportWarning — catch it here so the bot degrades to unencrypted rooms
    # instead of crashing, mirroring the previous graceful-fallback behaviour.
    client_config = AsyncClientConfig(store_sync_tokens=True)
    encryption_wanted = config.encryption
    if encryption_wanted:
        # matrix-nio's default crypto-DB filename is f"{user_id}_{device_id}.db".
        # Matrix user IDs always contain a ':' (e.g. "@bot:example.org"), and on
        # Windows/NTFS a colon in a filename isn't an error — it silently opens
        # an alternate data stream instead, splitting the real (100+KB) SQLite
        # database into an invisible stream on a 0-byte file. That's fragile:
        # plain copy/zip, most backup tools, and non-NTFS filesystems (Docker
        # volumes, WSL, cloud sync) silently drop ADS streams, which would
        # destroy the E2E keys. Override the filename with the colon stripped
        # so the store is always a normal, portable file.
        safe_store_name = config.user_id.replace(":", "_") + ".db"
        try:
            client_config = AsyncClientConfig(
                encryption_enabled=True, store_sync_tokens=True, store_name=safe_store_name
            )
        except ImportWarning:
            print(
                "[matrix] ERROR: encryption=true is set but 'libolm' is not installed. "
                "Install it via your system package manager (e.g. 'libolm-dev' on "
                "Debian/Ubuntu, 'libolm' via Homebrew on macOS) and reinstall "
                "matrix-nio[e2e]. Continuing WITHOUT encryption.",
                file=sys.stderr,
            )
            encryption_wanted = False

    store_path = ""
    if encryption_wanted and crypto_store_dir:
        # SqliteStore expects an existing directory — it does not create one.
        crypto_store_dir.mkdir(parents=True, exist_ok=True)
        store_path = str(crypto_store_dir)

    # AsyncClient holds the HTTP session and all Matrix state (rooms, sync token, etc).
    # device_id="" lets the server assign a new device ID if none is pre-configured.
    client = AsyncClient(
        homeserver=config.homeserver,
        user=config.user_id,
        device_id=config.device_id or "",
        store_path=store_path,
        config=client_config,
    )

    # --- Method 1: pre-issued access token ---
    # This is the recommended approach: generate a token once via Element or the
    # Admin API, store it in config.json, and never need to handle passwords.
    if config.access_token:
        client.access_token = config.access_token
        user_id = config.user_id
        device_id = config.device_id

        if encryption_wanted and not device_id:
            # E2E encryption is keyed to a specific device_id, and the access
            # token was already minted for a real device on the server. We
            # must reuse that same device_id rather than inventing one
            # locally, or the local crypto identity won't match what the
            # homeserver has on record for this token. whoami() looks it up.
            whoami = await client.whoami()
            device_id = getattr(whoami, "device_id", None)
            user_id = getattr(whoami, "user_id", None) or user_id
            if not device_id:
                print(
                    "[matrix] ERROR: encryption=true requires a stable deviceId "
                    "but none was configured and it could not be discovered "
                    "automatically. Set 'deviceId' in config.json under "
                    "channels.matrix. Continuing WITHOUT encryption.",
                    file=sys.stderr,
                )
                encryption_wanted = False

        if encryption_wanted:
            # restore_login() sets user_id/device_id/access_token AND calls
            # matrix-nio's internal load_store(), which builds both the
            # SQLite store and client.olm (the actual encrypt/decrypt engine
            # sync_forever() uses). Plain attribute assignment (the previous
            # approach) skips load_store() entirely, so client.olm never gets
            # set — no real encryption occurs even if the class name it was
            # importing existed.
            client.restore_login(user_id=user_id, device_id=device_id, access_token=config.access_token)
        else:
            # Manually set user_id and device_id because we're bypassing login().
            client.user_id = user_id
            if device_id:
                client.device_id = device_id
        return client

    # --- Method 2: password login ---
    # Less preferred because passwords in config files are a security risk, but
    # useful for local development or homeservers without token management UI.
    if config.password:
        resp = await client.login(
            password=config.password,
            device_name=config.device_name,  # human-readable device name in Element
        )
        # Successful login: resp has an access_token attribute.
        if hasattr(resp, "access_token"):
            return client
        await client.close()
        raise RuntimeError(
            f"Matrix password login failed: {getattr(resp, 'message', resp)}"
        )

    # --- Method 3: SSO ---
    # SSO (Single Sign-On) is used by homeservers that delegate auth to an
    # external provider (e.g. LDAP, Google, Keycloak).  We can't complete SSO
    # non-interactively, so we just print the URL and tell the user what to do.
    try:
        login_flows_resp = await client.get_login_flows()
        flows = getattr(login_flows_resp, "flows", [])
        sso_supported = any(
            getattr(f, "type", "") in ("m.login.sso", "m.login.cas") for f in flows
        )
    except Exception:
        sso_supported = False

    if sso_supported:
        sso_url = f"{config.homeserver}/_matrix/client/r0/login/sso/redirect"
        print(f"\n[matrix] SSO login required. Open this URL in your browser:\n  {sso_url}")
        try:
            webbrowser.open(sso_url)
        except Exception:
            pass
        await client.close()
        raise RuntimeError(
            "SSO login cannot be completed non-interactively. "
            "Complete the SSO flow in your browser, obtain an access token, "
            "and set 'accessToken' in config.json."
        )

    await client.close()
    raise RuntimeError(
        "No usable Matrix auth method: set 'accessToken' or 'password' in "
        "config.json under channels.matrix."
    )

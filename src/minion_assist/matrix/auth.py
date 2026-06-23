"""Matrix authentication helpers.

Resolves an authenticated ``matrix-nio`` ``AsyncClient`` from the three
supported auth methods in priority order:

1. **Access token** — set directly on the client; no login call needed.
2. **Password** — call ``client.login(password)`` to obtain a token.
3. **SSO** — print the SSO URL (or open the browser) and poll for completion.

Requires ``matrix-nio`` (installed with the ``matrix`` optional dependency).

Raises:
    RuntimeError: If neither ``accessToken`` nor ``password`` is configured.
    RuntimeError: If password login fails (server returns an error).
"""

from __future__ import annotations

import sys
import webbrowser

from .config import MatrixConfig


async def resolve_matrix_auth(config: MatrixConfig):
    """Return an authenticated matrix-nio ``AsyncClient`` for ``config``.

    Args:
        config: The :class:`~minion_assist.matrix.config.MatrixConfig` with
                homeserver URL, user ID, and credentials.

    Returns:
        An ``AsyncClient`` instance ready for use.  The caller is responsible
        for calling ``client.close()`` when done.

    Raises:
        RuntimeError: If no usable auth method is available or login fails.
    """
    try:
        # matrix-nio is the Python Matrix SDK.  It's an optional dependency —
        # only installed when the user adds the [matrix] extra group.
        from nio import AsyncClient  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "matrix-nio is not installed. "
            "Run: uv add --optional matrix 'matrix-nio[e2e]' aiosqlite"
        ) from exc

    # AsyncClient holds the HTTP session and all Matrix state (rooms, sync token, etc).
    # device_id="" lets the server assign a new device ID if none is pre-configured.
    client = AsyncClient(
        homeserver=config.homeserver,
        user=config.user_id,
        device_id=config.device_id or "",
    )

    # --- Method 1: pre-issued access token ---
    # This is the recommended approach: generate a token once via Element or the
    # Admin API, store it in config.json, and never need to handle passwords.
    if config.access_token:
        client.access_token = config.access_token
        # Manually set user_id and device_id because we're bypassing login().
        client.user_id = config.user_id
        if config.device_id:
            client.device_id = config.device_id
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

"""OpenAI Codex device-code OAuth authentication for minion-assist.

Flow (same as the openclaw reference implementation):
  1. POST /api/accounts/deviceauth/usercode  → {device_auth_id, user_code, interval}
  2. User visits https://auth.openai.com/codex/device and enters the code
  3. Poll POST /api/accounts/deviceauth/token until approved
     → {authorization_code, code_verifier}
  4. POST /oauth/token to exchange  → {access_token, refresh_token, expires_in}
  5. Decode JWT payload to extract chatgpt_account_id and email
  6. Save to ~/.minion-assist/codex-auth.json

Token file layout:
  {
    "access_token": "...",
    "refresh_token": "...",
    "expires_at": 1234567890.0,   # Unix timestamp (seconds)
    "account_id": "...",
    "email": "..."                 # optional
  }

Usage:
  codex-login              (CLI entry point registered in pyproject.toml)
  python -m minion_assist.auth.codex_auth
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import webbrowser
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

_AUTH_BASE = "https://auth.openai.com"
_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_DEVICE_CALLBACK_URL = f"{_AUTH_BASE}/deviceauth/callback"
_DEVICE_TIMEOUT_S = 15 * 60        # 15 minutes — matches the server-side code expiry window
_DEFAULT_POLL_INTERVAL_S = 5
# Refresh 5 minutes before expiry so the binary session startup (initialize +
# account/login/start) completes before the token actually expires.
_REFRESH_BUFFER_S = 5 * 60
_TOKEN_FILENAME = "codex-auth.json"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _token_path() -> Path:
    """Return the path to the Codex token file, creating the parent directory if needed."""
    home = Path(os.environ.get("MINION_ASSIST_HOME", "~/.minion-assist")).expanduser()
    home.mkdir(parents=True, exist_ok=True)
    return home / _TOKEN_FILENAME


def save_token(
    access_token: str,
    refresh_token: str,
    expires_at: float,
    account_id: str,
    email: str | None = None,
) -> None:
    """Persist a token dict to the Codex auth JSON file on disk."""
    data: dict = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": expires_at,
        "account_id": account_id,
    }
    if email:
        data["email"] = email
    _token_path().write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_raw() -> dict | None:
    """Return the raw stored token dict without refreshing. None if not logged in."""
    path = _token_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def _decode_jwt_payload(token: str) -> dict:
    """Base64url-decode the JWT payload section. Returns {} on any error."""
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    padded = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        decoded = json.loads(base64.urlsafe_b64decode(padded))
        # JSON can decode to any type; guard so callers can always use .get().
        return decoded if isinstance(decoded, dict) else {}
    except Exception:
        return {}


def _extract_identity(access_token: str) -> tuple[str, str | None]:
    """Return (account_id, email) from the JWT claims.

    account_id comes from ``chatgpt_account_id`` in the OpenAI auth claim;
    falls back to ``iss|sub`` or ``sub`` when absent.
    """
    payload = _decode_jwt_payload(access_token)
    auth = payload.get("https://api.openai.com/auth") or {}
    account_id = (auth.get("chatgpt_account_id") or "").strip()
    profile = payload.get("https://api.openai.com/profile") or {}
    email = (profile.get("email") or "").strip() or None
    if not account_id:
        iss = (payload.get("iss") or "").strip()
        sub = (payload.get("sub") or "").strip()
        account_id = f"{iss}|{sub}" if iss and sub else sub
    return account_id, email


def _token_expiry_from_jwt(access_token: str) -> float | None:
    """Extract the ``exp`` epoch from a JWT, or None if absent."""
    payload = _decode_jwt_payload(access_token)
    exp = payload.get("exp")
    return float(exp) if isinstance(exp, (int, float)) and exp > 0 else None


# ---------------------------------------------------------------------------
# HTTP helpers (thin wrappers so tests can patch them cleanly)
# ---------------------------------------------------------------------------

# auth.openai.com sits behind Cloudflare, which returns HTTP 530 for requests
# with Python's default User-Agent ("Python-urllib/3.x").  Both headers below
# are required: User-Agent to pass Cloudflare's bot check, and originator to
# satisfy OpenAI's own API validation layer.
_REQUEST_HEADERS = {
    "User-Agent": "minion-assist/0.1",
    "originator": "minion-assist",
}


def _post_json(url: str, body: dict) -> dict:
    """POST a JSON body to *url* and return the decoded JSON response."""
    data = json.dumps(body).encode()
    req = Request(url, data=data, headers={**_REQUEST_HEADERS, "Content-Type": "application/json"})
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _post_form(url: str, body: dict) -> dict:
    """POST a URL-encoded form body to *url* and return the decoded JSON response."""
    data = urlencode(body).encode()
    req = Request(url, data=data, headers={**_REQUEST_HEADERS, "Content-Type": "application/x-www-form-urlencoded"})
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Token lifecycle
# ---------------------------------------------------------------------------

def _do_refresh(raw: dict) -> dict | None:
    """Attempt to refresh using the stored refresh_token. Returns updated dict or None."""
    refresh_tok = (raw.get("refresh_token") or "").strip()
    if not refresh_tok:
        return None
    try:
        resp = _post_form(f"{_AUTH_BASE}/oauth/token", {
            "grant_type": "refresh_token",
            "refresh_token": refresh_tok,
            "client_id": _CLIENT_ID,
        })
    except Exception:
        return None
    new_access = (resp.get("access_token") or "").strip()
    new_refresh = (resp.get("refresh_token") or "").strip()
    if not new_access or not new_refresh:
        return None
    expires_in = resp.get("expires_in", 0)
    expires_at = time.time() + (expires_in or 3600)
    jwt_exp = _token_expiry_from_jwt(new_access)
    if jwt_exp:
        expires_at = jwt_exp
    account_id, email = _extract_identity(new_access)
    account_id = account_id or raw.get("account_id", "")
    email = email or raw.get("email")
    save_token(new_access, new_refresh, expires_at, account_id, email)
    result: dict = {
        "access_token": new_access,
        "refresh_token": new_refresh,
        "expires_at": expires_at,
        "account_id": account_id,
    }
    if email:
        result["email"] = email
    return result


def load_token() -> dict | None:
    """Load stored token, auto-refreshing when near expiry.

    Returns the token dict, or None if the user has never logged in.
    If refresh fails, returns the (possibly expired) stored token so the
    caller can decide whether to abort or try anyway.
    """
    raw = load_raw()
    if not raw:
        return None
    expires_at = float(raw.get("expires_at") or 0)
    if time.time() >= expires_at - _REFRESH_BUFFER_S:
        refreshed = _do_refresh(raw)
        if refreshed:
            return refreshed
    return raw


# ---------------------------------------------------------------------------
# Device-code login
# ---------------------------------------------------------------------------

def login(open_browser: bool = True) -> dict:
    """Run the OpenAI Codex device-code OAuth flow.

    Prints instructions to stdout, optionally opens the browser, and
    blocks until the user completes the login (up to 15 minutes).

    Returns:
        The saved token dict (same shape as ``load_token()``).

    Raises:
        RuntimeError: on unexpected API responses.
        TimeoutError: if the user does not complete login within 15 minutes.
    """
    print("Requesting device code from OpenAI...")
    resp = _post_json(f"{_AUTH_BASE}/api/accounts/deviceauth/usercode", {
        "client_id": _CLIENT_ID,
    })
    device_auth_id = (resp.get("device_auth_id") or "").strip()
    user_code = (resp.get("user_code") or resp.get("usercode") or "").strip()
    if not device_auth_id or not user_code:
        raise RuntimeError(f"Unexpected device code response: {resp}")

    try:
        interval = int(resp.get("interval") or _DEFAULT_POLL_INTERVAL_S)
    except (TypeError, ValueError):
        interval = _DEFAULT_POLL_INTERVAL_S

    verification_url = f"{_AUTH_BASE}/codex/device"
    print(f"\n  Visit:      {verification_url}")
    print(f"  Enter code: {user_code}\n")

    if open_browser:
        webbrowser.open(verification_url)

    print("Waiting for browser login... (timeout: 15 minutes)")
    deadline = time.monotonic() + _DEVICE_TIMEOUT_S

    while time.monotonic() < deadline:
        time.sleep(interval)
        try:
            token_resp = _post_json(f"{_AUTH_BASE}/api/accounts/deviceauth/token", {
                "device_auth_id": device_auth_id,
                "user_code": user_code,
            })
        except HTTPError as exc:
            # OpenAI signals "still waiting" as 403/404 rather than a 200
            # with a pending status field.  Any other status is a real error.
            if exc.code in (403, 404):
                continue
            raise

        auth_code = (token_resp.get("authorization_code") or "").strip()
        code_verifier = (token_resp.get("code_verifier") or "").strip()
        if not auth_code or not code_verifier:
            continue   # still pending

        print("Exchanging authorization code for tokens...")
        exchange = _post_form(f"{_AUTH_BASE}/oauth/token", {
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": _DEVICE_CALLBACK_URL,
            "client_id": _CLIENT_ID,
            "code_verifier": code_verifier,
        })

        access_token = (exchange.get("access_token") or "").strip()
        refresh_token = (exchange.get("refresh_token") or "").strip()
        if not access_token or not refresh_token:
            raise RuntimeError(f"Token exchange failed — missing tokens: {exchange}")

        expires_in = exchange.get("expires_in", 0)
        expires_at = time.time() + (expires_in or 3600)
        jwt_exp = _token_expiry_from_jwt(access_token)
        if jwt_exp:
            expires_at = jwt_exp

        account_id, email = _extract_identity(access_token)
        save_token(access_token, refresh_token, expires_at, account_id, email)

        name = email or account_id or "(unknown)"
        print(f"Authenticated as: {name}")
        return load_raw() or {}

    raise TimeoutError("Device authorization timed out after 15 minutes.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    """Entry point for the ``codex-login`` CLI command."""
    try:
        result = login()
        name = result.get("email") or result.get("account_id") or "(unknown)"
        print(f"\nToken stored at: {_token_path()}")
        print(f"Logged in as:    {name}")
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)
    except Exception as exc:
        print(f"\nLogin failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli_main()

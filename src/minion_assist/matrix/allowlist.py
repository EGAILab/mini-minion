"""Matrix user ID normalisation and allowlist checking.

Matrix user IDs follow the format ``@localpart:server``.  Config entries may
include a ``matrix:`` URI prefix or inconsistent casing.  All comparisons are
done against normalised lowercase IDs.

The special wildcard ``"*"`` in an allowlist permits any user.
"""

from __future__ import annotations


def normalise_matrix_user_id(raw: str) -> str:
    """Normalise a Matrix user ID string from config or an event.

    Strips the ``matrix:`` URI prefix if present, strips whitespace, and
    lowercases the result so allowlist comparisons are case-insensitive.

    Args:
        raw: Raw string from config or event (e.g. ``"matrix:@Alice:example.org"``).

    Returns:
        Normalised lowercase user ID, e.g. ``"@alice:example.org"``.
    """
    cleaned = raw.strip()
    # Some Matrix URI schemes prefix IDs with "matrix:" — strip it so we compare
    # bare user IDs like "@alice:example.org" in all cases.
    if cleaned.lower().startswith("matrix:"):
        cleaned = cleaned[len("matrix:"):]
    return cleaned.strip().lower()


def check_allowlist(user_id: str, allowlist: list[str]) -> bool:
    """Check whether ``user_id`` is permitted by ``allowlist``.

    Args:
        user_id:   Normalised (lowercase) Matrix user ID to check.
        allowlist: List of allowed user IDs from config.  ``"*"`` permits all.

    Returns:
        True if the user is on the allowlist (or the list contains ``"*"``).
    """
    if not allowlist:
        return False
    # Normalise the incoming user ID in case the caller hasn't done so already.
    normalised_user = normalise_matrix_user_id(user_id)
    for entry in allowlist:
        if entry.strip() == "*":
            # Wildcard: any authenticated Matrix user is permitted.
            return True
        if normalise_matrix_user_id(entry) == normalised_user:
            return True
    return False

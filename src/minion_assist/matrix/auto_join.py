"""Matrix auto-join invite handler.

When the bot receives a room invite, this module decides whether to accept it
based on the ``autoJoin`` policy in config:

- ``"always"``    — join any invited room.
- ``"allowlist"`` — join only rooms listed in ``autoJoinAllowlist``.
- ``"off"``       — never join automatically (default).
"""

from __future__ import annotations

from .config import MatrixConfig


async def handle_invite(
    client,
    room_id: str,
    inviter_id: str,
    config: MatrixConfig,
) -> None:
    """Accept or ignore a Matrix room invite according to the autoJoin policy.

    Args:
        client:     An authenticated matrix-nio ``AsyncClient``.
        room_id:    The room ID of the invite.
        inviter_id: The Matrix user ID who sent the invite.
        config:     The active :class:`~minion_assist.matrix.config.MatrixConfig`.
    """
    policy = config.auto_join

    # "off" (default): ignore all invites — the bot stays only in pre-configured rooms.
    if policy == "off":
        return

    # "always": trust every inviter and join unconditionally.
    if policy == "always":
        await client.join(room_id)
        return

    # "allowlist": join only rooms explicitly listed in autoJoinAllowlist.
    # This is the recommended setting for semi-public bots.
    if policy == "allowlist":
        if room_id in config.auto_join_allowlist:
            await client.join(room_id)

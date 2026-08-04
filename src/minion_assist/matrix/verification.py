"""Interactive SAS ("emoji") device verification for the Matrix channel.

matrix-nio implements the full cryptographic side of the verification
protocol internally (accepting a start event, exchanging keys, computing and
checking MACs, marking the device verified on success) — see
``nio.crypto.olm_machine.Olm.handle_key_verification``. What it does *not*
do on its own is decide whether to accept a request, show the emoji to a
human, or act on their yes/no — that's this module's job.

Flow
----
1. Someone starts verification with Ada from one of their own sessions.
2. ``handle_verification_start`` checks the sender is in the allowed list
   (``exec_approvals.approvers`` — the same "people Ada trusts" list used for
   exec-approval DMs) and accepts or cancels.
3. Once key material is exchanged, ``handle_verification_key`` reads the
   emoji short-authentication-string off the now-populated ``Sas`` object and
   DMs it to the requester with ✅/❌ reaction instructions.
4. ``handle_reaction`` resolves that DM's reaction to a confirm or reject
   call, and reports the final result back via another DM once matrix-nio
   finishes the exchange (``handle_verification_mac``) or the other side
   cancels (``handle_verification_cancel``).
"""

from __future__ import annotations

import sys

from .outbound import MatrixOutbound

_CONFIRM_EMOJI = "✅"
_REJECT_EMOJI = "❌"


class MatrixVerificationHandler:
    """Responds to interactive SAS device-verification requests.

    Args:
        client:    An authenticated matrix-nio ``AsyncClient``.
        outbound:  :class:`~minion_assist.matrix.outbound.MatrixOutbound` for sending DMs.
        approvers: User IDs allowed to initiate verification with Ada
                   (``config.exec_approvals.approvers``).
    """

    def __init__(self, client, outbound: MatrixOutbound, approvers: list[str]) -> None:
        self._client = client
        self._outbound = outbound
        self._approvers = approvers
        # Maps the DM event_id we sent the emoji in → the verification's
        # transaction_id, so handle_reaction() knows which SAS a reaction is for.
        self._pending: dict[str, str] = {}

    async def handle_verification_start(self, event) -> None:
        """Accept or reject an incoming ``KeyVerificationStart`` to-device event.

        Only users in ``approvers`` may verify with Ada — anyone else's
        request is cancelled immediately rather than left to hang.
        """
        if event.sender not in self._approvers:
            print(
                f"[matrix] Ignoring verification request from unapproved user {event.sender}",
                file=sys.stderr,
            )
            try:
                await self._client.cancel_key_verification(event.transaction_id)
            except Exception:
                pass  # nothing useful to do if even the cancel fails
            return

        try:
            await self._client.accept_key_verification(event.transaction_id)
        except Exception as exc:
            print(f"[matrix] Failed to accept verification request: {exc}", file=sys.stderr)

    async def handle_verification_key(self, event) -> None:
        """Show the SAS emoji to the requester once key material is exchanged.

        By the time this fires, matrix-nio has already processed the
        ``KeyVerificationKey`` event internally and populated the ``Sas``
        object's emoji — this just needs to read and display it.
        """
        sas = self._client.key_verifications.get(event.transaction_id)
        if sas is None or sas.canceled:
            return

        emoji_lines = "\n".join(f"{char}  {name}" for char, name in sas.get_emoji())
        body = (
            f"**Device verification**\n\n"
            f"Compare these with what your other session shows:\n\n"
            f"{emoji_lines}\n\n"
            f"React with {_CONFIRM_EMOJI} if they match, or {_REJECT_EMOJI} if they don't."
        )
        dm_room_id = await self._outbound.resolve_or_create_dm(event.sender)
        if not dm_room_id:
            # Can't reach the requester — cancel rather than leave it hanging.
            try:
                await self._client.cancel_key_verification(event.transaction_id)
            except Exception:
                pass
            return

        dm_event_id = await self._outbound.send_text(dm_room_id, body)
        self._pending[dm_event_id] = event.transaction_id

    async def handle_reaction(self, target_event_id: str, emoji: str) -> None:
        """Confirm or reject a pending verification based on a DM reaction.

        No-ops if ``target_event_id`` isn't a pending verification DM (e.g.
        it belongs to an exec-approval request instead).
        """
        transaction_id = self._pending.get(target_event_id)
        if transaction_id is None:
            return
        if emoji not in (_CONFIRM_EMOJI, _REJECT_EMOJI):
            return

        # Only resolve once — a second reaction on the same message should
        # not re-trigger confirm/cancel.
        del self._pending[target_event_id]

        try:
            if emoji == _CONFIRM_EMOJI:
                await self._client.confirm_short_auth_string(transaction_id)
            else:
                await self._client.cancel_key_verification(transaction_id, reject=True)
        except Exception as exc:
            print(f"[matrix] Failed to resolve verification: {exc}", file=sys.stderr)

    async def handle_verification_mac(self, event) -> None:
        """Notify the requester once matrix-nio finishes MAC verification."""
        sas = self._client.key_verifications.get(event.transaction_id)
        if sas is None:
            return

        dm_room_id = await self._outbound.resolve_or_create_dm(event.sender)
        if not dm_room_id:
            return

        if sas.verified:
            await self._outbound.send_text(dm_room_id, "✅ Device verified successfully.")
        elif sas.canceled:
            await self._outbound.send_text(dm_room_id, "❌ Verification failed or was cancelled.")

    async def handle_verification_cancel(self, event) -> None:
        """Notify the requester if they cancel verification from their side."""
        dm_room_id = await self._outbound.resolve_or_create_dm(event.sender)
        if dm_room_id:
            await self._outbound.send_text(dm_room_id, "❌ Verification was cancelled.")

"""Tests for matrix/verification.py — MatrixVerificationHandler.

matrix-nio's Sas/Olm machinery is mocked entirely — these tests only check
that MatrixVerificationHandler makes the right accept/cancel/confirm calls
and sends the right DMs in response to each stage of the protocol, not the
underlying cryptography (that's matrix-nio's job, already covered by its own
test suite).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from minion_assist.matrix.outbound import MatrixOutbound
from minion_assist.matrix.verification import (
    MatrixVerificationHandler,
    _CONFIRM_EMOJI,
    _REJECT_EMOJI,
)


def _make_client():
    client = MagicMock()
    client.key_verifications = {}
    client.accept_key_verification = AsyncMock()
    client.cancel_key_verification = AsyncMock()
    client.confirm_short_auth_string = AsyncMock()
    return client


def _make_outbound(event_id="$sas_dm", dm_room_id="!dm:ex.org"):
    out = MagicMock(spec=MatrixOutbound)
    out.send_text = AsyncMock(return_value=event_id)
    out.resolve_or_create_dm = AsyncMock(return_value=dm_room_id)
    return out


def _make_sas(canceled=False, verified=False, emoji=None):
    sas = MagicMock()
    sas.canceled = canceled
    sas.verified = verified
    sas.get_emoji = MagicMock(return_value=emoji or [("🐶", "Dog"), ("🐱", "Cat")])
    return sas


def _make_event(sender="@eric:ex.org", transaction_id="tx1"):
    event = MagicMock()
    event.sender = sender
    event.transaction_id = transaction_id
    return event


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# handle_verification_start
# ---------------------------------------------------------------------------

def test_start_from_approved_user_accepts():
    client = _make_client()
    handler = MatrixVerificationHandler(client, _make_outbound(), ["@eric:ex.org"])

    _run(handler.handle_verification_start(_make_event(sender="@eric:ex.org")))

    client.accept_key_verification.assert_called_once_with("tx1")
    client.cancel_key_verification.assert_not_called()


def test_start_from_unapproved_user_cancels():
    client = _make_client()
    handler = MatrixVerificationHandler(client, _make_outbound(), ["@eric:ex.org"])

    _run(handler.handle_verification_start(_make_event(sender="@stranger:ex.org")))

    client.cancel_key_verification.assert_called_once_with("tx1")
    client.accept_key_verification.assert_not_called()


# ---------------------------------------------------------------------------
# handle_verification_key
# ---------------------------------------------------------------------------

def test_key_sends_emoji_dm_and_tracks_pending():
    client = _make_client()
    client.key_verifications["tx1"] = _make_sas()
    outbound = _make_outbound(event_id="$sas_dm")
    handler = MatrixVerificationHandler(client, outbound, ["@eric:ex.org"])

    _run(handler.handle_verification_key(_make_event()))

    outbound.send_text.assert_called_once()
    body = outbound.send_text.call_args.args[1]
    assert "🐶" in body and "Dog" in body
    assert handler._pending["$sas_dm"] == "tx1"


def test_key_with_no_sas_sends_nothing():
    client = _make_client()  # key_verifications stays empty
    outbound = _make_outbound()
    handler = MatrixVerificationHandler(client, outbound, ["@eric:ex.org"])

    _run(handler.handle_verification_key(_make_event()))

    outbound.send_text.assert_not_called()


def test_key_with_canceled_sas_sends_nothing():
    client = _make_client()
    client.key_verifications["tx1"] = _make_sas(canceled=True)
    outbound = _make_outbound()
    handler = MatrixVerificationHandler(client, outbound, ["@eric:ex.org"])

    _run(handler.handle_verification_key(_make_event()))

    outbound.send_text.assert_not_called()


def test_key_dm_failure_cancels_verification():
    client = _make_client()
    client.key_verifications["tx1"] = _make_sas()
    outbound = _make_outbound()
    outbound.resolve_or_create_dm = AsyncMock(return_value=None)  # can't reach requester
    handler = MatrixVerificationHandler(client, outbound, ["@eric:ex.org"])

    _run(handler.handle_verification_key(_make_event()))

    client.cancel_key_verification.assert_called_once_with("tx1")


# ---------------------------------------------------------------------------
# handle_reaction
# ---------------------------------------------------------------------------

def test_reaction_confirm_calls_confirm_short_auth_string():
    client = _make_client()
    handler = MatrixVerificationHandler(client, _make_outbound(), ["@eric:ex.org"])
    handler._pending["$sas_dm"] = "tx1"

    _run(handler.handle_reaction("$sas_dm", _CONFIRM_EMOJI))

    client.confirm_short_auth_string.assert_called_once_with("tx1")
    client.cancel_key_verification.assert_not_called()
    assert "$sas_dm" not in handler._pending


def test_reaction_reject_calls_cancel_with_reject_true():
    client = _make_client()
    handler = MatrixVerificationHandler(client, _make_outbound(), ["@eric:ex.org"])
    handler._pending["$sas_dm"] = "tx1"

    _run(handler.handle_reaction("$sas_dm", _REJECT_EMOJI))

    client.cancel_key_verification.assert_called_once_with("tx1", reject=True)
    client.confirm_short_auth_string.assert_not_called()


def test_reaction_unknown_event_is_safe():
    client = _make_client()
    handler = MatrixVerificationHandler(client, _make_outbound(), ["@eric:ex.org"])

    _run(handler.handle_reaction("$unrelated", _CONFIRM_EMOJI))  # should not raise

    client.confirm_short_auth_string.assert_not_called()
    client.cancel_key_verification.assert_not_called()


def test_reaction_unrelated_emoji_leaves_pending_untouched():
    client = _make_client()
    handler = MatrixVerificationHandler(client, _make_outbound(), ["@eric:ex.org"])
    handler._pending["$sas_dm"] = "tx1"

    _run(handler.handle_reaction("$sas_dm", "👍"))

    client.confirm_short_auth_string.assert_not_called()
    client.cancel_key_verification.assert_not_called()
    assert handler._pending["$sas_dm"] == "tx1"


# ---------------------------------------------------------------------------
# handle_verification_mac / handle_verification_cancel
# ---------------------------------------------------------------------------

def test_mac_verified_sends_success_dm():
    client = _make_client()
    client.key_verifications["tx1"] = _make_sas(verified=True)
    outbound = _make_outbound()
    handler = MatrixVerificationHandler(client, outbound, ["@eric:ex.org"])

    _run(handler.handle_verification_mac(_make_event()))

    outbound.send_text.assert_called_once()
    assert "verified" in outbound.send_text.call_args.args[1].lower()


def test_mac_canceled_sends_failure_dm():
    client = _make_client()
    client.key_verifications["tx1"] = _make_sas(canceled=True)
    outbound = _make_outbound()
    handler = MatrixVerificationHandler(client, outbound, ["@eric:ex.org"])

    _run(handler.handle_verification_mac(_make_event()))

    outbound.send_text.assert_called_once()
    assert "failed" in outbound.send_text.call_args.args[1].lower() or "cancel" in outbound.send_text.call_args.args[1].lower()


def test_cancel_sends_notice_dm():
    client = _make_client()
    outbound = _make_outbound()
    handler = MatrixVerificationHandler(client, outbound, ["@eric:ex.org"])

    _run(handler.handle_verification_cancel(_make_event()))

    outbound.send_text.assert_called_once()
    assert "cancel" in outbound.send_text.call_args.args[1].lower()

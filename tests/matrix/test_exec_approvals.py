"""Tests for matrix/exec_approvals.py — MatrixExecApprovalHandler."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from minion_assist.matrix.config import MatrixExecApprovalsConfig
from minion_assist.matrix.exec_approvals import (
    MatrixExecApprovalHandler,
    _APPROVE_EMOJI,
    _DENY_EMOJI,
)
from minion_assist.matrix.outbound import MatrixOutbound


def _make_config(approvers=None, enabled=True):
    cfg = MagicMock(spec=MatrixExecApprovalsConfig)
    cfg.approvers = approvers or ["@admin:ex.org"]
    cfg.enabled = enabled
    return cfg


def _make_outbound(event_id="$req_event", dm_room_id="!dm:ex.org"):
    out = MagicMock(spec=MatrixOutbound)
    out.send_text = AsyncMock(return_value=event_id)
    out.resolve_or_create_dm = AsyncMock(return_value=dm_room_id)
    return out


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_approve_reaction_returns_allow_once():
    outbound = _make_outbound(event_id="$req")
    cfg = _make_config()
    handler = MatrixExecApprovalHandler(outbound, cfg)

    async def _run_with_reaction():
        loop = asyncio.get_running_loop()
        handler._loop = loop

        async def _simulate_approval():
            await asyncio.sleep(0.05)
            handler.handle_reaction("$req", _APPROVE_EMOJI)

        task = asyncio.create_task(_simulate_approval())
        result = await handler.request_approval("rm -rf /tmp/test")
        await task
        return result

    result = _run(_run_with_reaction())
    assert result == "allow_once"


def test_deny_reaction_returns_deny():
    outbound = _make_outbound(event_id="$req")
    cfg = _make_config()
    handler = MatrixExecApprovalHandler(outbound, cfg)

    async def _run_with_denial():
        async def _simulate_denial():
            await asyncio.sleep(0.05)
            handler.handle_reaction("$req", _DENY_EMOJI)

        task = asyncio.create_task(_simulate_denial())
        result = await handler.request_approval("dangerous command")
        await task
        return result

    result = _run(_run_with_denial())
    assert result == "deny"


def test_timeout_returns_deny():
    """With no approvers list, should return deny immediately."""
    outbound = _make_outbound()
    cfg = _make_config(approvers=[])  # no approvers → immediate deny
    handler = MatrixExecApprovalHandler(outbound, cfg)

    result = _run(handler.request_approval("some command"))
    assert result == "deny"


def test_handle_reaction_no_pending_is_safe():
    """handle_reaction with an unknown event_id should not raise."""
    handler = MatrixExecApprovalHandler(_make_outbound(), _make_config())
    handler.handle_reaction("$unknown_event", _APPROVE_EMOJI)  # should not raise


def test_handle_reaction_only_resolves_matching_event():
    outbound = _make_outbound(event_id="$req_a")
    cfg = _make_config()
    handler = MatrixExecApprovalHandler(outbound, cfg)

    async def _run_test():
        fut = asyncio.get_event_loop().create_future()
        handler._pending["$req_a"] = fut

        handler.handle_reaction("$req_b", _APPROVE_EMOJI)  # wrong event
        assert not fut.done()

        handler.handle_reaction("$req_a", _APPROVE_EMOJI)  # correct event
        assert fut.done()
        return await fut

    result = _run(_run_test())
    assert result is True

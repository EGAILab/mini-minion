"""Tests for matrix/auto_join.py — handle_invite()."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from minion_assist.matrix.auto_join import handle_invite
from minion_assist.matrix.config import MatrixConfig


def _config(auto_join="off", allowlist=None):
    cfg = MagicMock(spec=MatrixConfig)
    cfg.auto_join = auto_join
    cfg.auto_join_allowlist = allowlist or []
    return cfg


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_client():
    client = MagicMock()
    client.join = AsyncMock()
    return client


def test_off_policy_never_joins():
    client = _make_client()
    cfg = _config(auto_join="off")
    _run(handle_invite(client, "!room:ex.org", "@alice:ex.org", cfg))
    client.join.assert_not_called()


def test_always_policy_joins_any_room():
    client = _make_client()
    cfg = _config(auto_join="always")
    _run(handle_invite(client, "!room:ex.org", "@alice:ex.org", cfg))
    client.join.assert_called_once_with("!room:ex.org")


def test_allowlist_policy_joins_allowed_room():
    client = _make_client()
    cfg = _config(auto_join="allowlist", allowlist=["!room:ex.org"])
    _run(handle_invite(client, "!room:ex.org", "@alice:ex.org", cfg))
    client.join.assert_called_once_with("!room:ex.org")


def test_allowlist_policy_skips_unallowed_room():
    client = _make_client()
    cfg = _config(auto_join="allowlist", allowlist=["!other:ex.org"])
    _run(handle_invite(client, "!room:ex.org", "@alice:ex.org", cfg))
    client.join.assert_not_called()


def test_allowlist_policy_empty_list_does_not_join():
    client = _make_client()
    cfg = _config(auto_join="allowlist", allowlist=[])
    _run(handle_invite(client, "!room:ex.org", "@alice:ex.org", cfg))
    client.join.assert_not_called()

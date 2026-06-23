"""Tests for matrix/channel.py — MatrixChannel lifecycle."""

import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from minion_assist.matrix.channel import MatrixChannel
from minion_assist.matrix.config import MatrixConfig


def _make_config():
    raw = {
        "homeserver": "https://matrix.example.org",
        "userId": "@bot:example.org",
        "accessToken": "syt_abc123",
    }
    return MatrixConfig.from_dict(raw)


def _patch_monitor():
    """Patch monitor_matrix so no real Matrix connection is made."""
    return patch("minion_assist.matrix.monitor.monitor_matrix", new=AsyncMock())


def test_start_launches_daemon_thread(tmp_path):
    cfg = _make_config()

    started = threading.Event()

    async def _fake_monitor(config, sessions, stop_event, workspace):
        started.set()
        await stop_event.wait()

    with patch("minion_assist.matrix.channel.MatrixChannel._run_monitor", new=_fake_monitor):
        # _run_monitor is an instance method; patch it as unbound so self is ignored
        pass

    # Use a simpler approach: patch at the channel._run_monitor level via instance
    channel = MatrixChannel(cfg, tmp_path)

    async def _fake_run(sessions):
        started.set()
        await channel._stop_event.wait()

    channel._run_monitor = _fake_run
    channel.start({})
    assert started.wait(timeout=2.0), "Monitor coroutine did not start"
    assert channel._thread is not None
    assert channel._thread.daemon is True
    channel.stop()


def test_stop_joins_thread(tmp_path):
    cfg = _make_config()
    channel = MatrixChannel(cfg, tmp_path)

    finished = threading.Event()

    async def _fake_run(sessions):
        await channel._stop_event.wait()
        finished.set()

    channel._run_monitor = _fake_run
    channel.start({})
    time.sleep(0.05)
    channel.stop()
    assert finished.wait(timeout=2.0), "Monitor did not finish cleanly"
    assert not channel._thread.is_alive()


def test_double_start_is_safe(tmp_path):
    cfg = _make_config()
    channel = MatrixChannel(cfg, tmp_path)

    async def _fake_run(sessions):
        await channel._stop_event.wait()

    channel._run_monitor = _fake_run
    channel.start({})
    thread_before = channel._thread
    channel.start({})  # second call should be no-op
    assert channel._thread is thread_before
    channel.stop()


def test_double_stop_is_safe(tmp_path):
    cfg = _make_config()
    channel = MatrixChannel(cfg, tmp_path)

    async def _fake_run(sessions):
        await channel._stop_event.wait()

    channel._run_monitor = _fake_run
    channel.start({})
    time.sleep(0.05)
    channel.stop()
    channel.stop()  # second stop should not raise


def test_stop_before_start_is_safe(tmp_path):
    cfg = _make_config()
    channel = MatrixChannel(cfg, tmp_path)
    channel.stop()  # should not raise

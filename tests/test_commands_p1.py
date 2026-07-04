"""Tests for new P1 slash commands: /audit, /fork, /export, /provider test, /plugin list."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from minion_assist.agents.definitions import AgentConfig
from minion_assist.agents.session import AgentSession
from minion_assist.commands import CommandContext, CommandResult, dispatch_command
from minion_assist.context import Compactor
from minion_assist.memory.short_term import ShortTermMemory
from minion_assist.providers.base import LLMResponse
from minion_assist.session import SessionStore
from minion_assist.tools import ToolRegistry, default_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_provider(text="ok"):
    provider = Mock()
    provider.chat = Mock(return_value=LLMResponse(text=text, finish_reason="stop"))
    return provider


def _make_session(tmp_path, provider=None, agent_id="main"):
    if provider is None:
        provider = _mock_provider()
    short_term = ShortTermMemory(tmp_path / "sessions")
    session_store = SessionStore(tmp_path / "sessions.json")
    compactor = Compactor(context_window=100_000, preserve_tokens=2_000)
    return AgentSession(
        agent_id=agent_id,
        agent=AgentConfig(name="Ada", soul="soul"),
        provider=provider,
        max_output_tokens=512,
        tools=default_registry(root=tmp_path),
        compactor=compactor,
        short_term=short_term,
        session_store=session_store,
    )


def _ctx(tmp_path, command, args="", provider=None, agent_id="main"):
    session = _make_session(tmp_path, provider=provider, agent_id=agent_id)
    store = SessionStore(tmp_path / "sessions.json")
    return CommandContext(
        raw=f"/{command} {args}".strip(),
        command=f"/{command}",
        args=args,
        target_agent_id=agent_id,
        sessions={agent_id: session},
        agents_cfg={},
        session_store=store,
    ), session


# ---------------------------------------------------------------------------
# /audit
# ---------------------------------------------------------------------------

def test_audit_empty_log(tmp_path):
    ctx, _ = _ctx(tmp_path, "audit")
    result = dispatch_command(ctx)
    assert result.handled
    assert "empty" in result.message.lower()


def test_audit_shows_denied_entries(tmp_path):
    ctx, session = _ctx(tmp_path, "audit")
    # Record a denial via the policy.
    policy = session.registry.policy
    policy.audit_log.record(
        __import__("minion_assist.tools.audit", fromlist=["AuditEntry"]).AuditEntry(
            timestamp="2026-01-01T00:00:00+00:00",
            tool_name="bash",
            args_repr="rm -rf /",
            decision="denied",
            reason="user cancel",
        )
    )
    result = dispatch_command(ctx)
    assert result.handled
    assert "denied" in result.message
    assert "bash" in result.message


def test_audit_respects_count_argument(tmp_path):
    from minion_assist.tools.audit import AuditEntry, _utcnow
    ctx, session = _ctx(tmp_path, "audit", args="2")
    policy = session.registry.policy
    for i in range(5):
        policy.audit_log.record(AuditEntry(
            timestamp=_utcnow(), tool_name="bash", args_repr=f"cmd{i}", decision="allowed",
        ))
    result = dispatch_command(ctx)
    assert result.handled
    # Should show at most 2 entries.
    lines = [l for l in result.message.splitlines() if "bash" in l]
    assert len(lines) <= 2


# ---------------------------------------------------------------------------
# /fork
# ---------------------------------------------------------------------------

def test_fork_requires_new_id(tmp_path):
    ctx, _ = _ctx(tmp_path, "fork", args="")
    result = dispatch_command(ctx)
    assert result.handled
    assert "usage" in result.message.lower()


def test_fork_rejects_existing_session_id(tmp_path):
    ctx, _ = _ctx(tmp_path, "fork", args="main")  # "main" already exists
    result = dispatch_command(ctx)
    assert result.handled
    assert "already exists" in result.message.lower()


def test_fork_creates_new_session(tmp_path):
    ctx, session = _ctx(tmp_path, "fork", args="copy")
    session.send("original message")
    result = dispatch_command(ctx)
    assert result.handled
    assert "forked" in result.message.lower() or "copy" in result.message


# ---------------------------------------------------------------------------
# /export
# ---------------------------------------------------------------------------

def test_export_requires_path(tmp_path):
    ctx, _ = _ctx(tmp_path, "export")
    result = dispatch_command(ctx)
    assert result.handled
    assert "usage" in result.message.lower()


def test_export_writes_md_file(tmp_path):
    ctx, session = _ctx(tmp_path, "export", args=str(tmp_path / "out.md"))
    session.send("hello")
    result = dispatch_command(ctx)
    assert result.handled
    out_file = tmp_path / "out.md"
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "**User:**" in content


def test_export_writes_html_file(tmp_path):
    ctx, session = _ctx(tmp_path, "export", args=f"--html {tmp_path / 'out.html'}")
    session.send("hello")
    result = dispatch_command(ctx)
    assert result.handled
    out_file = tmp_path / "out.html"
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content


# ---------------------------------------------------------------------------
# /provider test
# ---------------------------------------------------------------------------

def test_provider_test_ok(tmp_path):
    provider = _mock_provider(text="ok")
    ctx, _ = _ctx(tmp_path, "provider", args="test main", provider=provider)
    result = dispatch_command(ctx)
    assert result.handled
    assert "ok" in result.message.lower()


def test_provider_test_failed_provider(tmp_path):
    provider = Mock()
    provider.chat = Mock(side_effect=RuntimeError("connection refused"))
    ctx, _ = _ctx(tmp_path, "provider", args="test main", provider=provider)
    result = dispatch_command(ctx)
    assert result.handled
    assert "failed" in result.message.lower()


def test_provider_test_unknown_agent(tmp_path):
    ctx, _ = _ctx(tmp_path, "provider", args="test nobody")
    result = dispatch_command(ctx)
    assert result.handled
    assert "unknown" in result.message.lower()


def test_provider_test_no_subcommand(tmp_path):
    ctx, _ = _ctx(tmp_path, "provider", args="")
    result = dispatch_command(ctx)
    assert result.handled
    assert "usage" in result.message.lower()


# ---------------------------------------------------------------------------
# /plugin list
# ---------------------------------------------------------------------------

def test_plugin_list_shows_tool_names(tmp_path):
    ctx, session = _ctx(tmp_path, "plugin", args="list")
    result = dispatch_command(ctx)
    assert result.handled
    # default_registry registers "bash" among others.
    assert "bash" in result.message


def test_plugin_list_unknown_subcommand(tmp_path):
    ctx, _ = _ctx(tmp_path, "plugin", args="unknown")
    result = dispatch_command(ctx)
    assert result.handled
    assert "usage" in result.message.lower()

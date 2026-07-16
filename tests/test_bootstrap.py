"""Tests for the bootstrap prompt injection layer.

Covers file discovery, security (path-escape rejection), budget enforcement,
rendering, and AgentSession integration.  All tests are self-contained and
require no provider calls — the bootstrap module is purely synchronous I/O
and string manipulation.
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from minion_assist.bootstrap import (
    BootstrapFile,
    ContextFile,
    _BOOTSTRAP_FILES,
    build_bootstrap_context_files,
    build_bootstrap_prompt_block,
    clear_bootstrap_block_cache,
    load_bootstrap_files,
    read_user_name,
    render_bootstrap_pending_context,
    render_project_context,
    render_truncation_warning,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    enabled: bool = True,
    max_chars: int = 20_000,
    total_max_chars: int = 60_000,
    truncation_warning: str = "always",
) -> SimpleNamespace:
    """Build a minimal duck-typed bootstrap config object."""
    return SimpleNamespace(
        enabled=enabled,
        max_chars=max_chars,
        total_max_chars=total_max_chars,
        truncation_warning=truncation_warning,
    )


def _write_file(directory: Path, name: str, content: str) -> Path:
    """Write a bootstrap file into directory and return the path."""
    p = directory / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# load_bootstrap_files — file discovery
# ---------------------------------------------------------------------------

def test_load_bootstrap_files_recognizes_known_names(tmp_path):
    """load_bootstrap_files finds all recognized bootstrap filenames."""
    # Write a subset of the canonical names.
    _write_file(tmp_path, "AGENTS.md", "agents content")
    _write_file(tmp_path, "SOUL.md", "soul content")
    _write_file(tmp_path, "USER.md", "user content")

    files = load_bootstrap_files(tmp_path)

    # All recognized names must appear in the result, in canonical order.
    names = [f.name for f in files]
    assert names == list(_BOOTSTRAP_FILES)

    # Present files have content; absent files are marked missing.
    present = {f.name: f for f in files if not f.missing}
    assert present["AGENTS.md"].content == "agents content"
    assert present["SOUL.md"].content == "soul content"
    assert present["USER.md"].content == "user content"


def test_load_bootstrap_files_omits_absent_files_marks_missing(tmp_path):
    """Files that don't exist on disk are represented with missing=True, content=None."""
    # Write only one file; the rest are absent.
    _write_file(tmp_path, "AGENTS.md", "hello")

    files = load_bootstrap_files(tmp_path)

    missing = [f for f in files if f.missing]
    # All canonical names except AGENTS.md should be missing.
    expected_missing = [n for n in _BOOTSTRAP_FILES if n != "AGENTS.md"]
    assert [f.name for f in missing] == expected_missing
    # All missing files have no content.
    assert all(f.content is None for f in missing)


def test_load_bootstrap_files_empty_dir_all_missing(tmp_path):
    """When the root directory is empty, every file is marked missing."""
    files = load_bootstrap_files(tmp_path)
    assert all(f.missing for f in files)
    assert len(files) == len(_BOOTSTRAP_FILES)


def test_load_bootstrap_files_returns_canonical_order(tmp_path):
    """Files are always returned in _BOOTSTRAP_FILES order, not filesystem order."""
    # Write files in reverse order to prove ordering isn't filesystem-driven.
    for name in reversed(_BOOTSTRAP_FILES):
        _write_file(tmp_path, name, f"content of {name}")

    files = load_bootstrap_files(tmp_path)
    assert [f.name for f in files] == list(_BOOTSTRAP_FILES)


def test_bootstrap_loader_rejects_path_escape(tmp_path):
    """A symlink pointing outside the bootstrap root is treated as missing."""
    outside = tmp_path / "outside"
    outside.mkdir()
    _write_file(outside, "SECRET.md", "secret data")

    root = tmp_path / "root"
    root.mkdir()

    # Create a symlink from root/AGENTS.md → ../outside/SECRET.md
    link = root / "AGENTS.md"
    try:
        link.symlink_to(outside / "SECRET.md")
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported on this platform")

    files = load_bootstrap_files(root)
    agents_file = next(f for f in files if f.name == "AGENTS.md")

    # Symlink escaping the root must be treated as missing (not read).
    assert agents_file.missing or agents_file.content is None


# ---------------------------------------------------------------------------
# build_bootstrap_context_files — budget enforcement
# ---------------------------------------------------------------------------

def test_build_context_files_applies_per_file_limit(tmp_path):
    """Content exceeding max_chars is truncated to a head+tail excerpt."""
    long_content = "A" * 100  # 100 chars
    _write_file(tmp_path, "AGENTS.md", long_content)

    files = load_bootstrap_files(tmp_path)
    ctx_files = build_bootstrap_context_files(files, max_chars=20, total_max_chars=60_000)

    agents_ctx = next(f for f in ctx_files if f.path.name == "AGENTS.md")
    assert agents_ctx.truncated is True
    assert agents_ctx.raw_chars == 100
    # The injected content includes a truncation marker; its raw content section
    # must be at most ~20 chars (head + tail), plus a marker.
    assert "...truncated" in agents_ctx.content
    assert agents_ctx.injected_chars < 100


def test_build_context_files_full_content_when_within_limit(tmp_path):
    """Content within max_chars is injected without truncation."""
    _write_file(tmp_path, "SOUL.md", "Short soul.")

    files = load_bootstrap_files(tmp_path)
    ctx_files = build_bootstrap_context_files(files, max_chars=20_000, total_max_chars=60_000)

    soul_ctx = next(f for f in ctx_files if f.path.name == "SOUL.md")
    assert soul_ctx.truncated is False
    assert soul_ctx.content == "Short soul."


def test_build_context_files_applies_total_limit(tmp_path):
    """Processing stops when the total character budget is exhausted."""
    # Write three files, each 30 chars.
    _write_file(tmp_path, "AGENTS.md", "A" * 30)
    _write_file(tmp_path, "SOUL.md",   "S" * 30)
    _write_file(tmp_path, "TOOLS.md",  "T" * 30)

    # Total budget: 50 — should stop before TOOLS.md is fully included.
    files = load_bootstrap_files(tmp_path)
    ctx_files = build_bootstrap_context_files(files, max_chars=20_000, total_max_chars=50)

    names = [f.path.name for f in ctx_files]
    # AGENTS.md (30) + SOUL.md (30) = 60 > 50, so SOUL.md should be truncated
    # or TOOLS.md omitted entirely. At minimum AGENTS.md must be present.
    assert "AGENTS.md" in names
    # TOOLS.md must not be present (total budget consumed by then).
    assert "TOOLS.md" not in names


def test_build_context_files_skips_missing_files(tmp_path):
    """Missing files are not represented in the output list."""
    _write_file(tmp_path, "AGENTS.md", "exists")

    files = load_bootstrap_files(tmp_path)
    ctx_files = build_bootstrap_context_files(files, max_chars=20_000, total_max_chars=60_000)

    names = [f.path.name for f in ctx_files]
    assert names == ["AGENTS.md"]


def test_build_context_files_skips_empty_content(tmp_path):
    """Files with whitespace-only content are skipped."""
    _write_file(tmp_path, "AGENTS.md", "   \n\n  ")
    _write_file(tmp_path, "SOUL.md", "actual content")

    files = load_bootstrap_files(tmp_path)
    ctx_files = build_bootstrap_context_files(files, max_chars=20_000, total_max_chars=60_000)

    names = [f.path.name for f in ctx_files]
    # AGENTS.md is whitespace-only → skipped; SOUL.md has content → included.
    assert names == ["SOUL.md"]


# ---------------------------------------------------------------------------
# render_project_context — rendering
# ---------------------------------------------------------------------------

def test_render_project_context_orders_files(tmp_path):
    """Render output respects the canonical file ordering from _BOOTSTRAP_FILES."""
    _write_file(tmp_path, "AGENTS.md", "agents")
    _write_file(tmp_path, "SOUL.md",   "soul")

    files = load_bootstrap_files(tmp_path)
    ctx_files = build_bootstrap_context_files(files, max_chars=20_000, total_max_chars=60_000)
    rendered = render_project_context(ctx_files)

    # Both files must appear, AGENTS.md before SOUL.md.
    assert "## AGENTS.md" in rendered
    assert "## SOUL.md" in rendered
    assert rendered.index("## AGENTS.md") < rendered.index("## SOUL.md")


def test_render_project_context_starts_with_header(tmp_path):
    """Rendered block starts with the # Project Context heading."""
    _write_file(tmp_path, "AGENTS.md", "content")

    files = load_bootstrap_files(tmp_path)
    ctx_files = build_bootstrap_context_files(files, max_chars=20_000, total_max_chars=60_000)
    rendered = render_project_context(ctx_files)

    assert rendered.startswith("# Project Context")


def test_render_project_context_empty_list():
    """render_project_context returns an empty string when given no files."""
    assert render_project_context([]) == ""


# ---------------------------------------------------------------------------
# render_bootstrap_pending_context
# ---------------------------------------------------------------------------

def test_bootstrap_pending_guidance_when_bootstrap_md_present(tmp_path):
    """BOOTSTRAP.md present in context files → pending guidance block is returned."""
    _write_file(tmp_path, "BOOTSTRAP.md", "First-run checklist: do X, then Y.")

    files = load_bootstrap_files(tmp_path)
    ctx_files = build_bootstrap_context_files(files, max_chars=20_000, total_max_chars=60_000)
    pending = render_bootstrap_pending_context(ctx_files)

    assert "<bootstrap_pending>" in pending
    assert "BOOTSTRAP.md" in pending


def test_bootstrap_pending_guidance_absent_when_no_bootstrap_md(tmp_path):
    """No BOOTSTRAP.md → render_bootstrap_pending_context returns empty string."""
    _write_file(tmp_path, "AGENTS.md", "agents")

    files = load_bootstrap_files(tmp_path)
    ctx_files = build_bootstrap_context_files(files, max_chars=20_000, total_max_chars=60_000)
    pending = render_bootstrap_pending_context(ctx_files)

    assert pending == ""


# ---------------------------------------------------------------------------
# render_truncation_warning
# ---------------------------------------------------------------------------

def test_truncation_warning_emitted_when_truncated():
    """Warning is returned when at least one ContextFile is truncated."""
    ctx_file = ContextFile(
        path=Path("AGENTS.md"),
        content="...",
        truncated=True,
        raw_chars=100,
        injected_chars=40,
    )
    warning = render_truncation_warning([ctx_file], mode="always")
    assert "[Bootstrap truncation warning]" in warning


def test_truncation_warning_suppressed_when_mode_off():
    """mode='off' suppresses the warning even when files are truncated."""
    ctx_file = ContextFile(
        path=Path("AGENTS.md"),
        content="...",
        truncated=True,
        raw_chars=100,
        injected_chars=40,
    )
    warning = render_truncation_warning([ctx_file], mode="off")
    assert warning == ""


def test_truncation_warning_absent_when_no_truncation():
    """No warning is returned when no files are truncated."""
    ctx_file = ContextFile(
        path=Path("AGENTS.md"),
        content="full content",
        truncated=False,
        raw_chars=12,
        injected_chars=12,
    )
    warning = render_truncation_warning([ctx_file], mode="always")
    assert warning == ""


def test_truncation_warning_once_mode_fires_only_once(tmp_path, monkeypatch):
    """mode='once' emits the warning only on the first truncated turn."""
    import minion_assist.bootstrap as _bmod

    # Reset the module-level flag so this test starts clean.
    monkeypatch.setattr(_bmod, "_truncation_warned", False)

    ctx_file = ContextFile(
        path=Path("AGENTS.md"),
        content="...",
        truncated=True,
        raw_chars=100,
        injected_chars=40,
    )

    first = render_truncation_warning([ctx_file], mode="once")
    second = render_truncation_warning([ctx_file], mode="once")

    assert "[Bootstrap truncation warning]" in first
    assert second == ""  # second call suppressed


# ---------------------------------------------------------------------------
# build_bootstrap_prompt_block — integration
# ---------------------------------------------------------------------------

def test_build_bootstrap_prompt_block_returns_empty_when_disabled(tmp_path):
    """build_bootstrap_prompt_block returns '' when enabled=False."""
    _write_file(tmp_path, "AGENTS.md", "content")
    cfg = _make_config(enabled=False)

    result = build_bootstrap_prompt_block(tmp_path, cfg)
    assert result == ""


def test_build_bootstrap_prompt_block_returns_empty_when_no_files(tmp_path):
    """build_bootstrap_prompt_block returns '' when no bootstrap files exist."""
    cfg = _make_config()
    result = build_bootstrap_prompt_block(tmp_path, cfg)
    assert result == ""


def test_build_bootstrap_prompt_block_contains_project_context(tmp_path):
    """build_bootstrap_prompt_block includes a # Project Context heading when files exist."""
    _write_file(tmp_path, "AGENTS.md", "My agent rules.")
    cfg = _make_config()

    result = build_bootstrap_prompt_block(tmp_path, cfg)

    assert "# Project Context" in result
    assert "My agent rules." in result


def test_build_bootstrap_prompt_block_includes_pending_when_bootstrap_md(tmp_path):
    """build_bootstrap_prompt_block prepends bootstrap-pending guidance when BOOTSTRAP.md present."""
    _write_file(tmp_path, "BOOTSTRAP.md", "Run setup first.")
    cfg = _make_config()

    result = build_bootstrap_prompt_block(tmp_path, cfg)

    assert "<bootstrap_pending>" in result
    # Pending guidance must appear before the project context.
    assert result.index("<bootstrap_pending>") < result.index("# Project Context")


# ---------------------------------------------------------------------------
# AgentSession integration
# ---------------------------------------------------------------------------

def _make_session(tmp_path, bootstrap_context=None, provider=None):
    """Build a minimal AgentSession for bootstrap integration tests."""
    from minion_assist.agents.definitions import AgentConfig
    from minion_assist.agents.session import AgentSession
    from minion_assist.context import Compactor
    from minion_assist.memory.short_term import ShortTermMemory
    from minion_assist.session import SessionStore
    from minion_assist.tools import ToolRegistry

    if provider is None:
        from unittest.mock import Mock
        from minion_assist.providers.base import LLMResponse
        provider = Mock()
        provider.chat = Mock(return_value=LLMResponse(text="ok", finish_reason="stop"))

    return AgentSession(
        agent_id="main",
        session_id="test-session",
        agent=AgentConfig(name="Ada", soul="You are Ada."),
        provider=provider,
        max_output_tokens=512,
        tools=ToolRegistry(),
        compactor=Compactor(context_window=100_000, preserve_tokens=2_000),
        short_term=ShortTermMemory(tmp_path / "sessions"),
        session_store=SessionStore(tmp_path / "sessions.json"),
        bootstrap_context=bootstrap_context,
    )


def test_agent_session_injects_bootstrap_after_soul(tmp_path):
    """Bootstrap block appears in the system prompt after the soul text."""
    captured_systems: list[str] = []

    def _fake_run_turn(provider, name, system, max_tokens, tools, messages, **kwargs):
        captured_systems.append(system)
        messages.append({"role": "assistant", "content": "ok"})
        return None

    bootstrap_fn = lambda: "# Project Context\n\n## AGENTS.md\nWorkspace rules."
    session = _make_session(tmp_path, bootstrap_context=bootstrap_fn)

    with patch("minion_assist.agents.session.run_turn", side_effect=_fake_run_turn):
        session.send("hello")

    assert captured_systems, "run_turn was never called"
    system = captured_systems[0]

    # Soul must be present.
    assert "You are Ada." in system
    # Bootstrap must be present.
    assert "# Project Context" in system
    # Bootstrap must come after the soul.
    assert system.index("You are Ada.") < system.index("# Project Context")


def test_agent_session_reads_bootstrap_each_turn(tmp_path):
    """bootstrap_context callable is invoked on every send() call."""
    call_count = 0

    def _counting_bootstrap():
        nonlocal call_count
        call_count += 1
        return f"# Project Context\n\n## AGENTS.md\nTurn {call_count}"

    def _fake_run_turn(provider, name, system, max_tokens, tools, messages, **kwargs):
        messages.append({"role": "assistant", "content": "ok"})
        return None

    session = _make_session(tmp_path, bootstrap_context=_counting_bootstrap)

    with patch("minion_assist.agents.session.run_turn", side_effect=_fake_run_turn):
        session.send("first")
        session.send("second")
        session.send("third")

    # Called once per turn, not once at startup.
    assert call_count == 3


def test_agent_session_without_bootstrap_context(tmp_path):
    """AgentSession works normally when bootstrap_context is None."""
    captured_systems: list[str] = []

    def _fake_run_turn(provider, name, system, max_tokens, tools, messages, **kwargs):
        captured_systems.append(system)
        messages.append({"role": "assistant", "content": "ok"})
        return None

    session = _make_session(tmp_path, bootstrap_context=None)

    with patch("minion_assist.agents.session.run_turn", side_effect=_fake_run_turn):
        session.send("hello")

    system = captured_systems[0]
    assert "You are Ada." in system
    # No bootstrap content injected.
    assert "# Project Context" not in system


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_config_validates_bootstrap_budget_fields():
    """_validate() catches invalid budget values in the bootstrap section."""
    import copy
    from minion_assist.config import _validate

    # A minimal valid base config (no bootstrap section — should be fine).
    base = {
        "models": {
            "providers": {
                "lmstudio": {
                    "api": "lmstudio",
                    "baseUrl": "http://localhost:1234/v1",
                    "models": [
                        {"id": "qwen-9b", "contextWindow": 8192, "maxOutputTokens": 4096}
                    ],
                }
            }
        },
        "agents": {
            "main": {"model": "lmstudio/qwen-9b"},
        },
    }

    # Non-positive max_chars should produce an issue.
    raw = copy.deepcopy(base)
    raw["bootstrap"] = {"max_chars": -1}
    issues = _validate(raw)
    assert any("bootstrap.max_chars" in i.path for i in issues)

    # Non-positive total_max_chars should produce an issue.
    raw = copy.deepcopy(base)
    raw["bootstrap"] = {"total_max_chars": 0}
    issues = _validate(raw)
    assert any("bootstrap.total_max_chars" in i.path for i in issues)


def test_config_validates_bootstrap_truncation_warning():
    """_validate() catches invalid truncation_warning values."""
    import copy
    from minion_assist.config import _validate

    base = {
        "models": {
            "providers": {
                "lmstudio": {
                    "api": "lmstudio",
                    "baseUrl": "http://localhost:1234/v1",
                    "models": [
                        {"id": "qwen-9b", "contextWindow": 8192, "maxOutputTokens": 4096}
                    ],
                }
            }
        },
        "agents": {"main": {"model": "lmstudio/qwen-9b"}},
    }

    raw = copy.deepcopy(base)
    raw["bootstrap"] = {"truncation_warning": "banana"}
    issues = _validate(raw)
    assert any("bootstrap.truncation_warning" in i.path for i in issues)


def test_config_validates_bootstrap_enabled_must_be_bool():
    """_validate() rejects a non-boolean 'enabled' field in bootstrap config."""
    import copy
    from minion_assist.config import _validate

    base = {
        "models": {
            "providers": {
                "lmstudio": {
                    "api": "lmstudio",
                    "baseUrl": "http://localhost:1234/v1",
                    "models": [
                        {"id": "qwen-9b", "contextWindow": 8192, "maxOutputTokens": 4096}
                    ],
                }
            }
        },
        "agents": {"main": {"model": "lmstudio/qwen-9b"}},
    }

    raw = copy.deepcopy(base)
    raw["bootstrap"] = {"enabled": "yes"}  # string instead of bool
    issues = _validate(raw)
    assert any("bootstrap.enabled" in i.path for i in issues)


# ---------------------------------------------------------------------------
# session_type — subagent bootstrap filtering
# ---------------------------------------------------------------------------

def test_load_bootstrap_files_with_allowed_names(tmp_path):
    """load_bootstrap_files with allowed_names only loads the permitted files."""
    from minion_assist.bootstrap import _SUBAGENT_BOOTSTRAP_FILES

    for name in ("AGENTS.md", "SOUL.md", "TOOLS.md", "IDENTITY.md"):
        _write_file(tmp_path, name, f"{name} content")

    files = load_bootstrap_files(tmp_path, allowed_names=_SUBAGENT_BOOTSTRAP_FILES)
    names = [f.name for f in files]

    # Only allowed names appear — order matches allowed_names tuple.
    assert names == list(_SUBAGENT_BOOTSTRAP_FILES)


def test_load_bootstrap_files_subagent_files_have_content(tmp_path):
    """Files matching allowed_names have their content loaded."""
    from minion_assist.bootstrap import _SUBAGENT_BOOTSTRAP_FILES

    _write_file(tmp_path, "AGENTS.md", "agents content")
    _write_file(tmp_path, "TOOLS.md", "tools content")

    files = load_bootstrap_files(tmp_path, allowed_names=_SUBAGENT_BOOTSTRAP_FILES)
    content_map = {f.name: f.content for f in files if not f.missing}

    assert content_map["AGENTS.md"] == "agents content"
    assert content_map["TOOLS.md"] == "tools content"


def test_build_bootstrap_prompt_block_root_includes_all_files(tmp_path):
    """session_type='root' (default) includes all present bootstrap files."""
    _write_file(tmp_path, "AGENTS.md", "agents")
    _write_file(tmp_path, "SOUL.md", "soul")
    _write_file(tmp_path, "TOOLS.md", "tools")
    cfg = _make_config()

    result = build_bootstrap_prompt_block(tmp_path, cfg, session_type="root")

    assert "agents" in result
    assert "soul" in result
    assert "tools" in result


def test_build_bootstrap_prompt_block_subagent_excludes_soul(tmp_path):
    """session_type='subagent' excludes SOUL.md, IDENTITY.md, and USER.md."""
    _write_file(tmp_path, "AGENTS.md", "agents")
    _write_file(tmp_path, "SOUL.md", "soul personality")
    _write_file(tmp_path, "TOOLS.md", "tools")
    _write_file(tmp_path, "IDENTITY.md", "identity")
    _write_file(tmp_path, "USER.md", "user profile")
    cfg = _make_config()

    result = build_bootstrap_prompt_block(tmp_path, cfg, session_type="subagent")

    assert "agents" in result
    assert "tools" in result
    assert "soul personality" not in result
    assert "identity" not in result
    assert "user profile" not in result


def test_build_bootstrap_prompt_block_subagent_only_agents_and_tools(tmp_path):
    """session_type='subagent' produces a block with only AGENTS.md and TOOLS.md."""
    _write_file(tmp_path, "AGENTS.md", "AGENTS_CONTENT")
    _write_file(tmp_path, "TOOLS.md", "TOOLS_CONTENT")
    cfg = _make_config()

    result = build_bootstrap_prompt_block(tmp_path, cfg, session_type="subagent")

    assert "AGENTS_CONTENT" in result
    assert "TOOLS_CONTENT" in result


# ---------------------------------------------------------------------------
# read_user_name()
# ---------------------------------------------------------------------------

def test_read_user_name_returns_none_when_no_file(tmp_path):
    """Returns None when USER.md does not exist."""
    assert read_user_name(tmp_path) is None


def test_read_user_name_finds_name_colon_line(tmp_path):
    """Extracts the name from a 'Name: <value>' line."""
    _write_file(tmp_path, "USER.md", "Name: Alice\nOther: stuff\n")
    assert read_user_name(tmp_path) == "Alice"


def test_read_user_name_finds_name_with_dash_prefix(tmp_path):
    """Extracts the name from '- Name: <value>' (list-item format)."""
    _write_file(tmp_path, "USER.md", "- Name: Bob\n")
    assert read_user_name(tmp_path) == "Bob"


def test_read_user_name_finds_name_with_star_prefix(tmp_path):
    """Extracts the name from '* Name: <value>' (bullet format)."""
    _write_file(tmp_path, "USER.md", "* Name: Carol\n")
    assert read_user_name(tmp_path) == "Carol"


def test_read_user_name_case_insensitive(tmp_path):
    """Key matching is case-insensitive ('name:' works as well as 'Name:')."""
    _write_file(tmp_path, "USER.md", "name: Dave\n")
    assert read_user_name(tmp_path) == "Dave"


def test_read_user_name_falls_back_to_h1_heading(tmp_path):
    """Falls back to the first H1 heading when no Name: key is present."""
    _write_file(tmp_path, "USER.md", "# Eve\n\nSome content.\n")
    assert read_user_name(tmp_path) == "Eve"


def test_read_user_name_returns_none_when_no_match(tmp_path):
    """Returns None when USER.md has no Name: line or H1 heading."""
    _write_file(tmp_path, "USER.md", "Just some prose with no name field.\n")
    assert read_user_name(tmp_path) is None


def test_read_user_name_bold_key_with_colon_outside(tmp_path):
    """Extracts name from '- **Name:** <value>' (bold key, colon inside bold)."""
    _write_file(tmp_path, "USER.md", "- **Name:** Eric\n")
    assert read_user_name(tmp_path) == "Eric"


def test_read_user_name_bold_key_no_prefix(tmp_path):
    """Extracts name from '**Name:** <value>' without list prefix."""
    _write_file(tmp_path, "USER.md", "**Name:** Eric\n")
    assert read_user_name(tmp_path) == "Eric"


def test_read_user_name_prefers_name_key_over_heading(tmp_path):
    """Name: key takes priority over a H1 heading present in the same file."""
    _write_file(tmp_path, "USER.md", "# Frank\nName: Grace\n")
    assert read_user_name(tmp_path) == "Grace"


def test_read_user_name_strips_whitespace(tmp_path):
    """Trailing whitespace in the extracted name is stripped."""
    _write_file(tmp_path, "USER.md", "Name:   Heidi   \n")
    assert read_user_name(tmp_path) == "Heidi"


def test_build_bootstrap_prompt_block_default_session_type_is_root(tmp_path):
    """Omitting session_type defaults to 'root' (all files included)."""
    _write_file(tmp_path, "SOUL.md", "soul content")
    cfg = _make_config()

    result_explicit = build_bootstrap_prompt_block(tmp_path, cfg, session_type="root")
    result_default = build_bootstrap_prompt_block(tmp_path, cfg)

    assert result_explicit == result_default


# ---------------------------------------------------------------------------
# build_bootstrap_prompt_block — mtime cache
# ---------------------------------------------------------------------------

def test_bootstrap_cache_returns_same_object_on_hit(tmp_path):
    """Second call with unchanged files returns the same string (cache hit).

    A cache hit means the exact same string object is returned without
    re-reading any file.  We verify identity (``is``) not just equality.
    """
    clear_bootstrap_block_cache()
    _write_file(tmp_path, "AGENTS.md", "agents content")
    cfg = _make_config()

    first = build_bootstrap_prompt_block(tmp_path, cfg)
    second = build_bootstrap_prompt_block(tmp_path, cfg)

    # Same string object — proves we returned the cache entry, not a rebuild.
    assert first is second


def test_bootstrap_cache_rebuilds_when_file_content_changes(tmp_path):
    """Cache is invalidated when a bootstrap file is modified (mtime changes)."""
    import time

    clear_bootstrap_block_cache()
    p = _write_file(tmp_path, "AGENTS.md", "original content")
    cfg = _make_config()

    first = build_bootstrap_prompt_block(tmp_path, cfg)
    assert "original content" in first

    # Force a new mtime by bumping it one second into the future.
    # This is more reliable than sleeping because filesystem mtime resolution
    # varies (1 s on FAT32, ~100 ns on NTFS, 1 ns on ext4).
    new_mtime = p.stat().st_mtime + 1.0
    p.write_text("updated content", encoding="utf-8")
    import os
    os.utime(p, (new_mtime, new_mtime))

    second = build_bootstrap_prompt_block(tmp_path, cfg)

    assert "updated content" in second
    assert first is not second


def test_bootstrap_cache_rebuilds_when_new_file_appears(tmp_path):
    """Cache is invalidated when a new bootstrap file appears on disk."""
    clear_bootstrap_block_cache()
    _write_file(tmp_path, "AGENTS.md", "agents content")
    cfg = _make_config()

    first = build_bootstrap_prompt_block(tmp_path, cfg)
    assert "soul content" not in first

    # A new file appears — cache must miss and the new content must be included.
    _write_file(tmp_path, "SOUL.md", "soul content")

    second = build_bootstrap_prompt_block(tmp_path, cfg)

    assert "soul content" in second
    assert first is not second


def test_bootstrap_cache_miss_on_different_config(tmp_path):
    """Different config limits produce independent cache entries."""
    clear_bootstrap_block_cache()
    _write_file(tmp_path, "AGENTS.md", "a" * 100)
    cfg_small = _make_config(max_chars=50)
    cfg_large = _make_config(max_chars=200)

    result_small = build_bootstrap_prompt_block(tmp_path, cfg_small)
    result_large = build_bootstrap_prompt_block(tmp_path, cfg_large)

    # Different char limits → different truncation → different blocks.
    assert result_small is not result_large


def test_clear_bootstrap_block_cache_forces_rebuild(tmp_path):
    """clear_bootstrap_block_cache() causes the next call to rebuild from disk."""
    clear_bootstrap_block_cache()
    _write_file(tmp_path, "AGENTS.md", "content v1")
    cfg = _make_config()

    first = build_bootstrap_prompt_block(tmp_path, cfg)

    # Clear the cache, then call again without touching any file.
    # Even though mtimes haven't changed, the cache is gone so we rebuild.
    clear_bootstrap_block_cache()
    second = build_bootstrap_prompt_block(tmp_path, cfg)

    # Content must be identical (same files) but objects will differ after rebuild.
    assert first == second

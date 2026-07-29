"""Tests for memory/cli.py: the `minion-assist memory ...` subcommand tree.

`_run_migrate()` reads `workspace`/`agents_cfg` from `minion_assist.config`
via a local import inside the function, so monkeypatching those module
attributes before calling `main()` is sufficient to redirect it at a tmp_path
workspace without touching the real ~/.minion-assist config.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import minion_assist.config as config
from minion_assist.memory import cli
from minion_assist.memory.files import MemoryFileRepository


def _patch_config(monkeypatch, tmp_path, agent_ids=("main",)):
    monkeypatch.setattr(config, "workspace", tmp_path)
    monkeypatch.setattr(config, "agents", {aid: {} for aid in agent_ids})
    # _resolve_agent_root() falls back to bootstrap_cfg.path/cwd when an agent
    # has no workspace directory (see test_doctor_warns_about_missing_workspace)
    # — pin that fallback inside tmp_path so a test can never touch the real
    # working directory's filesystem.
    monkeypatch.setattr(config, "bootstrap", SimpleNamespace(path=str(tmp_path)))
    # No database by default: _build_index() (Stage One Phase 3, slice C)
    # must not silently pick up this machine's real configured/reachable
    # dev database and start querying a live, unrelated index for these
    # file-based tests. Tests that specifically want an index configured
    # pass their own database SimpleNamespace instead (see test_reindex_*).
    monkeypatch.setattr(config, "database", SimpleNamespace(url=None))


def _agent_root(tmp_path, agent_id: str):
    """Pre-create workspaces/{agent_id}/ so agent_workspace_root() resolves it
    directly, without falling back to bootstrap_cfg.path/cwd (irrelevant here)."""
    root = tmp_path / "workspaces" / agent_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def _patch_index(monkeypatch, mock_index):
    """Make `_build_index()` return a mock instead of touching a real database."""
    monkeypatch.setattr(cli, "_build_index", lambda: mock_index)


def test_migrate_dry_run_is_default(monkeypatch, tmp_path, capsys):
    _patch_config(monkeypatch, tmp_path)
    exit_code = cli.main(["migrate"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "dry run" in out.lower()
    # Nothing should have been written by a dry run.
    assert not (tmp_path / "workspaces").exists()


def test_migrate_apply_creates_workspace_and_prints_manifest(monkeypatch, tmp_path, capsys):
    notes_dir = tmp_path / "memory" / "main"
    notes_dir.mkdir(parents=True)
    (notes_dir / "note.md").write_text("hello", encoding="utf-8")

    _patch_config(monkeypatch, tmp_path)
    exit_code = cli.main(["migrate", "--apply"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert (tmp_path / "workspaces" / "main" / "memory" / "topics" / "note.md").read_text(
        encoding="utf-8"
    ) == "hello"
    assert "Manifest:" in out
    assert "--rollback" in out


def test_migrate_rollback_dispatches_to_migration_module(monkeypatch, tmp_path, capsys):
    _patch_config(monkeypatch, tmp_path)
    missing_manifest = tmp_path / "no-such-manifest.json"

    exit_code = cli.main(["migrate", "--rollback", str(missing_manifest)])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "Could not read manifest" in out


def test_main_rejects_unknown_subcommand():
    parser_error_raised = False
    try:
        cli.main(["not-a-real-subcommand"])
    except SystemExit:
        parser_error_raised = True
    assert parser_error_raised


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def test_status_reports_note_counts(monkeypatch, tmp_path, capsys):
    root = _agent_root(tmp_path, "main")
    MemoryFileRepository(root).remember("goal", "content")
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["status"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "main:" in out
    assert "1 topic" in out


def test_status_filters_by_agent(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _agent_root(tmp_path, "researcher")
    _patch_config(monkeypatch, tmp_path, agent_ids=("main", "researcher"))

    exit_code = cli.main(["status", "--agent", "researcher"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "researcher:" in out
    assert "main:" not in out


def test_status_rejects_unknown_agent(monkeypatch, tmp_path):
    _patch_config(monkeypatch, tmp_path)
    try:
        cli.main(["status", "--agent", "nope"])
        raised = False
    except SystemExit:
        raised = True
    assert raised


def test_status_deep_without_a_database_reports_unavailable(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["status", "--deep"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "no database configured" in out


def test_status_deep_with_an_index_reports_chunk_counts(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_index = Mock()
    mock_index.index_summary.return_value = {
        "total_chunks": 4, "file_count": 2,
        "by_corpus": {"durable": 3, "daily": 1}, "last_indexed_at": 1234.0,
    }
    _patch_index(monkeypatch, mock_index)

    exit_code = cli.main(["status", "--deep"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "4 chunk(s) across 2 file(s)" in out
    assert "durable=3" in out


def test_status_without_deep_never_builds_an_index(monkeypatch, tmp_path, capsys):
    def _fail_if_called():
        raise AssertionError("status without --deep must not call _build_index()")

    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_build_index", _fail_if_called)

    exit_code = cli.main(["status"])

    assert exit_code == 0


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def test_list_shows_topic_keys(monkeypatch, tmp_path, capsys):
    root = _agent_root(tmp_path, "main")
    MemoryFileRepository(root).remember("project-goals", "content")
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["list"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "project-goals" in out


def test_list_empty_store_reports_zero(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["list"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "0 note(s)" in out


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

def test_get_reads_bounded_slice(monkeypatch, tmp_path, capsys):
    root = _agent_root(tmp_path, "main")
    MemoryFileRepository(root).remember("note", "line1\nline2\nline3")
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["get", "memory/topics/note.md", "--agent", "main", "--from-line", "2"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "line2" in out
    assert "line3" in out
    assert "line1" not in out
    assert "lines 2-3 of 3" in out


def test_get_reports_missing_file(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["get", "memory/topics/missing.md", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "Error" in out
    assert "not found" in out.lower()


def test_get_rejects_path_outside_root(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["get", "../../etc/passwd", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "outside the memory root" in out


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def test_search_finds_matching_note(monkeypatch, tmp_path, capsys):
    root = _agent_root(tmp_path, "main")
    MemoryFileRepository(root).remember("api-notes", "REST API best practices")
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["search", "REST"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "api-notes" in out
    assert "[topic]" in out


def test_search_no_matches_reports_zero(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["search", "xyzzy"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "0 match(es)" in out


def test_search_with_an_index_passes_corpus_through_and_shows_citation(
    monkeypatch, tmp_path, capsys
):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_index = Mock()
    mock_index.hybrid_search.return_value = [{
        "rel_path": "MEMORY.md", "source_kind": "durable", "chunk_index": 0,
        "heading_path": "", "content": "REST API best practices", "start_line": 1,
        "end_line": 1, "score": 0.5,
    }]
    _patch_index(monkeypatch, mock_index)

    exit_code = cli.main(["search", "REST", "--corpus", "durable"])
    out = capsys.readouterr().out

    assert exit_code == 0
    mock_index.hybrid_search.assert_called_once_with(
        "main", "REST", corpus="durable", max_results=20
    )
    assert "MEMORY.md:1-1" in out


# ---------------------------------------------------------------------------
# reindex
# ---------------------------------------------------------------------------

def test_reindex_without_a_database_reports_an_error(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["reindex"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "no database configured" in out


def test_reindex_without_force_calls_reconcile(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_index = Mock()
    mock_index.reconcile_agent.return_value = 2
    _patch_index(monkeypatch, mock_index)

    exit_code = cli.main(["reindex"])
    out = capsys.readouterr().out

    assert exit_code == 0
    mock_index.reconcile_agent.assert_called_once()
    mock_index.force_rebuild_agent.assert_not_called()
    assert "reindexed 2 file(s)" in out


def test_reindex_reports_already_up_to_date_when_nothing_changed(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_index = Mock()
    mock_index.reconcile_agent.return_value = 0
    _patch_index(monkeypatch, mock_index)

    exit_code = cli.main(["reindex"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "already up to date" in out


def test_reindex_with_force_calls_force_rebuild(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_index = Mock()
    mock_index.force_rebuild_agent.return_value = 7
    _patch_index(monkeypatch, mock_index)

    exit_code = cli.main(["reindex", "--force"])
    out = capsys.readouterr().out

    assert exit_code == 0
    mock_index.force_rebuild_agent.assert_called_once()
    mock_index.reconcile_agent.assert_not_called()
    assert "force-reindexed — 7 chunk(s)" in out


def test_reindex_filters_by_agent(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _agent_root(tmp_path, "researcher")
    _patch_config(monkeypatch, tmp_path, agent_ids=("main", "researcher"))
    mock_index = Mock()
    mock_index.reconcile_agent.return_value = 0
    _patch_index(monkeypatch, mock_index)

    exit_code = cli.main(["reindex", "--agent", "researcher"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "researcher:" in out
    assert "main:" not in out


# ---------------------------------------------------------------------------
# pin / unpin / pins
# ---------------------------------------------------------------------------

def test_pin_without_a_database_reports_an_error(monkeypatch, tmp_path, capsys):
    root = _agent_root(tmp_path, "main")
    MemoryFileRepository(root).remember("goal", "content")
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["pin", "goal", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "no database configured" in out


def test_pin_an_existing_note_succeeds(monkeypatch, tmp_path, capsys):
    root = _agent_root(tmp_path, "main")
    MemoryFileRepository(root).remember("goal", "content")
    _patch_config(monkeypatch, tmp_path)
    mock_index = Mock()
    _patch_index(monkeypatch, mock_index)

    exit_code = cli.main(["pin", "goal", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    mock_index.pin_file.assert_called_once_with("main", "memory/topics/goal.md")
    assert "pinned 'goal'" in out


def test_pin_a_missing_note_reports_an_error(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_index = Mock()
    _patch_index(monkeypatch, mock_index)

    exit_code = cli.main(["pin", "never-saved", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "No note found" in out
    mock_index.pin_file.assert_not_called()


def test_unpin_without_a_database_reports_an_error(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["unpin", "goal", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "no database configured" in out


def test_unpin_succeeds_even_for_a_missing_note(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_index = Mock()
    _patch_index(monkeypatch, mock_index)

    exit_code = cli.main(["unpin", "goal", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    mock_index.unpin_file.assert_called_once_with("main", "memory/topics/goal.md")
    assert "unpinned 'goal'" in out


def test_pins_without_a_database_reports_an_error(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["pins"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "no database configured" in out


def test_pins_lists_every_pinned_note(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_index = Mock()
    mock_index.pinned_files.return_value = ["memory/topics/a.md", "memory/topics/b.md"]
    _patch_index(monkeypatch, mock_index)

    exit_code = cli.main(["pins"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "2 pinned note(s)" in out
    assert "- a" in out
    assert "- b" in out


def test_pins_filters_by_agent(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _agent_root(tmp_path, "researcher")
    _patch_config(monkeypatch, tmp_path, agent_ids=("main", "researcher"))
    mock_index = Mock()
    mock_index.pinned_files.return_value = []
    _patch_index(monkeypatch, mock_index)

    exit_code = cli.main(["pins", "--agent", "researcher"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "researcher:" in out
    assert "main:" not in out


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

def test_doctor_reports_ok_when_nothing_pending(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["doctor"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "OK: no un-migrated legacy data found." in out


def test_doctor_warns_about_pending_legacy_notes(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    legacy_dir = tmp_path / "memory" / "main"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "old-note.md").write_text("legacy content", encoding="utf-8")
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["doctor"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "not yet migrated" in out


def test_doctor_warns_about_missing_workspace(monkeypatch, tmp_path, capsys):
    # No _agent_root() call — the agent has no workspace directory at all.
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["doctor"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "no workspace directory yet" in out


# ---------------------------------------------------------------------------
# _build_index (Stage One Phase 4, slice C: embedding provider construction)
# ---------------------------------------------------------------------------

def test_build_index_passes_no_embedding_provider_without_embeddings_config(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    import minion_assist.config as config

    monkeypatch.setattr(config, "database", SimpleNamespace(url="postgresql://fake/db"))
    monkeypatch.setattr(config, "embeddings", None)
    mock_index_cls = Mock()

    with patch("minion_assist.memory.postgres_index.PostgresMemoryIndex", mock_index_cls):
        cli._build_index()

    mock_index_cls.assert_called_once_with(
        "postgresql://fake/db", embedding_dimensions=None, embedding_provider=None
    )


def test_build_index_constructs_an_embedding_provider_when_configured(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    import minion_assist.config as config

    monkeypatch.setattr(config, "database", SimpleNamespace(url="postgresql://fake/db"))
    monkeypatch.setattr(config, "embeddings", SimpleNamespace(
        provider=SimpleNamespace(base_url="http://localhost:1234/v1", api_key="k"),
        model="nomic-embed-text",
        dimensions=768,
    ))
    mock_index_cls = Mock()
    mock_provider_cls = Mock()

    with patch("minion_assist.memory.postgres_index.PostgresMemoryIndex", mock_index_cls), \
         patch("minion_assist.providers.embeddings.EmbeddingProvider", mock_provider_cls):
        cli._build_index()

    mock_provider_cls.assert_called_once_with(
        base_url="http://localhost:1234/v1", api_key="k",
        model="nomic-embed-text", dimensions=768,
    )
    call_kwargs = mock_index_cls.call_args.kwargs
    assert call_kwargs["embedding_dimensions"] == 768
    assert call_kwargs["embedding_provider"] is mock_provider_cls.return_value

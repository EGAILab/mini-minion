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


def _patch_db(monkeypatch, mock_db):
    """Make `_build_db()` return a mock instead of touching a real database."""
    monkeypatch.setattr(cli, "_build_db", lambda: mock_db)


def _patch_provider(monkeypatch, mock_provider):
    """Make `_build_provider()` return a mock instead of constructing a real one."""
    monkeypatch.setattr(cli, "_build_provider", lambda agent_id: mock_provider)


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


def test_status_without_deep_never_builds_a_db(monkeypatch, tmp_path, capsys):
    def _fail_if_called():
        raise AssertionError("status without --deep must not call _build_db()")

    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_build_db", _fail_if_called)

    exit_code = cli.main(["status"])

    assert exit_code == 0


# --- status --deep: queue lag / embedding coverage (R2-GAP-016) ---

def _mock_lag(**overrides) -> dict:
    lane = {
        "pending_count": 0, "oldest_pending_age_s": None,
        "running_count": 0, "oldest_running_age_s": None,
        "failed_count": 0,
    }
    lag = {"capture": dict(lane), "commitment": dict(lane), "message_embedding": dict(lane)}
    for key, patch_dict in overrides.items():
        lag[key].update(patch_dict)
    return lag


def test_status_deep_reports_queue_lag_from_the_database(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_index(monkeypatch, None)
    mock_db = Mock()
    mock_db.queue_lag_summary.return_value = _mock_lag(capture={"pending_count": 3})
    mock_db.embedding_coverage_summary.return_value = None
    monkeypatch.setattr(cli, "_build_db", lambda: mock_db)

    exit_code = cli.main(["status", "--deep"])
    out = capsys.readouterr().out

    assert exit_code == 0
    mock_db.queue_lag_summary.assert_called_once_with("main")
    assert "capture_pending=3" in out


def test_status_deep_reports_stuck_running_and_failed_jobs(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_index(monkeypatch, None)
    mock_db = Mock()
    mock_db.queue_lag_summary.return_value = _mock_lag(
        capture={"running_count": 1}, message_embedding={"failed_count": 2},
    )
    mock_db.embedding_coverage_summary.return_value = None
    monkeypatch.setattr(cli, "_build_db", lambda: mock_db)

    exit_code = cli.main(["status", "--deep"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "capture_running=1" in out
    assert "message_embedding_failed=2" in out


def test_status_deep_reports_missing_embedding_coverage(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_index(monkeypatch, None)
    mock_db = Mock()
    mock_db.queue_lag_summary.return_value = _mock_lag()
    mock_db.embedding_coverage_summary.return_value = {
        "missing_count": 9, "model_identity": "test-endpoint::test-model",
    }
    monkeypatch.setattr(cli, "_build_db", lambda: mock_db)

    exit_code = cli.main(["status", "--deep"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "9 message(s)" in out
    assert "test-endpoint::test-model" in out


def test_status_deep_without_a_database_reports_queues_unavailable(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_index(monkeypatch, None)

    exit_code = cli.main(["status", "--deep"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "queues: no database configured" in out


def test_status_deep_survives_a_queue_lag_query_failure(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_index(monkeypatch, None)
    mock_db = Mock()
    mock_db.queue_lag_summary.side_effect = Exception("connection lost")
    mock_db.embedding_coverage_summary.return_value = None
    monkeypatch.setattr(cli, "_build_db", lambda: mock_db)

    exit_code = cli.main(["status", "--deep"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "queues: unavailable" in out
    assert "connection lost" in out


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
    mock_index.get_boundary.return_value = None  # Stage One Phase 6, slice A
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
# retention (MEM-GAP-015)
# ---------------------------------------------------------------------------

def _mock_db_for_retention(counts: dict | None = None) -> Mock:
    db = Mock()
    db.prune_operational_tables.return_value = counts or {
        "capture_jobs": 0, "commitment_jobs": 0, "message_embedding_jobs": 0,
    }
    return db


def test_retention_without_a_database_reports_an_error(monkeypatch, tmp_path, capsys):
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["retention"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "no database configured" in out


def test_retention_dry_run_is_default(monkeypatch, tmp_path, capsys):
    _patch_config(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "memory_retention", SimpleNamespace(retention_days=30))
    mock_db = _mock_db_for_retention({"capture_jobs": 2, "commitment_jobs": 0, "message_embedding_jobs": 1})
    _patch_db(monkeypatch, mock_db)
    _patch_index(monkeypatch, None)

    exit_code = cli.main(["retention"])
    out = capsys.readouterr().out

    assert exit_code == 0
    mock_db.prune_operational_tables.assert_called_once_with(30, dry_run=True)
    assert "would be deleted" in out
    assert "3 row(s)" in out
    assert "Re-run with --apply" in out


def test_retention_apply_deletes_and_reports_counts(monkeypatch, tmp_path, capsys):
    _patch_config(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "memory_retention", SimpleNamespace(retention_days=30))
    mock_db = _mock_db_for_retention({"capture_jobs": 5, "commitment_jobs": 0, "message_embedding_jobs": 0})
    _patch_db(monkeypatch, mock_db)
    _patch_index(monkeypatch, None)

    exit_code = cli.main(["retention", "--apply"])
    out = capsys.readouterr().out

    assert exit_code == 0
    mock_db.prune_operational_tables.assert_called_once_with(30, dry_run=False)
    assert "5 row(s)" in out
    assert "deleted:" in out
    assert "Re-run with --apply" not in out


def test_retention_days_override(monkeypatch, tmp_path, capsys):
    _patch_config(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "memory_retention", SimpleNamespace(retention_days=30))
    mock_db = _mock_db_for_retention()
    _patch_db(monkeypatch, mock_db)
    _patch_index(monkeypatch, None)

    cli.main(["retention", "--days", "7"])

    mock_db.prune_operational_tables.assert_called_once_with(7, dry_run=True)


def test_retention_skips_the_index_when_none_configured(monkeypatch, tmp_path, capsys):
    _patch_config(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "memory_retention", SimpleNamespace(retention_days=30))
    mock_db = _mock_db_for_retention({"capture_jobs": 1, "commitment_jobs": 0, "message_embedding_jobs": 0})
    _patch_db(monkeypatch, mock_db)
    _patch_index(monkeypatch, None)

    exit_code = cli.main(["retention"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "recall_events" not in out


def test_retention_merges_index_counts_when_configured(monkeypatch, tmp_path, capsys):
    _patch_config(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "memory_retention", SimpleNamespace(retention_days=30))
    mock_db = _mock_db_for_retention({"capture_jobs": 1, "commitment_jobs": 0, "message_embedding_jobs": 0})
    _patch_db(monkeypatch, mock_db)
    mock_index = Mock()
    mock_index.prune_operational_tables.return_value = {
        "recall_events": 4, "consolidation_previews": 0, "import_previews": 0,
    }
    _patch_index(monkeypatch, mock_index)

    exit_code = cli.main(["retention"])
    out = capsys.readouterr().out

    assert exit_code == 0
    mock_index.prune_operational_tables.assert_called_once_with(30, dry_run=True)
    assert "recall_events: 4" in out


# ---------------------------------------------------------------------------
# verify-deletions (R2-GAP-007)
# ---------------------------------------------------------------------------

def _incomplete_tombstone(**overrides) -> dict:
    base = {
        "agent_id": "main", "session_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "jsonl_deleted": True, "db_deleted": False, "evidence_cleaned": False,
        "proposal_ids": [],
    }
    base.update(overrides)
    return base


def test_verify_deletions_without_a_database_reports_an_error(monkeypatch, tmp_path, capsys):
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["verify-deletions"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "no database configured" in out


def test_verify_deletions_reports_none_found(monkeypatch, tmp_path, capsys):
    _patch_config(monkeypatch, tmp_path)
    mock_db = Mock()
    mock_db.list_incomplete_deletion_tombstones.return_value = []
    _patch_db(monkeypatch, mock_db)

    exit_code = cli.main(["verify-deletions"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "No incomplete session deletions found." in out


def test_verify_deletions_without_retry_only_lists(monkeypatch, tmp_path, capsys):
    _patch_config(monkeypatch, tmp_path)
    mock_db = Mock()
    mock_db.list_incomplete_deletion_tombstones.return_value = [_incomplete_tombstone()]
    _patch_db(monkeypatch, mock_db)

    exit_code = cli.main(["verify-deletions"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "1 incomplete session deletion" in out
    assert "db=PENDING" in out
    assert "Re-run with --retry" in out
    mock_db.delete_session.assert_not_called()


def test_verify_deletions_retry_finishes_db_and_evidence_phases(monkeypatch, tmp_path, capsys):
    _patch_config(monkeypatch, tmp_path)
    mock_db = Mock()
    mock_db.list_incomplete_deletion_tombstones.return_value = [_incomplete_tombstone()]
    mock_db.delete_session.return_value = {"messages": 2, "proposal_ids": [5]}
    _patch_db(monkeypatch, mock_db)
    _patch_index(monkeypatch, None)
    mock_service = Mock()
    monkeypatch.setattr(cli, "_build_service", lambda *a, **kw: mock_service)

    exit_code = cli.main(["verify-deletions", "--retry"])
    out = capsys.readouterr().out

    assert exit_code == 0
    mock_db.delete_session.assert_called_once_with("main", "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    mock_db.mark_deletion_db_done.assert_called_once_with(
        "main", "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", [5]
    )
    mock_service.forget_proposals.assert_called_once_with([5])
    mock_db.mark_deletion_evidence_done.assert_called_once_with(
        "main", "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    )
    assert "finished" in out


def test_verify_deletions_retry_skips_a_missing_jsonl_phase(monkeypatch, tmp_path, capsys):
    _patch_config(monkeypatch, tmp_path)
    mock_db = Mock()
    mock_db.list_incomplete_deletion_tombstones.return_value = [
        _incomplete_tombstone(jsonl_deleted=False)
    ]
    _patch_db(monkeypatch, mock_db)

    exit_code = cli.main(["verify-deletions", "--retry"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "needs manual attention" in out
    mock_db.delete_session.assert_not_called()


def test_verify_deletions_retry_skips_the_db_phase_when_already_done(monkeypatch, tmp_path, capsys):
    _patch_config(monkeypatch, tmp_path)
    mock_db = Mock()
    mock_db.list_incomplete_deletion_tombstones.return_value = [
        _incomplete_tombstone(db_deleted=True, proposal_ids=[9])
    ]
    _patch_db(monkeypatch, mock_db)
    _patch_index(monkeypatch, None)
    mock_service = Mock()
    monkeypatch.setattr(cli, "_build_service", lambda *a, **kw: mock_service)

    cli.main(["verify-deletions", "--retry"])

    mock_db.delete_session.assert_not_called()
    mock_service.forget_proposals.assert_called_once_with([9])


def test_verify_deletions_retry_skips_evidence_cleanup_when_no_proposals(
    monkeypatch, tmp_path, capsys
):
    _patch_config(monkeypatch, tmp_path)
    mock_db = Mock()
    mock_db.list_incomplete_deletion_tombstones.return_value = [_incomplete_tombstone()]
    mock_db.delete_session.return_value = {"messages": 1, "proposal_ids": []}
    _patch_db(monkeypatch, mock_db)
    _patch_index(monkeypatch, None)
    mock_service = Mock()
    monkeypatch.setattr(cli, "_build_service", lambda *a, **kw: mock_service)

    cli.main(["verify-deletions", "--retry"])

    mock_service.forget_proposals.assert_not_called()
    mock_db.mark_deletion_evidence_done.assert_called_once()


def test_verify_deletions_retry_reports_a_repeated_db_failure_without_crashing(
    monkeypatch, tmp_path, capsys
):
    _patch_config(monkeypatch, tmp_path)
    mock_db = Mock()
    mock_db.list_incomplete_deletion_tombstones.return_value = [_incomplete_tombstone()]
    mock_db.delete_session.side_effect = Exception("still down")
    _patch_db(monkeypatch, mock_db)

    exit_code = cli.main(["verify-deletions", "--retry"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "database cleanup failed again" in out
    assert "still down" in out
    mock_db.mark_deletion_db_done.assert_not_called()


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


# ---------------------------------------------------------------------------
# consolidate (Stage One Phase 5, slice D)
# ---------------------------------------------------------------------------

def _fake_preview(**overrides) -> dict:
    from minion_assist.memory.consolidation import _hash_text

    base = {
        "id": 5, "agent_id": "main", "proposal_id": 1, "target_kind": "new_topic",
        "target_key": "dark-mode", "based_on_content_hash": _hash_text(""),
        "drafted_content": "Drafted.", "rationale": "New preference.",
    }
    base.update(overrides)
    return base


def test_consolidate_list_shows_ranked_proposals(monkeypatch, tmp_path, capsys):
    import minion_assist.memory.consolidation as consolidation

    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_db(monkeypatch, Mock())
    _patch_index(monkeypatch, Mock())
    monkeypatch.setattr(consolidation, "rank_proposals", lambda db, index, agent_id: [
        {"id": 1, "score": 8, "claim_text": "User prefers dark mode."},
    ])

    exit_code = cli.main(["consolidate", "list", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "#1" in out
    assert "User prefers dark mode." in out


def test_consolidate_list_respects_top(monkeypatch, tmp_path, capsys):
    import minion_assist.memory.consolidation as consolidation

    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_db(monkeypatch, Mock())
    _patch_index(monkeypatch, Mock())
    monkeypatch.setattr(consolidation, "rank_proposals", lambda db, index, agent_id: [
        {"id": i, "score": 0, "claim_text": f"Claim {i}."} for i in range(1, 6)
    ])

    exit_code = cli.main(["consolidate", "list", "--agent", "main", "--top", "2"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert out.count("Claim") == 2


def test_consolidate_list_reports_no_pending_proposals(monkeypatch, tmp_path, capsys):
    import minion_assist.memory.consolidation as consolidation

    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_db(monkeypatch, Mock())
    _patch_index(monkeypatch, Mock())
    monkeypatch.setattr(consolidation, "rank_proposals", lambda db, index, agent_id: [])

    exit_code = cli.main(["consolidate", "list", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "no pending proposals" in out


def test_consolidate_list_without_a_database_reports_an_error(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["consolidate", "list", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "no database configured" in out


def test_consolidate_preview_shows_the_drafted_report(monkeypatch, tmp_path, capsys):
    import minion_assist.memory.consolidation as consolidation

    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_db(monkeypatch, Mock())
    _patch_index(monkeypatch, Mock())
    _patch_provider(monkeypatch, Mock())
    mock_consolidator = Mock()
    mock_consolidator.preview.return_value = _fake_preview()
    monkeypatch.setattr(consolidation, "MemoryConsolidator", lambda *a, **kw: mock_consolidator)

    exit_code = cli.main(["consolidate", "preview", "1", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "dark-mode" in out
    assert "Preview id: 5" in out
    mock_consolidator.preview.assert_called_once_with(1)


def test_consolidate_preview_reports_an_unknown_proposal(monkeypatch, tmp_path, capsys):
    import minion_assist.memory.consolidation as consolidation

    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_db(monkeypatch, Mock())
    _patch_index(monkeypatch, Mock())
    _patch_provider(monkeypatch, Mock())
    mock_consolidator = Mock()
    mock_consolidator.preview.side_effect = ValueError("No proposal with id 999")
    monkeypatch.setattr(consolidation, "MemoryConsolidator", lambda *a, **kw: mock_consolidator)

    exit_code = cli.main(["consolidate", "preview", "999", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "Error:" in out


def test_consolidate_explain_shows_the_report(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_index = Mock()
    mock_index.get_consolidation_preview.return_value = _fake_preview()
    _patch_index(monkeypatch, mock_index)
    mock_db = Mock()
    mock_db.get_proposal.return_value = {"status": "pending", "rejected_reason": ""}
    _patch_db(monkeypatch, mock_db)

    exit_code = cli.main(["consolidate", "explain", "5", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "dark-mode" in out
    assert "WARNING: stale" not in out  # nothing on disk yet, hash matches ""


def test_consolidate_explain_flags_a_stale_preview(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_index = Mock()
    # based_on_content_hash won't match "" (empty/never-existed) since the
    # note was hand-edited on disk after the preview was drafted.
    mock_index.get_consolidation_preview.return_value = _fake_preview(
        target_kind="revise_topic", based_on_content_hash="not-the-real-hash"
    )
    _patch_index(monkeypatch, mock_index)
    _patch_db(monkeypatch, Mock(get_proposal=Mock(return_value=None)))
    files = MemoryFileRepository(_agent_root(tmp_path, "main"))
    files.remember("dark-mode", "Hand-edited content.")

    exit_code = cli.main(["consolidate", "explain", "5", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "WARNING: stale" in out


def test_consolidate_explain_reports_an_unknown_preview(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_index = Mock()
    mock_index.get_consolidation_preview.return_value = None
    _patch_index(monkeypatch, mock_index)

    exit_code = cli.main(["consolidate", "explain", "999", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "Error:" in out


def test_consolidate_approve_applies_and_reports_success(monkeypatch, tmp_path, capsys):
    import minion_assist.memory.consolidation as consolidation

    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_db(monkeypatch, Mock())
    _patch_index(monkeypatch, Mock())
    mock_consolidator = Mock()
    mock_consolidator.approve.return_value = {
        "proposal_id": 1, "target_key": "dark-mode", "rel_path": "memory/topics/dark-mode.md",
    }
    monkeypatch.setattr(consolidation, "MemoryConsolidator", lambda *a, **kw: mock_consolidator)

    exit_code = cli.main(["consolidate", "approve", "5", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "applied" in out
    mock_consolidator.approve.assert_called_once_with(5)


def test_consolidate_approve_reports_a_stale_preview(monkeypatch, tmp_path, capsys):
    import minion_assist.memory.consolidation as consolidation
    from minion_assist.memory.consolidation import StaleProposalError

    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_db(monkeypatch, Mock())
    _patch_index(monkeypatch, Mock())
    mock_consolidator = Mock()
    mock_consolidator.approve.side_effect = StaleProposalError("Topic changed since draft.")
    monkeypatch.setattr(consolidation, "MemoryConsolidator", lambda *a, **kw: mock_consolidator)

    exit_code = cli.main(["consolidate", "approve", "5", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "Error:" in out


def test_consolidate_reject_marks_rejected(monkeypatch, tmp_path, capsys):
    import minion_assist.memory.consolidation as consolidation

    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_db(monkeypatch, Mock())
    mock_consolidator = Mock()
    monkeypatch.setattr(consolidation, "MemoryConsolidator", lambda *a, **kw: mock_consolidator)

    exit_code = cli.main(["consolidate", "reject", "3", "--agent", "main", "--reason", "Not useful."])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "rejected" in out
    mock_consolidator.reject.assert_called_once_with(3, reason="Not useful.")


def test_consolidate_reject_never_builds_an_index(monkeypatch, tmp_path, capsys):
    import minion_assist.memory.consolidation as consolidation

    def _fail_if_called():
        raise AssertionError("_build_index should not be called by reject")

    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_db(monkeypatch, Mock())
    monkeypatch.setattr(cli, "_build_index", _fail_if_called)
    monkeypatch.setattr(consolidation, "MemoryConsolidator", lambda *a, **kw: Mock())

    exit_code = cli.main(["consolidate", "reject", "3", "--agent", "main"])

    assert exit_code == 0


def test_consolidate_rollback_restores_and_reports(monkeypatch, tmp_path, capsys):
    import minion_assist.memory.consolidation as consolidation

    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_db(monkeypatch, Mock())
    _patch_index(monkeypatch, Mock())
    mock_consolidator = Mock()
    mock_consolidator.rollback.return_value = {
        "target_key": "dark-mode", "proposal_id": 1, "restored_content": "",
    }
    monkeypatch.setattr(consolidation, "MemoryConsolidator", lambda *a, **kw: mock_consolidator)

    exit_code = cli.main(["consolidate", "rollback", "dark-mode", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "rolled back" in out
    mock_consolidator.rollback.assert_called_once_with("dark-mode")


def test_consolidate_rollback_reports_no_history(monkeypatch, tmp_path, capsys):
    import minion_assist.memory.consolidation as consolidation

    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_db(monkeypatch, Mock())
    _patch_index(monkeypatch, Mock())
    mock_consolidator = Mock()
    mock_consolidator.rollback.side_effect = ValueError("No revision history for topic 'x' to roll back")
    monkeypatch.setattr(consolidation, "MemoryConsolidator", lambda *a, **kw: mock_consolidator)

    exit_code = cli.main(["consolidate", "rollback", "x", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "Error:" in out


def test_consolidate_backfill_reports_enqueued_count(monkeypatch, tmp_path, capsys):
    import minion_assist.memory.consolidation as consolidation

    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    config.agents["main"] = SimpleNamespace(model=SimpleNamespace(id="test-model"))
    _patch_db(monkeypatch, Mock())
    monkeypatch.setattr(consolidation, "backfill_agent", lambda db, agent_id, model_id: 3)

    exit_code = cli.main(["consolidate", "backfill", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "enqueued 3" in out


def test_consolidate_backfill_without_a_database_reports_an_error(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["consolidate", "backfill", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "no database configured" in out


# ---------------------------------------------------------------------------
# import (Stage One Phase 7, slice E)
# ---------------------------------------------------------------------------

def _fake_import_preview(**overrides) -> dict:
    from minion_assist.memory.import_review import _hash_text

    base = {
        "id": 5, "agent_id": "main", "import_key": "_auto_extracted", "target_kind": "new_topic",
        "target_key": "dark-mode", "based_on_content_hash": _hash_text(""),
        "drafted_content": "Drafted.", "rationale": "New preference.",
    }
    base.update(overrides)
    return base


def test_import_list_shows_each_import_key(monkeypatch, tmp_path, capsys):
    root = _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    MemoryFileRepository(root).remember_import("_auto_extracted", "Some content.")

    exit_code = cli.main(["import", "list", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "_auto_extracted" in out


def test_import_list_reports_no_imports(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["import", "list", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "no quarantined imports" in out


def test_import_preview_shows_the_drafted_report(monkeypatch, tmp_path, capsys):
    import minion_assist.memory.import_review as import_review

    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_index(monkeypatch, Mock())
    _patch_provider(monkeypatch, Mock())
    mock_reviewer = Mock()
    mock_reviewer.preview.return_value = _fake_import_preview()
    monkeypatch.setattr(import_review, "ImportReviewer", lambda *a, **kw: mock_reviewer)

    exit_code = cli.main(["import", "preview", "_auto_extracted", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "dark-mode" in out
    assert "Preview id: 5" in out
    mock_reviewer.preview.assert_called_once_with("_auto_extracted")


def test_import_preview_reports_an_unknown_key(monkeypatch, tmp_path, capsys):
    import minion_assist.memory.import_review as import_review

    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_index(monkeypatch, Mock())
    _patch_provider(monkeypatch, Mock())
    mock_reviewer = Mock()
    mock_reviewer.preview.side_effect = ValueError("No quarantined import with key 'x'")
    monkeypatch.setattr(import_review, "ImportReviewer", lambda *a, **kw: mock_reviewer)

    exit_code = cli.main(["import", "preview", "x", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "Error:" in out


def test_import_preview_without_a_database_reports_an_error(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_index(monkeypatch, None)

    exit_code = cli.main(["import", "preview", "_auto_extracted", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "no database configured" in out


def test_import_explain_shows_the_report(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_index = Mock()
    mock_index.get_import_preview.return_value = _fake_import_preview()
    _patch_index(monkeypatch, mock_index)

    exit_code = cli.main(["import", "explain", "5", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "dark-mode" in out
    assert "WARNING: stale" not in out  # nothing on disk yet, hash matches ""


def test_import_explain_flags_a_stale_preview(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_index = Mock()
    mock_index.get_import_preview.return_value = _fake_import_preview(
        target_kind="revise_topic", based_on_content_hash="not-the-real-hash"
    )
    _patch_index(monkeypatch, mock_index)
    files = MemoryFileRepository(_agent_root(tmp_path, "main"))
    files.remember("dark-mode", "Hand-edited content.")

    exit_code = cli.main(["import", "explain", "5", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "WARNING: stale" in out


def test_import_explain_reports_an_unknown_preview(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_index = Mock()
    mock_index.get_import_preview.return_value = None
    _patch_index(monkeypatch, mock_index)

    exit_code = cli.main(["import", "explain", "999", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "Error:" in out


def test_import_approve_applies_and_reports_success(monkeypatch, tmp_path, capsys):
    import minion_assist.memory.import_review as import_review

    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_index(monkeypatch, Mock())
    mock_reviewer = Mock()
    mock_reviewer.approve.return_value = {
        "import_key": "_auto_extracted", "target_key": "dark-mode",
        "rel_path": "memory/topics/dark-mode.md",
    }
    monkeypatch.setattr(import_review, "ImportReviewer", lambda *a, **kw: mock_reviewer)

    exit_code = cli.main(["import", "approve", "5", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "applied" in out
    assert "retired" in out
    mock_reviewer.approve.assert_called_once_with(5)


def test_import_approve_reports_a_stale_preview(monkeypatch, tmp_path, capsys):
    import minion_assist.memory.import_review as import_review
    from minion_assist.memory.import_review import StaleImportError

    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_index(monkeypatch, Mock())
    mock_reviewer = Mock()
    mock_reviewer.approve.side_effect = StaleImportError("Topic changed since draft.")
    monkeypatch.setattr(import_review, "ImportReviewer", lambda *a, **kw: mock_reviewer)

    exit_code = cli.main(["import", "approve", "5", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "Error:" in out


def test_import_approve_without_a_database_reports_an_error(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_index(monkeypatch, None)

    exit_code = cli.main(["import", "approve", "5", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "no database configured" in out


def test_import_reject_retires_the_import(monkeypatch, tmp_path, capsys):
    import minion_assist.memory.import_review as import_review

    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_index(monkeypatch, Mock())
    mock_reviewer = Mock()
    monkeypatch.setattr(import_review, "ImportReviewer", lambda *a, **kw: mock_reviewer)

    exit_code = cli.main(
        ["import", "reject", "_auto_extracted", "--agent", "main", "--reason", "Not useful."]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "rejected" in out
    mock_reviewer.reject.assert_called_once_with("_auto_extracted", reason="Not useful.")


def test_import_reject_reports_an_unknown_key(monkeypatch, tmp_path, capsys):
    import minion_assist.memory.import_review as import_review

    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_index(monkeypatch, Mock())
    mock_reviewer = Mock()
    mock_reviewer.reject.side_effect = ValueError("No quarantined import with key 'x'")
    monkeypatch.setattr(import_review, "ImportReviewer", lambda *a, **kw: mock_reviewer)

    exit_code = cli.main(["import", "reject", "x", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "Error:" in out


def test_import_reject_without_a_database_reports_an_error(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_index(monkeypatch, None)

    exit_code = cli.main(["import", "reject", "_auto_extracted", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "no database configured" in out


# ---------------------------------------------------------------------------
# commitments (Stage One Phase 6, slice C)
# ---------------------------------------------------------------------------

def _fake_commitment(**overrides) -> dict:
    base = {
        "id": 1, "agent_id": "main", "session_id": "sess-1", "channel": "!room:example.org",
        "kind": "open_loop", "sensitivity": "routine", "source": "inferred_user_context",
        "status": "pending", "reason": "User mentioned an interview.",
        "suggested_text": "How did it go?", "dedupe_key": "interview:2026-08-01",
        "confidence": 0.8, "due_earliest": 2_000_000_000.0, "due_latest": 2_000_010_000.0,
    }
    base.update(overrides)
    return base


def test_commitments_list_shows_each_commitment(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_db = Mock()
    mock_db.list_commitments.return_value = [_fake_commitment()]
    _patch_db(monkeypatch, mock_db)

    exit_code = cli.main(["commitments", "list", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "#1" in out
    assert "User mentioned an interview." in out
    mock_db.list_commitments.assert_called_once_with("main", status=None, channel=None)


def test_commitments_list_passes_status_and_channel_filters(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_db = Mock()
    mock_db.list_commitments.return_value = []
    _patch_db(monkeypatch, mock_db)

    cli.main(["commitments", "list", "--agent", "main", "--status", "pending", "--channel", "cli"])

    mock_db.list_commitments.assert_called_once_with("main", status="pending", channel="cli")


def test_commitments_list_reports_none_found(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_db = Mock()
    mock_db.list_commitments.return_value = []
    _patch_db(monkeypatch, mock_db)

    exit_code = cli.main(["commitments", "list", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "no commitments" in out


def test_commitments_list_without_a_database_reports_an_error(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["commitments", "list", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "no database configured" in out


def test_commitments_dismiss_marks_dismissed(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_db = Mock()
    mock_db.get_commitment.return_value = _fake_commitment()
    _patch_db(monkeypatch, mock_db)

    exit_code = cli.main(["commitments", "dismiss", "1", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "dismissed" in out.lower()
    mock_db.mark_commitment_dismissed.assert_called_once_with(1)


def test_commitments_dismiss_reports_an_unknown_id(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_db = Mock()
    mock_db.get_commitment.return_value = None
    _patch_db(monkeypatch, mock_db)

    exit_code = cli.main(["commitments", "dismiss", "999", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "Error:" in out
    mock_db.mark_commitment_dismissed.assert_not_called()


def test_commitments_dismiss_refuses_a_wrong_agent(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_db = Mock()
    mock_db.get_commitment.return_value = _fake_commitment(agent_id="researcher")
    _patch_db(monkeypatch, mock_db)

    exit_code = cli.main(["commitments", "dismiss", "1", "--agent", "main"])

    assert exit_code == 1
    mock_db.mark_commitment_dismissed.assert_not_called()


def test_commitments_dismiss_reports_an_already_handled_commitment(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_db = Mock()
    mock_db.get_commitment.return_value = _fake_commitment(status="sent")
    _patch_db(monkeypatch, mock_db)

    exit_code = cli.main(["commitments", "dismiss", "1", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "already" in out.lower()
    mock_db.mark_commitment_dismissed.assert_not_called()


def test_commitments_delete_removes_the_row(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_db = Mock()
    mock_db.delete_commitment.return_value = True
    _patch_db(monkeypatch, mock_db)

    exit_code = cli.main(["commitments", "delete", "1", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "deleted" in out.lower()
    mock_db.delete_commitment.assert_called_once_with("main", 1)


def test_commitments_delete_reports_an_unknown_id(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_db = Mock()
    mock_db.delete_commitment.return_value = False
    _patch_db(monkeypatch, mock_db)

    exit_code = cli.main(["commitments", "delete", "999", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "Error:" in out


def test_commitments_delete_without_a_database_reports_an_error(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)

    exit_code = cli.main(["commitments", "delete", "1", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "no database configured" in out


# ---------------------------------------------------------------------
# memory knowledge dashboard (Stage One Phase 7, slice C)
# ---------------------------------------------------------------------


def _empty_knowledge_index():
    """A mock PostgresMemoryIndex where every dashboard query returns nothing."""
    mock_index = Mock()
    mock_index.list_contradictions.return_value = []
    mock_index.list_stale_claims.return_value = []
    mock_index.list_low_confidence_claims.return_value = []
    mock_index.list_claims_missing_evidence.return_value = []
    mock_index.list_claims.return_value = []
    mock_index.list_claims_needing_privacy_review.return_value = []
    mock_index.list_claims_needing_reevaluation.return_value = []
    return mock_index


def test_knowledge_dashboard_shows_all_sections_by_default(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_index = _empty_knowledge_index()
    mock_index.list_contradictions.return_value = [
        {
            "from_claim_id": "c-1", "from_text": "The sky is green.", "from_status": "contested",
            "to_claim_id": "c-2", "to_text": "The sky is blue.", "to_status": "contested",
        }
    ]
    mock_index.list_stale_claims.return_value = [
        {"id": "c-3", "rel_path": "topics/x.md", "text": "Old fact.", "status": "supported",
         "observed_at": 0.0, "freshness": 0.1}
    ]
    mock_index.list_low_confidence_claims.return_value = [
        {"id": "c-4", "rel_path": "topics/x.md", "text": "Shaky fact.", "status": "unknown",
         "confidence": None}
    ]
    mock_index.list_claims_missing_evidence.return_value = [
        {"id": "c-5", "rel_path": "topics/x.md", "text": "Unsourced fact.", "status": "supported"}
    ]
    mock_index.list_claims.return_value = [
        {"id": "c-6", "rel_path": "topics/x.md", "text": "Open question.", "status": "unknown"}
    ]
    mock_index.list_claims_needing_privacy_review.return_value = [
        {"id": "c-7", "rel_path": "topics/x.md", "text": "Unclassified fact.", "status": "supported"}
    ]
    mock_index.list_claims_needing_reevaluation.return_value = [
        {"id": "c-8", "rel_path": "topics/x.md", "text": "Forgotten-source fact.", "status": "unknown"}
    ]
    _patch_index(monkeypatch, mock_index)

    exit_code = cli.main(["knowledge", "dashboard", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Contradictions" in out and "The sky is green." in out and "The sky is blue." in out
    assert "Stale claims" in out and "Old fact." in out
    assert "Low confidence" in out and "Shaky fact." in out
    assert "Missing provenance" in out and "Unsourced fact." in out
    assert "Open questions" in out and "Open question." in out
    assert "Privacy review" in out and "Unclassified fact." in out
    assert "Deletion coverage" in out and "Forgotten-source fact." in out
    mock_index.list_claims.assert_called_once_with("main", status="unknown")


def test_knowledge_dashboard_reports_none_for_empty_sections(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_index(monkeypatch, _empty_knowledge_index())

    exit_code = cli.main(["knowledge", "dashboard", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert out.count("(none)") == 7


def test_knowledge_dashboard_shows_a_dangling_contradiction_reference(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_index = _empty_knowledge_index()
    mock_index.list_contradictions.return_value = [
        {
            "from_claim_id": "c-1", "from_text": "Some claim.", "from_status": "contested",
            "to_claim_id": "c-typo", "to_text": None, "to_status": None,
        }
    ]
    _patch_index(monkeypatch, mock_index)

    exit_code = cli.main(["knowledge", "dashboard", "--agent", "main", "--section", "contradictions"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "dangling reference" in out


def test_knowledge_dashboard_section_flag_restricts_to_one_section(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_index = _empty_knowledge_index()
    _patch_index(monkeypatch, mock_index)

    exit_code = cli.main(["knowledge", "dashboard", "--agent", "main", "--section", "stale"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Stale claims" in out
    assert "Contradictions" not in out
    assert "Low confidence" not in out
    mock_index.list_stale_claims.assert_called_once()
    mock_index.list_contradictions.assert_not_called()
    mock_index.list_low_confidence_claims.assert_not_called()
    mock_index.list_claims.assert_not_called()


def test_knowledge_dashboard_without_a_database_reports_an_error(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_index(monkeypatch, None)

    exit_code = cli.main(["knowledge", "dashboard", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "no database configured" in out


# ---------------------------------------------------------------------
# memory knowledge compile (Stage One Phase 7, slice D)
# ---------------------------------------------------------------------


def test_knowledge_compile_writes_the_digest_file(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_index = Mock()
    mock_index.list_claims.return_value = [
        {"id": "c-1", "rel_path": "topics/x.md", "text": "Alice prefers dark mode.",
         "status": "supported"}
    ]
    _patch_index(monkeypatch, mock_index)

    exit_code = cli.main(["knowledge", "compile", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    digest_path = tmp_path / "workspaces" / "main" / "KNOWLEDGE_DIGEST.md"
    assert digest_path.exists()
    assert "Alice prefers dark mode." in digest_path.read_text(encoding="utf-8")
    assert "compiled 1 supported claim" in out
    mock_index.list_claims.assert_called_once_with("main", status="supported")


def test_knowledge_compile_reports_an_empty_digest_when_no_claims(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_index = Mock()
    mock_index.list_claims.return_value = []
    _patch_index(monkeypatch, mock_index)

    exit_code = cli.main(["knowledge", "compile", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 0
    digest_path = tmp_path / "workspaces" / "main" / "KNOWLEDGE_DIGEST.md"
    assert digest_path.exists()
    assert digest_path.read_text(encoding="utf-8") == ""
    assert "no supported claims yet" in out


def test_knowledge_compile_passes_max_chars_through(monkeypatch, tmp_path):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    mock_index = Mock()
    mock_index.list_claims.return_value = [
        {"id": "c-1", "rel_path": "topics/x.md", "text": "Fact.", "status": "supported"}
    ]
    _patch_index(monkeypatch, mock_index)

    cli.main(["knowledge", "compile", "--agent", "main", "--max-chars", "500"])

    digest_path = tmp_path / "workspaces" / "main" / "KNOWLEDGE_DIGEST.md"
    assert digest_path.exists()  # max_chars is exercised via compile_digest directly (see test_knowledge.py)


def test_knowledge_compile_without_a_database_reports_an_error(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_index(monkeypatch, None)

    exit_code = cli.main(["knowledge", "compile", "--agent", "main"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "no database configured" in out


# ---------------------------------------------------------------------
# memory knowledge forget (Stage One Phase 7, slice F)
# ---------------------------------------------------------------------


def test_knowledge_forget_reports_reevaluated_and_still_grounded_claims(monkeypatch, tmp_path, capsys):
    import minion_assist.memory.forgetting as forgetting

    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_index(monkeypatch, Mock())
    monkeypatch.setattr(
        forgetting, "forget_source",
        lambda index, files, agent_id, source_kind, source_ref: {
            "source_kind": source_kind, "source_ref": source_ref,
            "reevaluated": ["c-1"], "still_grounded": ["c-2"], "skipped_manual_review": [],
        },
    )

    exit_code = cli.main(
        ["knowledge", "forget", "--agent", "main", "--source-kind", "proposal", "--source-ref", "42"]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "forgot proposal:42" in out
    assert "c-1" in out
    assert "c-2" in out


def test_knowledge_forget_reports_claims_needing_manual_review(monkeypatch, tmp_path, capsys):
    import minion_assist.memory.forgetting as forgetting

    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_index(monkeypatch, Mock())
    monkeypatch.setattr(
        forgetting, "forget_source",
        lambda index, files, agent_id, source_kind, source_ref: {
            "source_kind": source_kind, "source_ref": source_ref,
            "reevaluated": [], "still_grounded": [],
            "skipped_manual_review": [{"claim_id": "c-1", "rel_path": "MEMORY.md"}],
        },
    )

    exit_code = cli.main(
        ["knowledge", "forget", "--agent", "main", "--source-kind", "proposal", "--source-ref", "42"]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "manual review" in out
    assert "MEMORY.md" in out


def test_knowledge_forget_reports_a_harmless_no_op(monkeypatch, tmp_path, capsys):
    import minion_assist.memory.forgetting as forgetting

    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_index(monkeypatch, Mock())
    monkeypatch.setattr(
        forgetting, "forget_source",
        lambda index, files, agent_id, source_kind, source_ref: {
            "source_kind": source_kind, "source_ref": source_ref,
            "reevaluated": [], "still_grounded": [], "skipped_manual_review": [],
        },
    )

    exit_code = cli.main(
        ["knowledge", "forget", "--agent", "main", "--source-kind", "proposal", "--source-ref", "999"]
    )
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "nothing to do" in out


def test_knowledge_forget_without_a_database_reports_an_error(monkeypatch, tmp_path, capsys):
    _agent_root(tmp_path, "main")
    _patch_config(monkeypatch, tmp_path)
    _patch_index(monkeypatch, None)

    exit_code = cli.main(
        ["knowledge", "forget", "--agent", "main", "--source-kind", "proposal", "--source-ref", "42"]
    )
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "no database configured" in out

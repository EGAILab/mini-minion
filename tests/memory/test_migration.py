"""Tests for memory/migration.py: Phase 0 legacy-notes merge tooling.

All tests are self-contained and use tmp_path as the fake workspace root —
no real filesystem side effects outside the test's own temp directory.
"""

from __future__ import annotations

import json

from minion_assist.memory.migration import (
    CLASSIFY_CONFLICT,
    CLASSIFY_MIGRATE,
    CLASSIFY_UNCHANGED,
    MigrationPlan,
    PlannedChange,
    apply_migration,
    discover_legacy_notes,
    dry_run_report,
    plan_migration,
    rollback_migration,
)


def _write_legacy_note(workspace, agent_id: str, key: str, content: str) -> None:
    """Create a legacy memory/{agent_id}/{key}.md note for a test."""
    notes_dir = workspace / "memory" / agent_id
    notes_dir.mkdir(parents=True, exist_ok=True)
    (notes_dir / f"{key}.md").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# discover_legacy_notes
# ---------------------------------------------------------------------------

def test_discover_legacy_notes_returns_empty_when_dir_missing(tmp_path):
    """No memory/{agent_id}/ directory at all -> empty list, no error."""
    assert discover_legacy_notes(tmp_path, "main") == []


def test_discover_legacy_notes_finds_files_sorted_by_key(tmp_path):
    """Notes are returned sorted by filename for deterministic output."""
    _write_legacy_note(tmp_path, "main", "zeta-notes", "z")
    _write_legacy_note(tmp_path, "main", "alpha-notes", "a")
    notes = discover_legacy_notes(tmp_path, "main")
    assert [n.key for n in notes] == ["alpha-notes", "zeta-notes"]


def test_discover_legacy_notes_computes_content_hash_and_size(tmp_path):
    """Hash/size reflect actual file bytes."""
    _write_legacy_note(tmp_path, "main", "note", "hello")
    [note] = discover_legacy_notes(tmp_path, "main")
    assert note.size == len(b"hello")
    assert len(note.content_hash) == 64  # sha256 hex digest length


def test_discover_legacy_notes_only_matches_md_files(tmp_path):
    """Non-.md files in the legacy notes dir are ignored."""
    notes_dir = tmp_path / "memory" / "main"
    notes_dir.mkdir(parents=True)
    (notes_dir / "note.md").write_text("x", encoding="utf-8")
    (notes_dir / "ignore.txt").write_text("y", encoding="utf-8")
    notes = discover_legacy_notes(tmp_path, "main")
    assert [n.key for n in notes] == ["note"]


# ---------------------------------------------------------------------------
# plan_migration: key mapping rules
# ---------------------------------------------------------------------------

def test_plan_migration_maps_user_context_to_user_md(tmp_path):
    """user_context.md is the always-resident profile; maps to USER.md."""
    _write_legacy_note(tmp_path, "main", "user_context", "I like tea.")
    plan = plan_migration(tmp_path, ["main"])
    [change] = plan.changes
    assert change.dest_path == tmp_path / "workspaces" / "main" / "USER.md"
    assert change.classification == CLASSIFY_MIGRATE


def test_plan_migration_quarantines_auto_extracted(tmp_path):
    """_auto_extracted.md is unreviewed extraction output -> memory/imports/."""
    _write_legacy_note(tmp_path, "main", "_auto_extracted", "fact 1\nfact 2")
    plan = plan_migration(tmp_path, ["main"])
    [change] = plan.changes
    expected = tmp_path / "workspaces" / "main" / "memory" / "imports" / "_auto_extracted.md"
    assert change.dest_path == expected


def test_plan_migration_quarantines_daily_notes_key(tmp_path):
    """_notes_YYYY-MM-DD.md (tools/memory.py's daily-log tool) -> memory/imports/."""
    _write_legacy_note(tmp_path, "main", "_notes_2026-01-01", "did a thing")
    plan = plan_migration(tmp_path, ["main"])
    [change] = plan.changes
    expected = tmp_path / "workspaces" / "main" / "memory" / "imports" / "_notes_2026-01-01.md"
    assert change.dest_path == expected


def test_plan_migration_maps_regular_note_to_topics(tmp_path):
    """An explicit save_memory note maps to memory/topics/{key}.md."""
    _write_legacy_note(tmp_path, "main", "api-rest-notes", "REST tips")
    plan = plan_migration(tmp_path, ["main"])
    [change] = plan.changes
    expected = tmp_path / "workspaces" / "main" / "memory" / "topics" / "api-rest-notes.md"
    assert change.dest_path == expected


# ---------------------------------------------------------------------------
# plan_migration: classification
# ---------------------------------------------------------------------------

def test_plan_migration_classifies_migrate_when_dest_absent(tmp_path):
    _write_legacy_note(tmp_path, "main", "note", "content")
    plan = plan_migration(tmp_path, ["main"])
    assert plan.changes[0].classification == CLASSIFY_MIGRATE


def test_plan_migration_classifies_unchanged_when_dest_identical(tmp_path):
    _write_legacy_note(tmp_path, "main", "note", "same content")
    dest = tmp_path / "workspaces" / "main" / "memory" / "topics" / "note.md"
    dest.parent.mkdir(parents=True)
    dest.write_text("same content", encoding="utf-8")

    plan = plan_migration(tmp_path, ["main"])
    assert plan.changes[0].classification == CLASSIFY_UNCHANGED


def test_plan_migration_classifies_conflict_when_dest_differs(tmp_path):
    _write_legacy_note(tmp_path, "main", "note", "legacy content")
    dest = tmp_path / "workspaces" / "main" / "memory" / "topics" / "note.md"
    dest.parent.mkdir(parents=True)
    dest.write_text("different content already there", encoding="utf-8")

    plan = plan_migration(tmp_path, ["main"])
    assert plan.changes[0].classification == CLASSIFY_CONFLICT


def test_plan_migration_tracks_workspaces_to_create(tmp_path):
    """Agents with no workspaces/{agent_id}/ dir yet are flagged."""
    plan = plan_migration(tmp_path, ["main", "researcher"])
    assert set(plan.workspaces_to_create) == {"main", "researcher"}


def test_plan_migration_does_not_flag_existing_workspace(tmp_path):
    (tmp_path / "workspaces" / "main").mkdir(parents=True)
    plan = plan_migration(tmp_path, ["main", "researcher"])
    assert plan.workspaces_to_create == ["researcher"]


def test_plan_migration_is_read_only(tmp_path):
    """plan_migration never creates or modifies any file."""
    _write_legacy_note(tmp_path, "main", "note", "content")
    plan_migration(tmp_path, ["main"])
    assert not (tmp_path / "workspaces").exists()


def test_plan_migration_counts(tmp_path):
    _write_legacy_note(tmp_path, "main", "a", "1")
    _write_legacy_note(tmp_path, "main", "b", "2")
    plan = plan_migration(tmp_path, ["main"])
    counts = plan.counts()
    assert counts[CLASSIFY_MIGRATE] == 2
    assert counts[CLASSIFY_UNCHANGED] == 0
    assert counts[CLASSIFY_CONFLICT] == 0


# ---------------------------------------------------------------------------
# dry_run_report
# ---------------------------------------------------------------------------

def test_dry_run_report_mentions_counts_and_agents(tmp_path):
    _write_legacy_note(tmp_path, "main", "note", "content")
    plan = plan_migration(tmp_path, ["main"])
    report = dry_run_report(plan)
    assert "main" in report
    assert "Notes to migrate: 1" in report
    assert "dry run" in report.lower()


def test_dry_run_report_warns_on_conflicts(tmp_path):
    _write_legacy_note(tmp_path, "main", "note", "legacy")
    dest = tmp_path / "workspaces" / "main" / "memory" / "topics" / "note.md"
    dest.parent.mkdir(parents=True)
    dest.write_text("different", encoding="utf-8")

    report = dry_run_report(plan_migration(tmp_path, ["main"]))
    assert "WARNING" in report


def test_dry_run_report_no_warning_without_conflicts(tmp_path):
    _write_legacy_note(tmp_path, "main", "note", "content")
    report = dry_run_report(plan_migration(tmp_path, ["main"]))
    assert "WARNING" not in report


def test_dry_run_report_is_deterministic(tmp_path):
    _write_legacy_note(tmp_path, "main", "b-note", "content")
    _write_legacy_note(tmp_path, "main", "a-note", "content")
    plan = plan_migration(tmp_path, ["main"])
    assert dry_run_report(plan) == dry_run_report(plan)


# ---------------------------------------------------------------------------
# apply_migration
# ---------------------------------------------------------------------------

def test_apply_migration_copies_note_to_destination(tmp_path):
    _write_legacy_note(tmp_path, "main", "note", "hello world")
    plan = plan_migration(tmp_path, ["main"])
    result = apply_migration(plan)

    dest = tmp_path / "workspaces" / "main" / "memory" / "topics" / "note.md"
    assert dest.read_text(encoding="utf-8") == "hello world"
    assert result.applied == 1
    assert result.ok


def test_apply_migration_does_not_delete_or_modify_source(tmp_path):
    """Source files under memory/{agent_id}/ are copied, never moved."""
    _write_legacy_note(tmp_path, "main", "note", "hello world")
    plan = plan_migration(tmp_path, ["main"])
    apply_migration(plan)

    source = tmp_path / "memory" / "main" / "note.md"
    assert source.exists()
    assert source.read_text(encoding="utf-8") == "hello world"


def test_apply_migration_creates_missing_workspace_dirs(tmp_path):
    plan = plan_migration(tmp_path, ["main", "researcher"])
    apply_migration(plan)
    assert (tmp_path / "workspaces" / "main").is_dir()
    assert (tmp_path / "workspaces" / "researcher").is_dir()


def test_apply_migration_skips_conflicts(tmp_path):
    _write_legacy_note(tmp_path, "main", "note", "legacy")
    dest = tmp_path / "workspaces" / "main" / "memory" / "topics" / "note.md"
    dest.parent.mkdir(parents=True)
    dest.write_text("existing different content", encoding="utf-8")

    plan = plan_migration(tmp_path, ["main"])
    result = apply_migration(plan)

    assert result.applied == 0
    assert result.skipped == 1
    assert dest.read_text(encoding="utf-8") == "existing different content"


def test_apply_migration_skips_unchanged(tmp_path):
    _write_legacy_note(tmp_path, "main", "note", "same")
    dest = tmp_path / "workspaces" / "main" / "memory" / "topics" / "note.md"
    dest.parent.mkdir(parents=True)
    dest.write_text("same", encoding="utf-8")

    plan = plan_migration(tmp_path, ["main"])
    result = apply_migration(plan)
    assert result.applied == 0
    assert result.skipped == 1


def test_apply_migration_writes_manifest(tmp_path):
    _write_legacy_note(tmp_path, "main", "note", "content")
    plan = plan_migration(tmp_path, ["main"])
    result = apply_migration(plan)

    assert result.manifest_path is not None
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["entries"]) == 1
    assert manifest["entries"][0]["agent_id"] == "main"


def test_apply_migration_no_manifest_when_nothing_applied(tmp_path):
    """An empty plan (e.g. all conflicts) produces no manifest file."""
    plan = MigrationPlan(workspace=tmp_path, agent_ids=("main",))
    result = apply_migration(plan)
    assert result.manifest_path is None
    assert result.applied == 0


def test_apply_migration_backs_up_stale_destination_before_overwrite(tmp_path):
    """Defensive path: if the plan is stale and the destination now exists
    with content different from what plan_migration saw, apply_migration
    must still back it up rather than silently clobbering it.
    """
    _write_legacy_note(tmp_path, "main", "note", "new content")
    source = tmp_path / "memory" / "main" / "note.md"

    dest = tmp_path / "workspaces" / "main" / "memory" / "topics" / "note.md"
    dest.parent.mkdir(parents=True)
    dest.write_text("stale destination content", encoding="utf-8")

    # Hand-construct a plan that (incorrectly, as if stale) classifies this as
    # "migrate" even though the destination already exists, to exercise the
    # defensive backup-before-overwrite branch directly.
    note = discover_legacy_notes(tmp_path, "main")[0]
    plan = MigrationPlan(workspace=tmp_path, agent_ids=("main",))
    plan.changes.append(PlannedChange(note, dest, CLASSIFY_MIGRATE))

    result = apply_migration(plan)

    assert dest.read_text(encoding="utf-8") == "new content"
    assert result.manifest_path is not None
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    backup_path = manifest["entries"][0]["backup_path"]
    assert backup_path is not None
    assert open(backup_path, encoding="utf-8").read() == "stale destination content"

    assert source.read_text(encoding="utf-8") == "new content"


# ---------------------------------------------------------------------------
# rollback_migration
# ---------------------------------------------------------------------------

def test_rollback_migration_deletes_newly_created_file(tmp_path):
    """A file apply_migration created fresh (no prior backup) is removed."""
    _write_legacy_note(tmp_path, "main", "note", "content")
    plan = plan_migration(tmp_path, ["main"])
    result = apply_migration(plan)

    dest = tmp_path / "workspaces" / "main" / "memory" / "topics" / "note.md"
    assert dest.exists()

    rollback = rollback_migration(result.manifest_path)
    assert rollback.ok
    assert rollback.restored == 1
    assert not dest.exists()


def test_rollback_migration_restores_backed_up_content(tmp_path):
    """A file apply_migration overwrote is restored to its prior content."""
    _write_legacy_note(tmp_path, "main", "note", "new content")
    dest = tmp_path / "workspaces" / "main" / "memory" / "topics" / "note.md"
    dest.parent.mkdir(parents=True)
    dest.write_text("stale destination content", encoding="utf-8")

    note = discover_legacy_notes(tmp_path, "main")[0]
    plan = MigrationPlan(workspace=tmp_path, agent_ids=("main",))
    plan.changes.append(PlannedChange(note, dest, CLASSIFY_MIGRATE))
    result = apply_migration(plan)
    assert dest.read_text(encoding="utf-8") == "new content"

    rollback = rollback_migration(result.manifest_path)
    assert rollback.ok
    assert dest.read_text(encoding="utf-8") == "stale destination content"


def test_rollback_migration_handles_missing_manifest(tmp_path):
    result = rollback_migration(tmp_path / "does-not-exist.json")
    assert not result.ok
    assert result.restored == 0


def test_rollback_migration_handles_malformed_manifest(tmp_path):
    manifest_path = tmp_path / "bad-manifest.json"
    manifest_path.write_text("not json", encoding="utf-8")
    result = rollback_migration(manifest_path)
    assert not result.ok

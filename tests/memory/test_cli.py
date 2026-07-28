"""Tests for memory/cli.py: the `minion-assist memory ...` subcommand tree.

`_run_migrate()` reads `workspace`/`agents_cfg` from `minion_assist.config`
via a local import inside the function, so monkeypatching those module
attributes before calling `main()` is sufficient to redirect it at a tmp_path
workspace without touching the real ~/.minion-assist config.
"""

from __future__ import annotations

import minion_assist.config as config
from minion_assist.memory import cli


def _patch_config(monkeypatch, tmp_path, agent_ids=("main",)):
    monkeypatch.setattr(config, "workspace", tmp_path)
    monkeypatch.setattr(config, "agents", {aid: {} for aid in agent_ids})


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

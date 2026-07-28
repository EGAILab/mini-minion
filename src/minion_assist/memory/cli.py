"""``minion-assist memory ...`` — top-level CLI subcommands for memory operations.

This is a separate command surface from the in-REPL slash commands
(``/plan``, ``/audit``, etc. — see ``commands.py``). Memory operations like
migration are meant to run *before* the REPL starts (or from a script/CI
job), so they get their own ``argparse`` subparser tree, dispatched from
``minion.py``'s ``main()`` before any REPL setup happens.

Currently supported:

- ``minion-assist memory migrate``            — dry-run report (default).
- ``minion-assist memory migrate --apply``    — perform the Phase 0 merge.
- ``minion-assist memory migrate --rollback MANIFEST`` — undo a previous apply.

Talks to
--------
- ``memory/migration.py`` — the actual planning/apply/rollback logic; this
  module only parses arguments and prints results.
- ``minion.py`` — ``main()`` dispatches ``sys.argv[1] == "memory"`` here
  before falling through to the interactive REPL.
- ``config.py`` — reads ``workspace`` and ``agents`` to know which agents
  and directories to inventory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import migration


def _build_parser() -> argparse.ArgumentParser:
    """Construct the ``minion-assist memory`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="minion-assist memory", description="Memory subsystem operations."
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    migrate = sub.add_parser(
        "migrate",
        help="Merge legacy memory/{agent_id}/ notes into workspaces/{agent_id}/ (Phase 0).",
    )
    migrate.add_argument(
        "--apply",
        action="store_true",
        help="Perform the migration. Without this flag, only a dry-run report is printed.",
    )
    migrate.add_argument(
        "--rollback",
        metavar="MANIFEST",
        help="Undo a previous --apply run using its migration-manifest.json.",
    )

    return parser


def _run_migrate(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory migrate [--apply | --rollback MANIFEST]``."""
    # Imported lazily so `minion-assist memory --help` doesn't pay the cost
    # (and doesn't require full config validation) just to print usage.
    # config.py's attribute is named `agents` (see minion.py's own
    # `from .config import agents as agents_cfg` for the same alias).
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415

    if args.rollback:
        result = migration.rollback_migration(Path(args.rollback))
        print(result.message)
        return 0 if result.ok else 1

    plan = migration.plan_migration(workspace, sorted(agents_cfg))
    print(migration.dry_run_report(plan))

    if args.apply:
        print("")
        result = migration.apply_migration(plan)
        print(result.message)
        if result.manifest_path:
            print(f"Manifest: {result.manifest_path}")
            print(
                "Rollback with: minion-assist memory migrate "
                f'--rollback "{result.manifest_path}"'
            )
        return 0 if result.ok else 1

    return 0


def main(argv: list[str]) -> int:
    """Entry point for the ``memory`` CLI subcommand tree.

    Args:
        argv: Arguments *after* the leading ``memory`` token, e.g. for
            ``minion-assist memory migrate --apply`` this receives
            ``["migrate", "--apply"]``.

    Returns:
        int: Process exit code (0 on success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "migrate":
        return _run_migrate(args)

    parser.error(f"Unknown subcommand: {args.subcommand}")
    return 2  # pragma: no cover — argparse.error() already raises SystemExit

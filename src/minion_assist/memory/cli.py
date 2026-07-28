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
- ``minion-assist memory status [--agent ID]``          — note counts per agent.
- ``minion-assist memory list [--agent ID]``            — list topic note keys.
- ``minion-assist memory get PATH --agent ID``          — bounded exact read.
- ``minion-assist memory search QUERY [--agent ID]``    — keyword search.
- ``minion-assist memory doctor [--agent ID]``          — status + un-migrated-data check.

Talks to
--------
- ``memory/migration.py`` — the actual planning/apply/rollback logic; this
  module only parses arguments and prints results.
- ``memory/files.py``, ``memory/service.py`` — ``status``/``list``/``get``/
  ``search`` build a :class:`~minion_assist.memory.service.MemoryService` per
  selected agent and call straight through to it.
- ``minion.py`` — ``main()`` dispatches ``sys.argv[1] == "memory"`` here
  before falling through to the interactive REPL.
- ``config.py`` — reads ``workspace``, ``agents``, and ``bootstrap`` to know
  which agents/directories to inventory and how to resolve each agent's
  memory root (same fallback ``minion.py`` uses for the live agent loop).
- ``workspace.py`` — :func:`agent_workspace_root` resolves each agent's
  per-agent or shared workspace directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import migration
from .files import MemoryFileRepository
from .service import MemoryService


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

    status = sub.add_parser("status", help="Show note counts per agent.")
    status.add_argument("--agent", help="Limit to one agent ID. Default: every configured agent.")

    list_cmd = sub.add_parser("list", help="List explicit note keys (memory/topics/) per agent.")
    list_cmd.add_argument("--agent", help="Limit to one agent ID. Default: every configured agent.")

    get_cmd = sub.add_parser("get", help="Read an exact, bounded slice of one agent's memory file.")
    get_cmd.add_argument("path", help="Path to the file, relative to the agent's workspace root.")
    get_cmd.add_argument("--agent", required=True, help="Which agent's memory root to read from.")
    get_cmd.add_argument("--from-line", type=int, default=None, help="1-indexed starting line.")
    get_cmd.add_argument("--lines", type=int, default=None, help="Maximum lines to return.")

    search = sub.add_parser("search", help="Keyword search across one or every agent's memory.")
    search.add_argument("query", help="One or more keywords, space-separated.")
    search.add_argument("--agent", help="Limit to one agent ID. Default: every configured agent.")

    doctor = sub.add_parser(
        "doctor",
        help="Report note counts and flag any un-migrated legacy data per agent.",
    )
    doctor.add_argument("--agent", help="Limit to one agent ID. Default: every configured agent.")

    return parser


def _resolve_agent_root(workspace: Path, agent_id: str, bootstrap_cfg: object) -> Path:
    """Resolve the effective memory root for one agent.

    Mirrors ``minion.py``'s own fallback exactly: prefer the agent's
    per-agent (or shared ``main``) workspace directory; fall back to the
    configured bootstrap path (or the current working directory) so memory
    commands work even for an agent with no dedicated workspace configured.

    Args:
        workspace: The user's minion-assist home directory.
        agent_id: The agent whose root to resolve.
        bootstrap_cfg: The ``BootstrapConfig`` instance from ``config.py``.

    Returns:
        Path: The resolved memory root (an existing or creatable directory).
    """
    from ..workspace import agent_workspace_root  # noqa: PLC0415

    agent_root = agent_workspace_root(workspace, agent_id)
    if agent_root is not None:
        return agent_root
    if bootstrap_cfg.path is not None:
        return Path(bootstrap_cfg.path).expanduser()
    return Path.cwd()


def _selected_agents(agent_ids: list[str], selected: str | None) -> list[str]:
    """Resolve the ``--agent`` filter against the configured agent list.

    Args:
        agent_ids: Every agent ID configured in ``config.json``.
        selected: The ``--agent`` value, or ``None`` for "every agent."

    Returns:
        list[str]: Sorted list of agent IDs to operate on.

    Raises:
        SystemExit: ``selected`` does not match any configured agent.
    """
    if selected is None:
        return sorted(agent_ids)
    if selected not in agent_ids:
        raise SystemExit(
            f"Unknown agent: {selected!r}. Configured agents: {', '.join(sorted(agent_ids))}"
        )
    return [selected]


def _build_service(workspace: Path, agent_id: str, bootstrap_cfg: object) -> MemoryService:
    """Build the :class:`MemoryService` for one agent, resolving its root first."""
    root = _resolve_agent_root(workspace, agent_id, bootstrap_cfg)
    return MemoryService(MemoryFileRepository(root))


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


def _run_status(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory status [--agent ID]``."""
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415

    for agent_id in _selected_agents(sorted(agents_cfg), args.agent):
        service = _build_service(workspace, agent_id, bootstrap_cfg)
        status = service.status()
        print(
            f"{agent_id}: {status.topic_count} topic, {status.import_count} import, "
            f"{status.daily_count} daily note(s) — root: {status.root}"
        )
    return 0


def _run_list(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory list [--agent ID]``."""
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415

    for agent_id in _selected_agents(sorted(agents_cfg), args.agent):
        service = _build_service(workspace, agent_id, bootstrap_cfg)
        keys = service.list_keys()
        print(f"{agent_id}: {len(keys)} note(s)")
        for key in keys:
            print(f"  - {key}")
    return 0


def _run_get(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory get PATH --agent ID [--from-line N] [--lines N]``."""
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415

    [agent_id] = _selected_agents(sorted(agents_cfg), args.agent)
    service = _build_service(workspace, agent_id, bootstrap_cfg)

    try:
        excerpt = service.get(args.path, from_line=args.from_line, lines=args.lines)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    except FileNotFoundError:
        print(f"Error: file not found: {args.path}")
        return 1

    print(
        f"[{excerpt.path} lines {excerpt.start_line}-{excerpt.end_line} "
        f"of {excerpt.total_lines}]"
    )
    print(excerpt.text)
    return 0


def _run_search(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory search QUERY [--agent ID]``."""
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415

    for agent_id in _selected_agents(sorted(agents_cfg), args.agent):
        service = _build_service(workspace, agent_id, bootstrap_cfg)
        hits = service.search(args.query)
        print(f"{agent_id}: {len(hits)} match(es)")
        for hit in hits:
            snippet = hit.content.strip().splitlines()[0][:80] if hit.content.strip() else ""
            print(f"  [{hit.source}] {hit.key}: {snippet}")
    return 0


def _run_doctor(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory doctor [--agent ID]``.

    Reports each selected agent's note counts, then reuses
    :func:`migration.plan_migration` to flag any legacy notes that have not
    been migrated yet (or would conflict) — the closest thing Phase 1 has to
    a health check, since there is no database/job/index state yet to report
    on (see ``docs/adr/0004-degraded-operation.md``).
    """
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415

    selected = _selected_agents(sorted(agents_cfg), args.agent)

    for agent_id in selected:
        service = _build_service(workspace, agent_id, bootstrap_cfg)
        status = service.status()
        print(
            f"{agent_id}: {status.topic_count} topic, {status.import_count} import, "
            f"{status.daily_count} daily note(s) — root: {status.root}"
        )

    plan = migration.plan_migration(workspace, selected)
    counts = plan.counts()
    pending = counts[migration.CLASSIFY_MIGRATE]
    conflicts = counts[migration.CLASSIFY_CONFLICT]

    if plan.workspaces_to_create:
        print(
            f"WARNING: no workspace directory yet for: {', '.join(plan.workspaces_to_create)}"
        )
    if pending:
        print(f"WARNING: {pending} legacy note(s) not yet migrated — run `memory migrate --apply`.")
    if conflicts:
        print(f"WARNING: {conflicts} migration conflict(s) — run `memory migrate` to see details.")
    if not (plan.workspaces_to_create or pending or conflicts):
        print("OK: no un-migrated legacy data found.")

    return 0


_HANDLERS = {
    "migrate": _run_migrate,
    "status": _run_status,
    "list": _run_list,
    "get": _run_get,
    "search": _run_search,
    "doctor": _run_doctor,
}


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

    handler = _HANDLERS.get(args.subcommand)
    if handler is None:
        parser.error(f"Unknown subcommand: {args.subcommand}")
        return 2  # pragma: no cover — argparse.error() already raises SystemExit
    return handler(args)

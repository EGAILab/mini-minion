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
- ``minion-assist memory status [--agent ID] [--deep]``          — note counts (+ index health).
- ``minion-assist memory list [--agent ID]``            — list topic note keys.
- ``minion-assist memory get PATH --agent ID``          — bounded exact read.
- ``minion-assist memory search QUERY [--agent ID] [--corpus C]`` — keyword search.
- ``minion-assist memory doctor [--agent ID]``          — status + un-migrated-data check.
- ``minion-assist memory reindex [--agent ID] [--force]`` — rebuild the lexical index.
- ``minion-assist memory retention``                    — dry-run report (default).
- ``minion-assist memory retention --apply``             — prune stale operational/telemetry rows.
- ``minion-assist memory retention [--days N]``          — override the configured retention window.
- ``minion-assist memory verify-deletions``              — list incomplete /delete-session attempts (R2-GAP-007).
- ``minion-assist memory verify-deletions --retry``      — finish their remaining cross-store cleanup phases.
- ``minion-assist memory pin KEY --agent ID``            — pin a topic note.
- ``minion-assist memory unpin KEY --agent ID``          — unpin a topic note.
- ``minion-assist memory pins [--agent ID]``             — list pinned notes.
- ``minion-assist memory consolidate list --agent ID [--top N]``       — ranked pending proposals.
- ``minion-assist memory consolidate preview PROPOSAL_ID --agent ID``  — draft a preview.
- ``minion-assist memory consolidate explain PREVIEW_ID --agent ID``   — show a stored preview + staleness.
- ``minion-assist memory consolidate approve PREVIEW_ID --agent ID``   — apply a preview to disk.
- ``minion-assist memory consolidate reject PROPOSAL_ID --agent ID [--reason TEXT]`` — reject.
- ``minion-assist memory consolidate rollback TARGET_KEY --agent ID``  — undo the last approve.
- ``minion-assist memory consolidate backfill --agent ID``             — gap-fill historical capture jobs.
- ``minion-assist memory commitments list --agent ID [--status S] [--channel C]`` — list commitments.
- ``minion-assist memory commitments dismiss COMMITMENT_ID --agent ID`` — dismiss without sending.
- ``minion-assist memory commitments delete COMMITMENT_ID --agent ID``  — permanently delete.

Talks to
--------
- ``memory/migration.py`` — the actual planning/apply/rollback logic; this
  module only parses arguments and prints results.
- ``memory/files.py``, ``memory/service.py`` — ``status``/``list``/``get``/
  ``search``/``reindex`` build a :class:`~minion_assist.memory.service.MemoryService`
  per selected agent and call straight through to it.
- ``memory/postgres_index.py`` — :func:`_build_index` constructs the
  optional :class:`~minion_assist.memory.postgres_index.PostgresMemoryIndex`
  that ``search --corpus``, ``status --deep``, and ``reindex`` use, when a
  database is configured (Stage One Phase 3, slice C).
- ``memory/consolidation.py`` — :func:`rank_proposals`,
  :func:`format_preview_report`, :func:`is_preview_stale`,
  :func:`backfill_agent`, and :class:`MemoryConsolidator`, wired up by
  every ``consolidate`` subcommand (Stage One Phase 5, slice D).
- ``session/db.py`` — :func:`_build_db` constructs the
  :class:`~minion_assist.session.db.SessionDB` every ``consolidate``
  subcommand needs (proposals live there, not in the lexical index).
  ``retention`` calls :meth:`~minion_assist.session.db.SessionDB.prune_operational_tables`
  and, when an index is configured,
  :meth:`~minion_assist.memory.postgres_index.PostgresMemoryIndex.prune_operational_tables`
  — the same methods ``memory/retention_scheduler.py``'s daily scheduler
  calls, exposed here for an on-demand dry-run or manual run (MEM-GAP-015).
  ``verify-deletions`` reads :meth:`~minion_assist.session.db.SessionDB.list_incomplete_deletion_tombstones`
  and, with ``--retry``, calls :meth:`~minion_assist.session.db.SessionDB.delete_session`/
  :meth:`~minion_assist.session.db.SessionDB.mark_deletion_db_done`/
  :meth:`~minion_assist.session.db.SessionDB.mark_deletion_evidence_done`
  and (via :func:`_build_service`) :meth:`~minion_assist.memory.service.MemoryService.forget_proposals`
  to finish whatever ``commands.py``'s ``/delete-session`` left incomplete (R2-GAP-007).
- ``minion.py`` — ``main()`` dispatches ``sys.argv[1] == "memory"`` here
  before falling through to the interactive REPL.
- ``config.py`` — reads ``workspace``, ``agents``, ``bootstrap``, and
  ``database`` to know which agents/directories to inventory, how to
  resolve each agent's memory root (same fallback ``minion.py`` uses for
  the live agent loop), and whether a lexical index is available.
- ``workspace.py`` — :func:`agent_workspace_root` resolves each agent's
  per-agent or shared workspace directory.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING

from . import migration
from .files import MemoryFileRepository
from .service import MemoryService

if TYPE_CHECKING:
    from .postgres_index import PostgresMemoryIndex


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
    status.add_argument(
        "--deep",
        action="store_true",
        help=(
            "Also report lexical-index health (chunk counts, corpus breakdown, "
            "last-indexed time). Requires a configured database."
        ),
    )

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
    search.add_argument(
        "--corpus",
        choices=["durable", "daily", "import", "proposal"],
        default=None,
        help=(
            "Restrict to one corpus. Default: search every *reviewed* "
            "corpus (--corpus proposal is required to see unreviewed "
            "capture-job proposals, per Stage One Phase 5, slice B)."
        ),
    )

    doctor = sub.add_parser(
        "doctor",
        help="Report note counts and flag any un-migrated legacy data per agent.",
    )
    doctor.add_argument("--agent", help="Limit to one agent ID. Default: every configured agent.")

    reindex = sub.add_parser(
        "reindex",
        help="Rebuild one or every agent's lexical index. Requires a configured database.",
    )
    reindex.add_argument("--agent", help="Limit to one agent ID. Default: every configured agent.")
    reindex.add_argument(
        "--force",
        action="store_true",
        help=(
            "Crash-safe full rebuild via shadow-table swap (Stage One Phase 3, slice C). "
            "Without this flag, performs a cheaper hash-diff reconciliation instead — only "
            "reindexes files that actually changed."
        ),
    )

    retention = sub.add_parser(
        "retention",
        help=(
            "Report or prune stale rows from operational/telemetry tables "
            "(MEM-GAP-015). Requires a configured database."
        ),
    )
    retention.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete the stale rows. Without this flag, only a dry-run report is printed.",
    )
    retention.add_argument(
        "--days",
        type=int,
        default=None,
        help=(
            "Override retention_days from config.json's memory_retention section "
            "(falls back to that section's configured value, or its default of 30)."
        ),
    )

    verify_deletions = sub.add_parser(
        "verify-deletions",
        help=(
            "List (and, with --retry, finish) any /delete-session attempt left "
            "incomplete by a crash or transient failure (R2-GAP-007). "
            "Requires a configured database."
        ),
    )
    verify_deletions.add_argument(
        "--retry",
        action="store_true",
        help="Finish every incomplete deletion's remaining phases. Without this flag, only lists them.",
    )

    pin = sub.add_parser(
        "pin",
        help="Pin a topic note so it's always surfaced by the pinned fusion lane.",
    )
    pin.add_argument("key", help="The note's identifier, as given to save_memory.")
    pin.add_argument("--agent", required=True, help="Which agent's note to pin.")

    unpin = sub.add_parser("unpin", help="Unpin a topic note.")
    unpin.add_argument("key", help="The note's identifier.")
    unpin.add_argument("--agent", required=True, help="Which agent's note to unpin.")

    pins = sub.add_parser("pins", help="List every pinned note for one or every agent.")
    pins.add_argument("--agent", help="Limit to one agent ID. Default: every configured agent.")

    consolidate = sub.add_parser(
        "consolidate",
        help=(
            "Review, approve, reject, or roll back consolidation proposals "
            "(Stage One Phase 5, slice D). Requires a configured database."
        ),
    )
    consolidate_sub = consolidate.add_subparsers(dest="consolidate_subcommand", required=True)

    c_list = consolidate_sub.add_parser(
        "list", help="Rank and list one agent's pending (unreviewed) proposals."
    )
    c_list.add_argument("--agent", required=True, help="Which agent's proposals to list.")
    c_list.add_argument("--top", type=int, default=10, help="Show at most this many (default 10).")

    c_preview = consolidate_sub.add_parser(
        "preview", help="Draft a preview for one pending proposal — never writes to disk."
    )
    c_preview.add_argument("proposal_id", type=int)
    c_preview.add_argument("--agent", required=True)

    c_explain = consolidate_sub.add_parser(
        "explain", help="Show one stored preview's full review report, plus staleness."
    )
    c_explain.add_argument("preview_id", type=int)
    c_explain.add_argument("--agent", required=True)

    c_approve = consolidate_sub.add_parser(
        "approve", help="Apply a preview: write its drafted content to disk and reindex it."
    )
    c_approve.add_argument("preview_id", type=int)
    c_approve.add_argument("--agent", required=True)

    c_reject = consolidate_sub.add_parser(
        "reject", help="Reject a pending proposal — never promoted, no file touched."
    )
    c_reject.add_argument("proposal_id", type=int)
    c_reject.add_argument("--agent", required=True)
    c_reject.add_argument("--reason", default="", help="Optional reason, stored for audit.")

    c_rollback = consolidate_sub.add_parser(
        "rollback", help="Undo the most recent approve() for one topic note."
    )
    c_rollback.add_argument("target_key", help="The topic note's key.")
    c_rollback.add_argument("--agent", required=True)

    c_backfill = consolidate_sub.add_parser(
        "backfill",
        help=(
            "Enqueue capture jobs for historical message ranges no capture job has "
            "ever covered (gap-filled across every session)."
        ),
    )
    c_backfill.add_argument("--agent", required=True)

    import_cmd = sub.add_parser(
        "import",
        help=(
            "Review and promote quarantined imports (memory/imports/, Stage "
            "One Phase 7, slice E). Requires a configured database."
        ),
    )
    import_sub = import_cmd.add_subparsers(dest="import_subcommand", required=True)

    i_list = import_sub.add_parser("list", help="List one agent's quarantined import keys.")
    i_list.add_argument("--agent", required=True, help="Which agent's imports to list.")

    i_preview = import_sub.add_parser(
        "preview", help="Draft a preview for one quarantined import — never writes to disk."
    )
    i_preview.add_argument("import_key")
    i_preview.add_argument("--agent", required=True)

    i_explain = import_sub.add_parser(
        "explain", help="Show one stored import preview's full review report, plus staleness."
    )
    i_explain.add_argument("preview_id", type=int)
    i_explain.add_argument("--agent", required=True)

    i_approve = import_sub.add_parser(
        "approve",
        help=(
            "Apply an import preview: write its drafted content to disk, reindex it, "
            "and retire the reviewed import."
        ),
    )
    i_approve.add_argument("preview_id", type=int)
    i_approve.add_argument("--agent", required=True)

    i_reject = import_sub.add_parser(
        "reject", help="Discard a quarantined import — nothing promoted, the import is retired."
    )
    i_reject.add_argument("import_key")
    i_reject.add_argument("--agent", required=True)
    i_reject.add_argument("--reason", default="", help="Optional reason, echoed back only.")

    commitments = sub.add_parser(
        "commitments",
        help=(
            "List, dismiss, or delete inferred commitments "
            "(Stage One Phase 6, slice C). Requires a configured database."
        ),
    )
    commitments_sub = commitments.add_subparsers(dest="commitments_subcommand", required=True)

    m_list = commitments_sub.add_parser("list", help="List one agent's commitments.")
    m_list.add_argument("--agent", required=True, help="Which agent's commitments to list.")
    m_list.add_argument(
        "--status",
        choices=["pending", "sent", "dismissed", "snoozed", "expired"],
        default=None,
        help="Restrict to one status. Default: every status.",
    )
    m_list.add_argument("--channel", default=None, help="Restrict to one channel.")

    m_dismiss = commitments_sub.add_parser(
        "dismiss", help="Dismiss one pending commitment without sending anything."
    )
    m_dismiss.add_argument("commitment_id", type=int)
    m_dismiss.add_argument("--agent", required=True)

    m_delete = commitments_sub.add_parser(
        "delete", help="Permanently delete one commitment."
    )
    m_delete.add_argument("commitment_id", type=int)
    m_delete.add_argument("--agent", required=True)

    knowledge = sub.add_parser(
        "knowledge",
        help=(
            "Report on, compile, and forget from the knowledge layer's claims "
            "(Stage One Phase 7, slices C-F). Requires a configured database."
        ),
    )
    knowledge_sub = knowledge.add_subparsers(dest="knowledge_subcommand", required=True)

    k_dashboard = knowledge_sub.add_parser(
        "dashboard", help="Show claim-review reports: contradictions, stale, low-confidence, etc."
    )
    k_dashboard.add_argument("--agent", required=True, help="Which agent's claims to report on.")
    k_dashboard.add_argument(
        "--section",
        choices=[
            "contradictions", "stale", "low-confidence", "missing-provenance",
            "open-questions", "privacy-review", "deletion-coverage",
        ],
        default=None,
        help="Restrict to one section. Default: show every section.",
    )

    k_compile = knowledge_sub.add_parser(
        "compile",
        help=(
            "Compile status=\"supported\" claims into KNOWLEDGE_DIGEST.md "
            "(Stage One Phase 7, slice D) — the same thing the scheduler "
            "does on its daily timer, run on demand."
        ),
    )
    k_compile.add_argument("--agent", required=True, help="Which agent's claims to compile.")
    k_compile.add_argument(
        "--max-chars",
        type=int,
        default=8000,
        help="Soft cap on the compiled digest's length. Default 8000.",
    )

    k_forget = knowledge_sub.add_parser(
        "forget",
        help=(
            "Forget one evidence source: re-flag (or leave grounded) every claim "
            "citing it, editing the affected pages directly (Stage One Phase 7, "
            "slice F)."
        ),
    )
    k_forget.add_argument("--agent", required=True, help="Which agent's claims to search.")
    k_forget.add_argument(
        "--source-kind", required=True, help='Evidence kind to forget, e.g. "proposal".'
    )
    k_forget.add_argument(
        "--source-ref", required=True, help="Evidence reference to forget, e.g. a proposal id."
    )

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


def _build_service(
    workspace: Path,
    agent_id: str,
    bootstrap_cfg: object,
    index: PostgresMemoryIndex | None = None,
) -> MemoryService:
    """Build the :class:`MemoryService` for one agent, resolving its root first.

    Args:
        index: The shared lexical index (from :func:`_build_index`), or
            ``None`` — passed straight through to ``MemoryService`` so
            ``search``/``deep_status``/``force_reindex`` all use it.
    """
    root = _resolve_agent_root(workspace, agent_id, bootstrap_cfg)
    return MemoryService(MemoryFileRepository(root), index=index, agent_id=agent_id)


def _build_provider(agent_id: str):
    """Construct the LLM provider configured for one agent.

    Stage One Phase 5, slice D: only ``consolidate preview`` needs an
    actual drafting call — every other ``consolidate`` subcommand
    (``list``/``explain``/``approve``/``reject``/``rollback``/``backfill``)
    never touches a provider, so callers should only invoke this when they
    actually need one (constructing a provider implies a live API key,
    which shouldn't be a requirement for e.g. rejecting a proposal).

    Args:
        agent_id: Which configured agent's provider/model to build.

    Raises:
        KeyError: If ``agent_id`` isn't a configured agent — same failure
            mode :func:`_selected_agents` already guards against for every
            other subcommand, so callers should validate ``agent_id``
            first the same way.
    """
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from ..providers import create_provider  # noqa: PLC0415

    cfg = agents_cfg[agent_id]
    return create_provider(
        api=cfg.provider.api,
        base_url=cfg.provider.base_url,
        api_key=cfg.provider.api_key,
        model=cfg.model.id,
    )


def _build_index() -> PostgresMemoryIndex | None:
    """Construct the shared lexical index, or ``None`` if no database is configured/reachable.

    One instance is built per CLI invocation and shared across every agent
    that invocation processes (mirrors how ``minion.py`` builds one
    :class:`PostgresMemoryIndex` for the whole running app, not one per
    agent). Never raises — a construction failure (e.g. the database is
    unreachable) is reported and treated the same as "no database
    configured," matching every other database-optional code path in this
    project (see ``docs/adr/0004-degraded-operation.md``).
    """
    from ..config import database as database_cfg  # noqa: PLC0415
    from ..config import embeddings as embeddings_cfg  # noqa: PLC0415

    if not database_cfg.url:
        return None
    try:
        from .postgres_index import PostgresMemoryIndex  # noqa: PLC0415

        dims = embeddings_cfg.dimensions if embeddings_cfg else None
        embedding_provider = None
        if embeddings_cfg is not None:
            from ..providers.embeddings import EmbeddingProvider  # noqa: PLC0415
            embedding_provider = EmbeddingProvider(
                base_url=embeddings_cfg.provider.base_url,
                api_key=embeddings_cfg.provider.api_key,
                model=embeddings_cfg.model,
                dimensions=embeddings_cfg.dimensions,
            )
        return PostgresMemoryIndex(
            database_cfg.url, embedding_dimensions=dims, embedding_provider=embedding_provider
        )
    except Exception as exc:
        print(f"Warning: memory index unavailable ({exc}). Continuing without it.")
        return None


def _build_db():
    """Construct a :class:`~minion_assist.session.db.SessionDB`, or ``None`` if unconfigured/unreachable.

    Stage One Phase 5, slice D: every ``consolidate`` subcommand needs
    ``SessionDB`` — proposals live there, not in the lexical index. Same
    "never raises, report and degrade" contract as :func:`_build_index`.
    """
    from ..config import database as database_cfg  # noqa: PLC0415

    if not database_cfg.url:
        return None
    try:
        from ..session.db import SessionDB  # noqa: PLC0415

        return SessionDB(database_cfg.url)
    except Exception as exc:
        print(f"Warning: database unavailable ({exc}).")
        return None


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
    """Handle ``minion-assist memory status [--agent ID] [--deep]``."""
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415

    index = _build_index() if args.deep else None

    for agent_id in _selected_agents(sorted(agents_cfg), args.agent):
        service = _build_service(workspace, agent_id, bootstrap_cfg, index=index)
        status = service.status()
        print(
            f"{agent_id}: {status.topic_count} topic, {status.import_count} import, "
            f"{status.daily_count} daily note(s) — root: {status.root}"
        )
        if args.deep:
            deep = service.deep_status()
            if deep is None:
                print("  index: no database configured — lexical index unavailable")
            else:
                corpus_str = ", ".join(
                    f"{kind}={count}" for kind, count in sorted(deep["by_corpus"].items())
                ) or "none"
                last = (
                    f"{deep['last_indexed_at']:.0f}" if deep["last_indexed_at"] else "never"
                )
                print(
                    f"  index: {deep['total_chunks']} chunk(s) across {deep['file_count']} "
                    f"file(s) ({corpus_str}) — last indexed at {last}"
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
    """Handle ``minion-assist memory search QUERY [--agent ID] [--corpus C]``."""
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415

    index = _build_index()

    for agent_id in _selected_agents(sorted(agents_cfg), args.agent):
        service = _build_service(workspace, agent_id, bootstrap_cfg, index=index)
        hits = service.search(args.query, corpus=args.corpus)
        print(f"{agent_id}: {len(hits)} match(es)")
        for hit in hits:
            snippet = hit.content.strip().splitlines()[0][:80] if hit.content.strip() else ""
            locator = f" ({hit.rel_path}:{hit.start_line}-{hit.end_line})" if hit.rel_path else ""
            print(f"  [{hit.source}] {hit.key}{locator}: {snippet}")
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


def _run_reindex(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory reindex [--agent ID] [--force]``.

    Without ``--force``: a cheap hash-diff reconciliation
    (``MemoryService.reconcile_index``) — the same operation ``minion.py``
    runs automatically at startup, exposed here so it can be triggered
    on demand (e.g. right after manually editing a memory file, without
    waiting for the live filesystem watcher's debounce window or a
    restart).

    With ``--force``: a crash-safe full rebuild-and-swap
    (``MemoryService.force_reindex``) — see
    ``PostgresMemoryIndex.force_rebuild_agent``'s docstring for why this is
    safe to interrupt.
    """
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415

    index = _build_index()
    if index is None:
        print("Error: no database configured (or it's unreachable) — nothing to reindex.")
        return 1

    for agent_id in _selected_agents(sorted(agents_cfg), args.agent):
        service = _build_service(workspace, agent_id, bootstrap_cfg, index=index)
        if args.force:
            count = service.force_reindex()
            print(f"{agent_id}: force-reindexed — {count} chunk(s) now indexed.")
        else:
            touched = service.reconcile_index()
            if touched:
                print(f"{agent_id}: reindexed {touched} file(s).")
            else:
                print(f"{agent_id}: already up to date.")
    return 0


def _run_retention(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory retention [--apply] [--days N]``.

    Without ``--apply``: a dry-run report — counts rows past the retention
    window in every covered operational/telemetry table, without deleting
    anything (:meth:`SessionDB.prune_operational_tables`'s ``dry_run=True``
    path runs ``SELECT count(*)`` instead of ``DELETE``, same filter).
    With ``--apply``: actually deletes them. See
    ``memory/retention_scheduler.py``'s module docstring for exactly which
    tables this covers and which it deliberately never touches.
    """
    from ..config import memory_retention as memory_retention_cfg  # noqa: PLC0415

    db = _build_db()
    if db is None:
        print("Error: no database configured (or it's unreachable) — nothing to report.")
        return 1
    index = _build_index()

    retention_days = args.days if args.days is not None else memory_retention_cfg.retention_days
    dry_run = not args.apply

    counts = db.prune_operational_tables(retention_days, dry_run=dry_run)
    if index is not None:
        counts.update(index.prune_operational_tables(retention_days, dry_run=dry_run))

    total = sum(counts.values())
    verb = "would be deleted" if dry_run else "deleted"
    print(f"{total} row(s) older than {retention_days} day(s) {verb}:")
    for name, count in counts.items():
        print(f"  {name}: {count}")
    if dry_run:
        print("Re-run with --apply to actually delete these rows.")
    return 0


def _run_verify_deletions(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory verify-deletions [--retry]`` (R2-GAP-007).

    ``/delete-session``'s cross-store cleanup (JSONL file, ``SessionDB``
    rows, indexed evidence) runs as three separate phases, each recorded
    in a ``deletion_tombstones`` row as it completes — see
    ``session/db.py``'s ``_migration_007_deletion_tombstones`` docstring.
    This command is the only way to finish one that got stuck: the JSONL
    file is already gone by the time any later phase could fail, so
    re-running ``/delete-session`` on the same target no longer works
    (it can't be found by listing/index again).

    Without ``--retry``: lists every incomplete tombstone and which phases
    are still pending, without touching anything. With ``--retry``:
    attempts to finish each one's remaining phases in order — the
    JSONL phase is never retried here (this command has no
    ``ShortTermMemory`` wiring, and by construction the tombstone always
    already has it marked done or the file genuinely couldn't be deleted,
    which needs manual attention, not an automated retry).
    """
    db = _build_db()
    if db is None:
        print("Error: no database configured (or it's unreachable) — nothing to verify.")
        return 1

    tombstones = db.list_incomplete_deletion_tombstones()
    if not tombstones:
        print("No incomplete session deletions found.")
        return 0

    print(f"{len(tombstones)} incomplete session deletion(s):")
    for t in tombstones:
        print(
            f"  [{t['agent_id']}] {t['session_id'][:8]} — "
            f"jsonl={'done' if t['jsonl_deleted'] else 'PENDING'} "
            f"db={'done' if t['db_deleted'] else 'PENDING'} "
            f"evidence={'done' if t['evidence_cleaned'] else 'PENDING'}"
        )

    if not args.retry:
        print("Re-run with --retry to finish these.")
        return 0

    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415

    index = _build_index()
    for t in tombstones:
        agent_id, session_id = t["agent_id"], t["session_id"]
        if not t["jsonl_deleted"]:
            print(
                f"  [{agent_id}] {session_id[:8]}: JSONL deletion never completed — "
                "this needs manual attention, not an automated retry; skipping."
            )
            continue

        proposal_ids = t["proposal_ids"]
        if not t["db_deleted"]:
            try:
                pg_result = db.delete_session(agent_id, session_id)
            except Exception as exc:
                print(f"  [{agent_id}] {session_id[:8]}: database cleanup failed again ({exc}).")
                continue
            proposal_ids = (pg_result.get("proposal_ids") or []) if pg_result is not None else []
            db.mark_deletion_db_done(agent_id, session_id, proposal_ids)

        if not t["evidence_cleaned"]:
            if proposal_ids:
                service = _build_service(workspace, agent_id, bootstrap_cfg, index=index)
                try:
                    service.forget_proposals(proposal_ids)
                except Exception as exc:
                    print(
                        f"  [{agent_id}] {session_id[:8]}: evidence cleanup failed again ({exc})."
                    )
                    continue
            db.mark_deletion_evidence_done(agent_id, session_id)

        print(f"  [{agent_id}] {session_id[:8]}: finished.")
    return 0


def _run_pin(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory pin KEY --agent ID``."""
    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415

    index = _build_index()
    if index is None:
        print("Error: no database configured (or it's unreachable) — nothing to pin.")
        return 1

    service = _build_service(workspace, args.agent, bootstrap_cfg, index=index)
    try:
        service.pin(args.key)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1
    print(f"{args.agent}: pinned {args.key!r}.")
    return 0


def _run_unpin(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory unpin KEY --agent ID``."""
    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415

    index = _build_index()
    if index is None:
        print("Error: no database configured (or it's unreachable) — nothing to unpin.")
        return 1

    service = _build_service(workspace, args.agent, bootstrap_cfg, index=index)
    service.unpin(args.key)
    print(f"{args.agent}: unpinned {args.key!r}.")
    return 0


def _run_pins(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory pins [--agent ID]``."""
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415

    index = _build_index()
    if index is None:
        print("Error: no database configured (or it's unreachable) — nothing to list.")
        return 1

    for agent_id in _selected_agents(sorted(agents_cfg), args.agent):
        service = _build_service(workspace, agent_id, bootstrap_cfg, index=index)
        keys = service.list_pinned()
        print(f"{agent_id}: {len(keys)} pinned note(s)")
        for key in keys:
            print(f"  - {key}")
    return 0


def _run_consolidate_list(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory consolidate list --agent ID [--top N]``."""
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from .consolidation import rank_proposals  # noqa: PLC0415

    agent_id = _selected_agents(sorted(agents_cfg), args.agent)[0]
    db = _build_db()
    index = _build_index()
    if db is None or index is None:
        print("Error: no database configured (or it's unreachable) — nothing to list.")
        return 1

    ranked = rank_proposals(db, index, agent_id)[: args.top]
    if not ranked:
        print(f"{agent_id}: no pending proposals.")
        return 0
    for p in ranked:
        snippet = p["claim_text"][:80]
        print(f"#{p['id']}  score={p['score']}  {snippet}")
    return 0


def _run_consolidate_preview(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory consolidate preview PROPOSAL_ID --agent ID``."""
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415
    from .consolidation import MemoryConsolidator, format_preview_report  # noqa: PLC0415

    agent_id = _selected_agents(sorted(agents_cfg), args.agent)[0]
    db = _build_db()
    index = _build_index()
    if db is None or index is None:
        print("Error: no database configured (or it's unreachable) — nothing to preview.")
        return 1

    files = MemoryFileRepository(_resolve_agent_root(workspace, agent_id, bootstrap_cfg))
    provider = _build_provider(agent_id)
    consolidator = MemoryConsolidator(db, index, files, provider, agent_id=agent_id)
    try:
        preview = consolidator.preview(args.proposal_id)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    print(format_preview_report(preview))
    print(
        f"\nPreview id: {preview['id']} — approve with "
        f"`memory consolidate approve {preview['id']} --agent {agent_id}`."
    )
    return 0


def _run_consolidate_explain(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory consolidate explain PREVIEW_ID --agent ID``."""
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415
    from .consolidation import format_preview_report, is_preview_stale  # noqa: PLC0415

    agent_id = _selected_agents(sorted(agents_cfg), args.agent)[0]
    index = _build_index()
    if index is None:
        print("Error: no database configured (or it's unreachable) — nothing to explain.")
        return 1

    preview = index.get_consolidation_preview(args.preview_id)
    if preview is None:
        print(f"Error: no consolidation preview with id {args.preview_id!r}.")
        return 1

    files = MemoryFileRepository(_resolve_agent_root(workspace, agent_id, bootstrap_cfg))
    print(format_preview_report(preview))

    db = _build_db()
    if db is not None:
        proposal = db.get_proposal(preview["proposal_id"])
        if proposal is not None and proposal["status"] != "pending":
            print(f"\nActual current status: {proposal['status']}")
            if proposal["rejected_reason"]:
                print(f"Rejection reason: {proposal['rejected_reason']}")

    if is_preview_stale(files, preview):
        print(
            "\nWARNING: stale — the target has changed since this preview was drafted. "
            "Re-run `preview` before approving."
        )
    return 0


def _run_consolidate_approve(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory consolidate approve PREVIEW_ID --agent ID``."""
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415
    from .consolidation import MemoryConsolidator, StaleProposalError  # noqa: PLC0415

    agent_id = _selected_agents(sorted(agents_cfg), args.agent)[0]
    db = _build_db()
    index = _build_index()
    if db is None or index is None:
        print("Error: no database configured (or it's unreachable) — nothing to approve.")
        return 1

    files = MemoryFileRepository(_resolve_agent_root(workspace, agent_id, bootstrap_cfg))
    consolidator = MemoryConsolidator(db, index, files, None, agent_id=agent_id)
    try:
        result = consolidator.approve(args.preview_id)
    except (ValueError, StaleProposalError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"{agent_id}: applied to {result['rel_path']} — proposal #{result['proposal_id']} promoted.")
    return 0


def _run_consolidate_reject(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory consolidate reject PROPOSAL_ID --agent ID [--reason TEXT]``."""
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415
    from .consolidation import MemoryConsolidator  # noqa: PLC0415

    agent_id = _selected_agents(sorted(agents_cfg), args.agent)[0]
    db = _build_db()
    if db is None:
        print("Error: no database configured (or it's unreachable) — nothing to reject.")
        return 1

    files = MemoryFileRepository(_resolve_agent_root(workspace, agent_id, bootstrap_cfg))
    # reject() never touches the index — pass None rather than paying for
    # _build_index()'s connection just to satisfy an unused parameter.
    consolidator = MemoryConsolidator(db, None, files, None, agent_id=agent_id)
    consolidator.reject(args.proposal_id, reason=args.reason)
    print(f"{agent_id}: rejected proposal #{args.proposal_id}.")
    return 0


def _run_consolidate_rollback(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory consolidate rollback TARGET_KEY --agent ID``."""
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415
    from .consolidation import MemoryConsolidator  # noqa: PLC0415

    agent_id = _selected_agents(sorted(agents_cfg), args.agent)[0]
    db = _build_db()
    index = _build_index()
    if db is None or index is None:
        print("Error: no database configured (or it's unreachable) — nothing to roll back.")
        return 1

    files = MemoryFileRepository(_resolve_agent_root(workspace, agent_id, bootstrap_cfg))
    consolidator = MemoryConsolidator(db, index, files, None, agent_id=agent_id)
    try:
        result = consolidator.rollback(args.target_key)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"{agent_id}: rolled back {args.target_key!r} — proposal #{result['proposal_id']} back to pending.")
    return 0


def _run_consolidate_backfill(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory consolidate backfill --agent ID``."""
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from .consolidation import backfill_agent  # noqa: PLC0415

    agent_id = _selected_agents(sorted(agents_cfg), args.agent)[0]
    db = _build_db()
    if db is None:
        print("Error: no database configured (or it's unreachable) — nothing to backfill.")
        return 1

    model_id = agents_cfg[agent_id].model.id
    enqueued = backfill_agent(db, agent_id, model_id)
    print(f"{agent_id}: enqueued {enqueued} new capture job(s) from historical gaps.")
    return 0


_CONSOLIDATE_HANDLERS = {
    "list": _run_consolidate_list,
    "preview": _run_consolidate_preview,
    "explain": _run_consolidate_explain,
    "approve": _run_consolidate_approve,
    "reject": _run_consolidate_reject,
    "rollback": _run_consolidate_rollback,
    "backfill": _run_consolidate_backfill,
}


def _run_consolidate(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory consolidate <list|preview|explain|approve|reject|rollback|backfill>``."""
    handler = _CONSOLIDATE_HANDLERS[args.consolidate_subcommand]
    return handler(args)


def _run_import_list(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory import list --agent ID``."""
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415

    agent_id = _selected_agents(sorted(agents_cfg), args.agent)[0]
    files = MemoryFileRepository(_resolve_agent_root(workspace, agent_id, bootstrap_cfg))
    keys = files.list_import_keys()
    if not keys:
        print(f"{agent_id}: no quarantined imports.")
        return 0
    for key in keys:
        print(f"- {key}")
    return 0


def _run_import_preview(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory import preview IMPORT_KEY --agent ID``."""
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415
    from .import_review import ImportReviewer, format_import_preview_report  # noqa: PLC0415

    agent_id = _selected_agents(sorted(agents_cfg), args.agent)[0]
    index = _build_index()
    if index is None:
        print("Error: no database configured (or it's unreachable) — nothing to preview.")
        return 1

    files = MemoryFileRepository(_resolve_agent_root(workspace, agent_id, bootstrap_cfg))
    provider = _build_provider(agent_id)
    reviewer = ImportReviewer(index, files, provider, agent_id=agent_id)
    try:
        preview = reviewer.preview(args.import_key)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    print(format_import_preview_report(preview))
    print(
        f"\nPreview id: {preview['id']} — approve with "
        f"`memory import approve {preview['id']} --agent {agent_id}`."
    )
    return 0


def _run_import_explain(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory import explain PREVIEW_ID --agent ID``."""
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415
    from .import_review import format_import_preview_report, is_import_preview_stale  # noqa: PLC0415

    agent_id = _selected_agents(sorted(agents_cfg), args.agent)[0]
    index = _build_index()
    if index is None:
        print("Error: no database configured (or it's unreachable) — nothing to explain.")
        return 1

    preview = index.get_import_preview(args.preview_id)
    if preview is None:
        print(f"Error: no import preview with id {args.preview_id!r}.")
        return 1

    files = MemoryFileRepository(_resolve_agent_root(workspace, agent_id, bootstrap_cfg))
    print(format_import_preview_report(preview))

    if is_import_preview_stale(files, preview):
        print(
            "\nWARNING: stale — the target has changed since this preview was drafted. "
            "Re-run `preview` before approving."
        )
    return 0


def _run_import_approve(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory import approve PREVIEW_ID --agent ID``."""
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415
    from .import_review import ImportReviewer, StaleImportError  # noqa: PLC0415

    agent_id = _selected_agents(sorted(agents_cfg), args.agent)[0]
    index = _build_index()
    if index is None:
        print("Error: no database configured (or it's unreachable) — nothing to approve.")
        return 1

    files = MemoryFileRepository(_resolve_agent_root(workspace, agent_id, bootstrap_cfg))
    reviewer = ImportReviewer(index, files, None, agent_id=agent_id)
    try:
        result = reviewer.approve(args.preview_id)
    except (ValueError, StaleImportError) as exc:
        print(f"Error: {exc}")
        return 1

    print(
        f"{agent_id}: applied to {result['rel_path']} — "
        f"import {result['import_key']!r} retired."
    )
    return 0


def _run_import_reject(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory import reject IMPORT_KEY --agent ID [--reason TEXT]``."""
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415
    from .import_review import ImportReviewer  # noqa: PLC0415

    agent_id = _selected_agents(sorted(agents_cfg), args.agent)[0]
    index = _build_index()
    if index is None:
        print("Error: no database configured (or it's unreachable) — nothing to reject.")
        return 1

    files = MemoryFileRepository(_resolve_agent_root(workspace, agent_id, bootstrap_cfg))
    # Unlike MemoryConsolidator.reject() (just flips a memory_proposals
    # status, touches nothing else), ImportReviewer.reject() retires the
    # import outright — it deletes the file AND its index entries, so a
    # real index is required here (see memory/import_review.py's module
    # docstring for why imports have no separate "rejected" status to set).
    reviewer = ImportReviewer(index, files, None, agent_id=agent_id)
    try:
        reviewer.reject(args.import_key, reason=args.reason)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    suffix = f" ({args.reason})" if args.reason else ""
    print(f"{agent_id}: rejected import {args.import_key!r}{suffix}.")
    return 0


_IMPORT_HANDLERS = {
    "list": _run_import_list,
    "preview": _run_import_preview,
    "explain": _run_import_explain,
    "approve": _run_import_approve,
    "reject": _run_import_reject,
}


def _run_import(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory import <list|preview|explain|approve|reject>``."""
    handler = _IMPORT_HANDLERS[args.import_subcommand]
    return handler(args)


def _format_due(epoch: float) -> str:
    """Render an epoch-seconds timestamp for CLI display."""
    from datetime import datetime  # noqa: PLC0415

    return datetime.fromtimestamp(epoch).isoformat(timespec="minutes")


def _run_commitments_list(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory commitments list --agent ID [--status S] [--channel C]``."""
    from ..config import agents as agents_cfg  # noqa: PLC0415

    agent_id = _selected_agents(sorted(agents_cfg), args.agent)[0]
    db = _build_db()
    if db is None:
        print("Error: no database configured (or it's unreachable) — nothing to list.")
        return 1

    results = db.list_commitments(agent_id, status=args.status, channel=args.channel)
    if not results:
        print(f"{agent_id}: no commitments.")
        return 0
    for c in results:
        print(
            f"#{c['id']} [{c['status']}] {c['kind']} ({c['channel']}) "
            f"due {_format_due(c['due_earliest'])}: {c['reason']}"
        )
    return 0


def _run_commitments_dismiss(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory commitments dismiss COMMITMENT_ID --agent ID``."""
    from ..config import agents as agents_cfg  # noqa: PLC0415

    agent_id = _selected_agents(sorted(agents_cfg), args.agent)[0]
    db = _build_db()
    if db is None:
        print("Error: no database configured (or it's unreachable) — nothing to dismiss.")
        return 1

    commitment = db.get_commitment(args.commitment_id)
    if commitment is None or commitment["agent_id"] != agent_id:
        print(f"Error: no commitment with id {args.commitment_id} for agent {agent_id!r}.")
        return 1
    if commitment["status"] != "pending":
        print(f"{agent_id}: commitment #{args.commitment_id} is already {commitment['status']!r}.")
        return 0

    db.mark_commitment_dismissed(args.commitment_id)
    print(f"{agent_id}: dismissed commitment #{args.commitment_id}.")
    return 0


def _run_commitments_delete(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory commitments delete COMMITMENT_ID --agent ID``."""
    from ..config import agents as agents_cfg  # noqa: PLC0415

    agent_id = _selected_agents(sorted(agents_cfg), args.agent)[0]
    db = _build_db()
    if db is None:
        print("Error: no database configured (or it's unreachable) — nothing to delete.")
        return 1

    deleted = db.delete_commitment(agent_id, args.commitment_id)
    if not deleted:
        print(f"Error: no commitment with id {args.commitment_id} for agent {agent_id!r}.")
        return 1
    print(f"{agent_id}: deleted commitment #{args.commitment_id}.")
    return 0


_COMMITMENTS_HANDLERS = {
    "list": _run_commitments_list,
    "dismiss": _run_commitments_dismiss,
    "delete": _run_commitments_delete,
}


def _run_commitments(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory commitments <list|dismiss|delete>``."""
    handler = _COMMITMENTS_HANDLERS[args.commitments_subcommand]
    return handler(args)


_ALL_DASHBOARD_SECTIONS = (
    "contradictions", "stale", "low-confidence", "missing-provenance",
    "open-questions", "privacy-review", "deletion-coverage",
)


def _print_claim_rows(rows: list[dict]) -> None:
    """Shared row-printing for dashboard sections that just list claims."""
    if not rows:
        print("  (none)")
        return
    for c in rows:
        print(f"  {c['id']} [{c['status']}] ({c['rel_path']}): {c['text']}")


def _run_knowledge_dashboard(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory knowledge dashboard --agent ID [--section S]``."""
    from ..config import agents as agents_cfg  # noqa: PLC0415

    agent_id = _selected_agents(sorted(agents_cfg), args.agent)[0]
    index = _build_index()
    if index is None:
        print("Error: no database configured (or it's unreachable) — nothing to report.")
        return 1

    sections = [args.section] if args.section else list(_ALL_DASHBOARD_SECTIONS)
    now = time.time()

    if "contradictions" in sections:
        print("== Contradictions ==")
        rows = index.list_contradictions(agent_id)
        if not rows:
            print("  (none)")
        for r in rows:
            print(f"  {r['from_claim_id']} [{r['from_status']}]: {r['from_text']}")
            if r["to_text"] is None:
                print(f"    contradicts -> {r['to_claim_id']} (no such claim — dangling reference)")
            else:
                print(f"    contradicts -> {r['to_claim_id']} [{r['to_status']}]: {r['to_text']}")
        print()

    if "stale" in sections:
        print("== Stale claims ==")
        rows = index.list_stale_claims(agent_id, now)
        if not rows:
            print("  (none)")
        for c in rows:
            print(f"  {c['id']} [{c['status']}] (freshness {c['freshness']:.2f}): {c['text']}")
        print()

    if "low-confidence" in sections:
        print("== Low confidence ==")
        rows = index.list_low_confidence_claims(agent_id)
        if not rows:
            print("  (none)")
        for c in rows:
            confidence = "unrated" if c["confidence"] is None else f"{c['confidence']:.2f}"
            print(f"  {c['id']} [{c['status']}] (confidence {confidence}): {c['text']}")
        print()

    if "missing-provenance" in sections:
        print("== Missing provenance ==")
        _print_claim_rows(index.list_claims_missing_evidence(agent_id))
        print()

    if "open-questions" in sections:
        print("== Open questions ==")
        _print_claim_rows(index.list_claims(agent_id, status="unknown"))
        print()

    if "privacy-review" in sections:
        print("== Privacy review ==")
        _print_claim_rows(index.list_claims_needing_privacy_review(agent_id))
        print()

    if "deletion-coverage" in sections:
        print("== Deletion coverage ==")
        _print_claim_rows(index.list_claims_needing_reevaluation(agent_id))
        print()

    return 0


def _run_knowledge_compile(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory knowledge compile --agent ID [--max-chars N]``.

    Does exactly what ``KnowledgeDigestScheduler`` does on its daily
    timer (Stage One Phase 7, slice D) — fetch this agent's
    ``status="supported"`` claims, compile them, and overwrite
    ``KNOWLEDGE_DIGEST.md`` — run on demand instead of waiting for the
    schedule.
    """
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415
    from .knowledge import compile_digest  # noqa: PLC0415

    agent_id = _selected_agents(sorted(agents_cfg), args.agent)[0]
    index = _build_index()
    if index is None:
        print("Error: no database configured (or it's unreachable) — nothing to compile.")
        return 1

    claims = index.list_claims(agent_id, status="supported")
    digest = compile_digest(claims, max_chars=args.max_chars)

    files = MemoryFileRepository(_resolve_agent_root(workspace, agent_id, bootstrap_cfg))
    path = files.write_digest(digest)

    if not claims:
        print(f"{agent_id}: no supported claims yet — wrote an empty {path}.")
    else:
        print(f"{agent_id}: compiled {len(claims)} supported claim(s) into {path}.")
    return 0


def _run_knowledge_forget(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory knowledge forget --agent ID --source-kind K --source-ref R``.

    Stage One Phase 7, slice F. Immediate, not previewed — mirrors
    ``consolidate reject``/``import reject``'s "deliberate, explicitly-
    named cleanup action" precedent (see ``memory/forgetting.py``'s
    module docstring).
    """
    from ..config import agents as agents_cfg  # noqa: PLC0415
    from ..config import bootstrap as bootstrap_cfg  # noqa: PLC0415
    from ..config import workspace  # noqa: PLC0415
    from .forgetting import forget_source  # noqa: PLC0415

    agent_id = _selected_agents(sorted(agents_cfg), args.agent)[0]
    index = _build_index()
    if index is None:
        print("Error: no database configured (or it's unreachable) — nothing to forget.")
        return 1

    files = MemoryFileRepository(_resolve_agent_root(workspace, agent_id, bootstrap_cfg))
    result = forget_source(index, files, agent_id, args.source_kind, args.source_ref)

    total = (
        len(result["reevaluated"]) + len(result["still_grounded"])
        + len(result["skipped_manual_review"])
    )
    if total == 0:
        print(f"{agent_id}: nothing cites {args.source_kind}:{args.source_ref} — nothing to do.")
        return 0

    print(f"{agent_id}: forgot {args.source_kind}:{args.source_ref}.")
    if result["reevaluated"]:
        print(f"  re-flagged status=unknown: {', '.join(result['reevaluated'])}")
    if result["still_grounded"]:
        print(f"  still grounded by other evidence: {', '.join(result['still_grounded'])}")
    if result["skipped_manual_review"]:
        print("  needs manual review (not auto-edited):")
        for entry in result["skipped_manual_review"]:
            print(f"    {entry['claim_id']} in {entry['rel_path']}")
    return 0


_KNOWLEDGE_HANDLERS = {
    "dashboard": _run_knowledge_dashboard,
    "compile": _run_knowledge_compile,
    "forget": _run_knowledge_forget,
}


def _run_knowledge(args: argparse.Namespace) -> int:
    """Handle ``minion-assist memory knowledge <dashboard|compile|forget>``."""
    handler = _KNOWLEDGE_HANDLERS[args.knowledge_subcommand]
    return handler(args)


_HANDLERS = {
    "migrate": _run_migrate,
    "status": _run_status,
    "list": _run_list,
    "get": _run_get,
    "search": _run_search,
    "doctor": _run_doctor,
    "reindex": _run_reindex,
    "retention": _run_retention,
    "verify-deletions": _run_verify_deletions,
    "pin": _run_pin,
    "unpin": _run_unpin,
    "pins": _run_pins,
    "consolidate": _run_consolidate,
    "import": _run_import,
    "commitments": _run_commitments,
    "knowledge": _run_knowledge,
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

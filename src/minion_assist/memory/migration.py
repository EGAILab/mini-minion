"""Phase 0 migration tooling: merge the legacy notes root into the per-agent workspace root.

Background
----------
Minion Assist currently has **two separate per-agent directories**:

- ``{workspace}/workspaces/{agent_id}/`` — bootstrap identity files
  (``AGENTS.md``, ``SOUL.md``, etc.), resolved by :func:`workspace.agent_workspace_root`.
- ``{workspace}/memory/{agent_id}/`` — flat :class:`LongTermMemory` notes, one
  ``{key}.md`` file per note.

The Stage One memory implementation plan (see
``minion-assist-docs/improve/memory-implementation-plan.md``) merges these
into one root per agent, under the existing ``workspaces/{agent_id}/``
convention, so a single directory holds identity, durable memory, and daily
notes. This module implements that merge safely:

1. :func:`plan_migration` — read-only inventory + classification. Never
   writes anything. Safe to call at any time, including on every startup.
2. :func:`dry_run_report` — a human-readable rendering of a plan, used by
   ``minion-assist memory migrate`` (the default, no-flag action).
3. :func:`apply_migration` — performs the merge. Legacy source files under
   ``memory/{agent_id}/`` are **copied, never moved or deleted** — the
   original notes remain exactly where they were, so nothing is lost even if
   the merge turns out to be wrong. Every destination file apply touches
   (whether newly created or overwritten) is backed up first and recorded in
   a JSON manifest.
4. :func:`rollback_migration` — undoes a specific ``apply_migration()`` run
   using its manifest: restores overwritten files from backup, and deletes
   files that didn't exist before the apply.

Key mapping rules
-----------------
- ``user_context`` (i.e. ``memory/{agent_id}/user_context.md``) is Minion's
  existing always-loaded profile note (see ``agents/session.py``'s
  ``_load_user_context``). It maps to the new root's ``USER.md``.
- ``_auto_extracted`` (the daemon fact-extractor's rolling note, see
  ``memory/extractor.py``) and ``_notes_YYYY-MM-DD`` (the daily-log tool in
  ``tools/memory.py``) map into ``memory/imports/`` — quarantined,
  reviewable material — rather than being merged directly into curated
  ``MEMORY.md``/topic pages. This matches the plan's instruction not to
  auto-promote unreviewed extraction output.
- Every other key (an explicit ``save_memory`` note) maps to
  ``memory/topics/{key}.md``.

Talks to
--------
- ``memory/long_term.py`` — the legacy store this module reads from.
- ``workspace.py`` — :func:`ensure_workspace` creates the destination root.
- ``memory/cli.py`` — the ``minion-assist memory migrate`` command wraps
  these functions.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .. import workspace as _workspace_mod

# The LongTermMemory key that agents/session.py loads at startup as the
# always-resident user profile block. See agents/session.py:_load_user_context.
_USER_CONTEXT_KEY = "user_context"

# Name of the JSON manifest file written alongside each apply's backups.
MANIFEST_FILENAME = "migration-manifest.json"

# Classification labels used throughout the plan/report/apply pipeline.
CLASSIFY_MIGRATE = "migrate"
CLASSIFY_UNCHANGED = "unchanged"
CLASSIFY_CONFLICT = "conflict"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LegacyNote:
    """One note discovered under a legacy ``memory/{agent_id}/`` directory."""

    agent_id: str
    source_path: Path
    key: str              # filename stem, e.g. "user_context" or "api_rest-notes"
    content_hash: str      # sha256 of the file's current bytes
    size: int


@dataclass(frozen=True)
class PlannedChange:
    """One planned migration action for a single legacy note.

    ``classification`` is one of :data:`CLASSIFY_MIGRATE` (destination does
    not exist yet — safe to copy), :data:`CLASSIFY_UNCHANGED` (destination
    already holds byte-identical content — nothing to do), or
    :data:`CLASSIFY_CONFLICT` (destination exists with *different* content —
    requires manual review; never migrated automatically).
    """

    note: LegacyNote
    dest_path: Path
    classification: str
    detail: str = ""


@dataclass
class MigrationPlan:
    """A full, read-only migration plan across every configured agent."""

    workspace: Path
    agent_ids: tuple[str, ...]
    changes: list[PlannedChange] = field(default_factory=list)
    # Agent IDs that have no workspaces/{agent_id}/ directory yet — apply_migration()
    # creates these so the agent stops silently falling back to workspaces/main/.
    workspaces_to_create: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        """Tally how many planned changes fall into each classification."""
        counts = {CLASSIFY_MIGRATE: 0, CLASSIFY_UNCHANGED: 0, CLASSIFY_CONFLICT: 0}
        for change in self.changes:
            counts[change.classification] = counts.get(change.classification, 0) + 1
        return counts


@dataclass
class ApplyResult:
    """Outcome of an :func:`apply_migration` run."""

    ok: bool
    manifest_path: Path | None
    applied: int
    skipped: int
    message: str


@dataclass
class RollbackResult:
    """Outcome of a :func:`rollback_migration` run."""

    ok: bool
    restored: int
    message: str


# ---------------------------------------------------------------------------
# Inventory and planning (read-only)
# ---------------------------------------------------------------------------

def _hash_bytes(data: bytes) -> str:
    """SHA256 hex digest — used to detect byte-identical vs. conflicting content."""
    return hashlib.sha256(data).hexdigest()


def _dest_relpath_for(key: str) -> tuple[str, str]:
    """Map a legacy note key to (path relative to the agent's new root, reason).

    See the module docstring's "Key mapping rules" section for the rationale
    behind each case.
    """
    if key == _USER_CONTEXT_KEY:
        return "USER.md", "always-resident user profile"
    if key == "_auto_extracted" or key.startswith("_notes_"):
        return (
            f"memory/imports/{key}.md",
            "unreviewed extraction output — quarantined, not auto-promoted",
        )
    return f"memory/topics/{key}.md", "explicit note"


def discover_legacy_notes(workspace: Path, agent_id: str) -> list[LegacyNote]:
    """Scan ``workspace/memory/{agent_id}/*.md`` for legacy :class:`LongTermMemory` notes.

    Args:
        workspace: The user's minion-assist home directory.
        agent_id: Agent whose legacy notes directory to scan.

    Returns:
        list[LegacyNote]: One entry per ``.md`` file found, sorted by key for
            deterministic output. Empty if the directory doesn't exist.
    """
    notes_dir = workspace / "memory" / agent_id
    if not notes_dir.is_dir():
        return []

    notes: list[LegacyNote] = []
    for path in sorted(notes_dir.glob("*.md")):
        try:
            data = path.read_bytes()
        except OSError:
            # Unreadable file (permissions, race with concurrent delete) — skip
            # rather than fail the whole inventory; it will simply be absent
            # from the plan and thus never migrated.
            continue
        notes.append(
            LegacyNote(
                agent_id=agent_id,
                source_path=path,
                key=path.stem,
                content_hash=_hash_bytes(data),
                size=len(data),
            )
        )
    return notes


def plan_migration(workspace: Path, agent_ids: list[str]) -> MigrationPlan:
    """Build a full, read-only migration plan for every configured agent.

    This function never writes to disk — it only reads existing files to
    classify what *would* happen. Safe to call from ``--dry-run``, at
    startup, or repeatedly for progress checks.

    Args:
        workspace: The user's minion-assist home directory.
        agent_ids: Every agent ID configured in ``config.json``.

    Returns:
        MigrationPlan: One :class:`PlannedChange` per discovered legacy note,
            plus the list of agents missing a per-agent workspace directory.
    """
    plan = MigrationPlan(workspace=workspace, agent_ids=tuple(sorted(agent_ids)))

    for agent_id in plan.agent_ids:
        per_agent_root = workspace / "workspaces" / agent_id
        if not per_agent_root.exists():
            plan.workspaces_to_create.append(agent_id)

        for note in discover_legacy_notes(workspace, agent_id):
            rel_dest, reason = _dest_relpath_for(note.key)
            dest_path = per_agent_root / rel_dest

            if not dest_path.exists():
                plan.changes.append(PlannedChange(note, dest_path, CLASSIFY_MIGRATE, reason))
                continue

            try:
                dest_hash = _hash_bytes(dest_path.read_bytes())
            except OSError:
                dest_hash = None

            if dest_hash == note.content_hash:
                plan.changes.append(
                    PlannedChange(note, dest_path, CLASSIFY_UNCHANGED,
                                  "destination already holds identical content")
                )
            else:
                plan.changes.append(
                    PlannedChange(note, dest_path, CLASSIFY_CONFLICT,
                                  "destination exists with different content — resolve manually")
                )

    return plan


def dry_run_report(plan: MigrationPlan) -> str:
    """Render a deterministic, human-readable report of a migration plan.

    Args:
        plan: A plan built by :func:`plan_migration`.

    Returns:
        str: Multi-line report — summary counts, per-agent workspace gaps,
            then one line per planned change. Printed by
            ``minion-assist memory migrate`` (default action, and always
            before ``--apply`` proceeds).
    """
    lines: list[str] = []
    counts = plan.counts()

    lines.append("Memory migration plan (dry run — nothing has been changed)")
    lines.append(f"Workspace: {plan.workspace}")
    lines.append(f"Agents inventoried: {', '.join(plan.agent_ids) or '(none configured)'}")
    lines.append("")
    lines.append(
        f"Notes to migrate: {counts[CLASSIFY_MIGRATE]}  "
        f"unchanged: {counts[CLASSIFY_UNCHANGED]}  "
        f"conflicts: {counts[CLASSIFY_CONFLICT]}"
    )
    if plan.workspaces_to_create:
        lines.append(f"Per-agent workspaces to create: {', '.join(plan.workspaces_to_create)}")
    lines.append("")

    for change in plan.changes:
        lines.append(
            f"[{change.classification:>9}] {change.note.agent_id}/{change.note.key}.md"
            f" -> {change.dest_path}"
            + (f"  ({change.detail})" if change.detail else "")
        )

    if counts[CLASSIFY_CONFLICT]:
        lines.append("")
        lines.append(
            "WARNING: conflicting notes are left untouched by --apply. Resolve them "
            "manually (compare source and destination, then re-run) before they can migrate."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Apply and rollback (the only functions that write to disk)
# ---------------------------------------------------------------------------

def apply_migration(plan: MigrationPlan, *, backup_root: Path | None = None) -> ApplyResult:
    """Apply a previously computed plan.

    Creates any missing ``workspaces/{agent_id}/`` directories, then copies
    (never moves) every non-conflicting, non-unchanged note to its
    destination. Every destination file this function touches is backed up
    first (if it existed) and recorded in a JSON manifest, so
    :func:`rollback_migration` can undo the run exactly.

    Legacy source files are never modified or deleted by this function.

    Args:
        plan: A plan built by :func:`plan_migration`. Conflicts within the
            plan are skipped, never overwritten.
        backup_root: Where to store backups and the manifest. Defaults to
            ``{workspace}/memory-migration-backups/``.

    Returns:
        ApplyResult: Counts of applied/skipped changes and the manifest path
            (``None`` if nothing was applied).
    """
    backup_root = backup_root or (plan.workspace / "memory-migration-backups")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = backup_root / stamp

    manifest_entries: list[dict] = []
    applied = 0
    skipped = 0

    # Ensure every configured agent has its own workspace directory so it
    # stops silently falling back to workspaces/main/ (see workspace.py's
    # agent_workspace_root fallback). This alone satisfies the acceptance
    # criterion "no two agents share a memory root unless configuration says so"
    # for agents that don't yet have a per-agent workspace.
    for agent_id in plan.agent_ids:
        per_agent_root = plan.workspace / "workspaces" / agent_id
        _workspace_mod.ensure_workspace(per_agent_root)

    for change in plan.changes:
        if change.classification in (CLASSIFY_CONFLICT, CLASSIFY_UNCHANGED):
            skipped += 1
            continue

        dest_path = change.dest_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        existed_before = dest_path.exists()
        backup_path: Path | None = None
        if existed_before:
            # Preserve the destination's relative layout under the backup dir
            # so a human can browse the backup and understand what was there.
            backup_path = backup_dir / dest_path.relative_to(plan.workspace)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest_path, backup_path)

        shutil.copy2(change.note.source_path, dest_path)
        applied += 1

        manifest_entries.append(
            {
                "agent_id": change.note.agent_id,
                "source_path": str(change.note.source_path),
                "dest_path": str(dest_path),
                "existed_before": existed_before,
                "backup_path": str(backup_path) if backup_path else None,
                "content_hash": change.note.content_hash,
            }
        )

    manifest_path: Path | None = None
    if manifest_entries:
        backup_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = backup_dir / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(
                {
                    "created_at": stamp,
                    "workspace": str(plan.workspace),
                    "entries": manifest_entries,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return ApplyResult(
        ok=True,
        manifest_path=manifest_path,
        applied=applied,
        skipped=skipped,
        message=f"Applied {applied} change(s), skipped {skipped} (conflict/unchanged).",
    )


def rollback_migration(manifest_path: Path) -> RollbackResult:
    """Undo a previous :func:`apply_migration` run using its manifest.

    For each entry: if the destination existed before apply, its backup is
    copied back over it. If apply created the destination fresh (no prior
    file), it is deleted — restoring the exact pre-apply state.

    Args:
        manifest_path: Path to the ``migration-manifest.json`` written by the
            apply run being undone.

    Returns:
        RollbackResult: Whether the manifest was readable, and how many
            destination files were restored.
    """
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        message = f"Could not read manifest {manifest_path}: {exc}"
        return RollbackResult(ok=False, restored=0, message=message)
    except json.JSONDecodeError as exc:
        message = f"Malformed manifest {manifest_path}: {exc}"
        return RollbackResult(ok=False, restored=0, message=message)

    restored = 0
    for entry in manifest.get("entries", []):
        dest_path = Path(entry["dest_path"])
        backup_path = entry.get("backup_path")

        if backup_path:
            shutil.copy2(Path(backup_path), dest_path)
        elif dest_path.exists():
            dest_path.unlink()
        restored += 1

    message = f"Restored {restored} file(s) from {manifest_path}."
    return RollbackResult(ok=True, restored=restored, message=message)

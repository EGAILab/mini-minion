# ADR 0003: One merged memory root per agent, with an explicit scope model

## Status

Accepted (Stage One, Phase 0). Migration implemented in
`memory/migration.py`; runtime consumption of the merged root is Phase 1.

## Context

Minion Assist currently resolves **two separate directories** per agent:

- `{workspace}/workspaces/{agent_id}/` — bootstrap files (`AGENTS.md`,
  `SOUL.md`, etc.), via `workspace.agent_workspace_root()`. When a per-agent
  directory doesn't exist, this **silently falls back** to
  `workspaces/main/`, meaning two differently-configured agents can share
  bootstrap and (once merged) memory content without either operator
  choosing that.
- `{workspace}/memory/{agent_id}/` — flat `LongTermMemory` notes.

This split means "an agent's memory" is not one thing an operator can point
to, back up, or delete — it's scattered across two roots with different
fallback semantics.

## Decision

Merge both into one root per agent, under the existing
`workspaces/{agent_id}/` convention:

```text
workspaces/{agent_id}/
  AGENTS.md, SOUL.md, IDENTITY.md   # bootstrap (unchanged)
  USER.md                           # was memory/{agent_id}/user_context.md
  MEMORY.md
  DREAMS.md                          # narrative diary only, never a promotion source
  memory/
    YYYY-MM-DD.md                    # daily notes
    topics/<key>.md                  # was memory/{agent_id}/{key}.md
    imports/<key>.md                 # quarantined: _auto_extracted, _notes_*
    reviews/YYYY-MM-DD.md             # consolidation review (later phases)
```

The implicit shared-`main` fallback is not removed by this ADR alone —
`apply_migration()` (Phase 0) proactively creates `workspaces/{agent_id}/`
for every configured agent, so the fallback path is never actually
exercised once migration has run. Removing the fallback code itself is
deferred to when `MemoryService` (Phase 1) becomes the sole reader, to avoid
two unrelated behavior changes landing in the same commit.

Scope model for every memory read/write (adapted from Mem0's
user/agent/run scoping, extended for Minion's multi-channel reality):
`agent-private` (default), `user-shared`, `workspace`, `session-lineage`,
`channel`, `import-quarantine`. Scope is enforced in file-root selection and
database queries *before* ranking, never only in prompt formatting.

## Consequences

- Migration is additive and non-destructive: legacy
  `memory/{agent_id}/*.md` files are **copied, never deleted**, and every
  destination file `apply_migration()` touches is backed up with a
  rollback-capable manifest (see `memory/migration.py`).
- `_auto_extracted` and `_notes_YYYY-MM-DD` legacy notes land in
  `memory/imports/`, not directly in curated topic pages — consistent with
  ADR 0001's canonical/derived split and the plan's "no silent promotion"
  principle.
- A destination file that already exists with *different* content than the
  source is classified as a conflict and is never auto-migrated; an operator
  must resolve it manually before it migrates.
- `agent_workspace_root()`'s fallback-to-`main` code path itself is
  untouched by this ADR — tracked as follow-up work for Phase 1.

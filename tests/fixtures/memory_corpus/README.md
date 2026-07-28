# Memory evaluation fixture corpus

Deterministic, checked-in fixture data for measuring memory retrieval —
introduced in Stage One Phase 0 (see
`minion-assist-docs/improve/memory-implementation-plan.md`'s "Evaluation
corpus" section) and reused by every later retrieval phase's tests.

`notes/*.md` are plain, realistic user notes — deliberately free of any
"this note exists to test X" meta-commentary, so shared filler words in that
commentary don't create false-positive matches across unrelated notes. The
rationale for each note lives here instead:

| Note | Category | Why it exists |
| --- | --- | --- |
| `user-preferences.md` | preferences | A stable, evergreen preference — should always be findable and should never expire. |
| `coffee-preference-change.md` | changed preferences | An explicit narrative preference change — tests that retrieval/consolidation surfaces the *current* answer (tea), not the superseded one (coffee), once contradiction handling ships (Stage One Phase 5 / Stage Two Phase 8). |
| `people.md` | names | Proper nouns (person and pet names) — a common, easy-to-get-wrong retrieval case. |
| `project-symbols.md` | code identifiers | Exact-identifier retrieval (a lexical/path-lane requirement, not a semantic-paraphrase one). |
| `deployment-constraint.md` | temporary constraints | An expiring, action-sensitive claim — exercises Stage One Phase 6's action-boundary metadata once implemented. |
| `external-wiki-import.md` | untrusted imports | Content pasted from an external source — exercises the `import-quarantine` scope (Stage One Phase 1): searchable, but must never auto-promote into curated `MEMORY.md`/topic pages, and must always be framed as untrusted reference material, never instructions. |
| `session-2026-07-20-notes.md` | session-scoped facts | Should be recallable within its own session (and forks), but must not leak into an unrelated session for the same agent — exercises the `session-lineage` scope. |

`queries.json` pairs a natural-language query with the note key(s) expected
to satisfy it (`expected_keys`), tagged by `category`.

## Recorded baseline (Stage One Phase 0)

See `tests/memory/test_baseline.py` and its recorded numbers — the current
`LongTermMemory` linear-scan search does not achieve perfect recall on this
corpus. In particular, `"What is my dog's name?"` misses `people.md` today:
`search()` splits on whitespace without stripping punctuation, so the query
term `dog's` (with apostrophe) never matches the stored text `dog`. This is
a genuine, faithful measurement of the current system — not a fixture
artifact — and is exactly the kind of gap Stage One Phase 3's proper lexical
index is expected to close. Do not "fix" this by rewording the query; the
point of a baseline is to record the current system honestly.

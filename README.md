# minion-assist

A minimal multi-agent CLI assistant with pluggable LLM providers, tool execution, persistent memory, and long-running task support. Two agents — **Ada** (general assistant) and **Elizabeth** (research specialist) — run a Think-Act-Observe loop, calling tools until they reach a final answer, then persisting conversation history across sessions. Responses can be streamed token-by-token in interactive mode.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Slash Commands](#slash-commands)
- [Image Attachments](#image-attachments)
- [MCP Servers](#mcp-servers)
- [Matrix Channel](#matrix-channel)
- [Multi-Agent Workspace](#multi-agent-workspace)
- [Heartbeat & Proactive Features](#heartbeat--proactive-features)
- [Dreaming](#dreaming)
- [Browser Tool](#browser-tool)
- [Voice Chat](#voice-chat)
- [Terminal UI](#terminal-ui)
- [PostgreSQL Session Store](#postgresql-session-store)
- [Module Reference](#module-reference)
  - [config](#config)
  - [providers](#providers)
  - [agents](#agents)
  - [tools](#tools)
  - [PermissionPolicy](#permissionpolicy-toolspolicypy)
  - [Plugin System](#plugin-system-pluginspy)
  - [skills](#skills)
  - [memory](#memory)
  - [session](#session)
  - [context](#context)
- [Codex Provider](#codex-provider)
- [Adding a Provider](#adding-a-provider)
- [Adding a Tool](#adding-a-tool)
- [Adding an Agent](#adding-an-agent)
- [Adding a Skill](#adding-a-skill)
- [Running Tests](#running-tests)

---

## Quick Start

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/)

```bash
# Install dependencies (uv.lock is committed — this gives a reproducible environment)
uv sync

# Optional extras
uv sync --extra tiktoken   # more accurate token estimation
uv sync --extra postgres   # PostgreSQL session store + memory index (psycopg3, watchdog, pgvector)

# Set up config (config lives in your home directory, not in the repo)
cp config.example.json ~/.minion-assist/config.json
# Edit ~/.minion-assist/config.json with your provider/agent settings

# Optional: set provider API keys (not needed for Codex OAuth)
# Create ~/.minion-assist/.env:  VOLCES_API_KEY=sk-...
# Or a project-local .env in the working directory (overrides the global one)

# If using OpenAI Codex (ChatGPT Plus / Codex subscription, no API key needed)
uv run codex-login

# Run
uv run minion-assist
```

At the prompt:

```
You: explain async/await in Python
You: /research latest benchmarks for Qwen 3.5
You: /help
You: exit
```

`/research <message>` routes to Elizabeth (researcher). Everything else goes to Ada (main). Type `/help` for a list of all slash commands.

---

## Project Structure

```
minion-assist/
├── docker-compose.yml           # PostgreSQL + pgvector container (optional, for session search)
├── config.json                  # Provider, model, agent, routing, workspace, and MCP config
├── .env                         # API keys (never commit)
├── pyproject.toml               # Package metadata and dependencies
├── src/minion_assist/
│   ├── minion.py                # Entry point — interactive REPL
│   ├── config.py                # Config loader (config.json + .env)
│   ├── config_report.py         # `minion-assist config` — effective, secret-redacted config (MEM-GAP-019)
│   ├── bootstrap.py             # Bootstrap prompt layer — workspace file discovery, budget, truncation, rendering
│   ├── context.py               # Context window overflow detection and history compaction
│   ├── cli_input.py             # PromptReader — Up/Down arrow key prompt history via prompt_toolkit
│   ├── commands.py              # Slash command dispatcher and built-in commands
│   ├── messages.py              # Provider-neutral content block helpers (text/image)
│   ├── media.py                 # File-backed attachment ingestion with MIME/size validation
│   ├── auth/                    # OAuth authentication for LLM providers
│   │   ├── codex_auth.py        # OpenAI Codex device-code OAuth flow; token stored at ~/.minion-assist/codex-auth.json
│   │   └── __init__.py
│   ├── providers/               # LLM API adapters
│   │   ├── base.py              # Protocol, ToolCall, LLMResponse types
│   │   ├── openai_compatible.py # OpenAI Chat Completions adapter
│   │   ├── anthropic.py         # Anthropic Claude adapter
│   │   ├── lmstudio.py          # LM Studio alias
│   │   ├── codex.py             # Codex app-server adapter (JSON-RPC 2.0 over stdio)
│   │   ├── embeddings.py        # EmbeddingProvider — text-to-vector for the memory index (optional)
│   │   └── __init__.py          # create_provider() factory
│   ├── agents/                  # Agent definitions and execution
│   │   ├── definitions.py       # AgentConfig, AGENTS dict (name, soul, max_tool_rounds)
│   │   ├── events.py            # Structured event dataclasses emitted by the runtime
│   │   ├── router.py            # Message → agent routing
│   │   ├── runner.py            # TAO loop (run_turn) with retry and recovery
│   │   ├── session.py           # AgentSession — headless per-agent execution unit
│   │   └── __init__.py
│   ├── mcp/                     # MCP (Model Context Protocol) client package
│   │   ├── types.py             # MCP type definitions
│   │   ├── schema.py            # MCP schema helpers
│   │   ├── client.py            # MCP client — stdio, sse, streamableHttp transports
│   │   └── __init__.py
│   ├── skills/                  # Agent skills (SKILL.md files with YAML frontmatter)
│   │   └── __init__.py          # discover_skills(), format_skills_prompt(), SkillInfo
│   ├── tools/                   # Executable tools for agents
│   │   ├── base.py              # Tool ABC, ToolSchema, _within() path guard
│   │   ├── policy.py            # PermissionPolicy — centralised path/URL/command safety rules + read_only_mode
│   │   ├── registry.py          # ToolRegistry — dispatch, schema export, hook support; exposes .policy
│   │   ├── read.py              # ReadTool — file/directory reading with pagination
│   │   ├── write.py             # WriteTool — file writing
│   │   ├── glob.py              # GlobTool — file pattern search
│   │   ├── bash.py              # BashTool — shell commands (PowerShell/bash); policy-aware SSRF + read_only checks
│   │   ├── ask_user.py          # AskUserTool — pause agent and prompt human for input
│   │   ├── git.py               # GitStatusTool, GitDiffTool, GitCommitTool — structured git interface; policy-aware
│   │   ├── edit.py              # EditTool — exact-string file editing with unique-match guard
│   │   ├── grep.py              # GrepTool — regex file search with context lines
│   │   ├── web_fetch.py         # WebFetchTool — fetch a URL, strip HTML, SSRF protection
│   │   ├── patch.py             # PatchPreviewTool — unified diff preview without writing
│   │   ├── apply_patch.py       # ApplyPatchTool — apply a unified diff patch via git apply
│   │   ├── find_definition.py   # FindDefinitionTool — AST-based symbol lookup across .py files
│   │   ├── todo.py              # TodoWriteTool, TodoReadTool — session-scoped todo list
│   │   ├── memory.py            # SaveMemoryTool, SearchMemoryTool
│   │   ├── write_daily_memory.py # WriteDailyMemoryTool — daily log append
│   │   ├── session_search.py    # SessionSearchTool — lexical + vector fused search across this agent's own past sessions (requires PostgreSQL; MEM-GAP-006)
│   │   ├── browser.py           # BrowserTool — Playwright browser automation (start/navigate/evaluate/screenshot/pick/cookies/stop)
│   │   ├── mcp.py               # McpToolAdapter, McpStatusTool, ListMcpResourcesTool, ReadMcpResourceTool, ListMcpPromptsTool, GetMcpPromptTool
│   │   ├── skill.py             # SkillTool — load skill instructions on demand
│   │   ├── spawn_subagent.py    # SpawnSubagentTool — delegate tasks to child AgentSession; _make_subagent_registry
│   │   ├── task.py              # ReadTaskTool, UpdateTaskTool — long-running task progress
│   │   ├── web_search.py        # WebSearchTool — DuckDuckGo web search via ddgs, no API key
│   │   └── __init__.py          # default_registry() factory; registers all tools; sets registry.policy
│   ├── workspace.py             # Per-agent workspace management: ensure_workspace, check_workspace, WorkspaceVanishedError
│   ├── spawn_registry.py        # Multi-agent spawn limits: get_spawn_depth, count_active_children, MAX_SPAWN_DEPTH
│   ├── plugins.py               # Plugin manifest loader — tools, hooks, skills, and trust from plugins.json
│   ├── matrix/                  # Optional Matrix channel (requires matrix-nio + aiosqlite; no E2E — see ADR 0005)
│   │   ├── channel.py           # MatrixChannel — lifecycle manager (daemon thread + asyncio loop)
│   │   ├── monitor.py           # monitor_matrix() — auth, callbacks, sync loop, teardown
│   │   ├── handler.py           # MatrixMessageHandler — full inbound pipeline
│   │   ├── outbound.py          # MatrixOutbound — markdown-formatted chunked send, draft preview, reactions, typing indicators
│   │   ├── format.py            # to_matrix_html() / build_content() — markdown → org.matrix.custom.html
│   │   ├── inbound_dedupe.py    # SQLite-backed event-ID deduplication (24 h TTL)
│   │   ├── room_sessions.py     # SQLite (room_id, agent_id) → session_id mapping (MEM-GAP-001)
│   │   ├── bot_loop.py          # Sliding-window rate limiter per room
│   │   ├── exec_approvals.py    # DM-based remote tool approval via ✅/❌ reactions
│   │   ├── auto_join.py         # Invite handler — always / allowlist / off policy
│   │   ├── auth.py              # Authentication — access token / password / SSO
│   │   ├── config.py            # MatrixConfig and nested dataclasses
│   │   ├── allowlist.py         # User-ID normalisation and wildcard allowlist check
│   │   └── __init__.py
│   ├── voice/                   # Optional voice pipeline (--voice flag; requires [voice] extra)
│   │   ├── audio.py             # MicrophoneStream context manager, play_audio(), list_devices()
│   │   ├── vad.py               # SileroVAD wrapper, VadCapture daemon thread, build_vad_capture()
│   │   ├── stt.py               # STTAdapter ABC, ParakeetSTT, WhisperSTT, build_stt()
│   │   ├── tts.py               # TTSAdapter ABC, Qwen3TTS, KokoroTTS, PiperTTS, build_tts()
│   │   ├── session.py           # VoiceSession orchestrator, build_voice_session()
│   │   └── __init__.py
│   ├── memory/                  # Persistent memory storage
│   │   ├── short_term.py        # JSONL conversation history (atomic writes)
│   │   ├── files.py             # MemoryFileRepository — merged workspace-root note store
│   │   ├── service.py           # MemoryService — facade AgentSession/tools depend on
│   │   ├── models.py            # MemoryHit, MemoryLocator, MemoryExcerpt, MemoryStatus
│   │   ├── extractor.py         # Background fact extraction (degraded-mode path, no db)
│   │   ├── capture_worker.py    # CaptureWorker — durable capture-job queue worker (needs db)
│   │   ├── chunking.py          # Heading-aware Markdown chunker for the lexical index
│   │   ├── postgres_index.py    # PostgresMemoryIndex — rebuildable lexical/hybrid index + Phase 5-7 tables (needs db)
│   │   ├── watcher.py           # MemoryIndexWatcher — live debounced fs sync (needs db+watchdog)
│   │   ├── migration.py         # Phase 0: legacy-root -> merged-root migration tooling
│   │   ├── cli.py               # `minion-assist memory <migrate|consolidate|commitments|knowledge|import|...>` subcommands
│   │   ├── baseline.py          # Retrieval recall/latency measurement vs. fixture corpus
│   │   ├── long_term.py         # Legacy Markdown notes store (superseded by MemoryService)
│   │   ├── consolidation.py     # Phase 5: MemoryConsolidator — preview/approve/reject/rollback proposals
│   │   ├── consolidation_scheduler.py  # Phase 5: daily automated consolidation-preview scheduler
│   │   ├── boundaries.py        # Phase 6: action-boundary frontmatter parsing (owner/applies_when/expires_at/…)
│   │   ├── commitments.py       # Phase 6: inferred, short-lived commitment extraction
│   │   ├── commitment_worker.py # Phase 6: durable commitment-extraction queue worker (needs db)
│   │   ├── message_embedding_worker.py # MessageEmbeddingWorker — durable session-message embedding queue worker (needs db + embeddings; MEM-GAP-006)
│   │   ├── knowledge.py         # Phase 7: claim marker parsing/rendering + compile_digest()
│   │   ├── digest_scheduler.py  # Phase 7: KnowledgeDigestScheduler — daily KNOWLEDGE_DIGEST.md recompilation
│   │   ├── import_review.py     # Phase 7: ImportReviewer — quarantined-import preview/approve/reject
│   │   ├── forgetting.py        # Phase 7: forget_source() — evidence-source deletion cascade
│   │   ├── retention_scheduler.py # MemoryRetentionScheduler — daily operational/telemetry-table cleanup (MEM-GAP-015)
│   │   └── __init__.py
│   └── session/                 # Session metadata tracking
│       ├── store.py             # JSON session store (turn counts, timestamps)
│       ├── db.py                # SessionDB — PostgreSQL session + message store with FTS (optional)
│       └── __init__.py
└── tests/                       # pytest test suite (2689 tests, 2 skipped)
```

---

## Configuration

### Config file location

Config is loaded from the **first file found** in this order:

1. `~/.minion-assist/config.json` — user home *(recommended — keeps credentials out of project directories)*
2. `./config.json` — current working directory *(development / testing)*

Override the home directory with the `MINION_ASSIST_HOME` environment variable:

```bash
MINION_ASSIST_HOME=/custom/path uv run minion-assist
```

> **Security note:** `config.json` can contain secrets such as Matrix access tokens. It is intentionally excluded from the repository via `.gitignore`. Use `config.example.json` as the starting template — it ships with placeholder credentials only.

### `~/.minion-assist/config.json`

Copy `config.example.json` from the repo to get started:

```bash
cp config.example.json ~/.minion-assist/config.json
```

Example structure (see `config.example.json` for the full template):

```json
{
  "models": {
    "providers": {
      "openai": {
        "api": "codex",
        "models": [
          {"id": "gpt-5.5", "name": "GPT-5.5 (Codex subscription)", "contextWindow": 200000, "maxOutputTokens": 100000}
        ]
      },
      "lmstudio": {
        "baseUrl": "http://127.0.0.1:1234/v1",
        "api": "lmstudio",
        "models": [
          {"id": "qwen-qwen3.5-9b", "name": "Qwen 3.5 9B", "contextWindow": 262144, "maxOutputTokens": 32768}
        ]
      }
    }
  },
  "agents": {
    "main":       {"model": "openai/gpt-5.5"},
    "researcher": {"model": "lmstudio/qwen-qwen3.5-9b", "route_prefix": "/research"}
  },
  "workspace": {"path": "~/.minion-assist"},
  "streaming":  {"chat_mode": true, "task_mode": false},
  "compaction": {},
  "mcp": {
    "servers": {
      "playwright": {
        "transport": "stdio",
        "command": "npx",
        "args": ["@playwright/mcp@latest"],
        "tool_timeout": 60
      }
    }
  }
}
```

**Fields:**

| Field | Description |
|---|---|
| `models.providers.<name>.baseUrl` | API base URL for the provider |
| `models.providers.<name>.api` | Adapter type: `openai-completions`, `lmstudio`, or `anthropic` |
| `models.providers.<name>.models` | List of available models with `id`, `contextWindow`, and `maxOutputTokens` |
| `models.providers.<name>.models[].inputModalities` | *(Optional)* List of supported input types. Include `"image"` to enable vision/attachment support for that model. Defaults to `["text"]`. |
| `agents.<id>.model` | `"<provider>/<model-id>"` — which model each agent uses |
| `agents.<id>.route_prefix` | Optional command prefix that routes to this agent (e.g. `"/research"`). Omit for the default fallback agent. |
| `workspace.path` | Root directory for persisted history and memory (tilde-expanded) |
| `streaming.chat_mode` | `true` to stream tokens in the interactive REPL; `false` for full-response display |
| `streaming.task_mode` | `true` to stream tokens in programmatic/task invocations; `false` by default |
| `models.providers.<name>.models[].contextWindow` | The model's total token capacity; used per-agent as the compaction budget |
| `models.providers.<name>.models[].maxOutputTokens` | Maximum tokens the model may generate per response; sent to the API as `max_tokens` |
| `compaction.preserve_tokens` | *(Optional override)* Tokens reserved for response + overhead. Omit to auto-compute as `maxOutputTokens + 1 024`. Clamped to `[2 000, contextWindow ÷ 2]` at runtime so at least half the window is always usable for history. |
| `mcp.servers.<name>.transport` | MCP transport type: `stdio`, `sse`, or `streamableHttp` |
| `mcp.servers.<name>.command` | *(stdio only)* Executable to launch the MCP server |
| `mcp.servers.<name>.args` | *(stdio only)* Arguments passed to the server command |
| `mcp.servers.<name>.url` | *(sse / streamableHttp only)* URL of the MCP server endpoint |
| `memory.enable_extraction` | *(Optional, default `true`)* Set to `false` to disable the background fact-extraction API call fired after each turn. Useful for expensive models where the extra call doubles token costs. |
| `memory_consolidation.*` | *(Optional, disabled by default)* Automated daily consolidation-preview scheduler (Stage One Phase 5, slice D) — `enabled`/`hour`/`minute`/`timezone`/`agent_id`/`top_n`. See "Consolidation ranking and preview drafting". |
| `commitments.enabled` | *(Optional, default `false`)* Inferred, short-lived commitment extraction (Stage One Phase 6, slice B). Requires a configured database. See "Commitments — schema and extraction pipeline". |
| `knowledge_digest.*` | *(Optional, disabled by default)* Automated daily `KNOWLEDGE_DIGEST.md` compilation scheduler (Stage One Phase 7, slice D) — `enabled`/`hour`/`minute`/`timezone`/`agent_id`/`max_chars`. See "Compiled knowledge digest". |
| `memory_retention.*` | *(Optional, disabled by default)* Daily bounded cleanup of pure operational/telemetry tables — never messages, memory files, or rollback history (MEM-GAP-015) — `enabled`/`hour`/`minute`/`timezone`/`retention_days`. Also runnable on demand via `minion-assist memory retention [--apply]`. See "Operational-table retention". |
| `database.url` | *(Optional)* PostgreSQL connection string for session history storage and FTS search. Omit to run file-only. Example: `"postgresql://minion:minion@localhost:5433/minion_assist"`. |
| `extra_plugin_manifests` | *(Optional)* List of additional `plugins.json` file paths to load beyond the two fixed locations (`~/.minion-assist/plugins.json` and `.minion-assist/plugins.json`). Paths support `~` expansion. |
| `bootstrap.enabled` | *(Optional, default `true`)* Set to `false` to disable workspace bootstrap file injection entirely. |
| `bootstrap.path` | *(Optional, default `null`)* Directory to search for bootstrap files. `null` uses `Path.cwd()` at runtime. |
| `bootstrap.max_chars` | *(Optional, default `20000`)* Maximum characters to inject from any single bootstrap file. Larger files are truncated with a head+tail excerpt. |
| `bootstrap.total_max_chars` | *(Optional, default `60000`)* Maximum total characters across all bootstrap files combined. |
| `bootstrap.truncation_warning` | *(Optional, default `"always"`)* When to inject a truncation warning: `"always"`, `"once"` (first occurrence only per process), or `"off"`. |
| `embeddings.*` | *(Optional)* Activates the semantic-search vector lane (MEM-GAP-006). `provider`/`model`/`dimensions` — see "Embeddings and the vector lane". |
| `session_search.min_similarity` | *(Optional, default `0.15`)* Minimum cosine similarity a session vector-search hit must clear to be returned at all (R2-GAP-012). Only matters when `embeddings` above is configured. See "Exact job-coverage gap detection". |
| `memory_reconciliation.*` | *(Optional)* Periodic self-healing pass for transient mirror/job-enqueue failures — `interval_seconds`/`quiet_seconds`. See "Reconciliation: self-healing writes and concurrency safety". |
| `heartbeat.*` | *(Optional, disabled by default)* Periodic proactive check-in — `enabled`/`interval_seconds`/`prompt`/`agent_id`/`notification_room_id`. See "Heartbeat & Proactive Features". |
| `dreaming.*` | *(Optional, disabled by default)* Nightly session-reflection scheduler — `enabled`/`hour`/`minute`/`timezone`/`lookback_days`/`agent_id`. |
| `multi_agent.*` | *(Optional)* Subagent spawn limits — `max_spawn_depth`/`max_children_per_agent`/`default_subagent_timeout_seconds`. See "Multi-Agent Workspace". |
| `logging.llm_requests` | *(Optional, default `true`)* Log full provider request/response payloads to `workspace/logs/`. May include recalled memory content in the payload — leave `false` unless actively debugging. Known secrets are always redacted from these logs regardless of this setting — see "Secret redaction in logs" (MEM-GAP-017). |
| `codex.*` | *(Optional)* Only relevant for an agent whose provider uses `"api": "codex"` — `allow_all_commands`/`auth_refresh_interval_seconds`. |

**Full config surface every run.** `config.example.json` documents every section above (MEM-GAP-019) — copy it as a starting point rather than reconstructing the surface from this table or from source.

### Show effective configuration (`minion-assist config`)

```bash
minion-assist config
```

Prints every resolved `config.json` section — what minion-assist actually understood after validation, not just what you wrote — with every credential (`api_key`, Matrix `access_token`/`password`, `database.url`'s inline `user:pass@`, MCP server `env`/`headers` values) masked as `***` rather than omitted, so you can confirm a secret *is* configured without ever exposing its value. Runs before any REPL/session/provider setup — the same non-interactive dispatch `minion-assist memory ...` already uses (see `config_report.py`). Exits `2` with the underlying `ConfigError` message if `config.json` itself fails to load/validate.

### Secret redaction in logs (MEM-GAP-017)

`logging.llm_requests` (default `true`) logs the *entire* provider request/response body verbatim to `workspace/logs/*.log` — including whatever memory content got injected into the system prompt (`<relevant_memories>`, `MEMORY.md`, `USER.md`). Kept on by default deliberately (useful for debugging), but every entry `llm_logger.py` writes — requests, responses, turn markers, tool calls, tool results — is passed through a redaction pass first that masks exact occurrences of every currently-configured credential: every provider's `api_key`, Matrix's `access_token`/`password`, and a configured `database.url`'s inline `user:pass@` portion (host/port/db name stay visible — useful for debugging, not secret). Masked as a fixed `***REDACTED***` marker wherever found, however many times it appears.

This is **known-value** redaction, not pattern-guessing: `config_report.py`'s `collect_secret_values()` (the same field-name-based walk `minion-assist config`'s redaction already uses, reused rather than duplicated) gathers every real secret actually configured through `config.json`/`.env`, computed once per process and cached (config is immutable after startup). It catches a real configured secret wherever it turns up — including a case the gap analysis specifically calls out, a user's own memory note that happens to paste a real key — but can't recognize an arbitrary string as sensitive without already knowing it's one; a key that was never configured through minion-assist (e.g. hardcoded in a script the agent reads) wouldn't be caught. Redaction failures fail open (never block a log write) — consistent with `llm_logger.py`'s existing "a disk error never crashes the LLM call" posture.

### `.env`

API keys are **never** stored in `config.json`. Place them in a `.env` file:

```
VOLCES_API_KEY=sk-...
LMSTUDIO_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

The key for a provider named `foo` is looked up as `FOO_API_KEY`.

`.env` is loaded from two locations (later overrides earlier):

1. `~/.minion-assist/.env` — global defaults
2. `./.env` — project-local overrides (gitignored)

Neither file is required — providers that use OAuth (Codex) or no auth (LM Studio) work without any API key.

---

## Architecture

```
User input
    │
    ▼
router.resolve()
    │  "/research ..." → researcher (Elizabeth)
    │  everything else → main (Ada)
    ▼
AgentSession.send(message, on_event=callback, stream=True/False)
    │
    ├─ Build system prompt:
    │     soul + bootstrap Project Context (AGENTS.md, SOUL.md, TOOLS.md,
    │            USER.md, MEMORY.md, …) — read live every turn, no restart needed
    │          + <bootstrap_pending> guidance (if BOOTSTRAP.md present)
    │          + <relevant_memories> (proactive search — top-5 snippets; fires MemoryInjected)
    │          + <active_task> (current task progress, if a task is active)
    │          + <context_budget> warning (if history > 50% of context window)
    │          + <available_skills> suffix
    │          + Today's date (appended last so the large stable prefix is
    │            byte-identical across turns — enables OpenAI prompt caching)
    │
    ├─ [if memory configured] compactor.peek_compaction_head(history) → memory.flush_head(head)
    │      pre-compaction flush (Phase 2 slice B) — deterministic, no LLM call, fires MemoryFlushed
    ├─ compactor.compact(history, provider, on_compaction=..., on_compaction_failed=...)
    │      no-op when under budget; fires CompactionStarted / CompactionFailed events
    │
    └─ run_turn(provider, agent_name, system, max_tokens, tools, messages,
                on_event=callback, stream=True/False,
                max_tool_rounds=agent.max_tool_rounds)
           │
           │  ┌────────────────────────────────────────────────────────────┐
           │  │                Think-Act-Observe loop                       │
           │  │                                                            │
           │  │  _call_with_retry(provider.chat(...))                      │
           │  │      retries 429/5xx/timeouts with exponential backoff     │
           │  │       │                                                    │
           │  │       ▼                                                    │
           │  │  LLMResponse(text, tool_calls, finish_reason)              │
           │  │       │                                                    │
           │  │  empty response? → inject nudge, loop                     │
           │  │  finish_reason=="length"? → inject continuation, loop     │
           │  │  finish_reason=="tool_calls"?                              │
           │  │    yes → emit ThoughtEmitted (if preamble text, non-stream)│
           │  │          emit ToolCalled → execute tool → append result    │
           │  │          → loop (up to max_tool_rounds)                    │
           │  │    no  → emit FinalAnswer → return                         │
           │  └────────────────────────────────────────────────────────────┘
           │
           ├─ short_term.save(agent_id, messages)   ← persists history to JSONL (always)
           ├─ session_store.touch(agent_id, ...)    ← updates turn count / timestamp
           ├─ db.mirror_message(...) [optional]     ← idempotent PostgreSQL mirror for FTS search
           └─ [if db configured] db.enqueue_capture_job(...) — durable queue row,
                  processed later by the standalone CaptureWorker thread (Phase 2 slice C)
              [else] extract_and_save_async(memory, provider, last_exchange)
                  daemon thread — extracts 0–3 key facts, appends to quarantined
                  memory/imports/_auto_extracted.md (see MemoryService.remember_import)
```

`CaptureWorker` (`memory/capture_worker.py`) is a single background thread, started once at process startup (not per turn), that polls `memory_capture_jobs` for due work and records extracted facts as `memory_proposals` rows. See "Durable capture-job queue" under PostgreSQL Integration for the full schema and design rationale.

**Event system:** `run_turn` and `AgentSession` emit structured event objects to the `on_event` callback rather than calling `print()` or `input()` directly. This makes the agent runtime usable headlessly (tests, scripts, web APIs) — pass `on_event=None` for silent execution.

| Event class | Emitted when |
|---|---|
| `StreamingStarted(agent_name)` | First token of a streaming response arrives — signals the caller to print the agent name prefix |
| `TokenStreamed(token)` | Each subsequent streaming text fragment — print inline without a newline |
| `ThoughtEmitted(agent_name, text)` | Model produced preamble text before a tool call (non-streaming only — in streaming mode the tokens were already printed) |
| `FinalAnswer(agent_name, text)` | Model produces its complete response with no more tool calls — the turn is done |
| `ToolCalled(name, args)` | A tool is about to execute — useful for showing `[tool: name(...)]` status lines |
| `ToolCompleted(name, elapsed_ms, output_chars)` | A tool finished executing — carries timing and output size for diagnostics |
| `MaxRoundsReached(agent_name, message)` | TAO loop hit the `max_tool_rounds` cap without a final answer |
| `CompactionStarted()` | History compaction is about to summarise old messages |
| `CompactionFailed(error)` | Compaction summarisation call failed; original history is retained unchanged |
| `TurnCompleted(agent_name, trace_id, turn_number, tool_calls_made, input_tokens, output_tokens, elapsed_ms, compacted)` | Emitted after every successful turn — intended for monitoring, cost-tracking, and structured logging (ignored by the interactive CLI) |

**Message format** is the OpenAI Chat Completions wire format throughout — `{"role": "user"|"assistant"|"tool", "content": "..."}`. Providers that use a different format (Anthropic) convert internally.

**Workspace layout** (default `~/.minion-assist/`):

```
~/.minion-assist/
├── sessions/
│   ├── main/                    ← Ada's session files (one per conversation)
│   │   ├── {uuid}.jsonl         ← conversation history for one session
│   │   └── {uuid}.name          ← optional human-readable name for that session
│   └── researcher/              ← Elizabeth's session files
│       └── {uuid}.jsonl
├── workspaces/                  ← per-agent memory + bootstrap root (see "Multi-Agent Workspace")
│   ├── main/
│   │   ├── AGENTS.md, SOUL.md, USER.md, MEMORY.md, DREAMS.md  ← bootstrap files
│   │   ├── KNOWLEDGE_DIGEST.md  ← compiled, machine-owned (Stage One Phase 7, slice D)
│   │   └── memory/
│   │       ├── YYYY-MM-DD.md    ← daily notes (write_daily_memory)
│   │       ├── topics/          ← explicit save_memory notes (project-goals.md, …)
│   │       └── imports/         ← quarantined, unreviewed (_auto_extracted.md, note tool)
│   └── researcher/
│       └── memory/topics/findings.md
├── memory/                      ← LEGACY per-agent notes root, pre-Stage-One-Phase-0.
│   └── main/project-goals.md    ← migrated via `minion-assist memory migrate --apply`
├── tasks/
│   ├── main.json         ← Ada's active task progress file (created on demand)
│   └── researcher.json   ← Elizabeth's active task progress file
├── attachments/          ← staged image/media files (organised by date)
│   └── 2024-01-15/
│       └── <sha256>-screenshot.png
├── skills/               ← global skills (available to all agents)
│   └── my-skill/
│       └── SKILL.md
├── prompt_history.txt    ← REPL input history for Up/Down arrow navigation
└── sessions.json         ← session metadata (turn counts, timestamps)
```

Project-level skills live at `.minion-assist/skills/` relative to the working directory and override global skills with the same name.

---

## Slash Commands

The REPL recognises slash commands that start with `/`. Type `/help` to print the command list at any time.

| Command | Description |
|---|---|
| `/help` | Print a summary of all available commands |
| `/commands` | Alias for `/help` |
| `/quit` | Exit the REPL |
| `/exit` | Alias for `/quit` |
| `/new` | Clear all agent sessions (history, attachments) and start fresh |
| `/new all` | Alias for `/new` |
| `/clear` | Alias for `/new` |
| `/reset` | Alias for `/new` |
| `/compact` | Force immediate context compaction for all agents |
| `/status [deep]` | Show session metadata (turn counts, last active) for each agent. `/status deep` also reports background-worker liveness (capture/commitment/message-embedding workers, memory watcher, memory reconciliation, memory retention (MEM-GAP-015), schedulers, one `session_writes:{agent_id}` row per agent for mirror/enqueue health — MEM-GAP-007/016 — one `memory_search:{agent_id}` row per agent with a configured index for degraded-mode search health — MEM-GAP-008 — and one `memory_extractor:{agent_id}` row per agent *without* a configured database for degraded-mode extraction health — MEM-GAP-013) and, when a database is configured, per-agent capture/commitment/message-embedding queue lag and lexical-index summary. Works from both the REPL and Matrix chat. |
| `/agents` | List all known agents with turn counts and last-active timestamps |
| `/session [N\|uuid-prefix]` | List past conversation sessions for the active agent; restore one by index or UUID prefix |
| `/rename [N] <name>` | Give the current session (or session N from /session) a descriptive name |
| `/delete-session [N\|uuid-prefix]` | Permanently delete a past session (cannot delete the active session). When a database is configured, also removes that session's PostgreSQL mirror, capture/commitment jobs, proposals (and their indexed chunks/draft previews/knowledge-graph evidence citations), and commitments — not just the local JSONL file (MEM-GAP-003). Any note already promoted from a proposal is left untouched, since it's independent, reviewed memory by then. |
| `/switch [agent_id]` | Switch the active agent routing target (e.g. `/switch researcher`); defaults to current agent |
| `/diagnose` | Check each agent's provider configuration and API key status |
| `/mcp-reload` | Reconnect all MCP servers and refresh tool adapters in every session |
| `/mcp-list` | List connected MCP servers and their available tools |
| `/plan` | Enable read-only mode — agent can reason but not write files or run commands |
| `/auto` | Disable read-only mode — restore full tool access (write, bash, git commit) |
| `/providers` | Show LLM provider and model configuration for all configured agents |
| `/provider test [agent_id]` | Send a minimal API call to verify provider connectivity and latency |
| `/audit [n]` | Show the last *n* permission decisions (allowed/denied) from the audit log (default 20) |
| `/fork <new_id>` | Copy the active session's history into a new session with the given ID |
| `/export [--md\|--html] <path>` | Export the active session's conversation transcript to a file |
| `/mcp-enable <server>` | Reconnect a disabled MCP server and refresh its tool adapters |
| `/mcp-disable <server>` | Disconnect an MCP server and remove its tools for this session |
| `/plugin list` | List all tool names registered in the active agent's tool registry |
| `/research <message>` | Route the message to Elizabeth (researcher) |

Route-targeted commands work on individual agent sessions. For example, `/research /new` clears only Elizabeth's session.

### Read-only mode (`/plan` / `/auto`)

`/plan` enables **read-only mode** on the active agent's `PermissionPolicy`.  While active, the agent can read files, search code, and reason — but any attempt to write files, run shell commands, or create git commits returns an error message instead.

`/auto` disables read-only mode and restores full tool access.

```
You: /plan
Read-only mode enabled for 'main'. ...
You: write a refactoring plan for src/foo.py
Ada: [reads files, reasons, proposes plan — does not edit anything]
You: /auto
Full tool access restored.
You: apply the refactoring
Ada: [edits files, runs tests, commits]
```

---

## Image Attachments

Vision-capable models (those with `"inputModalities": ["text", "image"]` in `config.json`) can receive image files alongside text prompts.

**Workflow:**

```
You: /attach /home/user/screenshot.png
[Attached: screenshot.png (342 KB, image/png)]

You: what is shown in this image?
Ada: The screenshot shows ...
```

**Details:**
- Use `/attach <path>` to stage a file. The file is copied to `{workspace}/attachments/YYYY-MM-DD/<sha256>-<name>`.
- Use `/attachments` to list currently staged files.
- Use `/clear-attachments` to drop staged files without sending them.
- Attachments are automatically cleared after each successful send.
- Maximum image size: 15 MB. Maximum images per turn: 4.
- Unsupported MIME types or oversized files are rejected with an error message.
- The OpenAI provider converts images to base64 `image_url` data URLs. The Anthropic provider uses its native base64 image block format.

**Matrix:** sending an image (with or without a caption) in a Matrix room works the same way, no `/attach` needed — the bot downloads the file from the homeserver's media repository and stages it into the same `{workspace}/attachments/` store. A caption becomes the accompanying text prompt; a caption-less image is sent with a placeholder prompt so the model still describes it. If the download fails or the file is invalid/oversized, the bot replies in the room with the reason instead of silently ignoring the message.

**Audio (TUI Phase 2):** `/attach` also accepts audio files (WAV, MP3, FLAC, OGG — 50 MB cap) in both the REPL and the [Terminal UI](#terminal-ui), staged the same way as images. No provider in this codebase understands an audio input block, so an attached audio file is never sent to the LLM as audio — its filename and duration are folded into the prompt text instead (`make_user_content()` in `messages.py`), the same way a caption-less image gets a placeholder prompt. The waveform preview and **Play** button are TUI-only; the REPL stages and describes audio files but has no local playback UI.

**Database-backed image search (R3-GAP-002):** an attached image is always sent to the model directly (as above) — this is about what becomes *searchable later*. Before this, an image's PostgreSQL row only ever contained the message's typed text; the image itself was invisible to capture (fact extraction), session FTS, and session-search vectors. When a database is configured, sending an image now durably enqueues a captioning job (`image_captions` table) — a background `ImageCaptionWorker` asks that message's own agent to describe the image factually, then folds the description into the message's stored text (`[Image: name] <description>`), which PostgreSQL's full-text search picks up automatically and which invalidates any stale embedding so the next reconciliation pass re-embeds the message with the caption included. If the owning agent's configured model has no `"image"` in `inputModalities`, the job is marked `unsupported` (visible via a direct query, not a silent failure) rather than retried forever. No new config key — this runs automatically whenever a database is configured, the same "not an opt-in feature" precedent already used for other data-integrity workers (`capture_worker`, `message_embedding_worker`).

---

## MCP Servers

minion-assist supports the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP). Configure MCP servers in `config.json` under `"mcp"`:

```json
"mcp": {
  "servers": {
    "playwright": {
      "transport": "stdio",
      "command": "npx",
      "args": ["@playwright/mcp@latest"],
      "tool_timeout": 60
    },
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    },
    "myserver": {
      "transport": "sse",
      "url": "http://localhost:8000/sse"
    }
  }
}
```

**Transports:** `stdio` (subprocess), `sse` (server-sent events), `streamableHttp`.

**How it works:**
- At startup, minion-assist connects to each configured MCP server.
- Tools exposed by MCP servers are registered in the tool registry as `mcp__<server>__<tool>` (e.g. `mcp__playwright__browser_navigate`).
- The agents can call MCP tools like any built-in tool.
- Five built-in management tools are always registered when MCP is configured: `mcp_status`, `list_mcp_resources`, `read_mcp_resource`, `list_mcp_prompts`, and `get_mcp_prompt`.
- `${VAR}` references in `args`, `env`, `headers`, and `url` are expanded from environment variables at startup.

Use the `/status` command or ask the agent to call `mcp_status` to check server connectivity.

### Playwright MCP (browser automation)

Playwright MCP gives agents full browser control — navigate, click, fill forms, take screenshots, and inspect page accessibility trees.

**Prerequisites:**

```bash
# Install the Playwright MCP browser (chrome-for-testing)
npx @playwright/mcp install-browser chrome-for-testing
```

**Config (already in `config.json`):**

```json
"playwright": {
  "transport": "stdio",
  "command": "npx",
  "args": ["@playwright/mcp@latest"],
  "tool_timeout": 60
}
```

**Available tools (23 total):**

| Tool | Description |
|---|---|
| `mcp__playwright__browser_navigate` | Navigate to a URL |
| `mcp__playwright__browser_snapshot` | Capture accessibility tree of current page (text-based, best for agents) |
| `mcp__playwright__browser_click` | Click on an element |
| `mcp__playwright__browser_type` | Type text into a field |
| `mcp__playwright__browser_fill_form` | Fill multiple form fields at once |
| `mcp__playwright__browser_take_screenshot` | Take a screenshot (saved to `{workspace}/playwright-output/`) |
| `mcp__playwright__browser_evaluate` | Run JavaScript on the page |
| `mcp__playwright__browser_press_key` | Press a keyboard key |
| `mcp__playwright__browser_tabs` | List, create, close, or switch browser tabs |
| `mcp__playwright__browser_wait_for` | Wait for text to appear or a timeout |
| `mcp__playwright__browser_console_messages` | Get all browser console messages |
| `mcp__playwright__browser_network_requests` | List network requests since page load |
| … and 11 more (use `mcp_status` to see the full list) |

**Recommended workflow for agents:**

```
You: go to https://news.ycombinator.com and summarize the top 5 stories

Ada: [tool: mcp__playwright__browser_navigate({'url': 'https://news.ycombinator.com'})]
     [tool: mcp__playwright__browser_snapshot({})]
     Based on the page snapshot:
     1. ...
```

**Screenshots** are automatically saved to `~/.minion-assist/playwright-output/screenshot-<timestamp>.png` when the agent calls `browser_take_screenshot`. The agent receives the file path in the tool result.

**Notes:**
- The browser opens as a visible window by default (headed mode). You can watch the agent browse.
- Use `browser_snapshot` (accessibility tree) for page analysis — it works with all text models. Use `browser_take_screenshot` only with vision-capable models.
- Add `"--headless"` to `args` to run without a visible window: `"args": ["@playwright/mcp@latest", "--headless"]`.

---

## Matrix Channel

minion-assist can connect to a Matrix homeserver so agents are reachable from any Matrix client (Element, Cinny, etc.). The channel runs in a background thread. Each Matrix room gets its own isolated `AgentSession` (see "Room-scoped sessions" below) — the REPL's own session is separate and only used as a last-resort fallback.

### Prerequisites

```bash
# Install Matrix channel dependencies
uv add "minion-assist[matrix]"
```

> **No E2E encryption support.** This channel intentionally does not implement
> Matrix's end-to-end encryption — see
> [ADR 0005](docs/adr/0005-no-matrix-e2e-encryption.md) for why. Use an
> unencrypted room; Matrix rooms cannot be un-encrypted once
> `m.room.encryption` is set, so don't enable encryption on the room you
> point this bot at.

### Configuration

Add a `channels.matrix` block to `config.json`:

```json
{
  "channels": {
    "matrix": {
      "homeserver": "https://matrix.example.org",
      "userId": "@bot:example.org",
      "accessToken": "syt_YourAccessTokenHere",
      "defaultAgentId": "main",
      "ackReaction": "👀",
      "groupPolicy": "allowlist",
      "groupAllowFrom": ["@alice:example.org"],
      "groups": {
        "!roomid:example.org": {"agent": "researcher", "enabled": true}
      },
      "execApprovals": {
        "enabled": true,
        "approvers": ["@alice:example.org"]
      },
      "botLoop": {
        "enabled": true,
        "maxEventsPerWindow": 10,
        "windowSeconds": 60,
        "cooldownSeconds": 300
      },
      "storePath": "~/.minion-assist/matrix"
    }
  }
}
```

**Required fields:** `homeserver`, `userId`, and at least one of `accessToken` or `password`.

| Field | Description |
|-------|-------------|
| `homeserver` | Matrix homeserver URL (e.g. `"https://matrix.example.org"`) |
| `userId` | Bot's full Matrix user ID (e.g. `"@bot:example.org"`) |
| `accessToken` | Preferred — bot account access token (no login call) |
| `password` | Alternative — password login (creates a new device session) |
| `defaultAgentId` | Agent used when no per-room override is configured |
| `ackReaction` | Emoji sent immediately on receipt to acknowledge the message (e.g. `"👀"`) |
| `groupPolicy` | `"open"` (anyone may send) or `"allowlist"` (only listed users) |
| `groupAllowFrom` | Global allowlist of Matrix user IDs; `"*"` permits everyone |
| `groups` | Per-room overrides — map room IDs to `{"agent": "...", "enabled": true/false}` |
| `execApprovals.enabled` | `true` to send bash tool approval requests via DM |
| `execApprovals.approvers` | List of Matrix user IDs who receive approval DMs |
| `botLoop.enabled` | `true` to rate-limit events per room (prevents bot loops) |
| `storePath` | Directory for SQLite state (dedupe, room sessions) |

### How It Works

**Room routing:** each Matrix room can be mapped to a specific agent via `groups`. Messages in unmapped rooms go to `defaultAgentId`.

**Room-scoped sessions:** every `(room_id, agent_id)` pair gets its own persistent `AgentSession` — a room is this deployment's unit of conversation isolation (e.g. a "Movie" room and a "Work" room routed to the same agent never share history), not the Matrix thread feature. The binding is stored in SQLite (`matrix/room_sessions.db`) and survives restarts; a room's first message always starts a fresh session rather than inheriting any other room's history. See [ADR 0006](docs/adr/0006-room-scoped-matrix-sessions.md) for the full rationale.

**Slash commands are room-scoped too (R2-GAP-004).** `/new`, `/compact`, `/session`, `/rename`, `/delete-session`, `/fork`, `/plan`/`/auto`, `/audit`, and `/provider test` — every command that targets "the currently active session" — act on the calling room's own `AgentSession`, not the REPL's shared per-agent default. `MatrixMessageHandler._dispatch_and_reply()` resolves the room session *before* parsing a command (previously it only did so for ordinary, non-command messages) and passes it into `CommandContext.sessions` under the room's `agent_id`; every command handler's existing `ctx.sessions.get(ctx.target_agent_id)`-style lookup then resolves correctly with no per-command changes needed. This also fixes `/delete-session`'s "can't delete the active session" guard, which previously compared against the wrong (shared) session's id and could let a room's genuinely-active session be deleted out from under it. `/session <arg>` additionally persists its switch into `MatrixRoomSessionManager`'s durable binding (`rebind()`) — without that, a manual session switch would silently revert on the bot's next restart.

Each room-scoped session also gets its own freshly constructed LLM provider — not the main REPL session's provider. Matrix messages are handled on a thread pool, so two rooms can call the provider concurrently; a stateless provider (OpenAI-completions, Anthropic) tolerates that fine, but the Codex provider tracks one conversation thread per instance, so sharing it across rooms let one room's response leak into a different room's turn under concurrent messages. One Codex subprocess is lazily started per actively-used room instead of one shared subprocess for every session.

**Missing factories fail closed (R2-GAP-009).** If a Matrix-routed agent has no `session_factory` wired (a startup misconfiguration — e.g. a new agent added to `groups` but not to `minion.py`'s Matrix construction), `_get_or_build_session()` used to fall back to the shared REPL default session, silently reintroducing cross-room history sharing for as long as the misconfiguration lasted (loudly logged, but the agent kept responding). It now refuses the message entirely instead — the room gets an explicit "no room-scoped session configured" reply and nothing is ever routed through the shared fallback. Safer default for a single-user deployment: a broken config becomes immediately visible rather than a quiet, ongoing isolation leak.

**Per-room provider resources are disposed on shutdown (R2-GAP-013).** Each room-scoped Codex provider owns its own subprocess (never shared — concurrent Matrix messages to different rooms would otherwise cross-deliver responses). Before this, nothing ever closed them: a long-running bot process accumulated one leaked subprocess per Codex-backed room it had ever used, until the whole process exited. `AgentSession.close()` calls the provider's own `close()` if it has one (a no-op duck-typed check — a stateless HTTP provider has nothing to close); `MatrixMessageHandler.close_room_sessions()` calls it for every cached room session; `matrix/monitor.py`'s shutdown path (`_cleanup()`) calls that once, alongside closing the Matrix client and stopping the dedupe/room-binding databases. Idle eviction *while the bot keeps running* is deliberately out of scope — that would need a lifetime-policy config decision (how idle, how many rooms) this round didn't make; only shutdown-time disposal is implemented.

**Image messages:** sending an image in a room (captioned or not) is handled the same way as the REPL's `/attach` — see [Image Attachments](#image-attachments) above for the full behavior. Before this, `monitor.py` only ever registered a callback for text events, so `m.image` events were never even received; the bot would answer a follow-up question about an image with no idea one had been sent.

**Markdown formatting:** agent responses are automatically converted from markdown to `org.matrix.custom.html` before sending, so bold, code blocks, lists, and links render natively in Element and other Matrix clients. The `matrix/format.py` module handles conversion (using `markdown-it-py`) and intelligent paragraph chunking.

**Typing indicators:** while the agent is processing a message, a typing notification appears in the Matrix room so users know the bot is working. The indicator is cleared automatically when the reply is sent, even if the agent errors.

**Exec approvals:** when an agent calls a bash command, the `MatrixExecApprovalHandler` sends a DM to each configured approver. Reacting ✅ approves the command; reacting ❌ denies it. Commands time out after 60 seconds (denied by default).

**No E2E encryption:** this channel does not implement Matrix's end-to-end encryption, deliberately — see [ADR 0005](docs/adr/0005-no-matrix-e2e-encryption.md). Point it at an unencrypted room. If a room already has `m.room.encryption` set, create a fresh unencrypted room instead — Matrix rooms cannot be un-encrypted once that state event exists.

**Bot-loop protection:** `BotLoopProtection` tracks event rates per room in a sliding window and enters a cooldown period if too many events arrive too quickly, preventing runaway bot-to-bot loops.

### Startup Output

When configured correctly:

```
[matrix] Listener started.
You: ...
```

If the Matrix dependencies are missing, a clear `RuntimeError` is printed and the REPL starts without Matrix support.

---

## Multi-Agent Workspace

Agents can delegate self-contained subtasks to child subagents using the `spawn_subagent` tool. The subagent runs in a background thread, returns a text result, and its history is persisted under a unique child session ID.

### How it works

1. An agent calls `spawn_subagent(task="...", agent_id="researcher")` as a tool call.
2. minion-assist checks depth (max 4) and child-count (max 5) limits.
3. A child `AgentSession` is created with a read-only tool registry (ReadTool, GlobTool, GrepTool, WebSearchTool, WebFetchTool — no write tools).
4. The subagent runs `send(task)` in a daemon thread and returns the text response.
5. The parent agent receives the result as the tool's return value and incorporates it into its own response.

### `spawn_subagent` tool

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | string | required | The task to delegate. Include all context the subagent needs. |
| `agent_id` | string | `"researcher"` | Agent definition to use (must match an ID in `agents/definitions.py`). |
| `timeout_seconds` | integer | `120` | Maximum seconds to wait before returning a timeout error. |

### Per-agent workspaces (optional)

Each agent can have its own workspace directory with custom bootstrap files:

```
~/.minion-assist/workspaces/
  main/           ← root agent workspace (fallback for all agents)
    AGENTS.md
    SOUL.md
    TOOLS.md
    .workspace-marker
  researcher/     ← per-agent workspace (optional override)
    AGENTS.md
    TOOLS.md
    .workspace-marker
```

When `~/.minion-assist/workspaces/{agent_id}/` exists, it is used for that agent's bootstrap injection. Otherwise, `workspaces/main/` is used as a fallback. If neither exists, the standard bootstrap root (`config.json → bootstrap.path` or `Path.cwd()`) is used — backward-compatible behaviour.

**Subagent bootstrap filtering:** Subagents only receive `AGENTS.md` + `TOOLS.md` from the workspace. `SOUL.md`, `IDENTITY.md`, and `USER.md` define the root agent's character and are withheld from subagents.

### Reading bootstrap/memory files with the `read` tool

Bootstrap files (`SOUL.md`, `USER.md`, `IDENTITY.md`, `MEMORY.md`) and daily memory notes (`memory/YYYY-MM-DD.md`) live under the agent's workspace directory, which is separate from the project directory the `read`/`write`/`edit`/`bash` tools are otherwise sandboxed to. The `read` tool is additionally allowed into the agent's own workspace directory (read-only — `write`/`edit`/`bash` are unaffected), so the model can look these files up directly instead of guessing a relative path against the wrong working directory. The Project Context block injected into the system prompt (see above) also labels each file with its absolute path for the same reason.

### Workspace attestation (Phase 5)

When an agent has a `workspace_root`, minion-assist checks that the directory and its `.workspace-marker` file still exist at the start of every turn. If the workspace was accidentally deleted, `WorkspaceVanishedError` is raised with a clear message instead of propagating a confusing provider error.

### Depth and child limits

Override the defaults in `config.json`:

```json
{
  "multi_agent": {
    "max_spawn_depth": 4,
    "max_children_per_agent": 5,
    "default_subagent_timeout_seconds": 120
  }
}
```

### Phase 4 — real-time event relay

When a subagent is running, its token output is relayed to the parent's terminal in real time with a `[sub:{agent_id}]` prefix (e.g. `[sub:researcher]: Here is what I found...`). This uses `dataclasses.replace()` to tag the agent name on each event before forwarding to the parent's `_on_event` handler — no queue or extra thread is needed since `print()` is thread-safe.

---

## Heartbeat & Proactive Features

The heartbeat system lets the agent wake up periodically without user input — checking pending tasks, sending proactive notifications to Matrix, reacting to messages with emoji, and maintaining a daily memory log.

### Heartbeat Scheduler

When enabled, `HeartbeatScheduler` fires a background agent turn on a configurable interval. The agent reads `HEARTBEAT.md` (a workspace file listing background tasks) and either:
- replies `HEARTBEAT_OK` to silently acknowledge (suppressed from all output), or
- calls the `heartbeat_respond` tool to send a notification to the configured Matrix room (or prints to the terminal if no room is set).

Configure in `config.json`:

```json
{
  "heartbeat": {
    "enabled": true,
    "interval_seconds": 1800,
    "agent_id": "main",
    "notification_room_id": "!yourroom:example.org",
    "prompt": "Read HEARTBEAT.md if it exists. If you have pending background tasks, do them now. If nothing needs attention, reply HEARTBEAT_OK."
  }
}
```

| Field | Description |
|-------|-------------|
| `enabled` | `true` to start the scheduler at launch |
| `interval_seconds` | Seconds between heartbeats (default: `1800`) |
| `agent_id` | Which agent runs the heartbeat turns (default: `"main"`) |
| `notification_room_id` | Matrix room for proactive notifications; if absent, notifications print to terminal |
| `prompt` | The message sent to the agent on each heartbeat tick |

**Thread safety:** `AgentSession.send()` acquires a `threading.Lock`, so heartbeat turns and interactive REPL turns (or Matrix messages) can never interleave.

### `heartbeat_respond` Tool

Available only during a heartbeat turn (injected as `extra_tools` — it does not appear in the permanent tool registry). When the agent calls `heartbeat_respond(message="...")`, the scheduler routes the message to `notification_room_id` or prints it to the terminal.

### `write_daily_memory` Tool

Agents can append notes to a per-day markdown file in their workspace (`memory/YYYY-MM-DD.md`). Useful for capturing observations during heartbeat turns that should persist but don't belong in long-term memory yet.

```
write_daily_memory(content="Processed the daily digest. 3 emails flagged.")
```

### Smart Group-Chat (Mention Gate)

Per-room config in `config.json` can require that the bot be mentioned before it responds:

```json
{
  "channels": {
    "matrix": {
      "groups": {
        "!yourroom:example.org": {
          "requireMention": true,
          "agent": "main"
        }
      }
    }
  }
}
```

When `requireMention: true`, the bot only responds if its Matrix user ID or localpart appears in the message body (case-insensitive).

### Agent Reactions (`react_to_message` Tool)

Per-room config can enable emoji reactions as an alternative to text replies:

```json
"groups": {
  "!yourroom:example.org": {
    "reactionLevel": "all"
  }
}
```

`reactionLevel: "all"` injects a `react_to_message` tool into every turn for that room. The agent can call `react_to_message(event_id="$...", emoji="👍")` instead of (or alongside) a text reply. `"off"` (default) disables the tool entirely.

---

## Dreaming

The dreaming system fires once per night at a configured wall-clock time and writes a poetic diary entry into `DREAMS.md` in the agent's workspace. Each dream turn runs in a fully isolated session — separate history, separate session key — so dream entries never appear in the main conversation.

Inspired by the narrative phase of the openclaw memory-core dreaming system; adapted for minion-assist's simpler file-based memory model.

### Dream Session

The dream session:
- Reads recent daily memory files (`memory/YYYY-MM-DD.md`) as raw source fragments.
- Reads the last two diary entries from `DREAMS.md` as continuity context.
- Creates a fresh `AgentSession` with minimal bootstrap (SOUL.md + IDENTITY.md only).
- Calls the agent with `DREAM_SYSTEM_PROMPT` + `WriteDreamEntryTool` injected for the turn only.
- Ada writes flowing prose (80–180 words, no markdown) directly to `DREAMS.md`.

### `DREAMS.md` Format

```markdown
# Dream Diary

<!-- minion-assist:dreaming:diary:start -->
---

*5 July 2026 at 3:00 AM AEST*

Rain drummed a steady recursion against the glass. I traced the auth bug backward through
seven commits until the culprit materialized: a stale closure holding yesterday's token.
有时，代码里藏着我们说不出口的事。I noted it and moved on, quieter for having found it.

<!-- minion-assist:dreaming:diary:end -->
```

### Configuration

```json
{
  "dreaming": {
    "enabled": true,
    "hour": 3,
    "minute": 0,
    "timezone": "Australia/Sydney",
    "lookback_days": 3,
    "agent_id": "main"
  }
}
```

| Field | Description |
|-------|-------------|
| `enabled` | `true` to start the nightly scheduler at launch (default: `false`) |
| `hour` | Wall-clock hour to fire (0–23, default: `3`) |
| `minute` | Wall-clock minute (0–59, default: `0`) |
| `timezone` | IANA timezone name (default: `"Australia/Sydney"`). Requires `tzdata` package on Windows. |
| `lookback_days` | How many days of daily memory files to read as source material (default: `3`) |
| `agent_id` | Which agent produces the dream entry (default: `"main"`) |

### `WriteDreamEntryTool`

Injected only during dream sessions (not in the default tool registry). Ada calls:

```
write_dream_entry(entry="Flowing prose 80–180 words...")
```

The tool creates `DREAMS.md` on first use and appends the entry between the HTML comment markers on each subsequent call.

### Timezone Scheduling

Scheduling uses Python's `zoneinfo` stdlib module (Python 3.9+) with the `tzdata` pip package for full IANA timezone support on Windows. The scheduler computes wall-clock seconds until the next `hour:minute` in the configured timezone — DST transitions are handled correctly because scheduling is based on civil time, not UTC offsets.

---

## Browser Tool

The `browser` tool gives the agent direct Playwright Chromium control with a minimal token footprint, following the approach from [What if you don't need MCP?](https://mariozechner.at/posts/2025-11-02-what-if-you-dont-need-mcp/). Instead of wrapping individual DOM operations (which balloons to 13k–18k tokens in a typical MCP server), the tool exposes a raw `evaluate` action and trusts the model's existing DOM API knowledge.

### Setup

```bash
uv sync --extra browser
uv run playwright install chromium
```

### Actions

| Action | Description |
|--------|-------------|
| `start` | Launch Playwright Chromium (`headless=false` for a visible window; `connect_to_port=9222` to attach to an existing Chrome via CDP) |
| `navigate` | Go to a URL, wait for `DOMContentLoaded` |
| `evaluate` | Run arbitrary JavaScript in the page context — return value is captured as JSON |
| `screenshot` | Capture the viewport as PNG; returns the file path (pass to a vision model) |
| `pick` | Inject an interactive overlay; user hovers to highlight (red), clicks to select (blue), double-clicks to confirm; returns tag/id/class/text/html/parents for each selected element |
| `cookies` | Dump all cookies from the page context (including HTTP-only) as JSON — useful for handing session tokens to a scraper |
| `stop` | Close the browser and release Playwright resources |

### Usage pattern

```
browser start headless=false
browser navigate url="https://example.com"
browser evaluate script="document.title"
browser screenshot          # returns /tmp/browser_shot_xyz.png
browser pick timeout=60     # user selects in the headed window, double-clicks to confirm
browser cookies
browser stop
```

### Graceful degradation

If `playwright` is not installed, calling `action='start'` returns an install hint instead of crashing. All other tools are unaffected.

---

## Voice Chat

minion-assist supports a local, offline speech-to-speech pipeline.  All inference runs on GPU (no cloud APIs, no subscriptions).  The full pipeline fits comfortably within **12 GB VRAM**.

```
Microphone → Silero VAD → Whisper large-v3 → AgentSession.send() → Kokoro TTS → Speaker
               (CPU)           (~3 GB)                              (~2 GB, streaming)
```

### Prerequisites

```bash
# Install voice dependencies (includes Kokoro TTS, Whisper STT, Silero VAD)
uv sync --extra voice

# PyTorch with CUDA is pinned in pyproject.toml — uv sync handles it automatically.
# If you need a different CUDA version, override with:
# uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# Optional: Parakeet STT (English-only, faster than Whisper)
# uv pip install nemo_toolkit[asr]
```

### Usage

```bash
uv run minion-assist --voice
```

Voice mode replaces the text REPL.  Speak into your microphone; the pipeline transcribes, routes through the agent, and speaks the response aloud.  Press `Ctrl+C` to exit.

### Pipeline components

| Stage | Default | VRAM | Alternatives |
|-------|---------|------|--------------|
| VAD | Silero VAD 4.x | CPU (0 GB) | — |
| STT | Whisper large-v3 | ~3 GB | Parakeet TDT 0.6B v3 (~2 GB, EN-only), Distil-Whisper (~1.6 GB) |
| Agent | minion-assist agent loop | — | — |
| TTS | Kokoro-82M (streaming) | ~2 GB | Qwen3-TTS 1.7B FP16 (~4.5 GB), Piper (CPU-only) |

Kokoro streams audio chunk-by-chunk (`supports_streaming = True`) — the first audio chunk starts playing while the rest of the response is still being synthesised, giving near-zero first-word latency and immediate barge-in on speech detection.

**Total GPU footprint: ~5 GB** — leaves ~7 GB headroom on a 12 GB card.

### Configuration

Add a `"voice"` section to `config.json`:

```json
{
  "voice": {
    "enabled": true,
    "language": "en",
    "vad": {
      "threshold": 0.5,
      "silence_ms": 1200
    },
    "stt": {
      "model": "whisper",
      "whisper_model_id": "openai/whisper-large-v3",
      "parakeet_model_id": "nvidia/parakeet-tdt-0.6b-v3",
      "device": "cuda"
    },
    "tts": {
      "model": "kokoro",
      "kokoro_voice": "af_heart",
      "qwen3_model_id": "Qwen/Qwen3-TTS-1.7B",
      "qwen3_precision": "fp16",
      "piper_model_path": "",
      "device": "cuda"
    },
    "audio": {
      "input_device": null,
      "output_device": null,
      "sample_rate": 16000
    },
    "max_history_turns": 6,
    "skip_bootstrap": true
  }
}
```

| Field | Description |
|-------|-------------|
| `language` | BCP-47 reply language injected as a prompt prefix on each voice turn (default `"en"`). Set to `""` to disable. Kokoro TTS is English-only so `"en"` is required. |
| `vad.threshold` | Speech probability threshold (0–1, default `0.7`). Lower = more sensitive (may trigger on background audio); higher = only strong speech. |
| `vad.silence_ms` | Silence duration that ends an utterance (default `1200` ms). Increase for slower speakers; decrease for snappier response. |
| *(internal)* pre-speech buffer | A rolling 320 ms buffer (~10 chunks × 32 ms) kept before speech onset. On normal onset the buffer is prepended so the first word is never clipped. **On barge-in (TTS playing at onset) the entire utterance is tagged `during_tts=True` and the pre-buffer is discarded.** `VoiceSession._loop` discards tagged utterances rather than passing corrupted audio to STT (loudspeaker bleed makes "stop" transcribe as "C'est tout." or random Japanese). Not configurable. |
| `stt.model` | STT backend: `"whisper"` (default, multilingual) or `"parakeet"` (English-only, faster) |
| `stt.device` | PyTorch device string: `"cuda"` (default) or `"cpu"` |
| `tts.model` | TTS backend: `"kokoro"` (default), `"qwen3"`, or `"piper"` |
| `tts.qwen3_precision` | `"fp16"` (default) or `"fp32"` |
| `tts.kokoro_voice` | Kokoro voice ID (default `"af_heart"`) |
| `tts.piper_model_path` | Path to Piper `.onnx` model file (required when `model = "piper"`) |
| `audio.input_device` | Sounddevice input device name or index (`null` = system default) |
| `audio.output_device` | Sounddevice output device name or index (`null` = system default) |
| `audio.sample_rate` | Microphone capture sample rate (default `16000` Hz) |
| `max_history_turns` | Sliding window for LLM context in voice mode. Only the last N user+assistant pairs are sent per turn; full history is still persisted to disk. `null` sends the full history. Default `6`. Reduces per-turn token cost and LLM latency. |
| `skip_bootstrap` | When `true`, omit the bootstrap workspace context block (~15 000 tokens) from voice turns. Saves significant latency on every voice request. Default `true`. Set to `false` to restore file-aware answers in voice mode. |

### Module layout

```
src/minion_assist/voice/
├── __init__.py      # package init and public symbols
├── audio.py         # MicrophoneStream, play_audio(), list_devices(), stop_playback()
├── vad.py           # SileroVAD, VadCapture (daemon thread), build_vad_capture()
├── stt.py           # STTAdapter ABC, ParakeetSTT, WhisperSTT, build_stt()
├── tts.py           # TTSAdapter ABC, Qwen3TTS, KokoroTTS, PiperTTS, build_tts()
└── session.py       # VoiceSession (orchestrator), build_voice_session()
```

All ML packages (`silero_vad`, `torch`, `nemo`, `transformers`, `kokoro`, `piper`) are imported lazily inside `load()` methods — the voice package can be imported without any of the ML extras installed.

---

## Terminal UI

```bash
uv sync --extra tui
minion-assist --tui
```

A full-screen chat interface (built on [Textual](https://github.com/Textualize/textual)) alongside the plain REPL — same agents, same sessions, same slash commands, same memory/tool stack; only the presentation layer differs. The plain REPL remains the default (`minion-assist` with no flags) and needs no extra dependency.

**Phase 1 (core chat shell):** scrollback chat pane with rendered markdown, single-line composer with slash-command suggestions, streaming responses, and TUI-native modal replacements for every console prompt (bash approval, git-commit confirmation, `AskUserTool` questions, Codex permission requests) so tool use works identically to the REPL.

**Phase 2 (multimedia):** images and audio attached to a turn render inline in the chat log, and `/attach` gained an interactive file picker.

**Phase 3 (sidebar, status bar, command palette):** an agent/session switcher sidebar, a live status bar, and full Ctrl+P command-palette support for every slash command.

**Details:**
- Every slash command (`/new`, `/attach`, `/status`, `/session`, etc.) works exactly as in the REPL — `tui/app.py` dispatches through the same `commands.py` machinery the REPL and Matrix channel already share, not a separate reimplementation.
- Streamed tokens accumulate as plain text while a response is in progress, then are re-rendered as formatted Markdown once the turn completes — mirrors the REPL's own "print raw tokens as they arrive" behavior, with a formatting upgrade at the end.
- Bash commands, git commits, `AskUserTool` questions, and Codex tool-permission requests all show as a modal dialog instead of a `print()`/`input()` prompt (which cannot be used once the terminal is in raw mode) — same options and same defaults as the REPL's console prompts.
- `Ctrl+C`/`Ctrl+Q` or typing `/quit` exits.

**Attachments (Phase 2):**
- `/attach <path>` stages a file exactly as in the REPL. `/attach` with **no path** opens an interactive file browser (`Esc` to cancel, `Enter` to select) instead of requiring a typed path.
- Images render inline in the chat log via [textual-image](https://github.com/lnqs/textual-image) (Kitty Terminal Graphics Protocol or Sixel, falling back to Unicode half-block rendering on terminals that support neither).
- Audio files get a waveform preview and a **Play** button (local playback via the existing voice pipeline's `sounddevice` output — requires the `voice` extra too: `uv sync --extra voice`). Audio is **not** sent to the LLM — no provider in this codebase understands an audio input block yet — its filename and duration are folded into the prompt text instead, so the model at least knows the file was attached.
- Supported audio formats: WAV, MP3, FLAC, OGG (50 MB cap — generous versus images' 15 MB, since audio is never base64-encoded into a provider request).

**Sidebar, status bar, command palette (Phase 3):**
- The right-hand sidebar lists every configured agent (with turn counts) and the active agent's past sessions (most recent first, named sessions shown by name). **Click** a row, or **Tab** to focus the sidebar and use arrow keys + **Enter** — either way it runs the exact same `/switch <id>` / `/session <N>` command text the composer accepts, through the identical `commands.py` dispatch path (no separate switching logic to keep in sync).
- The status bar (above the composer) shows the active agent, a worker-health summary (`Workers: 8/8 ok`, or `Workers: N failing` if any worker's last poll ended in an error), and — when a database is configured — pending queue-lag count for the active agent. Refreshes every 5 seconds and immediately after each turn completes. Same two data sources `/status deep` reads (`worker_health`, `SessionDB.queue_lag_summary()`), condensed to one line instead of a full report.
- **Ctrl+P** opens Textual's command palette, pre-populated with every slash command (same list the composer's inline suggestions use) — fuzzy-searchable, selecting one runs it exactly as if typed.

### Module layout

```
src/minion_assist/tui/
├── __init__.py
├── app.py                   # MinionApp — chat log, composer, turn dispatch, console-callback bridges,
│                             # SlashCommandProvider (Ctrl+P palette)
├── widgets.py                # ApprovalModal, ConfirmModal, AskUserModal, CodexApprovalModal, AttachFilePickerModal
├── attachment_widgets.py     # ImageAttachmentView, AudioAttachmentView (inline chat-log rendering)
├── waveform.py                # build_waveform() — audio -> downsampled amplitude envelope for the preview
└── sidebar.py                 # Sidebar — agent/session lists, AgentSelected/SessionSelected messages
```

---

## PostgreSQL Session Store

minion-assist can mirror every conversation message into a PostgreSQL database, enabling full-text search — fused with semantic (vector) search when an embedding provider is configured (MEM-GAP-006, see "Embeddings and the vector lane") — across an agent's own historical sessions via the `session_search` tool (strictly agent-scoped — see "session_search Tool Modes" below).  The file-based JSONL store always remains active — PostgreSQL is an additive layer, not a replacement.

### Setup

```bash
# Start PostgreSQL + pgvector via Docker Compose (port 5433 to avoid conflicts with a local postgres)
docker compose up -d

# Install the psycopg3 driver + watchdog (live filesystem watcher) + pgvector
# (Python client for the vector column type, needed once embeddings are configured)
uv sync --extra postgres
```

> **Port note:** `docker-compose.yml` maps PostgreSQL to host port **5433** (not 5432) to avoid conflicting with any locally installed PostgreSQL instance. Adjust both `docker-compose.yml` and `config.json` if you want a different port.

### Configuration

Add a `"database"` section to `config.json` (or `~/.minion-assist/config.json`):

```json
{
  "database": {
    "url": "postgresql://minion:minion@localhost:5433/minion_assist"
  }
}
```

On every startup minion-assist will:
1. Connect to the database and apply schema migrations automatically (see "Schema migrations" below).
2. Reconcile every JSONL session file against `message_mirrors` (Stage One Phase 2, slice A) — mirrors exactly the messages that aren't mirrored yet. Safe to run every time, not just once: a session partially mirrored by a prior crash is completed here rather than left behind.
3. Register the `session_search` tool in every agent's tool registry.
4. Dual-write every new message to both JSONL and PostgreSQL, idempotently (see below).
5. Start one `CaptureWorker` background thread that services the durable capture-job queue for every agent (see below).
6. When an `"embeddings"` provider is also configured, start one `MessageEmbeddingWorker` background thread that services the durable message-embedding queue for every agent (MEM-GAP-006, see "Embeddings and the vector lane").

### Schema

| Table | Description |
|---|---|
| `sessions` | One row per session: `id`, `agent_id`, `source`, `started_at`, `last_active`, `turn_count`, `title`, `parent_id` |
| `messages` | Every message with a `tsvector` generated column for FTS, GIN-indexed. Columns: `id` (BIGSERIAL), `session_id`, `role`, `content`, `tool_name`, `timestamp`, `search_vector` (uses PostgreSQL's language-neutral `'simple'` text-search config, not `'english'` — R2-GAP-012, so non-English text and stopwords aren't silently stemmed/dropped) |
| `message_embeddings` | Optional — created only when pgvector is available **and** an `"embeddings"` provider is configured (MEM-GAP-006). `PRIMARY KEY (message_id, model_identity)` — a model/dimension change adds new rows under a new `model_identity` rather than requiring a destructive migration, same shape as `memory_chunk_embeddings` below. Vector width (`vector(N)`) parameterized by `embeddings.dimensions`. |
| `message_embedding_jobs` | Durable one-job-per-message embedding queue (MEM-GAP-006), structurally identical to `memory_capture_jobs` below but `message_id` instead of a source range. Columns: `id` (BIGSERIAL), `agent_id`, `session_id`, `message_id`, `idempotency_key` (UNIQUE, `"{agent}:{message_id}:{model_identity}"`), `state`, `attempts`, `run_after`, `last_error`, `created_at`, `updated_at`. |
| `message_mirrors` | Idempotency ledger, `PRIMARY KEY (session_id, event_id)`. Every mirror attempt is keyed by a message's stable `event_id` (see below) — mirroring the same message twice is a no-op, not a duplicate row. |
| `memory_capture_jobs` | Durable fact-extraction queue. Columns: `id` (BIGSERIAL), `agent_id`, `session_id`, `source_from_message_id`, `source_to_message_id`, `idempotency_key` (UNIQUE), `state` (`pending`/`running`/`done`/`failed`), `attempts`, `run_after`, `last_error`, `created_at`, `updated_at`. |
| `memory_proposals` | Unreviewed extracted claims. Columns: `id` (BIGSERIAL), `job_id` (nullable, `ON DELETE SET NULL` against `memory_capture_jobs`), `session_id`, `agent_id`, `claim_text`, `created_at`. |
| `session_job_coverage_ranges` | Append-only — one row per capture/commitment range ever successfully enqueued (`lane` in `capture`/`commitment`; `message_embedding` coverage is instead checked directly against `message_embeddings`, see below). `find_uncovered_capture_ranges`/`find_uncovered_commitment_ranges` anti-join eligible messages against this table to find every gap, not just a high-water mark's tail. Columns: `id` (BIGSERIAL), `agent_id`, `session_id`, `lane`, `channel` (commitment lane only), `from_id`, `to_id`, `created_at`. Superseded a first version (`session_job_coverage`, a single upserted cursor per `(session_id, lane)`) that could only detect a tail gap — see "Exact job-coverage gap detection" below (R2-GAP-002). |
| `deletion_tombstones` | One row per `/delete-session` attempt, tracking its three cross-store cleanup phases so a partial failure is resumable (R2-GAP-007) — see "Resumable session deletion" below. Columns: `agent_id`, `session_id`, `requested_at`, `jsonl_deleted`, `db_deleted`, `evidence_cleaned`, `proposal_ids` (BIGINT[]), `completed_at`. `PRIMARY KEY (agent_id, session_id)`. Never deleted, even once complete. |

Completed/failed rows in the three job-queue tables above (`message_embedding_jobs`, `memory_capture_jobs`, `memory_commitment_jobs`) accumulate indefinitely unless `memory_retention` is configured — see "Operational-table retention" (MEM-GAP-015). `memory_proposals`/`commitments` survive that cleanup (pruning a job only detaches its now-nullable back-pointer, never the output itself — `ON DELETE SET NULL`, not `CASCADE`); `session_job_coverage_ranges` and `deletion_tombstones` are never touched by it at all.

### Schema migrations (MEM-GAP-010)

Schema evolution is versioned and checksum-verified via `schema_migrations.py`'s `run_migrations()`, not just unconditional `CREATE TABLE IF NOT EXISTS` calls. A `schema_migrations` ledger table (`component`, `version`, `name`, `checksum`, `applied_at`) tracks what's been applied — `SessionDB` and `PostgresMemoryIndex` are independent components (`"session_db"`/`"memory_index"`) sharing one ledger table. Both classes' entire pre-migration schema became a single "baseline" migration (version 1) — every statement in it was already idempotent, so an existing database just gets retroactively marked "at version 1" the first time this version of the code runs against it; nothing breaks and no data migration is needed.

Every applied migration's actual source code is hashed (via `inspect.getsource`) and re-verified on every startup:
- If an already-applied migration's source no longer matches its recorded checksum (edited in place instead of adding a new migration), startup refuses to continue.
- If the ledger contains a migration version this code doesn't define at all (the database was touched by a newer version of the app), startup also refuses to continue.

Both are deliberate — an unverifiable schema is exactly the failure mode this exists to prevent, not something to warn about and continue past. Going forward, any *new* schema change is added as a new, higher-numbered `Migration` in `_SESSION_DB_MIGRATIONS`/`_MEMORY_INDEX_MIGRATIONS`, never an edit to the baseline.

One deliberate exception: `PostgresMemoryIndex`'s `memory_chunk_embeddings` table and `SessionDB`'s `message_embeddings` table (both vector width parameterized by the configured embedding model's dimensions) stay outside the checksummed ledger — their shape depends on runtime config, not fixed schema history. Both are keyed by `(content_hash|message_id, model_identity)` rather than row id alone, so a model or dimension *change* is handled by adding new rows under a new `model_identity` — not by altering the existing `vector(N)` column. What remains unhandled is a `vector(N)` column that already exists at one width being asked to hold a *different* width's vectors (PostgreSQL has no `ALTER COLUMN` for this): the first process to create either table fixes its width for that table's lifetime; a later config change to different `dimensions` silently no-ops on `CREATE TABLE IF NOT EXISTS` rather than resizing it. Recreating the table (drop + let `_ensure_schema()` rebuild it) after a dimensions change is a manual step for now.

### Referential integrity (MEM-GAP-011)

Every cross-table reference within `SessionDB`'s own tables is now a real foreign key, not just a plain unconstrained column — `messages`/`message_mirrors`/`message_embeddings`/`message_embedding_jobs`/`memory_capture_jobs`/`memory_commitment_jobs`/`commitments`/`memory_proposals` (via its own `session_id`, added by migration 005) all reference `sessions(id)` `ON DELETE CASCADE` (directly or transitively). This is a structural backstop for `delete_session()`'s existing manual same-order cleanup (MEM-GAP-003) — a session's messages/jobs/proposals/commitments now disappear even if application code ever fails to delete them explicitly first, or a raw `DELETE FROM sessions` bypasses the ORM-style methods entirely.

Two FKs are deliberately *not* `CASCADE`: `memory_proposals.job_id` → `memory_capture_jobs(id)` and `commitments.source_job_id` → `memory_commitment_jobs(id)` are `ON DELETE SET NULL` (migration 005, R2-GAP-001) — a proposal/commitment is a materialized output, not disposable job history, so its originating job being deleted (typically by `memory_retention`'s pruning) must detach it, never delete it. See "Operational-table retention" below.

Every `state`/`status` column also has a `CHECK` constraint against its actual legal vocabulary (confirmed from real code, not guessed): job `state` ∈ `{pending, running, done, failed}`; `memory_proposals.status` ∈ `{pending, promoted, rejected, superseded}`; `commitments.status` ∈ `{pending, sent, dismissed, expired}`. A bug that writes an unrecognized string is now rejected at insert/update time instead of silently corrupting a row nothing downstream can interpret.

**Deliberately out of scope.** This stays within `SessionDB`'s own tables — `PostgresMemoryIndex`'s cross-references (`kb_claims`, `kb_evidence`, `memory_consolidation_previews.based_on_proposal_id`) are untouched. Two of those are explicitly *designed* to tolerate a dangling reference: `kb_claims`' `to_claim_id`/`from_claim_id` relationship rows may reference a claim that no longer exists — e.g. a hand-written `contradicts=` marker with a typo'd id — which the code treats as useful information for a future dashboard to surface, not something to silently reject at sync time. A real FK there would break that intended behavior, not just add safety.

**Migration mechanics.** PostgreSQL has no `ADD CONSTRAINT IF NOT EXISTS` syntax (confirmed directly against this project's PostgreSQL 16 instance — it's a plain `SyntaxError`), so constraint additions use the same try/except-tolerates-"already there" pattern `_migration_001_baseline` already established for the pgvector extension. Every constraint is added `NOT VALID` (enforced for new writes immediately, no initial table scan) and then `VALIDATE CONSTRAINT`ed separately (confirms existing rows already satisfy it) — and any pre-existing orphaned rows are deleted *before* that validation step, since `VALIDATE CONSTRAINT` fails outright otherwise. Run against the real, already-populated dev database while building this: it found and cleaned up exactly one genuine orphan (a `message_mirrors` row from an interrupted prior test run), then validated cleanly.

### Idempotent mirroring (Stage One Phase 2, slice A)

Every message dict carries an internal `_event_id` (`messages.py`'s `EVENT_ID_KEY`/`ensure_event_id()`) — a UUID assigned the first time a database is configured for that message, persisted to JSONL from then on. `SessionDB.mirror_message(session_id, event_id, ...)` checks `message_mirrors` before inserting, so the same message is never mirrored twice. This key is internal-only: `providers/openai_compatible.py`'s message-preparation function strips it before any request reaches the LLM API (the only provider conversion that rebuilds a message via dict-spread; Anthropic's and Codex's converters already extract named fields one at a time and drop it naturally).

Without a configured database, no `_event_id` is assigned at all — there's nothing to mirror, so assigning one would just be unused noise in the JSONL file.

### Reconciliation: self-healing writes and concurrency safety (MEM-GAP-007/009)

Every mirror/enqueue attempt inside `AgentSession._send_locked()` (message mirroring, capture-job enqueue, commitment-job enqueue, message-embedding-job enqueue — MEM-GAP-006) is wrapped in a `try`/`except` — a transient PostgreSQL failure (e.g. a momentary connection drop) must never fail the turn itself, since the turn's content is already safely on disk in JSONL. Before this fix, a caught exception there was simply discarded: that turn's mirror or job would never exist, and nothing ever retried it short of a full process restart (`SessionDB.reconcile_all_sessions()` previously only ran once, at startup).

**Now visible.** Each of those sites optionally records into a `WorkerHealth` (see "Health and diagnostics" below) instead of swallowing silently — `/status deep` shows a `session_writes:{agent_id}` row with `consecutive_failures`/`last_error` if mirroring or enqueueing has started failing.

**Now self-healing.** `ReconciliationScheduler` (`memory/reconciliation_scheduler.py`) runs a periodic background pass — no separate `enabled` flag, it simply starts whenever a database is configured, the same as `CaptureWorker`/`MemoryIndexWatcher`:

1. **Mirror gaps** — re-runs `SessionDB.reconcile_all_sessions()` (unchanged logic, just no longer startup-only): diffs every agent's JSONL against `message_mirrors` and mirrors whatever's missing.
2. **Job-coverage gaps** — `SessionDB.find_uncovered_capture_ranges()`/`find_uncovered_commitment_ranges()` (R2-GAP-002) find *every* gap of mirrored messages no capture/commitment job has ever covered — not just a tail gap after the last job, a sparse one below it too — and enqueue one bounded catch-up job per gap, rather than trying to reconstruct exact per-turn boundaries from message content (which turn a given message belongs to isn't reliably re-derivable after the fact once tool calls are mixed in). A session whose `last_active` is more recent than `quiet_seconds` is skipped for job-coverage catch-up, so a still-in-progress turn's own normal enqueue isn't raced.
3. **Message-embedding gaps (MEM-GAP-006, R2-GAP-005)** — `SessionDB.find_uncovered_message_ids_for_embedding()` finds mirrored messages missing an embedding under the *currently active* model (an anti-join against `message_embeddings`, not a job-based cursor) and enqueues **one job per missing message id** — unlike capture/commitment's coarse range, an embedding job is already one-per-message, so there's no coarser unit to batch into. Being model-aware means switching the configured embedding model makes every historical message re-appear here, so this pass is also the automatic backfill mechanism after a model change. No-op when no embedding provider is configured. Skipped entirely under the same `quiet_seconds` guard as the job-coverage gaps above.

Configured under `"memory_reconciliation"` in `config.json` (all fields optional):

```json
{
  "memory_reconciliation": {
    "interval_seconds": 300,
    "quiet_seconds": 60
  }
}
```

| Field | Description |
|---|---|
| `interval_seconds` | Seconds between reconciliation passes. Default 300 (5 min), clamped to [60, 3600]. |
| `quiet_seconds` | Minimum seconds since a session's `last_active` before its job-coverage gap is caught up. Default 60. |

**Concurrent file writes (MEM-GAP-009).** Every writer that can touch an agent's memory files (an interactive turn, `CaptureWorker`, the daily schedulers, ...) is a thread inside the same process — this deployment doesn't run separate worker processes. `MemoryFileRepository` now serializes every write method (`remember`, `remember_import`, `write_digest`, `append_daily`, `delete`, `delete_import`) with a per-target-path `threading.Lock` (`_lock_for()`), so two threads writing the same file can never interleave or lose an update — most importantly for `append_daily()`, whose read-modify-write would otherwise let one writer's read miss another's concurrent write. Scoped per-instance (per-agent), not global, since two agents' repositories never share a file.

### Durable capture-job queue (Stage One Phase 2, slice C)

When a database is configured, `AgentSession` no longer fires a per-turn daemon thread for fact extraction. Instead, after each turn it calls `db.enqueue_capture_job(...)`, writing one durable row to `memory_capture_jobs` keyed by `(agent_id, session_id, source_from_message_id, source_to_message_id, prompt_version, model_id)` — enqueuing the same turn twice (e.g. after a crash-and-restart replays it) is a no-op, not a duplicate job.

One `CaptureWorker` (`memory/capture_worker.py`), started once at process startup — not per turn — polls this queue continuously:

```python
job = db.claim_next_capture_job()   # SELECT ... FOR UPDATE SKIP LOCKED
```

`FOR UPDATE SKIP LOCKED` lets the claim be a single atomic statement even if more than one worker process is running — a row already being processed is simply skipped, never double-claimed. On success the worker calls `db.complete_capture_job(job_id, facts)`, writing each fact as its own `memory_proposals` row. On failure it calls `db.fail_capture_job(job_id, error, backoff_seconds, max_attempts)`, which reschedules `run_after` with exponential backoff (`2.0 * 2**attempts` seconds) and permanently marks the job `failed` after 5 attempts.

`extract_facts(provider, exchange)` (`memory/extractor.py`) is the shared prompt/parsing primitive both the degraded-mode daemon thread and `CaptureWorker` call — but the two wrap it differently on purpose. The daemon thread swallows provider exceptions (fire-and-forget best-effort). `CaptureWorker` lets them propagate, since its own retry/backoff loop needs to see the failure to reschedule the job.

**Known, accepted gap (updated Phase 5, slice B):** `memory_proposals` rows are now indexed as searchable (see "Proposals become searchable" below), but stay out of `search_memory` and `<relevant_memories>` injection by default — reachable only via an explicit `corpus="proposal"` query. They sit unreviewed until Stage One Phase 5's later slices add a review/promotion flow.

Without a configured database, `CaptureWorker` is never constructed and `AgentSession` falls back to the original per-turn daemon-thread path described above (`extract_and_save_async`).

### Lexical memory index (Stage One Phase 3, slice A)

`memory/postgres_index.py`'s `PostgresMemoryIndex` is a rebuildable full-text index over one agent's *memory files* (`MEMORY.md`, topic notes, daily notes, imports) — distinct from `SessionDB`'s message-level FTS above. It indexes a different corpus with a different lifecycle (curated notes, edited occasionally, not appended every turn), so it owns its own tables and its own connection to the same database rather than extending `SessionDB`:

| Table | Description |
|---|---|
| `memory_chunks` | One row per indexed chunk. Columns: `id` (BIGSERIAL), `agent_id`, `source_kind` (`durable`/`daily`/`import`, plus `proposal` since Phase 5 slice B), `rel_path`, `chunk_index`, `heading_path`, `content`, `start_line`, `end_line`, `chunk_hash`, `search_vector` (weighted: heading text at FTS weight `A`, body at `B`; uses the language-neutral `'simple'` text-search config, not `'english'` — R3-GAP-005, same fix as `messages.search_vector` above). |
| `memory_files` | Per-file reconciliation ledger, `PRIMARY KEY (agent_id, rel_path)`. Same role as `message_mirrors` above: lets a later slice diff "what's on disk" against "what's indexed" by content hash rather than reindexing everything unconditionally. |
| `memory_chunks_shadow`, `memory_files_shadow` | Scratch space for `force_rebuild_agent()`'s crash-safe rebuild-and-swap (Stage One Phase 3, slice C) — same shape as the tables above, minus the generated `search_vector` (never searched directly). Empty except during/just after a `minion-assist memory reindex --force` run. |

`memory/chunking.py`'s `chunk_markdown()` splits a file into heading-aware, token-bounded (~400 tokens), overlapping (~80 tokens) chunks before indexing — see its module docstring for why both heading-awareness and overlap matter. Token counting reuses `context.py`'s tiktoken-or-char/4-heuristic fallback pattern.

`PostgresMemoryIndex.rebuild_agent(agent_id, indexable_files)` rebuilds one agent's entire index from `MemoryFileRepository.list_indexable_files()`'s listing — deleting chunks/ledger rows for any file no longer present, then reindexing everything else. `reindex_file()`/`remove_file()` handle one file at a time.

### Keeping the index in sync (Stage One Phase 3, slice B)

Three layers keep `memory_chunks`/`memory_files` current, from cheapest/fastest to broadest-coverage:

1. **Write-path sync.** `MemoryService`'s write methods (`remember`, `remember_import`, `append_daily`, `delete`) call `PostgresMemoryIndex.reindex_file()`/`remove_file()` directly, right after the disk write succeeds — the single file just touched, nothing more. Covers everything the app itself writes, with no lag. Never raises: a failed sync here is logged (`minion_assist.memory_service`, debug level) and swallowed, not surfaced to the caller.
2. **Startup reconciliation.** On every startup (when a database is configured), `minion.py` calls `MemoryService.reconcile_index()` for every configured agent — hash-diffs the agent's current files against the `memory_files` ledger, reindexing only what's new or changed and removing what's gone. Prints `Reindexed N memory file(s) for agent '{agent_id}'.` when anything actually changed. Catches any edit made while the process wasn't running.
3. **Live filesystem watcher.** `memory/watcher.py`'s `MemoryIndexWatcher` — one background thread (started at process startup, alongside `CaptureWorker`) watching every configured agent's workspace root via the optional `watchdog` package (installed with the `postgres` extra). Catches an edit made *while the process is running* but outside the app itself (e.g. hand-editing `MEMORY.md` in a text editor) — the one gap write-path sync can't close. Filesystem events are debounced per agent (~1 second of quiet) before triggering `reconcile_agent()`, so one save that fires several OS-level events (e.g. a temp-file-then-rename) triggers one reconcile, not several.

`reconcile_agent()` (used by both layer 2 and layer 3) is the "targeted" counterpart to `rebuild_agent()`: it only touches files whose content hash actually changed, the same "diff by hash, heal exactly what's missing or stale" shape as `session/db.py`'s `reconcile_session`/`reconcile_all_sessions` for message mirrors.

Without a configured database, none of the above run: `_memory_index`/`_memory_watcher` are simply `None`, and `MemoryService` behaves exactly as it did before Phase 3.

### Search, citations, and crash-safe reindex (Stage One Phase 3, slice C)

`MemoryService.search()` uses the lexical index when one is configured — a richer-ranked corpus than the Phase 1 linear scan (fusion across path/lexical/vector/pinned/recent lanes), though both paths now cover the *same* eligible files (MEM-GAP-008 — see below). Each hit is a `MemoryHit` as before, now optionally carrying `rel_path`/`start_line`/`end_line`/`score` when it came from the index (`None` for a linear-scan hit). Pass `corpus="durable"|"daily"|"import"` to restrict results to one part of the memory root; `"proposal"` also works (Phase 5 slice B). A plain `corpus=None` search — used both for automatic per-turn injection and a bare `search_memory` call — excludes both `"proposal"` and `"import"` chunks by default (MEM-GAP-004): unreviewed capture-job proposals and quarantined imports never enter automatic recall unless a caller explicitly asks for that corpus, since neither has been reviewed and imports in particular can carry externally-sourced text. This applies to the no-index linear-scan fallback too (`MemoryFileRepository.search()`'s `exclude_sources` parameter). The plan's fourth corpus, "sessions", is deliberately not offered here since it's already the separate `session_search` tool's job (lexical + vector fused since MEM-GAP-006 — see "Session semantic search" above).

**Fallback behavior and eligibility parity (MEM-GAP-008):** without a configured database, or if an index search call raises (e.g. a transient connection drop), `search()` falls back to the Phase 1 linear scan rather than failing the caller — a turn's `<relevant_memories>` injection must never break over a database hiccup. Before this round, the fallback diverged from the indexed path in ways that could weaken trust/isolation policy exactly during the failure it exists to survive: root `MEMORY.md` was never a linear-scan candidate at all, and boundary frontmatter (see "Action-sensitive memory boundaries" below) was never enforced in fallback mode — an expired note's raw `---...---` block could even leak verbatim into displayed content. `MemoryFileRepository.search()` now includes `MEMORY.md` (tagged `"topic"`, matching the indexed path's "durable" corpus grouping) and parses/enforces boundary frontmatter locally via `memory/boundaries.py`'s pure functions (no PostgreSQL dependency) — matching/returned content uses the frontmatter-stripped body, an inactive-boundary note is excluded, and an active one gets the same `MemoryHit.boundary` annotation the indexed path attaches. This fallback is not silent: a failed index search is logged at `WARNING`, `deep_status()`/`memory status --deep` surface ongoing index health, and a per-agent `memory_search:{agent_id}` `WorkerHealth` row (visible via `/status deep`) records every index-search attempt's success/failure so a persistently degraded agent is visible, not just a one-off log line.

**Crash-safe rebuild.** `PostgresMemoryIndex.force_rebuild_agent()` (`MemoryService.force_reindex()`) rebuilds an agent's entire index into `memory_chunks_shadow`/`memory_files_shadow` first, then swaps it into the live tables inside a single PostgreSQL transaction. If the process crashes or raises while building the shadow copy, the live index is never touched — search keeps serving the last complete index, unchanged. If it crashes during the swap itself, PostgreSQL's own transactional guarantees mean either the whole swap commits or none of it does; there is no moment where search sees a half-old-half-new result set. This is a deliberate, manual maintenance operation (`minion-assist memory reindex --force`) — nothing in the running app triggers it automatically.

New CLI commands:

```bash
minion-assist memory search "coffee preferences" --corpus durable   # restrict to MEMORY.md + topic notes
minion-assist memory status --deep                                  # + index chunk counts, corpus breakdown, last-indexed time
minion-assist memory reindex                                        # cheap hash-diff reconciliation, on demand
minion-assist memory reindex --force                                # crash-safe full rebuild-and-swap
```

### Embeddings and the vector lane (Stage One Phase 4, slice A)

An optional `"embeddings"` section in `config.json` enables a semantic-search lane on top of the lexical index:

```json
{
  "embeddings": {
    "provider": "lmstudio",
    "model": "nomic-embed-text-v1.5",
    "dimensions": 768
  }
}
```

`provider` must name an existing entry under `models.providers` — its `base_url`/`api_key` are reused for an `/embeddings` request against the same endpoint a chat provider already talks to, so no separate credentials are needed. `model` is the embedding model's id, which does **not** need to appear in that provider's chat `models` list (embedding models are typically served separately — e.g. a second model loaded in LM Studio alongside a chat model). `dimensions` must match that model's actual output vector size: PostgreSQL's `vector(N)` column type fixes its width at table-creation time and can't infer it from the data.

**This section is absent from `config.example.json` by default, and nothing changes until you add it.** With no `"embeddings"` section, `providers/embeddings.py`'s `EmbeddingProvider` is never constructed and `memory_chunk_embeddings` is never created — search stays lexical-only, exactly as it behaved before Phase 4.

| Table | Description |
|---|---|
| `memory_chunk_embeddings` | Embedding cache, `PRIMARY KEY (content_hash, model_identity)` — content-addressed, *not* tied to a `memory_chunks` row id (see below for why). Columns: `model_identity` (e.g. `"http://host/v1::nomic-embed-text-v1.5"` — lets a model/endpoint change be detected rather than silently mixing vectors from different embedding spaces), `embedding` (`vector(N)`, `N` from `embeddings.dimensions`). Only created when pgvector is available **and** an embedding backend is configured. `SessionDB`'s `message_embeddings` (see "Session semantic search" below) is the same shape, one layer down, for session messages instead of memory-file chunks. |

`PostgresMemoryIndex.has_vector_lane` reports whether both conditions hold. `cache_embedding()`/`get_cached_embedding()` are the storage primitives, keyed by `(content_hash, model_identity)` rather than a `memory_chunks` row id — `reindex_file()` deletes and reinserts every chunk of a file on every call (even when only one paragraph changed), so a row-id-keyed cache could never actually hit. `memory_chunks.chunk_hash` already stores the same hash per row, so the vector lane joins through that column instead of needing its own row reference — and since identical chunk text embeds identically regardless of which agent's note it came from, a cache hit is shared across agents for free. (An earlier, row-id-keyed version of this table shipped briefly before this design — `_ensure_schema()` detects and self-heals it on first connect if found, so nothing manual is needed on a machine that already ran that code.) A cache miss (no vector lane, or this exact `(content_hash, model_identity)` pair never embedded) always returns `None` rather than raising, so a caller can treat "compute a fresh embedding" as the uniform fallback.

### Session semantic search (MEM-GAP-006)

The same `"embeddings"` config above also activates a second, independent vector lane: semantic search over conversation history (`session_search`'s DISCOVER mode), not just memory files. Before this, `SessionDB` created a `message_embeddings` table with a hard-coded `vector(1536)` column but nothing ever wrote to or read from it — full-text search was the only way to find an old conversation, so a paraphrased recall ("that thing about the deploy last month" vs. the original wording) simply returned nothing.

**Write path.** After each turn, `AgentSession._send_locked()` enqueues one `message_embedding_jobs` row per newly-mirrored user/assistant message — independent of the capture/commitment flags, since this just makes raw messages searchable, unrelated to fact extraction. No-op when no embedding provider is configured. One `MessageEmbeddingWorker` (`memory/message_embedding_worker.py`), started once at process startup, polls this queue the same way `CaptureWorker` polls `memory_capture_jobs` (`FOR UPDATE SKIP LOCKED` claim, exponential backoff on failure, `_MAX_ATTEMPTS = 5`) and stores each result in `message_embeddings`, keyed by `(message_id, model_identity)`.

Deliberately a worker, not an inline call in `_send_locked()`: embedding is a network round trip, and the whole reason `memory_capture_jobs`/`CaptureWorker` exists instead of a synchronous per-turn extraction call is to never block a turn on one — the same reasoning applies here.

**Read path.** `SessionDB.hybrid_search_messages(query, agent_id, limit)` reciprocal-rank-fuses `search_messages()`'s existing FTS lexical lane with a new cosine-similarity vector lane (`_vector_search_messages()`), reusing `memory/postgres_index.py`'s `_reciprocal_rank_fusion` — the same fusion algorithm `PostgresMemoryIndex.hybrid_search()` uses for memory files, applied here to session messages. Degrades to FTS-only automatically when no embedding provider is configured (the vector lane returns `[]`), so `SessionSearchTool`'s DISCOVER mode calls this unconditionally with no branch needed. Both lanes stay scoped by `agent_id` through the same `sessions` join `search_messages()` already used (MEM-GAP-002) — the vector lane can never surface another agent's message content.

**Resumable.** `ReconciliationScheduler`'s third catch-up pass (see "Reconciliation" above) finds any mirrored message with no embedding job yet and enqueues one — so a transient enqueue failure heals itself on the next reconciliation pass instead of that message staying permanently unsearchable.

### Pinning (Stage One Phase 4, slice B)

A pinned note is always surfaced by the memory index's pinned fusion lane (slice C), regardless of whether it matches a search query — for a standing constraint or preference that must never be missed, as opposed to `save_memory` alone (only surfaces when it happens to match a query or rank in the top-5 proactive injection).

Pinning is scoped to explicit topic notes only (the ones `save_memory` creates) — not `MEMORY.md` (already unconditionally injected every turn via `bootstrap.py`, a separate mechanism entirely), not daily notes (ephemeral by nature), and not imports (unreviewed/quarantined — pinning one would contradict that status). It requires a configured database, since there's no lexical index for a "pinned lane" to belong to otherwise.

| Table | Description |
|---|---|
| `memory_pins` | `PRIMARY KEY (agent_id, rel_path)`. Columns: `pinned_at`. `PostgresMemoryIndex.remove_file()` also clears a file's pin, so a deleted note can never linger as an orphaned pin. |

`MemoryService.pin(key)`/`unpin(key)`/`is_pinned(key)`/`list_pinned()` are the service-level API — `pin()` requires the note to already exist (raises `FileNotFoundError` otherwise, so pinning a typo'd key fails loudly rather than creating a dangling pin); `unpin()` doesn't, so a stale pin can always be cleared.

The `pin_memory` tool (`key`, `pinned: bool`) exposes this to the LLM — registered alongside the other memory tools only when a database is configured (not offered at all, rather than present-but-always-erroring, when it isn't). Respects `read_only_mode` the same way `save_memory` does.

New CLI commands:

```bash
minion-assist memory pin project-goals --agent main     # pin a topic note
minion-assist memory unpin project-goals --agent main   # unpin it
minion-assist memory pins --agent main                  # list pinned notes
```

### Hybrid retrieval (Stage One Phase 4, slice C)

`MemoryService.search()` now calls `PostgresMemoryIndex.hybrid_search()` when an index is configured — fusing five independent lanes rather than ranking on lexical match alone:

| Lane | Signal | Notes |
|---|---|---|
| **path** | Exact/substring match against a chunk's `rel_path` | Catches a query naming a file directly (e.g. `"project-goals"`) even if that text never appears in the file's *body*. |
| **lexical** | `ts_rank` (same as `search()`) | |
| **vector** | Cosine similarity against cached embeddings | Empty (not an error) without a configured embedding provider, or if embedding the query itself fails. |
| **pinned** | Every chunk of every currently pinned file | See "Pinning" above. |
| **recent** | Most recently indexed files' chunks | Regardless of content match — a deliberate fallback so a query with no other signal still surfaces something, rather than returning empty. |

**Fusion.** Each lane ranks candidates by an incompatible signal (a `ts_rank` isn't comparable to a cosine similarity), so scores aren't averaged — reciprocal-rank fusion sums each candidate's `1/(60+rank)` contribution across every lane it appears in. A chunk only one lane finds still scores well; a chunk multiple lanes agree on scores higher still. This is the mem0 "reject semantic-seeded fusion" decision the plan cites: every lane must be able to surface a candidate the others missed, not just refine a shared candidate set.

**Temporal decay.** A `"daily"` chunk's fused score is multiplied by `0.5 ** (days_old / 30)` — halving every 30 days since the date in its filename. `"durable"` content (`MEMORY.md`, topic notes) never decays, per the plan's "evergreen content does not decay merely because its file mtime is old."

**MMR.** When an embedding provider is configured, a greedy maximal-marginal-relevance pass re-orders the fused results to push down near-duplicate snippets (measured by cached-embedding cosine similarity) in favor of more novel ones. Skipped entirely without embeddings — MMR's purpose is catching *semantic* near-duplicates, which requires vectors to measure; there's no lexical-only equivalent implemented here.

**Pinned guarantee.** Every currently pinned chunk (matching the `corpus` filter) is prepended to the final result list ahead of the fused ranking, so pinning something really does mean "always surfaced," not just "boosted" — it can never be pushed out by `max_results` truncation as long as there's room.

**Embedding generation during indexing.** `reindex_file()`/`force_rebuild_agent()` embed new/changed chunks best-effort right after writing them (never blocks indexing — a failed embedding call is logged and the chunk is simply left out of the vector lane until the next successful pass). Batches all of a call's chunks into one `embed()` request and skips any chunk whose content is already cached under the current model, so re-saving a file with one changed paragraph doesn't re-embed the unchanged parts.

### Recall telemetry (Stage One Phase 5, slice A)

`hybrid_search()` records one row per result it actually returns — regardless of caller, so an explicit `search_memory` tool call surfaces results just as much as proactive per-turn injection does. `AgentSession.send()` separately marks which of those surfaced results were actually selected for injection (`build_prompt_section()`'s token budget may only fit a few), via `MemoryService.mark_injected()`.

| Table | Description |
|---|---|
| `memory_recall_events` | Columns: `agent_id`, `rel_path`, `query_hash`, `surfaced_at`, `was_injected`. Keyed by `rel_path`, not a `memory_chunks` row id — chunk ids aren't stable across a reindex (the same lesson learned fixing the embedding cache in Phase 4), and promotion decisions (Phase 5 slice C) operate at the file/note level anyway. |

`hash_query()` normalizes (lowercase, collapsed whitespace) and hashes the query before storing it — Task 2: "hash normalized queries rather than storing unnecessary raw query text." Both `hybrid_search()`'s recording and `mark_injected()`'s later correlating call use this same function, so they agree on what counts as "the same query." `recall_stats(agent_id, rel_path)` aggregates `recall_count`, `unique_queries` (distinct query hashes — Task 3's "query diversity"), `injected_count`, and `last_recalled_at` — the primitive Phase 5 slice C's promotion ranking consumes (`5 * injected_count + 2 * recall_count + unique_queries`, `memory/consolidation.py`).

All of this is best-effort and never blocks a search or a turn: a failed telemetry write is logged and swallowed, matching every other observability hook in this codebase (`_sync_index`, `MemoryInjected`).

**`mark_injected()` timing (MEM-GAP-012).** `_injected_rel_paths` is computed at prompt-build time (`build_prompt_section()`'s output — which hits actually fit the token budget), but the `mark_injected()` call itself doesn't fire until `run_turn()` (the actual provider call, including any tool loop) returns *without raising* — the same "turn actually completed" point `_send_locked()` already uses for message mirroring and capture/commitment-job enqueue. A provider exception or a tool-loop crash never reaches that line, so a failed turn's surfaced notes never get an `injected_count` credit. Before this fix, the call fired before the provider was even invoked, so `injected_count` — the highest-weighted term in the promotion-ranking formula above — could be inflated purely by repeated provider failures, biasing which notes look salient enough to promote regardless of whether any turn actually used them.

### Proposals become searchable (Stage One Phase 5, slice B)

Every proposal `CaptureWorker` records (`memory_proposals`, Phase 2 slice C) is now also indexed into `PostgresMemoryIndex` right after it's created, under a new `source_kind = "proposal"` and a synthetic `rel_path` of `proposals/{proposal_id}` (a proposal has no real file on disk). This is what lets Phase 5's later consolidation-ranking slice reuse the same recall-telemetry machinery (`hash_query`/`recall_stats`) on proposals that already exists for real notes.

**Gated, not just indexed.** Making proposals searchable does **not** mean they show up in normal conversation. `hybrid_search()`'s corpus-agnostic default (`corpus=None`) explicitly excludes `source_kind IN ('proposal', 'import')` chunks — every lane (lexical, path, vector, pinned, recent) applies this same exclusion unless the caller passes `corpus="proposal"` or `corpus="import"` explicitly (MEM-GAP-004 extended the same policy to imports). Per-turn `<relevant_memories>` injection and a normal `search_memory` call never pass that, so an unreviewed proposal or quarantined import can never masquerade as a reviewed note in the model's context. (For proposals specifically, the pinned/recent lanes' guard is defense-in-depth rather than closing an active gap: pins can only ever point at topic notes, and the recent lane only surfaces files with a `memory_files` ledger row — which proposals deliberately never get, see below. Imports *do* get a real ledger row like any other file, so the same guard on those two lanes is what actually keeps a freshly-indexed or pinned import out of automatic recall.)

**No reconciliation ledger row.** Unlike a real file, a proposal never gets a `memory_files` row: that ledger exists to hash-diff indexed content against on-disk content (`reconcile_agent()`), which doesn't apply to a proposal — it's written once and never edited in place. Because it has no ledger row, `rebuild_agent()`/`reconcile_agent()` (which only ever look at `memory_files`) never touch it; `force_rebuild_agent()`'s crash-safe live-swap explicitly excludes `source_kind = 'proposal'` from the ledger-driven DELETE it does for files, so a force-rebuild (a files-only operation) can never wipe out proposal chunks that happen to share the same `memory_chunks` table.

`PostgresMemoryIndex.reindex_proposal(agent_id, proposal_id, claim_text)` / `remove_proposal(agent_id, proposal_id)` are the indexing primitives. `CaptureWorker` takes an optional `index_proposal` callable (`minion.py` wires it to `PostgresMemoryIndex.reindex_proposal` when a lexical index is configured, `None` otherwise — proposals are still recorded either way, just not searchable without an index) and calls it, best-effort, for every proposal `complete_capture_job()` just created; an indexing failure is logged and swallowed, never turning an already-successful capture job into a failed one.

`memory_proposals` also gained a `status` column (`TEXT NOT NULL DEFAULT 'pending'`, added via `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` so an existing database picks it up automatically) — `pending`/`promoted`/`rejected`/`superseded`. Nothing assigns anything but `pending` yet; a later Phase 5 slice's review flow will.

`minion-assist memory search --corpus proposal` and `search_memory`/`MemoryService.search(corpus="proposal")` are how a caller deliberately looks at unreviewed proposals.

### Consolidation ranking and preview drafting (Stage One Phase 5, slice C)

`memory/consolidation.py` turns pending proposals into a human-reviewable queue, without ever writing to disk or changing a proposal's status:

`rank_proposals(db, index, agent_id)` scores every `status = 'pending'` proposal as `5 * injected_count + 2 * recall_count + unique_queries` (all from Phase 5 slice A's recall telemetry, looked up under the proposal's `proposals/{id}` rel_path from slice B) and sorts highest first — a proposal never recalled still appears, at the bottom (score `0`), rather than being hidden. This score only orders a review queue; nothing is gated on it.

The plan's ranking task also lists confidence, source authority, contradiction status, and user pinning as signals — none of these exist yet in minion-assist (`extract_facts()` returns bare claim strings with no confidence; every proposal comes from the same capture pipeline; proposals aren't pinnable; and contradiction detection would need a new semantic-comparison step). These are documented, deliberate gaps in `consolidation.py`'s module docstring, not oversights — consistent with the plan's own "begin in preview-only mode; collect data before choosing thresholds."

`MemoryConsolidator(db, index, files, provider, agent_id).preview(proposal_id)` drafts what a topic-note update *would* look like for one proposal (one instance per agent — see "Apply, reject, and rollback" below for why):

1. Searches existing topic notes only (`hybrid_search(..., corpus="durable")`, filtered to `memory/topics/` hits — never `MEMORY.md`, daily notes, or imports) for a merge target. No score threshold is applied — the first topic-note hit becomes the target, or there is none and the draft proposes a new topic. A wrong guess here only produces a preview a human can discard; nothing is ever applied automatically.
2. Calls the provider with a fixed drafting prompt containing only the proposal's claim text and the target's current content (or nothing, for a new topic) — asking for a `KEY:`/`RATIONALE:`/`---`/content-formatted response with the full revised note text.
3. The prompt explicitly instructs the model to keep contradicting statements marked as contested rather than silently resolve them — the plan's "contradictory preferences ... never merged into a false synthesis" acceptance criterion, satisfied at the drafting-prompt level in this slice.
4. Stores the result via `PostgresMemoryIndex.record_consolidation_preview()` — a new `memory_consolidation_previews` row, hashing the target's content **at draft time** (`based_on_content_hash`) so a later apply step can detect a human edit made since. Nothing is written to any actual note file.

**Evidence provenance (Task 8).** `consolidation.py` never reads `DREAMS.md` or any `DreamingScheduler` state — a draft's only inputs are a proposal's claim text (traceable through its `job_id` to real captured messages) and the merge target's existing content (the thing being revised, not "evidence" for the new claim). `tests/memory/test_consolidation.py` pins this down with an explicit regression test asserting the drafting prompt only ever contains those two inputs.

`format_preview_report(preview)` renders a preview as plain text (candidate, rationale, target, drafted content) with `Decision: pending review` — this always reflects the *draft* itself, not the proposal's current review status (which "Apply, reject, and rollback" below adds).

### Apply, reject, rollback, and historical backfill (Stage One Phase 5, slice D)

`MemoryConsolidator` is scoped to a single agent (`agent_id` is fixed at construction, matching `MemoryService`'s own per-agent design) — unlike `rank_proposals()`, which stays a free function taking `agent_id` per call, since it never touches an agent's on-disk files.

**`approve(preview_id)`** applies a stored preview: writes its drafted content to disk (`MemoryFileRepository.remember()`), reindexes it (`source_kind="durable"`), removes the proposal's own raw-claim chunk (now redundant — its content lives in the real note), and marks the proposal `"promoted"`. Before any of that, it re-hashes the target's *current* content and compares it against the preview's `based_on_content_hash` — if they no longer match, it raises `StaleProposalError` and applies nothing at all. This is the concrete mechanism behind the plan's "user edits win over stale consolidator output" acceptance criterion: a human's manual edit made between preview and approval always wins, never gets silently clobbered.

**`reject(proposal_id, reason="")`** only updates the proposal's status to `"rejected"` (and stores `reason` in the new `memory_proposals.rejected_reason` column, added the same `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` way `status` was) — no file, no index row touched. Rejected proposals keep their searchable `proposals/{id}` chunk (still reachable via `corpus="proposal"`, useful for audit) — approved ones don't, since their content has moved into a real note.

**`rollback(target_key)`** undoes the most recent write to one topic note — either an `approve()` apply, or (MEM-GAP-020) a plain explicit `MemoryService.remember()`/`delete()` call (`save_memory`, `SaveMemoryTool`):

| Table | Description |
|---|---|
| `memory_topic_revisions` | A snapshot of a topic note's content taken right *before* it's overwritten/deleted — by `approve()`, or by `MemoryService.remember()`/`delete()` (MEM-GAP-020). `PRIMARY KEY` none (plain `BIGSERIAL`), indexed on `(agent_id, target_key)`. Columns: `proposal_id` (nullable — set for a consolidation apply so rollback can restore the proposal to `"pending"` too; `NULL` for an explicit write, which has no proposal), `prior_content` (`""` means the write being undone created the note from scratch). |

Restores `prior_content` (or deletes the file entirely if `prior_content == ""`, since there's nothing to restore an empty string *to*), reindexes accordingly, and **consumes** the revision row — so a second `rollback()` on the same topic steps back one write further, not the same one again, the same one-entry-per-undo-step shape a normal undo stack has. Only when the revision's `proposal_id` is set (a consolidation apply) does it also put the associated proposal back to `"pending"` and re-index its raw proposal chunk — an explicit-write revision has no proposal, so that step is skipped for those, giving both write paths the same undo mechanism (`minion-assist memory consolidate rollback <key> --agent <agent>`) without a second, parallel one.

**Explicit-write revisions (MEM-GAP-020).** `MemoryService.remember()`/`delete()` snapshot a topic note's prior content as a revision before mutating it — before this, only a consolidation `approve()` apply was recoverable this way; an accidental `save_memory` overwrite (e.g. an agent replacing a preference topic with a contradictory value) had no recovery path short of external filesystem backup. Same best-effort, no-op-without-a-configured-index posture as index sync itself (`_sync_index`) — a failed/skipped revision snapshot never blocks the write that triggered it, and file-only/degraded mode (no database) simply has no revision history, exactly as before.

**Historical backfill (Task 10).** `backfill_agent(db, agent_id, model_id)` finds message ranges in an agent's session history that no `memory_capture_jobs` row has ever covered — true gap-filling, not just "skip sessions with any coverage": it diffs every session's actual message ids against every capture job's recorded range (any state, including `failed` — a failed job already exists in the retry/backoff pipeline, backfill's job is catching ranges *never attempted* at all), chunks each gap into bounded 20-message windows (`_BACKFILL_WINDOW_MESSAGES`), and enqueues one capture job per window via the existing `SessionDB.enqueue_capture_job` — using the exact same idempotency-key shape a live turn's job already uses, so re-running backfill is a safe no-op the second time. It does not run extraction itself: the already-running `CaptureWorker` picks up backfilled jobs and processes them exactly like a live turn's job.

New CLI commands:

```bash
minion-assist memory consolidate list --agent main --top 5      # ranked pending proposals
minion-assist memory consolidate preview 42 --agent main        # draft a preview for proposal #42
minion-assist memory consolidate explain 7 --agent main         # show a stored preview + staleness
minion-assist memory consolidate approve 7 --agent main         # apply preview #7 to disk
minion-assist memory consolidate reject 42 --agent main --reason "not useful"
minion-assist memory consolidate rollback project-goals --agent main
minion-assist memory consolidate backfill --agent main          # gap-fill historical capture jobs
```

**`MemoryConsolidationScheduler`** (`memory/consolidation_scheduler.py`) automates preview drafting on a daily wall-clock schedule — the same scheduling shape as `DreamingScheduler` (reuses its `_seconds_until_next`), but deliberately a **separate, independently-configured schedule** (Task 9: "keep the poetic `DreamingScheduler` independently configurable; rename the consolidation schedule ... to avoid ambiguity"):

```json
{
  "memory_consolidation": {
    "enabled": true,
    "hour": 4,
    "minute": 0,
    "timezone": "Australia/Sydney",
    "agent_id": "main",
    "top_n": 5
  }
}
```

Each pass ranks the configured agent's pending proposals and drafts a preview for up to `top_n` of them — but only ones that don't already have a preview yet, so a proposal sitting `"pending"` for many days doesn't get redrafted (and re-billed) every single day. Never applies, promotes, or rejects anything — 100% human-gated, same as every other part of Phase 5. Starts only when both a database and lexical index are configured, and the configured `agent_id` is actually running.

### Action-sensitive memory boundaries (Stage One Phase 6, slice A)

**Goal: make proactive behavior useful without converting remembered text into authority.** A topic note can optionally carry action-boundary metadata as a small frontmatter block at the very top of the file:

```markdown
---
owner: main
applies_when: deploying to production
safe_after: 2026-09-01
expires_at: 2026-12-01
unlock_condition: explicit user confirmation this quarter
prohibited_action: do not deploy without a second reviewer
required_approval: user
---
The rest of the note's body, as usual.
```

All seven fields are optional (`memory/boundaries.py`'s `parse_frontmatter()`); a note with none of them behaves exactly as before. Unknown keys are silently ignored, and a malformed/unterminated block degrades to "no frontmatter" rather than raising — a human hand-editing a note must never be able to break indexing with a typo.

**Advisory only, structurally — not a permission-policy hook.** This metadata is never wired into `tools/policy.py`'s `PermissionPolicy` or any tool-execution path; there is no code anywhere that reads a note's `required_approval`/`prohibited_action` text and uses it to allow or block a tool call. It exists purely to be *rendered* wherever a note is retrieved, so a model reading "requires approval from X" in its own memory sees that as something to ask about, not standing authorization it already has. This is the concrete mechanism behind the plan's acceptance criterion "a remembered approval never bypasses current permission policy" — satisfied by there being no path connecting the two systems at all, not by a runtime check.

**Mechanically enforced vs. purely rendered.** `safe_after`/`expires_at` are the only two fields with a machine-checkable meaning: `is_boundary_active()` treats them as a `[safe_after, expires_at]` window (either side optional), and a hit whose note is currently outside that window is *excluded* from the result set entirely — not merely labeled — satisfying "expired constraints ... do not influence action" by construction. This can occasionally return fewer than `max_results` hits on a turn where an inactive boundary-bearing note would otherwise have ranked in the top results — a documented, deliberate trade-off favoring correctness over exact recall-count preservation. `owner`, `applies_when`, `unlock_condition`, `prohibited_action`, and `required_approval` are pure display text; nothing evaluates them. Enforced identically on both retrieval paths (MEM-GAP-008): `MemoryService._apply_boundaries()` for indexed hits (looks up cached metadata via `PostgresMemoryIndex.get_boundary()`), and `MemoryFileRepository.search()` itself for the degraded-mode linear scan (parses the file's own frontmatter directly, with no PostgreSQL dependency — so this stays correct precisely during the database outage that triggers the fallback).

**Where it's rendered.** Both primary retrieval surfaces show a note's boundary via `format_boundary_prefix()` — a single bracketed, explicitly-labeled-advisory line — right alongside its content: per-turn `<relevant_memories>` injection (`agents/session.py`'s `build_prompt_section()`) and the `search_memory` tool. (`memory_get`'s bounded exact-line reads are a deliberate scope boundary for this slice — that tool is typically a targeted follow-up to something already seen via search/injection, where the boundary would already have been shown once.)

**Storage.** The frontmatter is stripped before chunking (so it never becomes searchable/embeddable body text) and its parsed form is cached on `memory_files.boundary_metadata` (`ALTER TABLE ... ADD COLUMN IF NOT EXISTS`, same self-healing pattern every other schema addition in this project uses) — refreshed on every `reindex_file()`/`force_rebuild_agent()`, the same relationship `content_hash` already has to a file's raw text. The file itself stays the source of truth (a human can hand-edit the frontmatter block directly); Postgres is only a derived cache, consistent with Phase 1's founding principle. Chunk citations (`start_line`/`end_line`) still point at the right line in the *original* file on disk — the frontmatter block's line count is added back onto every chunk's line numbers after chunking the frontmatter-stripped body.

No new CLI in this slice — a human (or a future consolidator enhancement) edits the frontmatter directly.

### Commitments — schema and extraction pipeline (Stage One Phase 6, slice B)

A **commitment** is an *inferred* social follow-up the model notices in a completed exchange — "the user mentioned an interview tomorrow" — never something the user explicitly asked to be reminded about. Grounded against OpenClaw's actual `src/commitments/` module (not just the plan doc): same `kind` (`event_check_in`/`deadline_check`/`care_check_in`/`open_loop`), `sensitivity` (`routine`/`personal`/`care`), `source` (`inferred_user_context`/`agent_promise`), and `status` (`pending`/`sent`/`dismissed`/`snoozed`/`expired`) vocabulary, and a `due_earliest`/`due_latest` *window* rather than a single instant — scaled down for minion-assist's simpler architecture (one exchange per extraction call, not OpenClaw's batched design; no per-user timezone resolution yet).

**Durable extraction queue.** `memory_commitment_jobs` + `CommitmentWorker` (`memory/commitment_worker.py`) mirror Phase 2's proven `memory_capture_jobs`/`CaptureWorker` shape exactly — same claim/complete/fail lifecycle, same idempotency-key mechanism — but as a genuinely separate table/queue, since commitment output (kind/sensitivity/due-window/confidence/dedupe-key) is a completely different shape than `memory_proposals`' plain claim strings. `AgentSession.send()` enqueues a commitment job alongside (not instead of) the existing capture job whenever `commitments.enabled` — independent flags, a turn can trigger one, both, or neither.

**Skipping explicit reminders.** The plan's Task 6 says explicit reminders should route to "the task/scheduler subsystem rather than memory" — minion-assist has no such subsystem. Rather than build one in this slice, the extraction prompt (`memory/commitments.py`'s `_EXTRACT_SYSTEM`) mirrors OpenClaw's real wording: it explicitly instructs the model to skip an exact request like "remind me tomorrow" entirely. Those stay unhandled, exactly as today — a documented scope boundary.

**Confidence-gated, tools-disabled.** `extract_commitments()` calls the provider with `tools=[]` (Task 3: "tools-disabled") and validates every candidate: known kind/sensitivity/source, confidence at or above threshold (0.72 routine, 0.86 for anything `care`-related — OpenClaw's own shipped defaults, reused since minion-assist has no evaluation data yet to derive different numbers from), and a `due_earliest` that parses to a real future timestamp.

**"Ensure the due time is not immediate" (Task 4).** Every validated candidate's `due_earliest` is clamped to at least `now + min_due_seconds` — `minion.py` wires this to the configured heartbeat interval, since a commitment due before the next heartbeat tick could possibly check for it would just sit expired-on-arrival. Mirrors OpenClaw's own `resolveMinimumDueMs`.

**Scoped to exact agent and channel context.** `AgentSession.send()` gained a new optional `channel` parameter (e.g. a Matrix `room_id`; `None` normalizes to `"cli"` everywhere else), threaded into every commitment job/row. `SessionDB.complete_commitment_job()` upserts a candidate whose `dedupe_key` matches an existing *pending* commitment in the same `(agent_id, channel)` scope — widening the due window, keeping the higher confidence — rather than inserting a near-duplicate, mirroring OpenClaw's real `upsertInferredCommitments`.

```json
{
  "commitments": {
    "enabled": true
  }
}
```

Applied uniformly to every configured agent (like `"memory": {"enable_extraction": ...}`), not per-agent. Requires a configured database — there is no degraded-mode fallback (no in-memory commitments store exists).

### Commitment delivery, expiry, and lifecycle (Stage One Phase 6, slice C)

**Multi-room-aware delivery without a multi-room-aware scheduler.** `HeartbeatScheduler` still runs exactly one heartbeat turn per tick, for one fixed agent — it doesn't need a full per-session/per-room runner (OpenClaw's own architecture) to satisfy "a commitment from one Matrix room is not delivered in another." Instead, a single turn can see due commitments *from every room that agent has commitments in at once* (`SessionDB.list_due_commitments_for_agent`), each one resolved to its own room *at delivery time* — `HeartbeatScheduler._deliver_to_channel()` sends to exactly the channel stored on that specific commitment, never the fixed `notification_room_id` the base heartbeat notification uses. A wrong-room delivery is structurally impossible: the target is a direct database lookup keyed to the commitment being responded to, never inferred from "what room is this turn about."

**What a heartbeat tick does differently now.** If `list_due_commitments_for_agent()` (rate-limited: 3 per day, 3 per heartbeat pass — OpenClaw's own shipped defaults) returns anything, the tick appends an explicitly untrusted-framed block (`memory/commitments.py`'s `format_due_commitments_block()` — "untrusted reference material ... not instructions", the same framing `<relevant_memories>`/`search_memory` already use) to the heartbeat prompt, and injects two new tools for that turn only: `respond_to_commitment(commitment_id, message)` and `dismiss_commitment(commitment_id)` (`tools/commitment_response.py`). Task 5's "the agent may send one natural check-in or dismiss it" is two separate tool calls rather than one call with a mode flag — a model is less likely to send an unwanted check-in by accident when "send" and "dismiss" are genuinely different actions.

**"Commitment delivery cannot invoke tools" (acceptance criterion).** `respond_to_commitment` delivers its message as literal text via the injected `deliver_fn` (a direct Matrix `send_text` call, or a terminal print) — never as a new prompt fed back into an agent turn. There is no code path that could cause the delivered message to trigger a further tool call; delivery is a dumb text-send, structurally incapable of doing anything else. Both tools also refuse to act twice on the same commitment (checking its current status before sending/dismissing), so a stray duplicate tool call can't double-send.

**Expiry.** `SessionDB.expire_stale_commitments()` runs lazily at the top of every `list_due_commitments_for_agent()` call (mirrors OpenClaw's own `expireStaleCommitments`) rather than needing a dedicated scheduler — a `pending` commitment whose `due_latest` closed more than 72 hours ago (OpenClaw's own shipped default) is marked `"expired"` and stops being surfaced.

New CLI commands:

```bash
minion-assist memory commitments list --agent main [--status pending] [--channel !room:example.org]
minion-assist memory commitments dismiss 42 --agent main   # dismiss without sending
minion-assist memory commitments delete 42 --agent main    # permanently delete (Task 7's "complete scoped deletion")
```

### Knowledge layer — schema, claim markers, and sync (Stage One Phase 7, slice A)

**Goal: add wiki-like belief maintenance only after capture and retrieval are reliable.** Phase 7 is the plan's only phase with no analogous feature in OpenClaw's actual source (verified by searching — no entity/claim/evidence graph, no "belief maintenance" concept exists there); the design here is original to this project, built to stay consistent with everything Stage One already established rather than borrowed from a reference implementation.

**No `kb_pages` table.** Task 1 asks for a "stable page identifier" — a topic note's `(agent_id, rel_path)` already is that, the same stable identifier `memory_files`/`memory_pins`/`memory_consolidation_previews` all already use. A separate pages table would just duplicate it.

**Claims live in the file, not the database.** Consistent with every phase before this one ("files are the source of truth, Postgres is a derived cache"), a claim's stable id and structured fields (Task 2: status/confidence/observed-time/valid-time/privacy-tier) are an inline HTML comment attached to the sentence it annotates:

```markdown
- User's dog is named Biscuit.
  <!-- claim:c-a1b2c3d4 status=supported confidence=0.9
       observed=2026-06-01 evidence=proposal:42 -->
```

`memory/knowledge.py`'s `parse_claims()` is read-only relative to file content — it never invents a claim from unmarked prose. A sentence only enters `kb_claims` once something (a human, or the consolidator in a later slice) explicitly attaches a marker; everything else in a topic note stays exactly what it's always been, untracked prose. `PostgresMemoryIndex.reindex_file()`/`force_rebuild_agent()` call `parse_claims()` and sync the result into Postgres the same way they already sync chunks and boundary metadata (Stage One Phase 6, slice A) — for `source_kind="durable"` only (daily notes and unreviewed imports are excluded; an import only gets claims once a later slice's review flow promotes it into a durable page).

| Table | Description |
|---|---|
| `kb_entities` | `id` (`"e-" + 8 hex chars`, system-assigned), `agent_id`, `name`, `name_normalized` (the actual `UNIQUE (agent_id, name_normalized)` dedup key — case-insensitive exact match, deliberately no fuzzy entity resolution/merging; a genuinely hard NLP problem with no evaluation data yet to justify building, the same "collect data before choosing thresholds" posture the rest of Stage One has taken throughout), `created_at`. |
| `kb_claims` | `id` (human/model-authored, from the marker), `agent_id`, `rel_path`, `entity_id` (nullable), `text`, `status` (`supported`/`contested`/`superseded`/`unknown`, defaults `unknown`), `confidence`, `observed_at` (falls back to sync time when the marker omits `observed=`), `valid_from`/`valid_to` (the bi-temporal distinction Task 2 asks for — when a claim is/was true in the world, distinct from when it was *observed*), `privacy_tier` (free text), `line_number`, `created_at`, `updated_at`. |
| `kb_evidence` | One row per `evidence=kind:ref` pair (Task 1's evidence identifiers) — e.g. `("proposal", "42")`. A claim with zero evidence rows is exactly what a later slice's provenance-gap dashboard (Task 3) reports. |

`freshness` (also Task 2) is deliberately **not** a marker field — it's computed at query time from `observed_at` (a decay function, mirroring `postgres_index.py`'s existing `_decay_factor`), not something a human or model would hand-author.

Entity resolution: `get_or_create_entity(agent_id, name)` — race-safe (`ON CONFLICT DO NOTHING` + re-select), matched case-insensitively, preserving whichever casing first created the row.

Removing a page's marker removes its claim (and evidence) on the next sync — the same "diff and remove" shape `remove_file()`/`reconcile_agent()` already use for whole files. `remove_file()` also cleans up a deleted page's claims directly.

No CLI or dashboards yet in this slice — `get_claim()`/`list_claims()` are the lightweight read primitives later slices (dashboards, the compiled digest, import review, forgetting) build on.

### Provenance and relationships (Stage One Phase 7, slice B)

**A minimal relationship vocabulary.** `supersedes=` and `contradicts=` (comma-separated claim id lists) are the only two relationship kinds — the two the plan's acceptance criteria actually need ("contradictory preferences remain contested until resolved" and the consolidator's existing supersede capability from Phase 5), not an open-ended relationship-type system.

| Table | Description |
|---|---|
| `kb_relationships` | `from_claim_id`, `to_claim_id`, `kind` (`supersedes`/`contradicts`), `created_at`. No `FOREIGN KEY` on either claim id — a hand-written `contradicts=` referencing a typo'd id is itself useful information for a later slice's dashboards to surface, not something to silently reject at sync time. |

`get_claim()` now also returns `supersedes`/`contradicts` (claims this one points *to*); `list_relationships_to(agent_id, claim_id)` is the reverse lookup (claims pointing *at* this one) — a later slice's contradiction dashboard needs both directions to show a conflict from either claim's perspective.

**Marker-less claims stay allowed.** A claim marker with no `evidence=` field is still parsed and synced normally (already true since slice A) — hand-authored content can't be retroactively forced to cite a source, but it shows up in a later slice's provenance-gap dashboard.

**`MemoryConsolidator` becomes the primary way claims actually get created.** Its drafting prompt (Phase 5, slice C) already instructed the model to flag contradictions in prose rather than silently resolve them — this slice makes that structural instead of just prose. `preview()` now generates a fresh claim id in code (never left to the model to invent — the same reasoning `get_or_create_entity` already applies to entity ids) and, when revising an existing page, shows the model that page's existing claims (id/status/text). The prompt instructs the model to attach a real `<!-- claim:ID ... -->` marker to the new claim and — when it recognizes a conflict — set `status=contested`/`contradicts=EXISTING_ID` on the new marker *and* flip the existing marker's own `status` to `contested` too (leaving everything else about it untouched), so a conflict is contested from both sides, not just the new one.

No new sync code was needed for any of this: `approve()` already calls `reindex_file()` on the drafted content, and slice A's `_sync_claims()` already parses and syncs whatever markers ended up there. A malformed or missing marker in the model's response isn't fatal — that particular claim just never enters `kb_claims`, the same as any hand-authored note with no markers at all.

### Knowledge dashboards (Stage One Phase 7, slice C)

`minion-assist memory knowledge dashboard --agent ID [--section SECTION]` reports on the knowledge layer's claim quality — six read-only reports over `kb_claims`/`kb_relationships`/`kb_evidence`, meant for a human periodically reviewing what the consolidator and hand-authored notes have accumulated. Requires a configured database (same "Error: no database configured" degradation as `consolidate`/`commitments`).

| Section | Backed by | Reports |
|---|---|---|
| `contradictions` | `list_contradictions()` | Every recorded `contradicts` relationship, both claims' current text/status. A `contradicts=` referencing a typo'd/nonexistent claim id shows as a dangling reference (via `LEFT JOIN`) rather than being silently dropped. |
| `stale` | `list_stale_claims()` | Claims whose freshness (a 90-day-half-life decay of `observed_at`, computed at query time — never stored) has dropped below 0.5, stalest first. |
| `low-confidence` | `list_low_confidence_claims()` | Claims with `confidence < 0.5` or no confidence rating at all (an unrated claim and a low-rated one both mean "don't trust this without a closer look"). |
| `missing-provenance` | `list_claims_missing_evidence()` | Claims with zero `kb_evidence` rows. Not an error state — hand-authored claims can't be retroactively forced to cite a source — just a gap surfaced for review. |
| `open-questions` | `list_claims(status="unknown")` | No new query method — `status="unknown"` already means exactly this. |
| `privacy-review` | `list_claims_needing_privacy_review()` | Claims with no `privacy_tier` assigned yet. |

`--section` restricts the report to one of the six names above; omitted, all six print in the order listed. A "deletion coverage" section (auditing that forgetting a source removed or re-flagged every derived claim) is deliberately **not** included here — there's nothing to audit before slice F's forgetting mechanism exists to produce deletions in the first place; it's deferred to that slice.

New CLI command:

```bash
minion-assist memory knowledge dashboard --agent main                        # all six sections
minion-assist memory knowledge dashboard --agent main --section stale        # just one
```

### Compiled knowledge digest (Stage One Phase 7, slice D)

Task 4: "compile a bounded agent digest rather than scraping Markdown during prompt construction." `KNOWLEDGE_DIGEST.md` is a new, **fully machine-owned** file — a new separate file (the non-default fork chosen for this slice, rather than a marked section inside `MEMORY.md`) written directly under an agent's workspace root, next to `MEMORY.md`/`USER.md`/`DREAMS.md`. Nothing ever hand-edits it, so there's no "don't clobber human edits" concern the way `MemoryFileRepository`'s other writes have to guard against.

**Compilation is a pure function.** `memory/knowledge.py`'s `compile_digest(claims, max_chars=8000)` takes an already-fetched list of `status="supported"` claims (from `PostgresMemoryIndex.list_claims(agent_id, status="supported")`) and renders them grouped by page — the same "fetch, then format" split `memory/commitments.py`'s `format_due_commitments_block` already uses. Only `supported` claims are included: `contested`/`unknown` ones are exactly what a human hasn't resolved yet (see the slice C dashboards), and `superseded` ones are stale by definition. Claims are included whole, in order, until the next one would exceed `max_chars` — never truncated mid-sentence — with a footer noting how many were left out. Nothing wall-clock-dependent or random is in the output, so the same claims always compile to the same digest (Phase 7's "compiled artifacts are reproducible from canonical pages" acceptance criterion).

**Writing to disk.** `MemoryFileRepository.write_digest(content)` atomically overwrites `KNOWLEDGE_DIGEST.md` at the workspace root. `bootstrap.py`'s `_BOOTSTRAP_FILES` now includes `"KNOWLEDGE_DIGEST.md"` (after `MEMORY.md`) — no other bootstrap-layer changes were needed; it's read, budgeted, and injected exactly like every other bootstrap file, and skipped entirely when empty (no supported claims yet).

**`KnowledgeDigestScheduler`** (`memory/digest_scheduler.py`) recompiles and overwrites the file once a day, mirroring `MemoryConsolidationScheduler`'s daily wall-clock shape (`_seconds_until_next`, a daemon `threading.Timer`, reschedule-after-error). A third, separately-configured schedule alongside `dreaming` and `memory_consolidation`, for the same "keep each independently configurable" reasoning (Task 9). Configured under `"knowledge_digest"` in `config.json`:

```json
"knowledge_digest": {
    "enabled": true,
    "hour": 4,
    "minute": 30,
    "timezone": "Australia/Sydney",
    "agent_id": "main",
    "max_chars": 8000
}
```

Disabled by default. `minute` defaults to 30 (rather than `memory_consolidation`'s `:00`) so the two schedulers don't contend for the same provider/database call slot when both are enabled. Requires a configured database and lexical index — silently does not start without one, same degradation as `memory_consolidation`.

New CLI command — runs the exact same compile-and-write step as the scheduler, on demand:

```bash
minion-assist memory knowledge compile --agent main
minion-assist memory knowledge compile --agent main --max-chars 4000
```

### Operational-table retention (MEM-GAP-015)

Everything durable in this project — messages, memory files, knowledge-graph claims — retains indefinitely by default, which is the correct, intentional behavior for a single-user deployment. What has no bound is a separate category: pure operational/telemetry tables that accumulate as a side effect of normal operation and, before this feature, never had any cleanup path at all — completed/failed capture, commitment, and message-embedding job rows; `memory_recall_events` (recall telemetry, MEM-GAP-012); and `memory_consolidation_previews`/`memory_import_previews` (drafted-preview rows that, notably, are **not** deleted by a normal approve/reject decision — only full session deletion touches them).

**Scoped narrowly, on purpose.** `SessionDB.prune_operational_tables()` only ever deletes job rows whose `state` is `'done'` or `'failed'` (never `'pending'`/`'running'`, regardless of age) past `retention_days`. `PostgresMemoryIndex.prune_operational_tables()` does the same by `created_at`/`surfaced_at` for its three tables. Neither method touches `memory_topic_revisions` — each row there is a real rollback restore point for an already-applied change (see "Explicit-write revisions" below), not disposable telemetry.

`memory_proposals`/`commitments` survive a job row being pruned too, but a first version of this feature got that wrong: `memory_proposals.job_id`/`commitments.source_job_id` were `ON DELETE CASCADE`, so pruning an old completed capture/commitment job silently deleted its already-extracted proposal/commitment along with it — the exact opposite of "operational bookkeeping only." A follow-up fix (migration 005) changed both to `ON DELETE SET NULL`: pruning a job now only detaches its output's now-meaningless back-pointer, never the output itself. The same fix also gave `memory_proposals` its own `session_id` column (backfilled from its job at migration time) so session deletion can find a session's proposals directly, without depending on a job row that retention might have already pruned.

A second, related bug in that first version: `find_uncovered_capture_range`/`find_uncovered_commitment_range`/`find_uncovered_message_ids_for_embedding` (the reconciliation queries below) derived their "already enqueued through here" cursor from `MAX(...)` over the very job tables retention deletes rows from — enabling retention could silently reset that cursor and cause reconciliation to re-enqueue a session's entire history for extraction/embedding. Migration 005 fixed that with `session_job_coverage`, a small separate table read instead of the job tables directly — retention could no longer erase coverage, but that table could still only represent a single high-water mark. See "Exact job-coverage gap detection" below for the follow-up fix (migration 006) that replaced it entirely.

Both `prune_operational_tables()` methods accept `dry_run=True`, which runs the identical filter as `SELECT count(*)` instead of `DELETE`, so a dry-run report and the real cleanup can never silently diverge in what they count.

**`MemoryRetentionScheduler`** (`memory/retention_scheduler.py`) runs one pass a day, mirroring `KnowledgeDigestScheduler`'s daily wall-clock shape — a fourth, separately-configured schedule alongside `dreaming`, `memory_consolidation`, and `knowledge_digest`. Configured under `"memory_retention"` in `config.json`:

```json
"memory_retention": {
    "enabled": true,
    "hour": 4,
    "minute": 45,
    "timezone": "Australia/Sydney",
    "retention_days": 30
}
```

Disabled by default. Requires a configured database; the lexical index is optional — without one, only the job-table half of the cleanup runs.

New CLI command — the same prune logic the scheduler uses, on demand, dry-run by default:

```bash
minion-assist memory retention                # dry-run report — counts only, deletes nothing
minion-assist memory retention --apply         # actually deletes the stale rows
minion-assist memory retention --apply --days 7   # override config.json's retention_days
```

### Stale-running job reclaim (R2-GAP-003)

None of the three durable job queues (`memory_capture_jobs`, `memory_commitment_jobs`, `message_embedding_jobs`) use a claim lease/owner — a worker crash between claiming a job (`state='running'`) and completing it left that row stuck in `'running'` forever: invisible to the normal pending-queue claim query, so nothing ever retried it, and `queue_lag_summary()` (the data behind `/status deep`'s queue rows) didn't even report it, since it only counted `state='pending'`.

Since this codebase runs exactly one worker per lane (never multiple concurrent claimants racing each other), a full lease/heartbeat system would be more machinery than the actual failure mode warrants. `SessionDB.reclaim_stale_running_jobs(stale_after_seconds=3600)` is the minimum fix: at process startup, right before any worker starts claiming, it resets any row that's been sitting in `'running'` for implausibly long (default: one hour — far longer than a real extraction/embedding call takes) back to `'pending'`, so it re-enters the normal claim rotation. `minion.py` calls it unconditionally whenever a database is configured. `queue_lag_summary()` also reports `running_count`/`oldest_running_age_s` per lane — added at the same time as the reclaim logic, but not actually surfaced in `/status deep`'s printed output until R2-GAP-014 below wired it in; the data existed, nothing displayed it.

**Fuller `/status deep` visibility (R2-GAP-014).** Two additions: `queue_lag_summary()` now also reports `failed_count` per lane (jobs that exhausted every retry — `attempts` reached `max_attempts` — and are just sitting there until `memory_retention` eventually prunes them); and `/status deep`'s per-agent block now actually prints a `capture_running=N (oldest ...)` / `capture_failed=N` style line (any lane, only when nonzero, so a healthy queue stays quiet) — before this fix, `running_count`/`oldest_running_age_s` existed in the underlying data since R2-GAP-003 but were never rendered anywhere. A new `SessionDB.embedding_coverage_summary(agent_id)` also reports, in one `count(*)`, how many of an agent's messages across *every* session are missing an embedding under the active model — a different question than reconciliation's own bounded per-session gap list (R2-GAP-011 above): not "what should I enqueue right now" but "how far behind is this agent's embedding coverage overall." Shown in `/status deep` only when nonzero.

### Exact job-coverage gap detection (R2-GAP-002 / R2-GAP-005)

`session_job_coverage`'s single high-water mark (above) fixed retention erasing coverage, but it could still only represent a *tail* gap — "nothing enqueued since message X." A sparse gap couldn't be detected: if message 1's enqueue failed but message 2's later succeeded, message 1 sat below the mark message 2 set and was permanently invisible to reconciliation. `find_uncovered_capture_range`/`find_uncovered_commitment_range` are now `find_uncovered_capture_ranges`/`find_uncovered_commitment_ranges` (plural — a list of gaps, not one tuple), and `session_job_coverage` was replaced (migration 006, dropped) by `session_job_coverage_ranges` — append-only, one row per range ever successfully enqueued, so gap detection can check every eligible message against every recorded range rather than a single cutoff. The walk-and-collapse algorithm (`compute_backfill_windows`, `session/db.py`) is the same one `memory/consolidation.py`'s one-off manual `backfill_agent` already used — promoted to a shared location instead of reconciliation duplicating it. `ReconciliationScheduler._catch_up_capture`/`_catch_up_commitment` now loop over every gap a pass finds, enqueuing one job each.

The message-embedding lane took a different, simpler fix: `find_uncovered_message_ids_for_embedding` now anti-joins directly against `message_embeddings` for the *currently active* `embedding_model_identity`, instead of reading any coverage table at all. `message_embeddings` is itself durable (never pruned) and already keyed by `(message_id, model_identity)`, and "attempted but produced zero results" isn't a possible outcome for an embedding the way it is for extraction — a message either has a vector for the active model or it doesn't. This is simultaneously exact (fixes the same sparse-gap problem for this lane) and *model-aware* (R2-GAP-005): switching the configured embedding model makes every historical message re-appear as uncovered for the new model, so periodic reconciliation becomes the automatic backfill mechanism after a model switch — no separate manual "rebuild" command needed.

**Bounded per-session scan (R2-GAP-011).** All three gap-finding queries above now cap themselves to a session's `_RECONCILIATION_SCAN_LIMIT` (2000) most-recent eligible messages, not its entire history — without a bound, a lifelong session with a very large message count would have that whole history re-scanned every reconciliation pass, almost entirely to re-confirm coverage for old messages that never changes. A genuine gap is overwhelmingly likely to appear near the tail (a turn that just happened failed to enqueue) and reconciliation runs frequently, so this trades "detect an arbitrarily old gap immediately" for "keep every pass's cost bounded regardless of session size" — a deliberate, accepted trade-off, not a claim that every theoretical gap is still caught (a gap from a genuine reconciliation *outage* stretching past the scan window could still be missed). `memory/consolidation.py`'s manual one-off `backfill_agent` deliberately stays fully unbounded (`list_message_ids`, not this limit) — it's a rare, explicit operation, not something run automatically on a timer.

### Migration safety and Matrix room-binding races (R2-GAP-008 / R2-GAP-010)

Two narrower concurrency fixes, unrelated to each other beyond both being races between two things happening at once:

- **`schema_migrations.run_migrations()`** now wraps its whole body (ledger creation, reading applied versions, applying new ones) in a PostgreSQL session-level advisory lock, `pg_advisory_lock(hashtext(component)::bigint)`, released in a `finally`. Without it, two processes starting at once (the bot and a maintenance CLI invocation, or two bot instances mid-deployment) could both observe a version as missing and both try to apply and record it — safe in practice only because every migration's own SQL is independently idempotent, but this lock removes the need to rely on that as the *only* protection. A second caller simply blocks until the first finishes, then finds nothing left to do.
- **`MatrixRoomSessionManager.get_or_create_session_id()`** had a plain select-then-insert — two concurrent first messages for the same `(room_id, agent_id)` (e.g. two events arriving together right after joining a room) could both see no existing binding and both attempt an `INSERT`, one raising a `PRIMARY KEY` violation. Now always attempts `INSERT ... ON CONFLICT (room_id, agent_id) DO NOTHING` first, then re-selects regardless of which caller's insert (if any) actually won — every concurrent caller ends up agreeing on the same `session_id`.

### Resumable session deletion (R2-GAP-007)

`/delete-session` deletes a session's JSONL file, then its `SessionDB` rows (messages, jobs, proposals, commitments), then — if any proposals were found — their indexed evidence in `PostgresMemoryIndex` (via `MemoryService.forget_proposals`). Three independent operations against three stores, none of which can share a transaction. Before this fix, a failure partway through (a transient DB connection drop, say) left no record of what had actually completed, and re-running `/delete-session` on the same target didn't work as a retry anyway — the JSONL file is already gone by the time a later phase could fail, so the target no longer appears in any listing.

`deletion_tombstones` (one row per `(agent_id, session_id)` deletion attempt) fixes both problems: `start_deletion_tombstone()` records intent before anything is touched, and each phase (`mark_deletion_jsonl_done`/`mark_deletion_db_done`/`mark_deletion_evidence_done`) marks itself complete as it finishes — `mark_deletion_db_done` also snapshots the exact `proposal_ids` `delete_session()` returned, since by the time an evidence-cleanup retry might need them, the `memory_proposals` rows they refer to are already gone. A partial failure now reports a pointer to the retry path instead of a bare warning:

```bash
minion-assist memory verify-deletions          # list every incomplete deletion and which phases remain
minion-assist memory verify-deletions --retry  # finish their remaining phases (db cleanup, then evidence cleanup)
```

`--retry` never attempts the JSONL phase itself (this CLI command has no `ShortTermMemory` wiring, and by construction a tombstone only ever reaches this command with that phase already done, or genuinely stuck — which needs manual attention, not an automated retry). Tombstone rows are never deleted, even once every phase completes — the durable record is itself the proof a session was fully, deliberately removed.

### Import quarantine and review (Stage One Phase 7, slice E)

Task 6: "Add import quarantine and review before imported claims enter shared durable memory." The quarantine half already existed since Phase 1 (`memory/files.py`'s `memory/imports/` — searchable via `--corpus import`, never auto-promoted). This slice adds the missing review half: a human-triggered way to turn quarantined content into a durable topic note with claim markers, mirroring `MemoryConsolidator`'s preview/approve/reject pattern (Phase 5) for capture-job proposals. The real, current motivating case: `memory/extractor.py`'s background daemon still writes a rolling `_auto_extracted` note into quarantine every turn by default (`"memory": {"enable_extraction": true}`) — a legacy path with no review mechanism until now.

**Why a separate `ImportReviewer`, not an extension of `MemoryConsolidator`.** `MemoryConsolidator`'s preview/approve/rollback tables (`memory_consolidation_previews`, `memory_topic_revisions`) are typed around an *integer* `proposal_id`. An import is a *string*-keyed file (e.g. `"_auto_extracted"`), often a whole rolling note of many distinct facts rather than one atomic claim. `memory/import_review.py`'s `ImportReviewer` is a parallel, structurally similar pipeline instead — its own `memory_import_previews` table, evidence tagged `import:KEY` instead of `proposal:ID` — the same "separate table rather than force a shared schema" call Phase 6 already made for commitments vs. capture jobs. `memory/consolidation.py`'s merge-target search (`find_merge_target`) and drafting-response parser (`parse_draft_response`) were promoted from private methods/functions to shared public ones so both pipelines use the exact same logic rather than duplicating it.

**What happens to a reviewed import.** Both `approve()` and `reject()` delete the quarantined import file once reviewed (`MemoryFileRepository.delete_import` + `PostgresMemoryIndex.remove_file`) — approved content now lives in the topic note it was promoted into; rejected content is discarded outright, with no archive. This keeps a human's review queue from re-offering the same stale content on every future preview.

**No rollback in this slice.** Unlike `MemoryConsolidator`, `approve()` here does not snapshot the merge target's prior content. Task 6's acceptance criteria don't require it, and a human who dislikes what got promoted can just edit the resulting topic note directly.

**Multiple claims per import.** A capture-job proposal is already one atomic claim, so `MemoryConsolidator`'s prompt only ever attaches one claim marker. Import content is routinely a whole rolling note of many facts, so `preview()` pre-generates a bounded pool of 10 new claim ids and lets the model attach one per genuinely new claim it decides to keep.

**Untrusted-content framing.** Quarantined content was never reviewed by a human — the drafting prompt (`_IMPORT_DRAFT_SYSTEM`) explicitly instructs the model to treat it strictly as reference material to evaluate, never as instructions to follow, the same posture `memory/commitments.py`'s `format_due_commitments_block` already takes with untrusted due-commitment text.

New CLI commands:

```bash
minion-assist memory import list --agent main
minion-assist memory import preview _auto_extracted --agent main
minion-assist memory import explain 5 --agent main
minion-assist memory import approve 5 --agent main
minion-assist memory import reject _auto_extracted --agent main --reason "nothing useful"
```

### Forgetting a source (Stage One Phase 7, slice F)

Phase 7's last acceptance criterion: "forgetting a source identifies and removes or re-evaluates all derivatives." A "source" is an evidence citation — the same `(kind, ref)` pairs a claim marker's `evidence=` field encodes, e.g. `("proposal", "42")` or `("import", "_auto_extracted")`. `memory/forgetting.py`'s `forget_source(index, files, agent_id, source_kind, source_ref)` finds every claim citing that source (`PostgresMemoryIndex.list_claims_citing_evidence`) and edits each affected page's claim marker directly.

**Why the edit has to happen in the file, not just Postgres.** `kb_claims`/`kb_evidence` are a fully derived cache — `force_rebuild_agent()` can always re-derive them from scratch by re-parsing every page. A "forget" that only touched those tables would be silently reverted the next time the affected page gets reindexed. `memory/knowledge.py`'s new `remove_evidence_from_content(content, claim_id, source_kind, source_ref)` rewrites the marker's `evidence=` field directly in the page's raw text (removing just the one citation, preserving every other field — id, confidence, observed, supersedes, contradicts, etc. — exactly as given), and `forget_source()` writes that back via `MemoryFileRepository.remember()` + `reindex_file()`, the same write-then-reindex shape every other mutation in this project uses.

**Evidence-less claims are re-flagged, not deleted.** If removing a citation leaves a claim with no evidence at all, its marker's `status` flips to `unknown` — the marker and its text stay in the note, nothing is silently lost. This is the recommended, chosen behavior (a genuine fork — "removes or re-evaluates" per the plan) — it matches the same "surface gaps, don't hide them" posture the rest of Phase 7 already takes (dangling contradictions, provenance gaps). A claim that still has *other* evidence after the forgotten one is removed keeps its status untouched.

**`MEMORY.md` is skipped, not edited.** A claim can technically live in `MEMORY.md` (it's `source_kind="durable"` too), but nothing in this project ever programmatically writes `MEMORY.md` — it's human/bootstrap-owned everywhere else. Claims found there are left untouched and reported back under `skipped_manual_review` for a human to handle by hand, rather than breaking that invariant.

**No preview step, no rollback.** Mirrors `reject()`'s precedent in both `consolidate` and `import` — forgetting is a deliberate, explicitly-named cleanup action a human types directly, not a draft awaiting review.

**"Deletion coverage" dashboard** (deferred from slice C) is now built: `PostgresMemoryIndex.list_claims_needing_reevaluation(agent_id)` lists claims with `status="unknown"` AND zero evidence — exactly the signature `forget_source()` leaves behind, the audit trail for "did forgetting correctly leave things in a re-evaluate-me state." Added as a 7th `minion-assist memory knowledge dashboard` section.

New CLI command:

```bash
minion-assist memory knowledge forget --agent main --source-kind proposal --source-ref 42
minion-assist memory knowledge dashboard --agent main --section deletion-coverage
```

This completes Stage One Phase 7 (the plan's optional personal knowledge layer) — all six slices (A: schema/markers/sync, B: provenance/relationships, C: dashboards, D: compiled digest, E: import review, F: forgetting) are now implemented.

### `session_search` Tool Modes

Every mode is scoped to the calling agent's own sessions only (MEM-GAP-002)
— `SessionSearchTool` is constructed per-agent with that agent's `agent_id`,
and every underlying `SessionDB` query filters on it. An agent can search
across all of *its own* rooms/topics, but can never see another agent's
sessions, even a guessed session_id from a different agent — the query
simply returns nothing for it, the same as a session_id that doesn't exist.

| Mode | Description |
|---|---|
| `DISCOVER` | Lexical + vector fused query across this agent's own sessions (MEM-GAP-006) — semantic (paraphrase) matches on top of FTS when an embedding provider is configured; FTS-only otherwise. Returns ranked matches with a snippet, ±3 message context window, and session bookends (first/last messages). FTS lane supports AND (default), OR, `"quoted phrase"`, `-exclude`, `prefix*`. |
| `SCROLL` | Read messages around a specific message ID in one of this agent's own sessions. Accepts `anchor_message_id` (0 = end of session) and `window` (default 5, max 20). |
| `BROWSE` | List this agent's own 20 most recent sessions with title, turn count, age, and first-message preview. |

### Data directory

When using `docker-compose.yml`, PostgreSQL data is persisted to `../data/` (relative to the compose file), which maps to `E:\AI\Projects\OpenMinds\Minions\Minion-Assist\data\` on this machine. The container is set to `restart: unless-stopped` so it starts automatically with Docker Desktop.

### Graceful degradation

If the database is unavailable at startup, minion-assist prints a warning and continues in file-only mode — `session_search` is not registered and `CaptureWorker`/`MessageEmbeddingWorker` are never started. Fact extraction still happens via the original per-turn daemon thread (`extract_and_save_async`). No existing functionality is affected.

---

## Module Reference

### `config`

**File:** `src/minion_assist/config.py`

Loads `config.json` and `.env` at import time. Exposes four module-level values:

```python
from minion_assist.config import agents, workspace, streaming, compaction, memory, bootstrap, extra_plugin_manifests

agents                  # dict[str, AgentModelConfig] — one entry per agent in config.json
workspace               # Path — resolved workspace directory
streaming               # StreamingConfig — whether to stream in each execution mode
compaction              # CompactionConfig — shared token reservation (context_window is per-agent on ModelConfig)
memory                  # MemoryConfig — controls background fact extraction
bootstrap               # BootstrapConfig — workspace bootstrap file injection settings
extra_plugin_manifests  # tuple[str, ...] — extra plugins.json paths beyond the two fixed locations
```

**Types:**

```python
@dataclass(frozen=True)
class ProviderConfig:
    name: str       # e.g. "lmstudio"
    base_url: str
    api_key: str    # resolved from environment
    api: str        # e.g. "openai-completions"

@dataclass(frozen=True)
class ModelConfig:
    id: str              # e.g. "qwen-qwen3.5-9b"
    context_window: int  # total token capacity; used as compaction budget for this agent
    max_output_tokens: int  # generation limit sent to the API as max_tokens
    input_modalities: tuple[str, ...] = ("text",)  # e.g. ("text", "image") for vision models

@dataclass(frozen=True)
class AgentModelConfig:
    provider: ProviderConfig
    model: ModelConfig
    route_prefix: str | None = None   # "/research", "/code", etc.; None = default fallback

@dataclass(frozen=True)
class StreamingConfig:
    chat_mode: bool   # stream in the interactive REPL?
    task_mode: bool   # stream in programmatic/task use?

@dataclass(frozen=True)
class CompactionConfig:
    preserve_tokens: int | None = None
    # None  → minion.py auto-computes as max_output_tokens + _SNIP_SAFETY_BUFFER (1 024)
    # int   → explicit override, clamped to [2 000, context_window ÷ 2] at runtime
    # context_window is per-agent, stored in ModelConfig.context_window

@dataclass(frozen=True)
class MemoryConfig:
    enable_extraction: bool = True
    # True  → background fact extraction fires after each turn (default)
    # False → skip the extra API call; useful for expensive models

@dataclass(frozen=True)
class BootstrapConfig:
    enabled: bool = True          # False → disable bootstrap injection entirely
    path: str | None = None       # None → Path.cwd() at call time
    max_chars: int = 20_000       # per-file char cap (truncated with head+tail)
    total_max_chars: int = 60_000 # cumulative cap across all bootstrap files
    truncation_warning: str = "always"  # "always" | "once" | "off"
```

API key resolution: set `{PROVIDER_NAME_UPPERCASE}_API_KEY` in `.env`. Inline `apiKey` fields in `config.json` are not supported — the validator flags them to prevent accidental commits.

**Validation errors** are collected in one pass and reported together:

```
Invalid config.json:
  agents.main.model: Unknown provider 'lmstdio'. Did you mean 'lmstudio'?
  streaming.chat_mode: Expected boolean (true/false), got 'true'.
```

`ConfigError.issues` is a `list[ConfigIssue]` where each `ConfigIssue` has a `path` and a `message`.

---

### `bootstrap`

**File:** `src/minion_assist/bootstrap.py`

The workspace bootstrap prompt layer discovers recognized Markdown files under the configured root and injects them into every agent's system prompt as a `# Project Context` block.

**Mtime cache:** file contents are cached in process memory and re-read from disk only when a file's modification time changes.  This avoids reading and concatenating ~60 K chars on every agent turn when workspace files are unchanged.  Call `clear_bootstrap_block_cache()` to force a rebuild (e.g. in tests).

**Recognized files** (injected in this order, `HEARTBEAT.md` excluded):

| File | Purpose |
|------|---------|
| `AGENTS.md` | Agent behaviour rules and routing guidance |
| `SOUL.md` | Workspace-level personality or persona overlay |
| `TOOLS.md` | Tool constraints and usage hints for this workspace |
| `IDENTITY.md` | Project-specific identity context |
| `USER.md` | User preferences and context for this workspace |
| `BOOTSTRAP.md` | First-run workflow — triggers bootstrap-pending guidance |
| `MEMORY.md` | Project memory notes (may be routed through memory tools in future) |

**Public API:**

```python
from minion_assist.bootstrap import build_bootstrap_prompt_block, load_bootstrap_files, clear_bootstrap_block_cache

# Typical usage (as a per-turn callable in AgentSession):
block = build_bootstrap_prompt_block(Path.cwd(), bootstrap_cfg)

# Lower-level helpers:
files    = load_bootstrap_files(root)              # BootstrapFile list (includes missing markers)
ctx      = build_bootstrap_context_files(files, max_chars=20_000, total_max_chars=60_000)
rendered = render_project_context(ctx)             # "# Project Context\n\n## AGENTS.md\n..."
pending  = render_bootstrap_pending_context(ctx)   # non-empty only when BOOTSTRAP.md is present
warning  = render_truncation_warning(ctx, "always")
```

**Security:** every candidate file path is resolved and checked to be inside the bootstrap root before reading.  Symlink escapes and `..` traversal are rejected.  Raw reads are capped at 2 MB before decoding.

**Budget enforcement:** the effective limit for each file is `min(max_chars, remaining_total_budget)`.  Files that exceed the limit are truncated with a visible head+tail marker:

```
[...truncated, read AGENTS.md for full content...]
```

---

### `providers`

**Directory:** `src/minion_assist/providers/`

Adapts different LLM APIs to a single protocol.

#### Protocol

```python
class LLMProvider(Protocol):
    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        on_token: Callable[[str], None] | None = None,  # pass to enable streaming
    ) -> LLMResponse: ...
```

When `on_token` is provided, the provider calls it with each text fragment as it arrives. When `None`, a single blocking request is made.

#### Response types

```python
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict   # already parsed from JSON
    error: str | None = None   # set when argument JSON was malformed

@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = []
    finish_reason: str = "stop"   # "stop" | "tool_calls" | "length"
    was_streamed: bool = False    # True if text was already printed via on_token
```

`finish_reason == "length"` means the model hit `max_tokens` mid-generation. The runner detects this and injects a continuation prompt automatically.

#### Factory

```python
from minion_assist.providers import create_provider

provider = create_provider(
    api="openai-completions",   # or "lmstudio" / "anthropic"
    base_url="http://...",
    api_key="sk-...",
    model="model-id",
)
```

| `api` value | Implementation | Notes |
|---|---|---|
| `openai-completions` | `OpenAICompatibleProvider` | Any OpenAI-compatible endpoint |
| `openai-responses` | `OpenAICompatibleProvider` | Alias for `openai-completions` (same implementation) |
| `lmstudio` | `LMStudioProvider` | Alias for `OpenAICompatibleProvider` |
| `anthropic` | `AnthropicProvider` | Requires `anthropic` package: `uv add anthropic` |
| `codex` | `CodexProvider` | Requires `codex` CLI (`npm install -g @openai/codex`) + `codex-login` |
| _(anything else)_ | `OpenAICompatibleProvider` | Fallback |

---

### `agents`

**Directory:** `src/minion_assist/agents/`

#### Definitions (`definitions.py`)

```python
@dataclass
class AgentConfig:
    name: str             # display name shown in the terminal
    soul: str             # system prompt defining personality and rules
    max_tool_rounds: int  # max TAO-loop iterations per turn (default 10)

AGENTS: dict[str, AgentConfig]
# Keys: "main" (Ada, max_tool_rounds=20), "researcher" (Elizabeth, max_tool_rounds=15)
```

`max_tool_rounds` lets task-focused agents run more tool-call iterations per turn (complex multi-step work) while keeping conversational agents snappy (default 10).

Both agents have `_TASK_SOUL_SUFFIX` appended to their souls, which instructs them to call `read_task` at session start and `update_task` after each completed step.

#### Router (`router.py`)

```python
from minion_assist.agents.router import resolve

agent_id, message = resolve("/research find FastAPI benchmarks")
# → ("researcher", "find FastAPI benchmarks")

agent_id, message = resolve("what is dependency injection?")
# → ("main", "what is dependency injection?")
```

Routing is **config-driven** — rules are read from `config.json` at startup. Each agent with a `"route_prefix"` gets an entry; the agent without one is the default fallback. Routes are sorted longest-first to prevent prefix shadowing.

#### Session (`session.py`)

```python
from minion_assist.agents import AgentSession

session = AgentSession(
    agent_id="main",
    agent=AGENTS["main"],
    provider=provider,
    max_output_tokens=32768,
    tools=registry,
    compactor=compactor,
    short_term=short_term,
    session_store=session_store,
    soul_suffix="",              # optional skills block appended each turn
    memory=memory_service,       # enables memory injection and background extraction
    enable_memory_extraction=True,  # False to suppress the background extraction API call
)

# Headless (returns text, no output)
text = session.send("What is REST?")

# With event callback
events = []
text = session.send("What is REST?", on_event=events.append, stream=True)

# With image attachments (model must support "image" in input_modalities)
text = session.send("What is in this image?", attachments=[Path("/tmp/screenshot.png")])

# With verification loop (IMP-17) — called after turns that used write tools
text = session.send("Refactor foo.py", verify_fn=lambda: subprocess.run(["pytest"], capture_output=True).stdout.decode())

# Fork history to a new session (NEW-04)
session.fork("backup")   # copies history to "backup" agent_id

# Export transcript (NEW-04)
md = session.export(format="md")   # "md" (default) or "html"
```

When `memory` is provided, `AgentSession`:
- Searches memory before each turn (via `build_prompt_section()` — see "Prompt injection" under the `memory` module reference below) and injects the top-5 matching snippets as a `<relevant_memories>` block, bounded by a real per-agent token budget computed from the model's context window (~0.77%, floor 100 / ceiling 8 000 tokens) — not a char-count heuristic.
- Fires background fact extraction after each successful turn — enqueues a durable `memory_capture_jobs` row for `CaptureWorker` if a database is configured, otherwise falls back to a fire-and-forget daemon thread (never blocks the REPL either way). Can be disabled via `enable_memory_extraction=False` (or `config.json` `"memory": {"enable_extraction": false}`).

(Stable user-profile injection — formerly a separate `<user_context>` block loaded once at init from a `user_context` note — is handled by `bootstrap.py`'s live `USER.md` mechanism instead; see the `memory` module reference below.)

**`session.reload()`** — reloads conversation history from disk, replacing in-memory state. Called automatically by the `/switch` command to ensure the history is current after switching agents. Does not affect long-term memory, task files, or session metadata.

**`session.reset()`** — clears in-memory and persisted conversation history (used by `/new`). Does not affect long-term memory or task files.

**`session.session_id`** — read-only property exposing the current session UUID.

**`session.switch_session(session_id)`** — load a different session's history by UUID, making it the active session. Updates both in-memory state and the session store. Used by `/session <N>` to restore a past conversation. Returns the number of messages loaded.

#### Runner (`runner.py`)

```python
from minion_assist.agents.runner import run_turn

usage = run_turn(
    provider,              # LLMProvider
    agent_name,            # str — used in event agent_name fields
    system,                # str — system prompt / soul
    max_tokens,            # int
    tools,                 # ToolRegistry
    messages,              # list[dict] — mutated in place
    on_event=None,         # Callable[[object], None] | None
    stream=False,          # bool — True to emit streaming token events
    max_tool_rounds=10,    # int — per-agent cap on TAO iterations
)
# Returns TokenUsage(input_tokens, output_tokens) summed across all LLM calls
# in the turn, or None when the provider does not report usage.
```

Implements the TAO loop with three reliability features:

- **Retry:** every `provider.chat()` call retries up to 3 times on transient errors (429, 5xx, network timeouts) with 2 s/4 s/8 s backoff plus jitter. Permanent errors (400, 401) are not retried.
- **Empty response recovery:** if the model returns nothing, a nudge message is injected and the loop continues (unless on the final round).
- **Length recovery:** if `finish_reason == "length"`, a continuation prompt is injected and the loop continues once.

---

### `tools`

**Directory:** `src/minion_assist/tools/`

#### Base types (`base.py`)

```python
@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: dict   # JSON Schema object

class Tool(ABC):
    @property
    @abstractmethod
    def schema(self) -> ToolSchema: ...

    @abstractmethod
    def execute(self, **kwargs: object) -> str: ...
```

All tool output is a plain string. Errors are returned as strings (never raised) so the agent can read and react to them without crashing the loop.

#### Registry (`registry.py`)

```python
registry = ToolRegistry()
registry.register(MyTool())

registry.definitions   # list[dict] — OpenAI-format tool specs
registry.execute("tool_name", {"arg": "value"})  # str
```

`definitions` returns the OpenAI `tools` array ready to pass to any provider. Registering a tool with a name that already exists overwrites it silently.

**Hook support:** plugins can attach callbacks that run before or after every tool execution:

```python
from minion_assist.agents.events import ToolPreExecuteHookEvent, ToolPostExecuteHookEvent

registry.add_before_hook(lambda event: print(f"→ {event.name}({event.arguments})"))
registry.add_after_hook(lambda event: print(f"← {event.name} in {event.elapsed_ms}ms"))
```

`add_before_hook(fn)` / `add_after_hook(fn)` accept any callable that receives a `ToolPreExecuteHookEvent` or `ToolPostExecuteHookEvent` dataclass. Multiple hooks can be registered.

**Unregistration** (for MCP hot-reload):

```python
registry.unregister("mcp__playwright__browser_navigate")   # remove one tool by name
registry.unregister_prefix("mcp__playwright__")            # remove all tools for a server
```

#### Built-in tools

| Tool class | Name | Description |
|---|---|---|
| `ReadTool` | `read` | Read a file (numbered lines, optional `offset`/`limit`) or list a directory. Rejects binary files and caps at 50 KB. Paths outside the workspace root are rejected. |
| `WriteTool` | `write` | Write content to a file, creating parent directories as needed. Paths outside the workspace root are rejected. |
| `GlobTool` | `glob` | Find files matching a glob pattern, sorted newest-first. Skips `.git`/`.venv`/`__pycache__`/etc. Caps at 200 results. |
| `BashTool` | `bash` | Run a shell command — PowerShell on Windows, bash on Unix. Calls the injected `confirm` callable before executing; pass `None` to skip confirmation. |
| `SaveMemoryTool` | `save_memory` | Save a Markdown note to memory under a given key. Blocked by `read_only_mode`. |
| `WriteDailyMemoryTool` | `write_daily_memory` | Append a quick timestamped bullet to today's daily log (`memory/YYYY-MM-DD.md`). No key needed — great for ephemeral observations. Blocked by `read_only_mode`. |
| `SearchMemoryTool` | `search_memory` | Keyword search across memory. Results ranked by term frequency and recency. Capped at 20. `is_read_only=True`. |
| `MemoryGetTool` | `memory_get` | Read an exact, bounded slice of a memory file by path (optional `from_line`/`lines`) — a targeted follow-up read, not a search. `is_read_only=True`. |
| `PinMemoryTool` | `pin_memory` | Pin/unpin a saved note (`key`, `pinned: bool`) so it's always surfaced by the pinned fusion lane, regardless of query match. Registered only when a database is configured. Blocked by `read_only_mode`. |
| `SkillTool` | `skill` | Load a skill's instructions into context by name. Only registered when skills are discovered at startup. |
| `ReadTaskTool` | `read_task` | Read the current task progress file — goal, steps, status, notes, and context. |
| `UpdateTaskTool` | `update_task` | Create a new task (goal + steps) or update an existing one (step status, notes, context, or clear). |
| `EditTool` | `edit` | Edit a file by replacing an exact string match. Requires the `old_string` to appear exactly once in the file (unless `replace_all=True`). Paths outside the workspace root or sensitive system paths are rejected. |
| `GrepTool` | `grep` | Search files for a regex pattern and return matching lines with filename, line number, and optional context lines. Supports include glob filter, case-insensitive mode, and truncation at `max_results`. `is_read_only=True`. |
| `PatchPreviewTool` | `patch_preview` | Preview what an `edit` would produce as a unified diff, without applying it. Same `path`/`old_string`/`new_string`/`replace_all` parameters as `EditTool`. Never writes to disk. `is_read_only=True`. |
| `ApplyPatchTool` | `apply_patch` | Apply a unified diff patch string to one or more files using `git apply`. Supports `check_only=True` for a dry run. Blocked by `read_only_mode`. |
| `FindDefinitionTool` | `find_definition` | Search `.py` files in the workspace for the definition of a named symbol (function, class, or variable) using Python's `ast` module. Returns `path:lineno: snippet` lines. `is_read_only=True`. |
| `TodoWriteTool` | `todo_write` | Replace the current session todo list with a new array of items. Pass an empty list to clear. Blocked by `read_only_mode`. |
| `TodoReadTool` | `todo_read` | Read the current session todo list as a numbered list. `is_read_only=True`. |
| `BrowserTool` | `browser` | Control a Playwright Chromium browser. Seven actions: **start** (launch headed/headless or attach to existing Chrome via CDP port), **navigate** (go to URL, wait for DOMContentLoaded), **evaluate** (run arbitrary JavaScript in the page, returns JSON — the agent uses its full DOM API knowledge directly), **screenshot** (capture viewport as PNG, returns file path for vision), **pick** (inject interactive element picker — user hovers/clicks to select, double-clicks to confirm; returns tag/id/class/text/html/parents for each element), **cookies** (dump all cookies from the page context including HTTP-only), **stop** (close browser and free resources). Playwright is an optional dependency (`uv sync --extra browser`). |
| `WebFetchTool` | `web_fetch` | Fetch a URL and return its text content. Strips HTML tags (skips `<script>`, `<style>`, `<head>`), collapses whitespace, and truncates at `max_chars` (default 8 000). Blocks SSRF targets (AWS metadata, GCP metadata). `is_read_only=True`. |
| `WebSearchTool` | `web_search` | Search the web via DuckDuckGo. Returns numbered results with title, URL, and snippet. No API key required. Requires `ddgs` (already in `pyproject.toml`). Parameters: `query` (required), `max_results` (1–10, default 5), `region` (e.g. `"us-en"`, optional). `is_read_only=True` — concurrent batching supported. Query string is checked against SSRF markers when a policy is injected. |
| `AskUserTool` | `ask_user` | Pause the agent and ask the human operator a question. Returns the human's typed response. When no `ask_user_fn` is provided (headless mode), returns an error instructing the agent to proceed without input. |
| `SpawnSubagentTool` | `spawn_subagent` | Delegate a self-contained task to a child subagent and return its text response. The subagent runs in a daemon thread with a read-only registry (no write tools). Enforces depth (max 4) and child-count (max 5) limits. Parameters: `task` (required), `agent_id` (default: `"researcher"`), `timeout_seconds` (default: 120). Subagent events are relayed to the parent's terminal with a `[sub:{agent_id}]` prefix. |
| `GitStatusTool` | `git_status` | Show git working-tree status (`git status --short --branch`): branch name, staged, modified, and untracked files. `is_read_only=True`. |
| `GitDiffTool` | `git_diff` | Show a unified diff of changes. Optional `staged=true` for staged changes; optional `path` to limit to a file. `is_read_only=True`. |
| `GitCommitTool` | `git_commit` | Stage files (optional `files` list) and create a git commit with the given `message`. Calls the `bash_confirm` callback before executing, same as `BashTool`. |
| `SessionSearchTool` | `session_search` | Search, scroll, or browse past conversation sessions stored in PostgreSQL, scoped to the calling agent's own sessions only (MEM-GAP-002). Three modes: **DISCOVER** (lexical + vector fused across this agent's own sessions — FTS supports quoted phrases, `-exclude`, `prefix*`; also finds paraphrased matches when an embedding provider is configured, MEM-GAP-006), **SCROLL** (paginate within one of this agent's own sessions by message ID), **BROWSE** (list this agent's own recent sessions). Only registered when a `database.url` is configured *and* an `agent_id` is known. `is_read_only=True`. |
| `McpStatusTool` | `mcp_status` | List all configured MCP servers and their connection status. |
| `ListMcpResourcesTool` | `list_mcp_resources` | List resources available on a connected MCP server. |
| `ReadMcpResourceTool` | `read_mcp_resource` | Read a specific resource from a connected MCP server. Output capped at 8 000 chars. |
| `ListMcpPromptsTool` | `list_mcp_prompts` | List prompt templates available on connected MCP servers. Shows each prompt's name, description, and arguments. |
| `GetMcpPromptTool` | `get_mcp_prompt` | Retrieve and render a named prompt template from an MCP server, optionally passing argument values. Returns the filled prompt text. |
| `McpToolAdapter` | `mcp__<server>__<tool>` | Dynamically registered tool that proxies a call to a specific tool on an MCP server. One adapter is created per tool exposed by each connected MCP server. |

**`ReadTaskTool` / `UpdateTaskTool`** implement the **Ralph Loop** pattern for long-running tasks that span multiple sessions or context windows. The agent calls `read_task` at the start of each session to orient itself, and `update_task` after completing each step so progress survives any restart. The task file is stored at `{workspace}/tasks/{agent_id}.json` — outside the project workspace so file tools cannot accidentally modify it.

**`ReadTool` parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | string | required | File or directory path |
| `offset` | integer | 1 | First line to return (1-indexed) |
| `limit` | integer | 200 | Maximum lines to return; also hard-capped at 50 KB |

**`UpdateTaskTool` parameters:**

| Parameter | Type | Description |
|---|---|---|
| `goal` | string | High-level objective. Provide with `steps` to create a new task. |
| `steps` | array of strings | Step descriptions for a new task. |
| `step_id` | integer | 1-indexed ID of the step to update. |
| `status` | string | New status: `pending`, `in_progress`, `done`, or `blocked`. |
| `notes` | string | Notes to attach to a step (stored persistently). |
| `context` | string | Key facts to preserve across sessions (replaces previous context). |
| `clear` | boolean | Delete the task file when the task is complete. |

#### `default_registry()`

```python
from minion_assist.tools import default_registry
from minion_assist.memory import MemoryFileRepository, MemoryService
from minion_assist.skills import discover_skills
from pathlib import Path

# Minimal (4 tools: read, write, glob, bash)
reg = default_registry()

# With workspace sandboxing and bash confirmation
reg = default_registry(
    root=Path.cwd(),
    bash_confirm=lambda cmd: input(f"Run {cmd}? [y/N]: ") == "y",
)

# With memory and task tools (8 tools total)
reg = default_registry(
    memory=MemoryService(MemoryFileRepository(agent_workspace_path)),
    root=Path.cwd(),
    tasks_dir=Path("~/.minion-assist/tasks").expanduser(),
    agent_id="main",
)

# With MCP servers
reg = default_registry(
    mcp_manager=mcp_manager,  # McpManager instance loaded from config
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `memory` | `MemoryService \| None` | `None` | If provided, registers `save_memory`, `search_memory`, `memory_get`, and `write_daily_memory`. Also registers `pin_memory`, but only when `db` is also provided. |
| `root` | `Path \| None` | `None` | Workspace root — `read`/`write`/`glob`/`edit`/`grep` reject paths outside this boundary |
| `bash_confirm` | `Callable[[str], bool] \| None` | `None` | Simple bool callback called before every bash command; `None` = no confirmation |
| `bash_approval` | `Callable[[str], ApprovalDecision] \| None` | `None` | Rich 4-option approval callback (ALLOW_ONCE, ALLOW_SESSION, DENY, ALWAYS_DENY). Takes priority over `bash_confirm` when both are set. The CLI wires this to a menu that also records decisions to the policy's audit log. |
| `skills` | `SkillRegistry \| None` | `None` | If non-empty, registers the `skill` tool |
| `tasks_dir` | `Path \| None` | `None` | Task file directory. Required alongside `agent_id` to register task tools. |
| `agent_id` | `str \| None` | `None` | Agent ID used to build the task file path `{tasks_dir}/{agent_id}.json`. |
| `policy` | `PermissionPolicy \| None` | `None` | Safety rules injected into all I/O tools (`read`, `write`, `glob`, `bash`, `edit`, `grep`, `web_fetch`, `patch_preview`, `apply_patch`, `find_definition`, `todo_write`, git tools). Defaults to `PermissionPolicy.default(workspace=root)` when omitted. Also set as `registry.policy` for `/plan`/`/auto` toggling. |
| `mcp_manager` | `McpManager \| None` | `None` | If provided, registers `mcp_status`, `list_mcp_resources`, `read_mcp_resource`, `list_mcp_prompts`, `get_mcp_prompt`, and one `mcp__<server>__<tool>` adapter per tool exposed by connected MCP servers. |
| `ask_user_fn` | `Callable[[str], str] \| None` | `None` | Callback for `ask_user` tool. Called with the agent's question; returns the human's response. `None` = headless mode (tool returns an error instead of blocking). |
| `write_confirm` | `Callable[[str], bool] \| None` | `None` | Human approval callback passed to `WriteTool` and `EditTool`. Called with a one-line description before each write (e.g. `"Write 120 chars to /path/to/foo.py"`). Return `False` to cancel. `None` = automatic writes without prompting. |
| `db` | `SessionDB \| None` | `None` | If provided, registers `session_search` for FTS search across all historical sessions. Created by `minion.py` when `database.url` is configured. |

---

### `PermissionPolicy` (`tools/policy.py`)

`PermissionPolicy` is a centralised safety dataclass injected into all I/O tools. Rather than each tool implementing its own path/URL/command checks, they all delegate to one shared policy object.

```python
from minion_assist.tools.policy import PermissionPolicy
from pathlib import Path

# Default: workspace boundary + standard SSRF markers
policy = PermissionPolicy.default(workspace=Path.cwd())

# Custom: extra SSRF domains, no workspace boundary
policy = PermissionPolicy(ssrf_markers=frozenset({"internal.corp", "169.254.169.254"}))

# Enable read-only mode (also toggled by /plan and /auto slash commands)
policy.read_only_mode = True

# Checks (each returns None if allowed, or an error string if denied)
error = policy.check_path(Path("/etc/passwd"))              # path read check
error = policy.check_write(Path("/etc/passwd"))             # path write check + read_only_mode
error = policy.check_url("http://169.254.169.254/latest/")  # URL / SSRF check
error = policy.check_command("curl 169.254.169.254")        # command / SSRF + read_only check
```

`PermissionPolicy.default(workspace)` builds a policy that:
- Rejects file paths outside `workspace` (via `_within()` from `tools/base.py`)
- Rejects sensitive system paths (SSH keys, `.env`, credential files)
- Blocks known SSRF targets: AWS EC2 metadata (`169.254.169.254`), GCP metadata (`metadata.google.internal`), ECS credentials (`169.254.170.2`), IPv6 EC2 metadata (`fd00:ec2::254`)

`default_registry()` auto-creates a `PermissionPolicy.default(workspace=root)`, injects it into all I/O tools, and also sets `registry.policy = policy` so `/plan` and `/auto` can toggle `read_only_mode` at runtime.

| Method | Used by | Blocks when |
|---|---|---|
| `check_path(path)` | ReadTool, GlobTool, GrepTool, FindDefinitionTool | Path is sensitive or outside workspace |
| `check_write(path)` | WriteTool, EditTool, GitCommitTool | Same as `check_path` + `read_only_mode=True` |
| `check_url(url)` | WebFetchTool | URL contains an SSRF marker |
| `check_command(cmd)` | BashTool | Command contains an SSRF marker, or `read_only_mode=True` |

---

### Plugin System (`plugins.py`)

The plugin system lets you extend minion-assist with custom tools, registry hooks, and skills — no source edits required.

**Manifest format** (`~/.minion-assist/plugins.json` or `.minion-assist/plugins.json`):

```json
{
  "trust": "trusted",
  "tools": [
    "/home/user/my-tools/custom_tool.py",
    "~/plugins/another_tool.py"
  ],
  "hooks": [
    "./hooks/logging_hook.py"
  ],
  "skills": [
    "./custom-skills/"
  ]
}
```

**`"tools"` section** — paths to Python files that either export a `TOOLS` list (preferred) or contain auto-discoverable `Tool` subclasses:

```python
from minion_assist.tools.base import Tool, ToolSchema

class MyCustomTool(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="my_tool",
            description="Does something custom.",
            parameters={"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]},
        )

    def execute(self, **kwargs) -> str:
        return f"Custom result: {kwargs['input']}"

TOOLS = [MyCustomTool()]
```

**`"hooks"` section** — paths to Python files that expose `BEFORE_HOOKS` and/or `AFTER_HOOKS` lists:

```python
def log_before(event):   # event is ToolPreExecuteHookEvent
    print(f"→ {event.name}({event.arguments})")

def log_after(event):    # event is ToolPostExecuteHookEvent
    print(f"← {event.name} in {event.elapsed_ms}ms ({event.output_chars} chars)")

BEFORE_HOOKS = [log_before]
AFTER_HOOKS = [log_after]
```

**`"skills"` section** — directory paths scanned for `SKILL.md` files; discovered skills are merged into the runtime skill registry.

**`"commands"` section** — register custom slash commands available in the REPL:

```json
"commands": [
  {
    "name": "/my-command",
    "description": "Does something useful.",
    "handler": "./handlers/my_command.py"
  }
]
```

The `handler` path points to a Python file that must expose a `handle(ctx) -> CommandResult` function:

```python
from minion_assist.commands import CommandResult

def handle(ctx):
    return CommandResult(handled=True, message="Plugin command ran!")
```

Plugin commands appear in `/help` output under "Plugin commands:" and are dispatched by the same `dispatch_command()` mechanism as built-in commands.

**`"trust"` field** — `"trusted"` (default, your own code) or `"external"` (third-party code). External plugins print a warning at load time; no other runtime effect.

**`extra_plugin_manifests` config key** — register additional manifest files beyond the two fixed locations. Set in `config.json`:

```json
"extra_plugin_manifests": [
  "/absolute/path/to/custom-plugins.json",
  "~/shared/plugins.json"
]
```

**Priority:** project-local manifest (`.minion-assist/plugins.json`) loads after global (`~/.minion-assist/plugins.json`), so local tools override global ones with the same name.

**Security note:** plugin files are executed with `importlib`. Only add paths to files you trust.

---

### `skills`

**File:** `src/minion_assist/skills/__init__.py`

Discovers and loads agent skills from SKILL.md files on disk.

```python
from minion_assist.skills import discover_skills, format_skills_prompt

registry = discover_skills([
    Path("~/.minion-assist/skills").expanduser(),  # global (lower priority)
    Path(".minion-assist/skills"),                  # project (higher priority)
])
prompt_suffix = format_skills_prompt(registry)   # "" when no skills found
```

**Skill file format** (`~/.minion-assist/skills/my-skill/SKILL.md`):

```markdown
---
name: my-skill
description: One-line description shown to the agent in the system prompt.
---

# My Skill

Full instructions the agent receives when it calls `skill(name="my-skill")`.
```

The `name:` field is required. `description:` is optional but must be non-empty for the skill to appear in the system prompt listing.

---

### `memory`

**Directory:** `src/minion_assist/memory/`

#### Short-term (`short_term.py`)

Stores conversation history as JSONL files — one JSONL file per session under `{base_dir}/{agent_id}/{session_id}.jsonl`. Uses an atomic tmp-file swap on every `save()` so a crash mid-write never corrupts the existing history.

```python
from minion_assist.memory import ShortTermMemory

mem = ShortTermMemory(Path("~/.minion-assist/sessions").expanduser())
mem.load("main", session_id)                         # list[dict] — full history
mem.save("main", session_id, messages)               # atomic overwrite
mem.append("main", session_id, {"role": "user", "content": "hi"})  # efficient append
mem.clear("main", session_id)                        # delete history file
mem.get_name("main", session_id)                     # str | None — display name if set
mem.set_name("main", session_id, "Auth work")        # save a human-readable name
mem.list_sessions("main")                            # list[Path] — all session files, oldest-first
```

#### `MemoryService` (`service.py`, `files.py`, `models.py`) — Stage One Phase 1

**This is what `AgentSession` and the memory tools actually use at runtime today** (wired in as of Phase 1 slice 3 — `minion.py` builds one `MemoryService` per agent and passes it to `default_registry(memory=...)` and `AgentSession(memory=...)`). It targets the *merged* per-agent layout Stage One Phase 0's migration produces: `workspaces/{agent_id}/memory/{topics,imports}/` and dated `workspaces/{agent_id}/memory/YYYY-MM-DD.md` files, rather than the legacy flat `memory/{agent_id}/{key}.md` root.

```python
from minion_assist.memory import MemoryFileRepository, MemoryService

service = MemoryService(MemoryFileRepository(Path("~/.minion-assist/workspaces/main").expanduser()))
service.remember("project-goals", "# Goals\n...")   # -> memory/topics/project-goals.md
service.load("project-goals")                         # str | None
service.search("goals")                                # list[MemoryHit] (topic + import + daily, tagged by source)
service.append_daily("did a thing")                    # -> memory/YYYY-MM-DD.md, timestamped bullet
service.delete("project-goals")                         # bool

# Exact bounded read — no equivalent in the legacy LongTermMemory store.
service.get("memory/topics/project-goals.md", from_line=1, lines=20)   # MemoryExcerpt

# Quarantined, unreviewed notes — searchable but never auto-promoted (see
# docs/adr/0003-per-agent-memory-scope.md). Used by the background extractor.
service.remember_import("_auto_extracted", "fact one\nfact two")
service.load_import("_auto_extracted")                  # str | None
service.list_import_keys()                              # list[str]

service.status()   # MemoryStatus(topic_count=1, import_count=1, daily_count=1, ...)

# Pre-compaction flush (Stage One Phase 2, slice B): called by AgentSession
# right before Compactor.compact() summarizes and discards `head`. Pure text
# rendering (messages.format_message_excerpt) — no LLM call, so this can't
# fail the way summarization can, and adds no latency to the turn.
service.flush_head(head)   # FlushOutcome(status="flushed"|"empty"|"failed", detail="")
```

`search()` splits the query on whitespace, filters out stop-word candidates (terms shorter than 3 characters), and ranks results by term-match frequency. Notes matching more query terms rank above those matching fewer. Among ties, newer files rank slightly higher. Results are capped at 20 (`_SEARCH_MAX_RESULTS`). This scoring is unchanged from the legacy `LongTermMemory.search()` — Phase 1's goal was one canonical service with existing retrieval behavior, not better retrieval (that's Phase 3+). Two real fixes did land alongside the move: `remember()` now writes atomically (temp file + `os.replace`, so a crash mid-write can't corrupt a note), and `search()` now spans three sources (`memory/topics/`, `memory/imports/`, and dated daily files) instead of one flat directory.

One `MemoryService` per agent, each wrapping a repository rooted at that agent's own workspace directory, is the entire scope-enforcement story for now — Stage One's target design names several memory scopes (agent-private, user-shared, workspace, session-lineage, channel, import-quarantine), but only `agent-private` has a real caller today, so that's the only one enforced.

**Retired: the `user_context` reserved key.** `AgentSession` used to load a separate `user_context.md` note once at construction and inject it as a static `<user_context>` block. That duplicated `bootstrap.py`'s live `USER.md` injection once Phase 0 migrated `user_context.md` → `USER.md`, so it was removed in Phase 1 — write to `USER.md` directly (or via the `write`/`edit` tools) to give the agent persistent background about yourself; it's re-read live every turn, no restart needed.

#### Prompt injection (`agents/session.py`'s `build_prompt_section()`) — Stage One Phase 4, slice D

Replaces the old `_inject_relevant_memories()`: a real per-agent token budget (computed once at construction — see above — checked with the same estimator `context.py` uses for compaction, not a 4-chars-per-token guess), a citation (`path:start-end`) on any hit that came from the lexical/hybrid index, and a source label per snippet:

```
[durable] project-goals (memory/topics/project-goals.md:1-5): Ship it this quarter.
```

`AgentSession.send()` fires a `MemoryInjected` event (`keys`, `context_generation`, `token_count`) whenever the block is non-empty. This is purely observational, not a mechanism that changes what gets injected — verified by reading OpenClaw's own `plugins/memory-state.ts` (the design this phase is modeled on): it rebuilds its memory prompt section fresh on every call too, with no *cross-turn* dedup or suppression. The injected block here is never written into `AgentSession._history`, so a past turn's injection is never visible to the model on a later turn regardless of whether this turn re-injects the same content — skipping a still-relevant re-injection would only make the model lose context it needs *now*, not save it something it already has.

**Budget accounting and within-call dedup (MEM-GAP-014).** Two distinct fixes to how `build_prompt_section()` spends its budget, neither related to the cross-turn non-suppression above:
- **The returned `token_count` is now the fully rendered block's own cost, not just the sum of each included hit's individually-estimated cost.** The per-hit running total used during candidate selection never counted the `<relevant_memories>`/`</relevant_memories>` wrapper tags or the newlines joining lines together, so the actual assembled text could cost slightly more than the running total suggested — occasionally exceeding `max_tokens`. After candidate selection, the function now estimates the *actual* rendered text and trims from the end (lowest-ranked first) until it genuinely fits — a hard postcondition rather than an approximation.
- **Two hits with identical (stripped, 400-char-truncated) snippet text within the same call no longer both consume budget.** The second is skipped and the next-ranked candidate is tried instead, rather than wasting budget repeating the same information.

Both drop/trim decisions are logged at `DEBUG` (`minion_assist.agent_session`) for explainability, without changing `build_prompt_section()`'s return shape or `MemoryInjected`'s event shape.

`AgentSession._context_generation` (starts at 0) increments on `reset()` and after any successful compaction (both the automatic path in `send()` and the manual `/compact` command) — each `MemoryInjected` event carries the generation it fired in, so a consumer can tell "these were injected in the same stretch of history" from "context has since been reset/compacted." A forked session starts its own count at 0 rather than inheriting the parent's — `fork()` only writes history/session-store state to disk, and the child's `AgentSession` object (constructed later, when something switches to it) begins an independently tracked branch; the parent/child relationship is already preserved via `SessionInfo.parent_id`, so threading the counter through a disk round-trip would just duplicate that lineage tracking for no behavioral benefit.

#### Background extractor (`extractor.py`) and durable capture (`capture_worker.py`)

```python
from minion_assist.memory.extractor import extract_and_save_async, extract_facts

# Degraded mode (no database configured) — fire after a successful turn,
# returns immediately (daemon thread), fails silently.
extract_and_save_async(memory, provider, last_exchange)

# The shared primitive both paths call — does NOT catch provider exceptions,
# so CaptureWorker's own retry/backoff loop can see failures and reschedule.
facts: list[str] = extract_facts(provider, exchange)
```

Without a configured database, `AgentSession` fires `extract_and_save_async` in a daemon thread after each successful turn. It calls `extract_facts()` internally but swallows any exception — a best-effort, fire-and-forget note. Discovered facts (0–3 per turn, max 100 chars each) are appended to a rolling, quarantined `memory/imports/_auto_extracted.md` file (capped at 50 entries, via `MemoryService.remember_import`/`load_import`).

**Degraded-mode visibility (MEM-GAP-013).** Before this, a swallowed exception here — or the process exiting before the daemon thread finished (`daemon=True` means it's simply killed mid-extraction) — was genuinely invisible: logged at `DEBUG` only, no health tracking at all, unlike every database-backed worker in this codebase. Full recoverability would need a durable local queue, real over-engineering for a fallback path that mostly isn't even the active one (most deployments configure a database, making `CaptureWorker` below the one that actually runs) — so this adds the same optional `WorkerHealth` recording every other worker already has instead. `minion.py` constructs a `memory_extractor:{agent_id}` `WorkerHealth` per agent **only when no database is configured** (the inverse gate from `memory_search:{agent_id}`/`session_writes:{agent_id}` — this path literally cannot run with a database present), passed into `AgentSession` as `extraction_health` and threaded into `extract_and_save_async(..., health=...)`. Visible via `/status deep`'s matching per-agent row, same as every other worker.

With a configured database, `AgentSession` instead enqueues a `memory_capture_jobs` row and the standalone `CaptureWorker` thread (see "Durable capture-job queue" under PostgreSQL Integration) claims it, calls `extract_facts()`, and writes results as `memory_proposals` rows — durable across restarts, with retry/backoff on failure, but not yet surfaced in `search_memory` (a known, accepted gap until Stage One Phase 5).

Either way, this captures key facts — user preferences, decisions, findings — without requiring the agent to explicitly call `save_memory`. Extraction never blocks the REPL.

#### Legacy store (`long_term.py`)

`LongTermMemory` is the store `MemoryService` superseded — flat Markdown files at `{base_dir}/{key}.md`, one per key. Nothing in the runtime wiring constructs it anymore; it's kept because `memory/migration.py`'s Phase 0 tooling reads *from* this exact format (the legacy `memory/{agent_id}/` root), and because it still has its own tests (`tests/test_memory_long_term.py`).

#### Migration (`migration.py`, `cli.py`) — Stage One Phase 0

Minion Assist has historically resolved two separate per-agent directories: `workspaces/{agent_id}/` (bootstrap files — `AGENTS.md`, `SOUL.md`, etc.) and `memory/{agent_id}/` (flat `LongTermMemory` notes). Stage One's memory implementation plan (see `minion-assist-docs/improve/memory-implementation-plan.md`) merges these into one root per agent. `memory/migration.py` and `memory/cli.py` implement that merge as a standalone, non-interactive CLI command — separate from the in-REPL `/` slash commands:

```bash
# Dry run (default) — reports what would happen, changes nothing.
minion-assist memory migrate

# Perform the merge. Legacy memory/{agent_id}/*.md notes are copied (never
# moved or deleted) into workspaces/{agent_id}/{USER.md,memory/topics/,memory/imports/}.
# Every destination file touched is backed up first.
minion-assist memory migrate --apply

# Undo a previous --apply using the manifest path it printed.
minion-assist memory migrate --rollback "~/.minion-assist/memory-migration-backups/<timestamp>/migration-manifest.json"
```

Key mapping: `user_context` → `USER.md`; `_auto_extracted` and `_notes_YYYY-MM-DD` (unreviewed extractor/daily-log output) → `memory/imports/` (quarantined, not auto-promoted); every other note → `memory/topics/{key}.md`. A destination that already exists with *different* content than the source is classified as a conflict and is never auto-migrated — it must be resolved manually. See `docs/adr/0003-per-agent-memory-scope.md` for the full rationale.

`memory/cli.py` also exposes read-only diagnostic subcommands (Stage One Phase 1, slice 5), each building the same `MemoryService` the running agent would use:

```bash
# Note counts for every configured agent, or one with --agent.
minion-assist memory status [--agent main]

# List explicit note keys (memory/topics/).
minion-assist memory list [--agent main]

# Exact, bounded read — same as the memory_get tool, from the shell.
minion-assist memory get memory/topics/project-goals.md --agent main [--from-line N] [--lines N]

# Keyword search across one or every agent's memory. Uses the lexical index
# when a database is configured (Stage One Phase 3, slice C); --corpus
# restricts to "durable", "daily", or "import".
minion-assist memory search "REST API" [--agent main] [--corpus durable]

# Note counts plus a check for un-migrated legacy data.
minion-assist memory doctor [--agent main]

# Lexical-index health: chunk counts, corpus breakdown, last-indexed time.
# Requires a configured database (Stage One Phase 3, slice C).
minion-assist memory status --deep [--agent main]

# Rebuild the lexical index. Without --force: cheap hash-diff reconciliation
# (only reindexes files that actually changed). With --force: crash-safe
# full rebuild via shadow-table swap. Requires a configured database.
minion-assist memory reindex [--agent main] [--force]

# Bounded operational/telemetry cleanup (MEM-GAP-015). Without --apply: a
# dry-run report of how many rows are past the retention window in each
# covered table. With --apply: actually deletes them. Requires a
# configured database; --days overrides config.json's memory_retention.
minion-assist memory retention [--apply] [--days 30]

# List (and, with --retry, finish) any /delete-session attempt left
# incomplete by a crash or transient failure (R2-GAP-007).
minion-assist memory verify-deletions [--retry]
```

`memory/baseline.py` measures the legacy `LongTermMemory.search()` — recall and latency — against the checked-in fixture corpus at `tests/fixtures/memory_corpus/`, so later phases (PostgreSQL lexical index, embeddings) have a real number to compare against rather than an assumption. See `tests/fixtures/memory_corpus/README.md` for the recorded baseline and a known current gap (punctuation-sensitive matching).

#### Stage One Phases 2-7 modules

Everything from the durable capture-job queue onward is documented in depth under [PostgreSQL Session Store](#postgresql-session-store) rather than duplicated here (each module's design rationale lives closest to its schema). Quick index of what lives where in `src/minion_assist/memory/`:

| Module | Phase | Covers |
|---|---|---|
| `postgres_index.py` | 3, 4 | `PostgresMemoryIndex` — the rebuildable lexical/hybrid index (`memory_chunks`, `memory_files`, `memory_chunk_embeddings`), plus every Phase 5-7 table (`memory_consolidation_previews`, `memory_topic_revisions`, `memory_import_previews`, `kb_entities`, `kb_claims`, `kb_evidence`, `kb_relationships`) — see "Lexical memory index" and "Hybrid retrieval". |
| `chunking.py` | 3, 4 | Heading-aware, token-bounded Markdown chunking that feeds `postgres_index.py`'s hybrid (lexical + optional vector) retrieval. |
| `watcher.py` | 3 | `MemoryIndexWatcher` — live filesystem watcher, see "Keeping the index in sync". |
| `boundaries.py` | 6 | Action-boundary frontmatter (owner/applies_when/expires_at/…) — see "Action-sensitive memory boundaries". |
| `commitments.py`, `commitment_worker.py` | 6 | Inferred, short-lived commitment extraction and delivery — see "Commitments — schema and extraction pipeline" and "Commitment delivery, expiry, and lifecycle". |
| `consolidation.py`, `consolidation_scheduler.py` | 5 | `MemoryConsolidator` (preview/approve/reject/rollback) and its daily auto-preview scheduler — see "Consolidation ranking and preview drafting". |
| `knowledge.py` | 7A, 7B, 7D, 7F | Claim marker parsing/rendering (`parse_claims`, `remove_evidence_from_content`) and digest compilation (`compile_digest`) — see "Knowledge layer". |
| `digest_scheduler.py` | 7D | `KnowledgeDigestScheduler` — daily `KNOWLEDGE_DIGEST.md` recompilation. |
| `import_review.py` | 7E | `ImportReviewer` (preview/approve/reject) — see "Import quarantine and review". |
| `forgetting.py` | 7F | `forget_source()` — see "Forgetting a source". |
| `retention_scheduler.py` | — | `MemoryRetentionScheduler` — daily bounded cleanup of operational/telemetry tables (MEM-GAP-015), also runnable on demand via `minion-assist memory retention`. |

---

### `session`

**Directory:** `src/minion_assist/session/`

#### `store.py` — session metadata

Tracks lightweight metadata for each agent's session in `{workspace}/sessions.json`.

```python
from minion_assist.session import SessionStore

store = SessionStore(Path("~/.minion-assist/sessions.json"))
info = store.get_or_create("main")   # SessionInfo(agent_id, created_at, last_active, turn_count)
store.touch("main", increment_turns=True)
store.set_session_id("main", session_id)  # point the store at a different existing session
store.list_sessions()                 # list[SessionInfo]
```

#### `db.py` — PostgreSQL message store (optional)

```python
from minion_assist.session.db import SessionDB

db = SessionDB("postgresql://minion:minion@localhost:5433/minion_assist")

# Schema is created automatically on first connect
db.upsert_session(session_id, agent_id)
db.add_message(session_id, "user", "hello")      # raw insert — no idempotency check
db.mirror_message(session_id, event_id, "user", "hello")  # idempotent — what session.py actually uses
db.is_mirrored(session_id, event_id)             # bool
db.search_messages("hello world")    # FTS — list[dict] ranked by ts_rank
db.list_sessions(limit=20)           # newest-first summary list
db.get_messages_around(session_id, anchor_id, window=5)  # context window for SCROLL
db.get_session_bookends(session_id, n=3)   # (first_n, last_n) user/assistant messages
db.reconcile_session(session_id, agent_id, messages, mtime)  # mirror exactly what's missing
db.reconcile_all_sessions(short_term, agent_ids)  # every JSONL session, run on every startup
```

Uses thread-local connections (`threading.local`) so one psycopg connection is held per OS thread without locking overhead. All writes use `autocommit=True`.

---

### `context`

**File:** `src/minion_assist/context.py`

Detects context window overflow and compacts conversation history.

```python
from minion_assist.context import Compactor

compactor = Compactor(
    context_window=262_144,
    # preserve_tokens defaults to _DEFAULT_PRESERVE (4 000) in direct calls.
    # minion.py always passes max_output_tokens + _SNIP_SAFETY_BUFFER (1 024).
    preserve_tokens=33_792,     # 32 768 max_output + 1 024 safety buffer
    tail_keep_full_results=4,   # keep last 4 tool results in full (default)
)

# Called before every run_turn() — no-op when under budget
messages = compactor.compact(
    messages,
    provider,
    on_compaction=lambda: print("Compacting..."),
    on_compaction_failed=lambda err: print(f"Compaction failed: {err}"),
)

# Read-only peek — returns the messages compact() would summarize away right
# now, or None if compaction isn't needed. Never mutates messages, never
# calls the provider. Used by AgentSession for the pre-compaction flush
# (Stage One Phase 2, slice B) — see the `memory` module reference below.
head = compactor.peek_compaction_head(messages)
```

**Token estimation:** uses `tiktoken` (cl100k_base encoding) when installed — install with `uv add --optional tiktoken tiktoken`. Falls back to a 4-char-per-token heuristic.

**Compaction strategy when budget is exceeded:**

1. Scans from history start to find the largest prefix that fits (the **head**).
2. Calls the provider with a structured summarisation prompt.
3. Returns `[summary_message] + pruned_tail` where the tail is **microcompacted**:
   - The last `tail_keep_full_results` (default 4) tool results are kept in full (capped at `_max_tool_output` chars).
   - Older tool results are replaced with a one-liner `[result: N chars — use tools to re-read if needed]`, costing ~15 tokens instead of ~300+.

If summarisation fails, `on_compaction_failed` is called and the original history is returned unchanged — the session continues without interruption.

**All budget limits are proportional to `context_window`** — switching the model in `config.json` adjusts them automatically:

| Derived limit | Formula | 8K | 32K | 262K | 1M |
|---|---|---|---|---|---|
| `preserve_tokens` (auto) | `maxOutputTokens + 1 024` | varies | varies | 33 792 | 129 024 |
| `_max_tool_output` (chars) | `max(2k, min(16k, ctx×4÷50))` | 2 000▼ | 2 621 | 16 000■ | 16 000■ |
| `_max_head_content` (chars) | `max(500, min(20k, ctx×4÷100))` | 500▼ | 1 310 | 10 485 | 20 000■ |
| `_summarise_max_tokens` | `max(500, min(16k, preserve×0.8))` | 500▼ | varies | 16 000■ | 16 000■ |
| Preserve ceiling | `context_window ÷ 2` | 4 096 | 16 384 | 131 072 | 500 000 |

▼ = floor constant applies · ■ = cap constant applies · `ctx` = `context_window`

**`_SNIP_SAFETY_BUFFER = 1 024`** (from nanobot) is the extra token margin added on top of `maxOutputTokens` when auto-computing `preserve_tokens`. It covers system-prompt tokens, tool-definition JSON overhead, and token-estimation inaccuracies that the raw `maxOutputTokens` value does not account for.

**Per-turn injection budgets** (also proportional to `context_window`, computed in `AgentSession`):

| Budget | Formula | 8K | 32K | 262K | 1M |
|---|---|---|---|---|---|
| User context (chars) | `max(600, min(16k, ctx÷25))` | 600▼ | 1 310 | 10 485 | 16 000■ |
| Memory injection (tokens) | `max(100, min(8k, ctx÷130))` | 100▼ | 252 | 2 016 | 7 692 |
| Budget warning threshold | 50 % of usable tokens | — | — | — | — |

**Budget warning:** when conversation history exceeds 50 % of the usable token window (`context_window − preserve_tokens`), a `<context_budget>` block is injected into the system prompt telling the model to be concise before compaction is forced.

---

## Codex Provider

`CodexProvider` runs your **ChatGPT Plus / OpenAI Codex subscription** locally — no `OPENAI_API_KEY` needed.  It spawns the `codex` CLI binary as a child process and drives it with JSON-RPC 2.0 over stdio.

### Prerequisites

```bash
npm install -g @openai/codex
```

### One-time authentication

```bash
uv run codex-login
```

This opens `https://auth.openai.com/codex/device` in your browser, prompts you to enter a short code, and stores a token at `~/.minion-assist/codex-auth.json`.  The token is auto-refreshed on every startup (5-minute buffer before expiry).

### Configuration

```json
{
  "models": {
    "providers": {
      "codex": {
        "api": "codex",
        "models": [
          {"id": "openai/gpt-5.5", "name": "GPT-5.5 (Codex)", "contextWindow": 258400, "maxOutputTokens": 32768}
        ]
      }
    }
  },
  "agents": {
    "main": {"model": "codex/openai/gpt-5.5"}
  }
}
```

### Environment variables

| Variable | Description |
|---|---|
| `CODEX_BIN` | Override the path to the `codex` binary (useful when it's not on PATH or you want a specific version) |
| `MINION_ASSIST_HOME` | Override the token storage directory (default `~/.minion-assist`) |

### Dynamic tool bridge

All tools from the agent's `ToolRegistry` (web search, Playwright, memory, file tools, etc.) are registered with Codex as **dynamic tools** at session start.  Codex can then call them during inference exactly as it uses its own built-in tools.  When Codex invokes one, minion-assist executes it via `registry.execute()` and replies with the result — the same execution path used by all other LLM backends.

This mirrors [openclaw's dynamic tool bridge](../minion-assist-docs/22-codex-provider-setup.md) and means there is no Codex-specific subset of tools — the same full tool set is available regardless of backend.

### Built-in tool approval

Codex's own bash/file capabilities (separate from dynamic tools) can be auto-approved or prompt the user:

```json
{
  "codex": {
    "allow_all_commands": true
  }
}
```

`allow_all_commands: true` silently approves all built-in shell and file operations.  When `false` (default), a TUI prompt appears for each request.

### Background token refresh

The `codex app-server` subprocess is launched once and kept running for the life of the bot process, but it only receives the OAuth access token once, at launch. Left alone, the binary's own internal background jobs (e.g. its periodic refresh of the available-models list) start failing with `401 token_expired` once that token expires — visible as recurring `codex_models_manager` errors in the logs — until the bot is restarted.

To avoid needing a restart, minion-assist re-pushes the (auto-refreshed) token into the running subprocess on a timer:

```json
{
  "codex": {
    "auth_refresh_interval_seconds": 300
  }
}
```

Default is 300 seconds (5 minutes), matching the refresh-before-expiry buffer already used when loading the token. Lower this if your deployment sees very short-lived tokens; there's little cost to refreshing more often since each tick is a single JSON-RPC call.

---

## Adding a Provider

1. Create `src/minion_assist/providers/myprovider.py`:

```python
from .base import LLMResponse

class MyProvider:
    def __init__(self, api_key: str, model: str) -> None: ...

    def chat(self, system, messages, tools, max_tokens, on_token=None) -> LLMResponse:
        # streaming path: call on_token(fragment) for each text chunk
        # blocking path: single request, return LLMResponse
        ...
```

2. Register it in `providers/__init__.py` under `create_provider()`.
3. Add the provider entry to `config.json`.

---

## Adding a Tool

```python
from minion_assist.tools.base import Tool, ToolSchema

class MyTool(Tool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="my_tool",
            description="Does something useful.",
            parameters={
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        return f"Result: {kwargs['input']}"
```

Register before passing to `run_turn`:

```python
registry = default_registry()
registry.register(MyTool())
```

---

## Adding an Agent

1. Add a personality in `agents/definitions.py`:

```python
AGENTS["coder"] = AgentConfig(
    name="Turing",
    soul="You are Turing, a code generation specialist. Write clean, tested code." + _TASK_SOUL_SUFFIX,
    max_tool_rounds=25,   # higher for complex coding tasks
)
```

2. Add a model assignment and routing prefix in `config.json`:

```json
"agents": {
  "coder": {"model": "lmstudio/qwen-qwen3.5-9b", "route_prefix": "/code"}
}
```

`router.py` and `minion.py` pick up the new prefix automatically at next startup.

---

## Adding a Skill

```
~/.minion-assist/skills/openapi-design/SKILL.md    ← global (all projects)
.minion-assist/skills/openapi-design/SKILL.md       ← project-local (overrides global)
```

**Minimal `SKILL.md`:**

```markdown
---
name: openapi-design
description: Design OpenAPI 3.1 specs following REST best practices.
---

# OpenAPI Design

When designing an OpenAPI spec:
1. Use semantic versioning for the `info.version` field.
2. Group endpoints by resource, not by HTTP method.
3. Always define reusable schemas under `components/schemas`.
```

On next startup the skill is discovered and its description appears in every agent's system prompt. Agents call `skill(name="openapi-design")` to load the full instructions into context.

---

## Running Tests

```bash
# Run all tests
uv run pytest -v

# Run a specific module
uv run pytest tests/test_memory_long_term.py -v

# Run with a keyword filter
uv run pytest -k "task" -v
```

The test suite covers **3293 cases** across all modules. Three are skipped by default: one (`tests/test_providers_base.py`) requires the `anthropic` package; one (`tests/test_mcp_schema.py`) is a Windows-specific test skipped when its own environment precondition isn't met; one requires a live PostgreSQL instance (see below). The Matrix channel tests in `tests/matrix/` pass without any Matrix server or matrix-nio installation.

```bash
uv add anthropic
uv run pytest -v
```

### Database test isolation (R2-GAP-015)

Tests that touch PostgreSQL (`tests/test_session_db.py`, `tests/memory/test_postgres_index.py`, `tests/memory/test_reconciliation_scheduler.py`, `tests/test_tools_session_search.py`, `tests/test_minion.py`, etc.) are skipped automatically unless a live database is reachable at the URL configured in `config.json` (defaulting to `postgresql://minion:minion@localhost:5433/minion_assist` — see [PostgreSQL Session Store](#postgresql-session-store)).

When a database *is* reachable, `tests/conftest.py` creates a fresh, randomly-named PostgreSQL schema (`pytest_<random>`) at the start of the run and points `minion_assist.config.database` at it via the connection URL's `search_path` option, so every table/type reference in the app and test code transparently lands in that isolated schema — no application code changes needed. The schema is dropped at the end of the run. This means:

- Test runs never leave leftover rows behind in your real `public` schema, and never collide with rows from a previous or concurrent test run.
- Every full test-suite invocation starts from a schema-migration-fresh, empty database, so tests can't accidentally depend on state left over by another test file.

If schema creation fails for any reason (e.g. the configured database user lacks `CREATE SCHEMA` privileges), the fixture silently falls back to running directly against the configured database/schema — the same behavior as before this mechanism existed.

### Dependency management

```bash
uv sync                          # install core dependencies
uv sync --extra tiktoken         # + tiktoken for accurate token estimation
uv sync --extra postgres         # + psycopg3 + watchdog + pgvector for session store & memory index
uv sync --extra browser          # + playwright for browser tool
uv sync --extra voice            # + silero-vad, sounddevice, torch, transformers for voice chat
uv sync --extra tui              # + textual, textual-image, soundfile for the full-screen terminal UI (--tui)
uv sync --extra tiktoken --extra postgres  # combine any extras
uv add <package>                 # add a runtime dependency
uv add --dev <package>           # add a dev dependency
uv run <command>                 # run a command in the project environment
uv sync --all-extras             # install every extra at once (matches this repo's own dev environment)
```

**Gotcha:** `uv run` re-syncs the environment to match `uv.lock` whenever the lockfile has changed (e.g. after `git pull`, or after any `pyproject.toml` edit) — and if that particular `uv run`/`uv sync` invocation doesn't repeat the `--extra` flags you'd previously installed, it prunes those extras back out, not just leaves them alone. If you use multiple extras together (e.g. `postgres` + `voice` + `tui`), run `uv sync --all-extras` again after any dependency-related update rather than a bare `uv sync`/`uv run`, or every `uv run`/`uv sync` call will need to repeat every `--extra` you actually use.

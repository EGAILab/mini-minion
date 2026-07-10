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
uv sync --extra postgres   # PostgreSQL session store (psycopg3 driver)

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
│   │   ├── memory.py            # SaveMemoryTool, SearchMemoryTool, NoteTool
│   │   ├── session_search.py    # SessionSearchTool — FTS search across all past sessions (requires PostgreSQL)
│   │   ├── mcp.py               # McpToolAdapter, McpStatusTool, ListMcpResourcesTool, ReadMcpResourceTool, ListMcpPromptsTool, GetMcpPromptTool
│   │   ├── skill.py             # SkillTool — load skill instructions on demand
│   │   ├── spawn_subagent.py    # SpawnSubagentTool — delegate tasks to child AgentSession; _make_subagent_registry
│   │   ├── task.py              # ReadTaskTool, UpdateTaskTool — long-running task progress
│   │   ├── web_search.py        # WebSearchTool — DuckDuckGo web search via ddgs, no API key
│   │   └── __init__.py          # default_registry() factory; registers all tools; sets registry.policy
│   ├── workspace.py             # Per-agent workspace management: ensure_workspace, check_workspace, WorkspaceVanishedError
│   ├── spawn_registry.py        # Multi-agent spawn limits: get_spawn_depth, count_active_children, MAX_SPAWN_DEPTH
│   ├── plugins.py               # Plugin manifest loader — tools, hooks, skills, and trust from plugins.json
│   ├── matrix/                  # Optional Matrix channel (requires matrix-nio[e2e] + aiosqlite)
│   │   ├── channel.py           # MatrixChannel — lifecycle manager (daemon thread + asyncio loop)
│   │   ├── monitor.py           # monitor_matrix() — auth, callbacks, sync loop, teardown
│   │   ├── handler.py           # MatrixMessageHandler — full inbound pipeline
│   │   ├── outbound.py          # MatrixOutbound — markdown-formatted chunked send, draft preview, reactions, typing indicators
│   │   ├── format.py            # to_matrix_html() / build_content() — markdown → org.matrix.custom.html
│   │   ├── inbound_dedupe.py    # SQLite-backed event-ID deduplication (24 h TTL)
│   │   ├── thread_bindings.py   # SQLite thread-root → agent session key mapping
│   │   ├── bot_loop.py          # Sliding-window rate limiter per room
│   │   ├── exec_approvals.py    # DM-based remote tool approval via ✅/❌ reactions
│   │   ├── auto_join.py         # Invite handler — always / allowlist / off policy
│   │   ├── crypto.py            # E2E encryption setup via SqliteCryptoStore + libolm
│   │   ├── auth.py              # Authentication — access token / password / SSO
│   │   ├── config.py            # MatrixConfig and nested dataclasses
│   │   ├── allowlist.py         # User-ID normalisation and wildcard allowlist check
│   │   └── __init__.py
│   ├── memory/                  # Persistent memory storage
│   │   ├── short_term.py        # JSONL conversation history (atomic writes)
│   │   ├── long_term.py         # Markdown notes store (ranked keyword search)
│   │   ├── extractor.py         # Background fact extraction after each turn
│   │   └── __init__.py
│   └── session/                 # Session metadata tracking
│       ├── store.py             # JSON session store (turn counts, timestamps)
│       ├── db.py                # SessionDB — PostgreSQL session + message store with FTS (optional)
│       └── __init__.py
└── tests/                       # pytest test suite (1104 tests, 2 skipped)
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
| `database.url` | *(Optional)* PostgreSQL connection string for session history storage and FTS search. Omit to run file-only. Example: `"postgresql://minion:minion@localhost:5433/minion_assist"`. |
| `extra_plugin_manifests` | *(Optional)* List of additional `plugins.json` file paths to load beyond the two fixed locations (`~/.minion-assist/plugins.json` and `.minion-assist/plugins.json`). Paths support `~` expansion. |
| `bootstrap.enabled` | *(Optional, default `true`)* Set to `false` to disable workspace bootstrap file injection entirely. |
| `bootstrap.path` | *(Optional, default `null`)* Directory to search for bootstrap files. `null` uses `Path.cwd()` at runtime. |
| `bootstrap.max_chars` | *(Optional, default `20000`)* Maximum characters to inject from any single bootstrap file. Larger files are truncated with a head+tail excerpt. |
| `bootstrap.total_max_chars` | *(Optional, default `60000`)* Maximum total characters across all bootstrap files combined. |
| `bootstrap.truncation_warning` | *(Optional, default `"always"`)* When to inject a truncation warning: `"always"`, `"once"` (first occurrence only per process), or `"off"`. |

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
    │     soul + bootstrap Project Context (AGENTS.md, SOUL.md, TOOLS.md, …)
    │          + <bootstrap_pending> guidance (if BOOTSTRAP.md present)
    │          + <user_context> (if user_context.md exists)
    │          + <relevant_memories> (proactive search — top-5 snippets)
    │          + <active_task> (current task progress, if a task is active)
    │          + <context_budget> warning (if history > 50% of context window)
    │          + <available_skills> suffix
    │
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
           ├─ db.add_message(...) [optional]        ← mirrors to PostgreSQL for FTS search
           └─ extract_and_save_async(long_term, provider, last_exchange)
                  daemon thread — extracts 0–3 key facts, appends to _auto_extracted.md
```

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
├── memory/
│   ├── main/             ← Ada's long-term notes (isolated per agent)
│   │   ├── user_context.md     ← injected into system prompt every turn (optional)
│   │   ├── _auto_extracted.md  ← rolling facts extracted automatically after turns
│   │   ├── project-goals.md
│   │   └── api-research.md
│   └── researcher/       ← Elizabeth's long-term notes (isolated)
│       └── findings.md
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
| `/status` | Show session metadata (turn counts, last active) for each agent |
| `/agents` | List all known agents with turn counts and last-active timestamps |
| `/session [N\|uuid-prefix]` | List past conversation sessions for the active agent; restore one by index or UUID prefix |
| `/rename [N] <name>` | Give the current session (or session N from /session) a descriptive name |
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

minion-assist can connect to a Matrix homeserver so agents are reachable from any Matrix client (Element, Cinny, etc.). The channel runs in a background thread and shares the same `AgentSession` instances as the REPL — both can be active simultaneously.

### Prerequisites

```bash
# Install Matrix channel dependencies
uv add "minion-assist[matrix]"

# E2E encryption also requires libolm at the OS level (optional but recommended)
# Ubuntu/Debian: apt install libolm-dev
# macOS: brew install libolm
```

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
      "threadBindings": {"enabled": true},
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
| `threadBindings.enabled` | `true` to persist Matrix thread → conversation mapping in SQLite |
| `execApprovals.enabled` | `true` to send bash tool approval requests via DM |
| `execApprovals.approvers` | List of Matrix user IDs who receive approval DMs |
| `botLoop.enabled` | `true` to rate-limit events per room (prevents bot loops) |
| `storePath` | Directory for SQLite state and E2E crypto store |

### How It Works

**Room routing:** each Matrix room can be mapped to a specific agent via `groups`. Messages in unmapped rooms go to `defaultAgentId`.

**Thread continuity:** when `threadBindings.enabled = true`, each Matrix thread maps to a dedicated `AgentSession` history key persisted in SQLite. Replies within the same thread continue the same conversation across restarts.

**Markdown formatting:** agent responses are automatically converted from markdown to `org.matrix.custom.html` before sending, so bold, code blocks, lists, and links render natively in Element and other Matrix clients. The `matrix/format.py` module handles conversion (using `markdown-it-py`) and intelligent paragraph chunking.

**Typing indicators:** while the agent is processing a message, a typing notification appears in the Matrix room so users know the bot is working. The indicator is cleared automatically when the reply is sent, even if the agent errors.

**Exec approvals:** when an agent calls a bash command, the `MatrixExecApprovalHandler` sends a DM to each configured approver. Reacting ✅ approves the command; reacting ❌ denies it. Commands time out after 60 seconds (denied by default).

**E2E encryption:** if libolm is installed, the bot automatically uses `SqliteCryptoStore` for end-to-end encryption. Without libolm, the bot falls back to unencrypted communication with a console warning — it does not fail to start.

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

## PostgreSQL Session Store

minion-assist can mirror every conversation message into a PostgreSQL database, enabling full-text search across all historical sessions via the `session_search` tool.  The file-based JSONL store always remains active — PostgreSQL is an additive layer, not a replacement.

### Setup

```bash
# Start PostgreSQL + pgvector via Docker Compose (port 5433 to avoid conflicts with a local postgres)
docker compose up -d

# Install the psycopg3 driver
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

On next startup minion-assist will:
1. Connect to the database and create the schema automatically.
2. Migrate all existing JSONL session files into the database (one-time, skips already-imported sessions).
3. Register the `session_search` tool in every agent's tool registry.
4. Dual-write every new message to both JSONL and PostgreSQL.

### Schema

| Table | Description |
|---|---|
| `sessions` | One row per session: `id`, `agent_id`, `source`, `started_at`, `last_active`, `turn_count`, `title`, `parent_id` |
| `messages` | Every message with a `tsvector` generated column for FTS, GIN-indexed. Columns: `id` (BIGSERIAL), `session_id`, `role`, `content`, `tool_name`, `timestamp`, `search_vector` |
| `message_embeddings` | Optional — created only when the `vector` extension (pgvector) is available. Holds `vector(1536)` embeddings for future semantic search. |

### `session_search` Tool Modes

| Mode | Description |
|---|---|
| `DISCOVER` | FTS query across all sessions. Returns ranked matches with a snippet, ±3 message context window, and session bookends (first/last messages). Supports AND (default), OR, `"quoted phrase"`, `-exclude`, `prefix*`. |
| `SCROLL` | Read messages around a specific message ID in one session. Accepts `anchor_message_id` (0 = end of session) and `window` (default 5, max 20). |
| `BROWSE` | List the 20 most recent sessions with title, turn count, age, and first-message preview. |

### Data directory

When using `docker-compose.yml`, PostgreSQL data is persisted to `../data/` (relative to the compose file), which maps to `E:\AI\Projects\OpenMinds\Minions\Minion-Assist\data\` on this machine. The container is set to `restart: unless-stopped` so it starts automatically with Docker Desktop.

### Graceful degradation

If the database is unavailable at startup, minion-assist prints a warning and continues in file-only mode — `session_search` is simply not registered. No existing functionality is affected.

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

The workspace bootstrap prompt layer discovers recognized Markdown files under the configured root and injects them into every agent's system prompt as a `# Project Context` block.  Files are re-read on every turn so edits take effect without restarting the process.

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
from minion_assist.bootstrap import build_bootstrap_prompt_block, load_bootstrap_files

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
    long_term=long_term,         # enables memory injection and background extraction
    memory_injection_tokens=600, # token budget for proactive memory injection (default 600)
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

When `long_term` is provided, `AgentSession`:
- Loads `user_context.md` from the memory directory at init and injects it into the system prompt on every turn as a `<user_context>` block.
- Searches long-term memory before each turn and injects the top-5 matching snippets as a `<relevant_memories>` block (capped at `memory_injection_tokens * 4` characters).
- Fires background fact extraction after each successful turn (daemon thread — never blocks the REPL). Can be disabled via `enable_memory_extraction=False` (or `config.json` `"memory": {"enable_extraction": false}`).

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
| `SaveMemoryTool` | `save_memory` | Save a Markdown note to long-term memory under a given key. Blocked by `read_only_mode`. |
| `NoteTool` | `note` | Append a quick timestamped bullet to today's daily log in memory (`_notes_YYYY-MM-DD.md`). No key needed — great for ephemeral observations. Blocked by `read_only_mode`. |
| `SearchMemoryTool` | `search_memory` | Keyword search across long-term memory. Results ranked by term frequency and recency. Capped at 20. `is_read_only=True`. |
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
| `WebFetchTool` | `web_fetch` | Fetch a URL and return its text content. Strips HTML tags (skips `<script>`, `<style>`, `<head>`), collapses whitespace, and truncates at `max_chars` (default 8 000). Blocks SSRF targets (AWS metadata, GCP metadata). `is_read_only=True`. |
| `WebSearchTool` | `web_search` | Search the web via DuckDuckGo. Returns numbered results with title, URL, and snippet. No API key required. Requires `ddgs` (already in `pyproject.toml`). Parameters: `query` (required), `max_results` (1–10, default 5), `region` (e.g. `"us-en"`, optional). `is_read_only=True` — concurrent batching supported. Query string is checked against SSRF markers when a policy is injected. |
| `AskUserTool` | `ask_user` | Pause the agent and ask the human operator a question. Returns the human's typed response. When no `ask_user_fn` is provided (headless mode), returns an error instructing the agent to proceed without input. |
| `SpawnSubagentTool` | `spawn_subagent` | Delegate a self-contained task to a child subagent and return its text response. The subagent runs in a daemon thread with a read-only registry (no write tools). Enforces depth (max 4) and child-count (max 5) limits. Parameters: `task` (required), `agent_id` (default: `"researcher"`), `timeout_seconds` (default: 120). Subagent events are relayed to the parent's terminal with a `[sub:{agent_id}]` prefix. |
| `GitStatusTool` | `git_status` | Show git working-tree status (`git status --short --branch`): branch name, staged, modified, and untracked files. `is_read_only=True`. |
| `GitDiffTool` | `git_diff` | Show a unified diff of changes. Optional `staged=true` for staged changes; optional `path` to limit to a file. `is_read_only=True`. |
| `GitCommitTool` | `git_commit` | Stage files (optional `files` list) and create a git commit with the given `message`. Calls the `bash_confirm` callback before executing, same as `BashTool`. |
| `SessionSearchTool` | `session_search` | Search, scroll, or browse past conversation sessions stored in PostgreSQL. Three modes: **DISCOVER** (FTS across all sessions — supports quoted phrases, `-exclude`, `prefix*`), **SCROLL** (paginate within a session by message ID), **BROWSE** (list recent sessions). Only registered when a `database.url` is configured. `is_read_only=True`. |
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
from minion_assist.memory import LongTermMemory
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
    long_term=LongTermMemory(some_path),
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
| `long_term` | `LongTermMemory \| None` | `None` | If provided, registers `save_memory` and `search_memory` |
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

#### Long-term (`long_term.py`)

Stores notes as Markdown files — one file per key at `{base_dir}/{key}.md`. Forward slashes in keys are replaced with underscores.

```python
from minion_assist.memory import LongTermMemory

mem = LongTermMemory(Path("~/.minion-assist/memory/main").expanduser())
mem.save("api-research", "# REST vs GraphQL\n...")
mem.load("api-research")       # str or None
mem.search("GraphQL REST")     # list[tuple[key, content]] — ranked results
mem.list_keys()                # list[str]
mem.delete("api-research")     # bool
```

`search()` splits the query on whitespace, filters out stop-word candidates (terms shorter than 3 characters), and ranks results by term-match frequency. Notes matching more query terms rank above those matching fewer. Among ties, newer files rank slightly higher. Results are capped at 20 (`_SEARCH_MAX_RESULTS`).

**Reserved key: `user_context`** — `AgentSession` loads this key at startup and injects its content into the system prompt every turn. Write to it with `save_memory(key="user_context", content="...")` to give the agent persistent background about yourself.

#### Background extractor (`extractor.py`)

```python
from minion_assist.memory.extractor import extract_and_save_async

# Fire after a successful turn — returns immediately (daemon thread)
extract_and_save_async(long_term, provider, last_exchange)
```

After each successful turn, `AgentSession` fires `extract_and_save_async` in a daemon thread. The function sends the last user↔assistant exchange to the provider with a short extraction prompt and appends any discovered facts (0–3 per turn, max 100 chars each) to a rolling `_auto_extracted.md` file (capped at 50 entries).

This captures key facts — user preferences, decisions, findings — without requiring the agent to explicitly call `save_memory`. Extraction never blocks the REPL and fails silently.

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
db.add_message(session_id, "user", "hello")
db.search_messages("hello world")    # FTS — list[dict] ranked by ts_rank
db.list_sessions(limit=20)           # newest-first summary list
db.get_messages_around(session_id, anchor_id, window=5)  # context window for SCROLL
db.get_session_bookends(session_id, n=3)   # (first_n, last_n) user/assistant messages
db.replay_jsonl(short_term, agent_ids)     # one-time migration from JSONL files
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

The test suite covers **1480 cases** across all modules. One test (`test_create_provider_anthropic`) is skipped unless the `anthropic` package is installed; one is skipped on non-Windows systems (`test_windows_npx_wrapped`). The Matrix channel tests in `tests/matrix/` pass without any Matrix server or matrix-nio installation.

```bash
uv add anthropic
uv run pytest -v
```

### Dependency management

```bash
uv sync                          # install core dependencies
uv sync --extra tiktoken         # + tiktoken for accurate token estimation
uv sync --extra postgres         # + psycopg3 for PostgreSQL session store
uv sync --extra tiktoken --extra postgres  # both extras
uv add <package>                 # add a runtime dependency
uv add --dev <package>           # add a dev dependency
uv run <command>                 # run a command in the project environment
```

# mini-minion

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

# Optional: install tiktoken for more accurate context-window token estimation
uv add --optional tiktoken tiktoken

# Set API keys
cp .env.example .env   # then fill in your keys

# Run
uv run mini-minion
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
mini-minion/
├── config.json                  # Provider, model, agent, routing, workspace, and MCP config
├── .env                         # API keys (never commit)
├── pyproject.toml               # Package metadata and dependencies
├── src/mini_minion/
│   ├── minion.py                # Entry point — interactive REPL
│   ├── config.py                # Config loader (config.json + .env)
│   ├── context.py               # Context window overflow detection and history compaction
│   ├── cli_input.py             # PromptReader — Up/Down arrow key prompt history via prompt_toolkit
│   ├── commands.py              # Slash command dispatcher and built-in commands
│   ├── messages.py              # Provider-neutral content block helpers (text/image)
│   ├── media.py                 # File-backed attachment ingestion with MIME/size validation
│   ├── providers/               # LLM API adapters
│   │   ├── base.py              # Protocol, ToolCall, LLMResponse types
│   │   ├── openai_compatible.py # OpenAI Chat Completions adapter
│   │   ├── anthropic.py         # Anthropic Claude adapter
│   │   ├── lmstudio.py          # LM Studio alias
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
│   │   ├── mcp.py               # McpToolAdapter, McpStatusTool, ListMcpResourcesTool, ReadMcpResourceTool, ListMcpPromptsTool, GetMcpPromptTool
│   │   ├── skill.py             # SkillTool — load skill instructions on demand
│   │   ├── task.py              # ReadTaskTool, UpdateTaskTool — long-running task progress
│   │   ├── web_search.py        # WebSearchTool — DuckDuckGo web search via ddgs, no API key
│   │   └── __init__.py          # default_registry() factory; registers all tools; sets registry.policy
│   ├── plugins.py               # Plugin manifest loader — tools, hooks, skills, and trust from plugins.json
│   ├── memory/                  # Persistent memory storage
│   │   ├── short_term.py        # JSONL conversation history (atomic writes)
│   │   ├── long_term.py         # Markdown notes store (ranked keyword search)
│   │   ├── extractor.py         # Background fact extraction after each turn
│   │   └── __init__.py
│   └── session/                 # Session metadata tracking
│       ├── store.py             # JSON session store (turn counts, timestamps)
│       └── __init__.py
└── tests/                       # pytest test suite (966 tests, 3 skipped)
```

---

## Configuration

### `config.json`

```json
{
  "models": {
    "providers": {
      "lmstudio": {
        "baseUrl": "http://127.0.0.1:1234/v1",
        "api": "lmstudio",
        "models": [
          {"id": "qwen-qwen3.5-9b", "name": "Qwen 3.5 9B", "contextWindow": 262144, "maxOutputTokens": 32768}
        ]
      },
      "aliyuncs": {
        "baseUrl": "https://coding.dashscope.aliyuncs.com/v1",
        "api": "openai-completions",
        "models": [
          {"id": "glm-5", "name": "glm-5", "contextWindow": 128000, "maxOutputTokens": 4096}
        ]
      },
      "anthropic": {
        "api": "anthropic",
        "models": [
          {"id": "claude-sonnet-4-5", "name": "Claude Sonnet", "contextWindow": 200000, "maxOutputTokens": 8096, "inputModalities": ["text", "image"]}
        ]
      }
    }
  },
  "agents": {
    "main":       {"model": "lmstudio/qwen-qwen3.5-9b"},
    "researcher": {"model": "aliyuncs/glm-5", "route_prefix": "/research"}
  },
  "workspace": {
    "path": "~/.mini-minion"
  },
  "streaming": {
    "chat_mode": true,
    "task_mode": false
  },
  "compaction": {},
  "mcp": {
    "servers": {
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
| `extra_plugin_manifests` | *(Optional)* List of additional `plugins.json` file paths to load beyond the two fixed locations (`~/.mini-minion/plugins.json` and `.mini-minion/plugins.json`). Paths support `~` expansion. |

### `.env`

API keys are **never** stored in `config.json`. Place them in `.env`:

```
ALIYUNCS_API_KEY=sk-...
LMSTUDIO_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

The key for a provider named `foo` is looked up as `FOO_API_KEY`.

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
    │     soul + <user_context> (if user_context.md exists)
    │          + <relevant_memories> (proactive search — top-5 snippets)
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
           ├─ short_term.save(agent_id, messages)   ← persists history to JSONL
           ├─ session_store.touch(agent_id, ...)    ← updates turn count / timestamp
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

**Workspace layout** (default `~/.mini-minion/`):

```
~/.mini-minion/
├── sessions/
│   ├── main.jsonl        ← Ada's conversation history
│   └── researcher.jsonl  ← Elizabeth's conversation history
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

Project-level skills live at `.mini-minion/skills/` relative to the working directory and override global skills with the same name.

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
| `/sessions` | List all agent sessions with turn counts and last-active timestamps |
| `/resume [agent_id]` | Switch the active agent (e.g. `/resume researcher`); defaults to current agent |
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

mini-minion supports the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP). Configure MCP servers in `config.json` under `"mcp"`:

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
- At startup, mini-minion connects to each configured MCP server.
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

**Screenshots** are automatically saved to `~/.mini-minion/playwright-output/screenshot-<timestamp>.png` when the agent calls `browser_take_screenshot`. The agent receives the file path in the tool result.

**Notes:**
- The browser opens as a visible window by default (headed mode). You can watch the agent browse.
- Use `browser_snapshot` (accessibility tree) for page analysis — it works with all text models. Use `browser_take_screenshot` only with vision-capable models.
- Add `"--headless"` to `args` to run without a visible window: `"args": ["@playwright/mcp@latest", "--headless"]`.

---

## Module Reference

### `config`

**File:** `src/mini_minion/config.py`

Loads `config.json` and `.env` at import time. Exposes four module-level values:

```python
from mini_minion.config import agents, workspace, streaming, compaction, memory, extra_plugin_manifests

agents                  # dict[str, AgentModelConfig] — one entry per agent in config.json
workspace               # Path — resolved workspace directory
streaming               # StreamingConfig — whether to stream in each execution mode
compaction              # CompactionConfig — shared token reservation (context_window is per-agent on ModelConfig)
memory                  # MemoryConfig — controls background fact extraction
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

### `providers`

**Directory:** `src/mini_minion/providers/`

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
from mini_minion.providers import create_provider

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
| `openai-responses` | `OpenAICompatibleProvider` | Alias, same implementation |
| `lmstudio` | `LMStudioProvider` | Alias for `OpenAICompatibleProvider` |
| `anthropic` | `AnthropicProvider` | Requires `anthropic` package: `uv add anthropic` |
| _(anything else)_ | `OpenAICompatibleProvider` | Fallback |

---

### `agents`

**Directory:** `src/mini_minion/agents/`

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
from mini_minion.agents.router import resolve

agent_id, message = resolve("/research find FastAPI benchmarks")
# → ("researcher", "find FastAPI benchmarks")

agent_id, message = resolve("what is dependency injection?")
# → ("main", "what is dependency injection?")
```

Routing is **config-driven** — rules are read from `config.json` at startup. Each agent with a `"route_prefix"` gets an entry; the agent without one is the default fallback. Routes are sorted longest-first to prevent prefix shadowing.

#### Session (`session.py`)

```python
from mini_minion.agents import AgentSession

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

**`session.reload()`** — reloads conversation history from disk, replacing in-memory state. Called automatically by the `/resume` command to ensure the history is current after switching agents. Does not affect long-term memory, task files, or session metadata.

**`session.reset()`** — clears in-memory and persisted conversation history (used by `/new`). Does not affect long-term memory or task files.

#### Runner (`runner.py`)

```python
from mini_minion.agents.runner import run_turn

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

**Directory:** `src/mini_minion/tools/`

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
from mini_minion.agents.events import ToolPreExecuteHookEvent, ToolPostExecuteHookEvent

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
| `GitStatusTool` | `git_status` | Show git working-tree status (`git status --short --branch`): branch name, staged, modified, and untracked files. `is_read_only=True`. |
| `GitDiffTool` | `git_diff` | Show a unified diff of changes. Optional `staged=true` for staged changes; optional `path` to limit to a file. `is_read_only=True`. |
| `GitCommitTool` | `git_commit` | Stage files (optional `files` list) and create a git commit with the given `message`. Calls the `bash_confirm` callback before executing, same as `BashTool`. |
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
from mini_minion.tools import default_registry
from mini_minion.memory import LongTermMemory
from mini_minion.skills import discover_skills
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
    tasks_dir=Path("~/.mini-minion/tasks").expanduser(),
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

---

### `PermissionPolicy` (`tools/policy.py`)

`PermissionPolicy` is a centralised safety dataclass injected into all I/O tools. Rather than each tool implementing its own path/URL/command checks, they all delegate to one shared policy object.

```python
from mini_minion.tools.policy import PermissionPolicy
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

The plugin system lets you extend mini-minion with custom tools, registry hooks, and skills — no source edits required.

**Manifest format** (`~/.mini-minion/plugins.json` or `.mini-minion/plugins.json`):

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
from mini_minion.tools.base import Tool, ToolSchema

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
from mini_minion.commands import CommandResult

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

**Priority:** project-local manifest (`.mini-minion/plugins.json`) loads after global (`~/.mini-minion/plugins.json`), so local tools override global ones with the same name.

**Security note:** plugin files are executed with `importlib`. Only add paths to files you trust.

---

### `skills`

**File:** `src/mini_minion/skills/__init__.py`

Discovers and loads agent skills from SKILL.md files on disk.

```python
from mini_minion.skills import discover_skills, format_skills_prompt

registry = discover_skills([
    Path("~/.mini-minion/skills").expanduser(),  # global (lower priority)
    Path(".mini-minion/skills"),                  # project (higher priority)
])
prompt_suffix = format_skills_prompt(registry)   # "" when no skills found
```

**Skill file format** (`~/.mini-minion/skills/my-skill/SKILL.md`):

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

**Directory:** `src/mini_minion/memory/`

#### Short-term (`short_term.py`)

Stores conversation history as JSONL files — one file per agent at `{base_dir}/{key}.jsonl`. Uses an atomic tmp-file swap on every `save()` so a crash mid-write never corrupts the existing history.

```python
from mini_minion.memory import ShortTermMemory

mem = ShortTermMemory(Path("~/.mini-minion/sessions").expanduser())
mem.load("main")                          # list[dict] — full history
mem.save("main", messages)               # atomic overwrite
mem.append("main", {"role": "user", "content": "hi"})  # efficient append
mem.clear("main")                        # delete history file
```

#### Long-term (`long_term.py`)

Stores notes as Markdown files — one file per key at `{base_dir}/{key}.md`. Forward slashes in keys are replaced with underscores.

```python
from mini_minion.memory import LongTermMemory

mem = LongTermMemory(Path("~/.mini-minion/memory/main").expanduser())
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
from mini_minion.memory.extractor import extract_and_save_async

# Fire after a successful turn — returns immediately (daemon thread)
extract_and_save_async(long_term, provider, last_exchange)
```

After each successful turn, `AgentSession` fires `extract_and_save_async` in a daemon thread. The function sends the last user↔assistant exchange to the provider with a short extraction prompt and appends any discovered facts (0–3 per turn, max 100 chars each) to a rolling `_auto_extracted.md` file (capped at 50 entries).

This captures key facts — user preferences, decisions, findings — without requiring the agent to explicitly call `save_memory`. Extraction never blocks the REPL and fails silently.

---

### `session`

**Directory:** `src/mini_minion/session/`

Tracks lightweight metadata for each agent's session in `{workspace}/sessions.json`.

```python
from mini_minion.session import SessionStore

store = SessionStore(Path("~/.mini-minion/sessions.json"))
info = store.get_or_create("main")   # SessionInfo(agent_id, created_at, last_active, turn_count)
store.touch("main", increment_turns=True)
store.list_sessions()                 # list[SessionInfo]
```

---

### `context`

**File:** `src/mini_minion/context.py`

Detects context window overflow and compacts conversation history.

```python
from mini_minion.context import Compactor

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

## Adding a Provider

1. Create `src/mini_minion/providers/myprovider.py`:

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
from mini_minion.tools.base import Tool, ToolSchema

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
~/.mini-minion/skills/openapi-design/SKILL.md    ← global (all projects)
.mini-minion/skills/openapi-design/SKILL.md       ← project-local (overrides global)
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

The test suite covers **966 cases** across all modules (966 passed, 3 skipped). One test (`test_create_provider_anthropic`) is skipped unless the `anthropic` package is installed; one is skipped on non-Windows systems (`test_windows_npx_wrapped`); one integration test is skipped when the `mcp` package is not installed.

```bash
uv add anthropic
uv run pytest -v
```

### Dependency management

```bash
uv sync                        # install all dependencies
uv add <package>               # add a runtime dependency
uv add --optional tiktoken tiktoken  # install optional tiktoken for better token estimation
uv add --dev <package>         # add a dev dependency
uv run <command>               # run a command in the project environment
```

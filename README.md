# mini-minion

A minimal multi-agent CLI assistant with pluggable LLM providers, tool execution, persistent memory, and long-running task support. Two agents — **Ada** (general assistant) and **Elizabeth** (research specialist) — run a Think-Act-Observe loop, calling tools until they reach a final answer, then persisting conversation history across sessions. Responses can be streamed token-by-token in interactive mode.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Module Reference](#module-reference)
  - [config](#config)
  - [providers](#providers)
  - [agents](#agents)
  - [tools](#tools)
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
You: exit
```

`/research <message>` routes to Elizabeth (researcher). Everything else goes to Ada (main).

---

## Project Structure

```
mini-minion/
├── config.json                  # Provider, model, agent, routing, and workspace config
├── .env                         # API keys (never commit)
├── pyproject.toml               # Package metadata and dependencies
├── src/mini_minion/
│   ├── minion.py                # Entry point — interactive REPL
│   ├── config.py                # Config loader (config.json + .env)
│   ├── context.py               # Context window overflow detection and history compaction
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
│   ├── skills/                  # Agent skills (SKILL.md files with YAML frontmatter)
│   │   └── __init__.py          # discover_skills(), format_skills_prompt(), SkillInfo
│   ├── tools/                   # Executable tools for agents
│   │   ├── base.py              # Tool ABC, ToolSchema, _within() path guard
│   │   ├── registry.py          # ToolRegistry — dispatch and schema export
│   │   ├── read.py              # ReadTool — file/directory reading with pagination
│   │   ├── write.py             # WriteTool — file writing
│   │   ├── glob.py              # GlobTool — file pattern search
│   │   ├── bash.py              # BashTool — shell commands (PowerShell/bash)
│   │   ├── memory.py            # SaveMemoryTool, SearchMemoryTool
│   │   ├── skill.py             # SkillTool — load skill instructions on demand
│   │   ├── task.py              # ReadTaskTool, UpdateTaskTool — long-running task progress
│   │   └── __init__.py          # default_registry() factory
│   ├── memory/                  # Persistent memory storage
│   │   ├── short_term.py        # JSONL conversation history (atomic writes)
│   │   ├── long_term.py         # Markdown notes store (ranked keyword search)
│   │   ├── extractor.py         # Background fact extraction after each turn
│   │   └── __init__.py
│   └── session/                 # Session metadata tracking
│       ├── store.py             # JSON session store (turn counts, timestamps)
│       └── __init__.py
└── tests/                       # pytest test suite (352 tests, 1 skipped)
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
  "compaction": {
    "preserve_tokens": 32768
  }
}
```

**Fields:**

| Field | Description |
|---|---|
| `models.providers.<name>.baseUrl` | API base URL for the provider |
| `models.providers.<name>.api` | Adapter type: `openai-completions`, `lmstudio`, or `anthropic` |
| `models.providers.<name>.models` | List of available models with `id`, `contextWindow`, and `maxOutputTokens` |
| `agents.<id>.model` | `"<provider>/<model-id>"` — which model each agent uses |
| `agents.<id>.route_prefix` | Optional command prefix that routes to this agent (e.g. `"/research"`). Omit for the default fallback agent. |
| `workspace.path` | Root directory for persisted history and memory (tilde-expanded) |
| `streaming.chat_mode` | `true` to stream tokens in the interactive REPL; `false` for full-response display |
| `streaming.task_mode` | `true` to stream tokens in programmatic/task invocations; `false` by default |
| `models.providers.<name>.models[].contextWindow` | The model's total token capacity; used per-agent as the compaction budget |
| `models.providers.<name>.models[].maxOutputTokens` | Maximum tokens the model may generate per response; sent to the API as `max_tokens` |
| `compaction.preserve_tokens` | Tokens reserved for the next response and overhead; clamped to `[2 000, 40 000]` |

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
├── skills/               ← global skills (available to all agents)
│   └── my-skill/
│       └── SKILL.md
└── sessions.json         ← session metadata (turn counts, timestamps)
```

Project-level skills live at `.mini-minion/skills/` relative to the working directory and override global skills with the same name.

---

## Module Reference

### `config`

**File:** `src/mini_minion/config.py`

Loads `config.json` and `.env` at import time. Exposes four module-level values:

```python
from mini_minion.config import agents, workspace, streaming, compaction

agents     # dict[str, AgentModelConfig] — one entry per agent in config.json
workspace  # Path — resolved workspace directory
streaming  # StreamingConfig — whether to stream in each execution mode
compaction # CompactionConfig — shared token reservation (context_window is per-agent on ModelConfig)
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
    preserve_tokens: int   # tokens reserved for response + overhead; clamped to [2k, 40k]
    # context_window is per-agent, stored in ModelConfig.context_window
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
    soul_suffix="",             # optional skills block appended each turn
    long_term=long_term,        # enables memory injection and background extraction
    memory_injection_tokens=600, # token budget for proactive memory injection (default 600)
)

# Headless (returns text, no output)
text = session.send("What is REST?")

# With event callback
events = []
text = session.send("What is REST?", on_event=events.append, stream=True)
```

When `long_term` is provided, `AgentSession`:
- Loads `user_context.md` from the memory directory at init and injects it into the system prompt on every turn as a `<user_context>` block.
- Searches long-term memory before each turn and injects the top-5 matching snippets as a `<relevant_memories>` block (capped at `memory_injection_tokens * 4` characters).
- Fires background fact extraction after each successful turn (daemon thread — never blocks the REPL).

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

#### Built-in tools

| Tool class | Name | Description |
|---|---|---|
| `ReadTool` | `read` | Read a file (numbered lines, optional `offset`/`limit`) or list a directory. Rejects binary files and caps at 50 KB. Paths outside the workspace root are rejected. |
| `WriteTool` | `write` | Write content to a file, creating parent directories as needed. Paths outside the workspace root are rejected. |
| `GlobTool` | `glob` | Find files matching a glob pattern, sorted newest-first. Skips `.git`/`.venv`/`__pycache__`/etc. Caps at 200 results. |
| `BashTool` | `bash` | Run a shell command — PowerShell on Windows, bash on Unix. Calls the injected `confirm` callable before executing; pass `None` to skip confirmation. |
| `SaveMemoryTool` | `save_memory` | Save a Markdown note to long-term memory under a given key. |
| `SearchMemoryTool` | `search_memory` | Keyword search across long-term memory. Results ranked by term frequency and recency. Capped at 20. |
| `SkillTool` | `skill` | Load a skill's instructions into context by name. Only registered when skills are discovered at startup. |
| `ReadTaskTool` | `read_task` | Read the current task progress file — goal, steps, status, notes, and context. |
| `UpdateTaskTool` | `update_task` | Create a new task (goal + steps) or update an existing one (step status, notes, context, or clear). |

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
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `long_term` | `LongTermMemory \| None` | `None` | If provided, registers `save_memory` and `search_memory` |
| `root` | `Path \| None` | `None` | Workspace root — `read`/`write`/`glob` reject paths outside this boundary |
| `bash_confirm` | `Callable[[str], bool] \| None` | `None` | Called before every bash command; `None` = no confirmation |
| `skills` | `SkillRegistry \| None` | `None` | If non-empty, registers the `skill` tool |
| `tasks_dir` | `Path \| None` | `None` | Task file directory. Required alongside `agent_id` to register task tools. |
| `agent_id` | `str \| None` | `None` | Agent ID used to build the task file path `{tasks_dir}/{agent_id}.json`. |

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
    preserve_tokens=32_768,
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
   - The last `tail_keep_full_results` (default 4) tool results are kept in full (capped at 2 000 chars each).
   - Older tool results are replaced with a one-liner `[result: N chars — use tools to re-read if needed]`, costing ~15 tokens instead of ~300+.

If summarisation fails, `on_compaction_failed` is called and the original history is returned unchanged — the session continues without interruption.

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

The test suite covers **352 cases** across all modules (352 passed, 1 skipped). One test (`test_create_provider_anthropic`) is skipped unless the `anthropic` package is installed.

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

# mini-minion

A minimal multi-agent CLI assistant with pluggable LLM providers, tool execution, and persistent memory. Two agents — **Ada** (general assistant) and **Elizabeth** (research specialist) — run a Think-Act-Observe loop, calling tools until they reach a final answer, then persisting conversation history across sessions. Responses can be streamed token-by-token in interactive mode.

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
│   │   ├── definitions.py       # AgentConfig, AGENTS dict
│   │   ├── events.py            # Structured event dataclasses emitted by the runner
│   │   ├── router.py            # Message → agent routing
│   │   ├── runner.py            # TAO loop (run_turn)
│   │   ├── session.py           # AgentSession — headless per-agent execution unit
│   │   └── __init__.py
│   ├── skills/                  # Agent skills (SKILL.md files with YAML frontmatter)
│   │   └── __init__.py          # discover_skills(), format_skills_prompt(), SkillInfo
│   ├── tools/                   # Executable tools for agents
│   │   ├── base.py              # Tool ABC, ToolSchema
│   │   ├── registry.py          # ToolRegistry
│   │   ├── read.py              # ReadTool — file/directory reading
│   │   ├── write.py             # WriteTool — file writing
│   │   ├── glob.py              # GlobTool — file pattern search
│   │   ├── bash.py              # BashTool — shell commands
│   │   ├── memory.py            # SaveMemoryTool, SearchMemoryTool
│   │   ├── skill.py             # SkillTool — load skill instructions on demand
│   │   └── __init__.py          # default_registry() factory
│   ├── memory/                  # Persistent memory storage
│   │   ├── short_term.py        # JSONL conversation history
│   │   ├── long_term.py         # Markdown notes store
│   │   └── __init__.py
│   └── session/                 # Session metadata tracking
│       ├── store.py             # JSON session store
│       └── __init__.py
└── tests/                       # pytest test suite (304 tests, 1 skipped)
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
          {"id": "qwen-qwen3.5-9b", "name": "Qwen 3.5 9B", "contextWindow": 8192, "maxOutputTokens": 4096}
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
    "preserve_tokens": 4000
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
| `compaction.preserve_tokens` | Tokens reserved for the next response and overhead; clamped to `[2000, 8000]` |

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
    ├─ compactor.compact(history, provider, on_compaction=...)
    │      no-op when under budget; fires CompactionStarted event when triggered
    │
    └─ run_turn(provider, agent_name, system, max_tokens, tools, messages,
                on_event=callback, stream=True/False)
           │
           │  ┌────────────────────────────────────────────────────┐
           │  │              Think-Act-Observe loop                 │
           │  │                                                    │
           │  │  provider.chat(system, messages, tools, n,         │
           │  │                on_token=callback if stream else None│
           │  │       │                                            │
           │  │       ▼                                            │
           │  │  LLMResponse(text, tool_calls, finish_reason)      │
           │  │       │                                            │
           │  │  finish_reason == "tool_calls"?                    │
           │  │    yes → emit ToolCalled event → execute tool      │
           │  │          → append results → loop                   │
           │  │    no  → emit FinalAnswer event → return           │
           │  └────────────────────────────────────────────────────┘
           │
           ▼
    short_term.save(agent_id, messages)   ← persists history to JSONL
    session_store.touch(agent_id, ...)    ← updates turn count and timestamp
```

**Event system:** `run_turn` and `AgentSession` emit structured event objects to the `on_event` callback rather than calling `print()` or `input()` directly. This makes the agent runtime usable headlessly (tests, scripts, web APIs) — pass `on_event=None` for silent execution.

| Event class | Emitted when |
|---|---|
| `StreamingStarted(agent_name)` | First token of a streaming response arrives |
| `TokenStreamed(token)` | Each subsequent streaming token |
| `FinalAnswer(agent_name, text)` | Model produces a complete (non-tool) response |
| `ToolCalled(name, args)` | A tool is about to be executed |
| `MaxRoundsReached(agent_name, message)` | TAO loop hits the turn limit |
| `CompactionStarted()` | History compaction is about to run |

**Message format** is the OpenAI Chat Completions wire format throughout — `{"role": "user"|"assistant"|"tool", "content": "..."}`. Providers that use a different format (Anthropic) convert internally.

**Workspace layout** (default `~/.mini-minion/`):

```
~/.mini-minion/
├── sessions/
│   ├── main.jsonl        ← Ada's conversation history
│   └── researcher.jsonl  ← Elizabeth's conversation history
├── memory/
│   ├── main/             ← Ada's long-term notes (isolated)
│   │   ├── project-goals.md
│   │   └── api-research.md
│   └── researcher/       ← Elizabeth's long-term notes (isolated)
│       └── findings.md
├── skills/               ← global skills (available to all agents)
│   └── my-skill/
│       └── SKILL.md
└── sessions.json         ← session metadata (turn counts, timestamps)
```

Project-level skills live at `.mini-minion/skills/` relative to the working directory. They override global skills with the same name.

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
    preserve_tokens: int   # tokens reserved for response + overhead; clamped to [2k, 8k]
    # context_window is per-agent, stored in ModelConfig.context_window
```

API key resolution: set `{PROVIDER_NAME_UPPERCASE}_API_KEY` in `.env` (or the environment directly). Inline `apiKey` fields in `config.json` are not supported — they risk being committed to version control.

**Validation errors:**

If `config.json` is missing, malformed JSON, or fails validation, startup raises `ConfigError` with a list of every problem found:

```
Invalid config.json:
  agents.main.model: Unknown provider 'lmstdio'. Did you mean 'lmstudio'?
  streaming.chat_mode: Expected boolean (true/false), got 'true'. JSON strings are not booleans.
```

All issues are collected in one pass — fix everything before restarting. The error object exposes `ConfigError.issues: list[ConfigIssue]`, where each `ConfigIssue` has a `path` (dot-separated JSON key) and a `message`.

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

When `on_token` is provided, the provider calls it with each text fragment as it arrives. When `on_token` is `None` (the default), a single blocking request is made.

#### Response types

```python
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict   # already parsed from JSON

@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = []
    finish_reason: str = "stop"   # "stop" | "tool_calls"
    was_streamed: bool = False    # True if text was already printed via on_token
```

`was_streamed` tells the runner whether the text has already been printed token-by-token. When `True`, the runner only needs to print a final newline rather than the full response text.

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

#### `AnthropicProvider`

Converts the OpenAI message format to Anthropic's format on each call:
- `tool_calls` in assistant messages → `tool_use` content blocks
- `role: "tool"` messages → `role: "user"` with `tool_result` content blocks
- Tool definitions: `parameters` → `input_schema`

The `anthropic` package is imported lazily — if not installed, construction raises `ImportError` with an install hint.

---

### `agents`

**Directory:** `src/mini_minion/agents/`

#### Definitions (`definitions.py`)

```python
@dataclass
class AgentConfig:
    name: str   # display name
    soul: str   # system prompt

AGENTS: dict[str, AgentConfig]
# Keys: "main" (Ada), "researcher" (Elizabeth)
```

To add an agent, add an entry to `AGENTS` and a matching entry under `agents` in `config.json`.

#### Router (`router.py`)

```python
from mini_minion.agents.router import resolve

agent_id, message = resolve("/research find FastAPI benchmarks")
# → ("researcher", "find FastAPI benchmarks")

agent_id, message = resolve("what is dependency injection?")
# → ("main", "what is dependency injection?")
```

Routing is **config-driven**: rules are read from `config.json` at startup via `_build_routes()`, not hard-coded. Each agent with a `"route_prefix"` field gets a routing entry; the agent without one is the default fallback.

- A prefix matches only when followed by a space (so `/research ` matches but `/researchfoo` does not).
- Routes are sorted longest-first to prevent prefix shadowing.
- Adding a new routed agent requires only a `"route_prefix"` entry in `config.json` — no Python changes.

#### Session (`session.py`)

```python
from mini_minion.agents import AgentSession

session = AgentSession(
    agent_id="main",
    agent=AGENTS["main"],
    provider=provider,
    max_output_tokens=4096,
    tools=registry,
    compactor=compactor,
    short_term=short_term,
    session_store=session_store,
    soul_suffix="",          # optional text appended to the system prompt each turn
)

# Headless (returns text, no output)
text = session.send("What is REST?")

# With event callback (drive any presentation layer)
events = []
text = session.send("What is REST?", on_event=events.append, stream=True)

# Read conversation history (defensive copy)
history = session.history
```

`AgentSession` encapsulates all per-agent state: history, provider, tools, compactor, memory persistence, and session tracking. It is the recommended entry point for non-CLI use (tests, scripts, web APIs). `minion.py` creates one instance per agent and drives them from the REPL.

#### Runner (`runner.py`)

```python
from mini_minion.agents.runner import run_turn

run_turn(
    provider,              # LLMProvider
    agent_name,            # str — used in event agent_name fields
    system,                # str — system prompt / soul
    max_tokens,            # int
    tools,                 # ToolRegistry
    messages,              # list[dict] — mutated in place
    on_event=None,         # Callable[[object], None] | None — receives structured events
    stream=False,          # bool — True to emit streaming token events
)
```

Implements the TAO loop. Emits structured events via `on_event` rather than calling `print()` directly. Pass `on_event=None` for silent execution. See the event table in the Architecture section for the full list of emitted types.

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

All tool output is a plain string. Errors are returned as strings (never raised) so the agent can read and react to them.

#### Registry (`registry.py`)

```python
registry = ToolRegistry()
registry.register(MyTool())

registry.definitions   # list[dict] — OpenAI-format tool specs
registry.execute("tool_name", {"arg": "value"})  # str
```

`definitions` returns the OpenAI `tools` array format directly, ready to pass to any provider. Registering a tool with a name that already exists overwrites it.

#### Built-in tools

| Tool class | Name | Description |
|---|---|---|
| `ReadTool` | `read` | Read a file (numbered lines, optional `offset`/`limit`) or list a directory. Rejects binary files and caps at 50 KB. Paths outside the workspace root are rejected. |
| `WriteTool` | `write` | Write content to a file, creating parent directories as needed. Paths outside the workspace root are rejected. |
| `GlobTool` | `glob` | Find files matching a glob pattern, sorted newest-first. Skips `.git`/`.venv`/`__pycache__`/etc. Caps at 200 results. Paths outside the workspace root are rejected. |
| `BashTool` | `bash` | Run a shell command — PowerShell on Windows, bash on Unix. Calls the injected `confirm` callable before executing; pass `None` to skip confirmation. |
| `SaveMemoryTool` | `save_memory` | Save a Markdown note to long-term memory |
| `SearchMemoryTool` | `search_memory` | Case-insensitive search across all long-term memory notes (capped at 20 results) |
| `SkillTool` | `skill` | Load a skill's instructions into context by name. Only registered when skills are discovered at startup. |

**`ReadTool` parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | string | required | File or directory path (must be inside the workspace root) |
| `offset` | integer | 1 | First line to return (1-indexed) |
| `limit` | integer | 200 | Maximum lines to return; also hard-capped at 50 KB |

**`GlobTool` parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pattern` | string | required | Glob pattern (e.g. `**/*.py`) |
| `path` | string | workspace root | Root directory to search in (must be inside the workspace root) |

**`BashTool` parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `command` | string | required | Shell command to execute |
| `timeout` | integer | 30 | Timeout in seconds |

#### `default_registry()`

```python
from mini_minion.tools import default_registry
from mini_minion.memory import LongTermMemory
from mini_minion.skills import discover_skills
from pathlib import Path

# Minimal (4 tools: read, write, glob, bash) — unrestricted paths, no confirmation
reg = default_registry()

# With workspace sandboxing and console bash confirmation (recommended for interactive use)
reg = default_registry(root=Path.cwd(), bash_confirm=lambda cmd: input(f"Run {cmd}? [y/N]: ") == "y")

# With memory tools (6 tools: + save_memory, search_memory)
reg = default_registry(long_term=LongTermMemory(some_path), root=Path.cwd())

# With skill tool (7 tools: + skill)
skills = discover_skills([Path("~/.mini-minion/skills").expanduser()])
reg = default_registry(skills=skills)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `long_term` | `LongTermMemory \| None` | `None` | If provided, also registers `save_memory` and `search_memory` tools |
| `root` | `Path \| None` | `None` | Workspace root — `read`/`write`/`glob` reject paths outside this boundary. `None` = unrestricted |
| `bash_confirm` | `Callable[[str], bool] \| None` | `None` | Called with the command string before execution; return `True` to allow, `False` to cancel. `None` = no confirmation |
| `skills` | `SkillRegistry \| None` | `None` | If provided (and non-empty), registers a `skill` tool that agents can call to load skill instructions |

---

### `skills`

**File:** `src/mini_minion/skills/__init__.py`

Discovers and loads agent skills from SKILL.md files on disk.

```python
from mini_minion.skills import discover_skills, format_skills_prompt, SkillInfo

# Scan one or more directories — later entries override earlier on name collision
registry = discover_skills([
    Path("~/.mini-minion/skills").expanduser(),  # global (lower priority)
    Path(".mini-minion/skills"),                  # project (higher priority)
])

# Build the <available_skills> block injected into every agent's system prompt
prompt_suffix = format_skills_prompt(registry)
# Returns "" when registry is empty or all skills lack descriptions
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

The `name:` field is required; a SKILL.md without it is silently skipped. The `description:` field is optional but must be non-empty for the skill to appear in the system prompt listing.

**`SkillInfo`:**

```python
@dataclass
class SkillInfo:
    name: str
    description: str
    path: Path      # path to SKILL.md
    content: str    # body text (frontmatter stripped)
```

Companion files (e.g. `EXAMPLES.md`, `template.py`) placed alongside `SKILL.md` are listed in the `skill` tool's response so the agent knows what supporting files are available.

---

### `memory`

**Directory:** `src/mini_minion/memory/`

#### Short-term (`short_term.py`)

Stores conversation history as JSONL files — one file per agent key at `{base_dir}/{key}.jsonl`. Each line is a JSON-serialised message dict.

```python
from mini_minion.memory import ShortTermMemory
from pathlib import Path

mem = ShortTermMemory(Path("~/.mini-minion/sessions").expanduser())

mem.load("main")                          # list[dict] — full history
mem.save("main", messages)               # overwrite entire history
mem.append("main", {"role": "user", "content": "hi"})  # add one message
mem.clear("main")                        # delete history file
```

`load()` on a key with no file returns `[]`. `append()` is efficient — opens the file in append mode rather than rewriting it.

#### Long-term (`long_term.py`)

Stores notes as Markdown files — one file per key at `{base_dir}/{key}.md`. Forward slashes in keys are replaced with underscores to stay filesystem-safe.

```python
from mini_minion.memory import LongTermMemory

mem = LongTermMemory(Path("~/.mini-minion/memory").expanduser())

mem.save("api-research", "# REST vs GraphQL\n...")
mem.load("api-research")               # str content or None
mem.search("GraphQL")                  # list[tuple[key, content]] — case-insensitive
mem.list_keys()                        # list[str] — all stored keys
mem.delete("api-research")            # bool — True if file existed
```

`search()` is a case-insensitive keyword scan across all `.md` files. It splits the query on whitespace and returns notes where any term appears in the content or key. Results are sorted alphabetically by key and capped at 20 (`_SEARCH_MAX_RESULTS`) to prevent context flooding. `SearchMemoryTool` appends a note to the output when the cap is hit.

---

### `session`

**Directory:** `src/mini_minion/session/`

Tracks lightweight metadata for each agent's session in a single JSON file at `{workspace}/sessions.json`.

```python
from mini_minion.session import SessionStore, SessionInfo

store = SessionStore(Path("~/.mini-minion/sessions.json"))

info = store.get_or_create("main")
# SessionInfo(agent_id="main", created_at="2026-...", last_active="2026-...", turn_count=0)

store.touch("main", increment_turns=True)
# Updates last_active, increments turn_count → returns updated SessionInfo

store.list_sessions()
# list[SessionInfo] — one entry per agent that has ever been used
```

Timestamps are ISO 8601 UTC strings. The file is created on first write; `list_sessions()` returns `[]` if the file does not exist yet.

### `context`

**File:** `src/mini_minion/context.py`

Detects context window overflow and compacts conversation history before it grows too large for the model.

```python
from mini_minion.context import Compactor

compactor = Compactor(context_window=32_768, preserve_tokens=4_000)

# Called before every run_turn() — no-op when under budget
messages = compactor.compact(messages, provider)

# With notification callback (called once when compaction actually runs)
messages = compactor.compact(messages, provider, on_compaction=lambda: print("Compacting..."))
```

`compact()` estimates the total token count of the history (4 chars ≈ 1 token). If the total exceeds `context_window - preserve_tokens`, it:

1. Scans from the start to find the largest prefix that fits the usable budget (the **head**).
2. Calls the provider with a structured summarisation prompt and `max_tokens=2_000`.
3. Truncates any tool outputs in the remaining tail that exceed 2 000 characters.
4. Returns `[summary_message] + pruned_tail`.

If the summarisation call fails, `compact()` returns the original list unchanged — the session continues without interruption.

`preserve_tokens` is clamped to `[2_000, 8_000]` at construction time, so misconfigured values never produce a negative usable budget.

---

## Adding a Provider

1. Create `src/mini_minion/providers/myprovider.py` implementing `chat()`:

```python
from .base import LLMResponse, ToolCall

class MyProvider:
    def __init__(self, api_key: str, model: str) -> None:
        ...

    def chat(self, system, messages, tools, max_tokens, on_token=None) -> LLMResponse:
        if on_token is not None:
            # streaming path: call on_token(fragment) for each text chunk
            ...
            return LLMResponse(text="...", tool_calls=[...], finish_reason="stop", was_streamed=True)
        # blocking path
        ...
        return LLMResponse(text="...", tool_calls=[...], finish_reason="stop")
```

2. Register it in `providers/__init__.py`:

```python
from .myprovider import MyProvider

def create_provider(api, base_url, api_key, model):
    match api:
        case "my-api":
            return MyProvider(api_key=api_key, model=model)
        ...
```

3. Add the provider entry to `config.json` and assign it to an agent.

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
                "properties": {
                    "input": {"type": "string", "description": "Input value"},
                },
                "required": ["input"],
            },
        )

    def execute(self, **kwargs: object) -> str:
        return f"Result: {kwargs['input']}"
```

Register it before passing the registry to `run_turn`:

```python
from mini_minion.tools import default_registry

registry = default_registry()
registry.register(MyTool())
```

---

## Adding an Agent

1. Add a personality in `agents/definitions.py`:

```python
AGENTS["coder"] = AgentConfig(
    name="Turing",
    soul="You are Turing, a code generation specialist. Write clean, tested code.",
)
```

2. Add a model assignment and routing prefix in `config.json`:

```json
"agents": {
  "coder": {"model": "lmstudio/qwen-qwen3.5-9b", "route_prefix": "/code"}
}
```

That's it — `router.py` and `minion.py` pick up the new prefix automatically at next startup. No Python changes needed for routing.

---

## Adding a Skill

Create a directory with a `SKILL.md` inside either the global or project skills location:

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

On the next startup, the skill is automatically discovered and its description appears in every agent's system prompt. Agents call `skill(name="openapi-design")` to load the full instructions into their context.

You can also place companion files (examples, templates) alongside `SKILL.md`; they are listed in the `skill` tool's response so the agent knows to `read` them if needed.

---

## Running Tests

```bash
# Run all tests
uv run pytest -v

# Run a specific module
uv run pytest tests/test_memory_long_term.py -v

# Run with a keyword filter
uv run pytest -k "session" -v
```

The test suite covers 304 cases across all modules (304 passed, 1 skipped). One test (`test_create_provider_anthropic`) is skipped unless the `anthropic` package is installed:

```bash
uv add anthropic
uv run pytest -v
```

### Dependency management

```bash
uv sync                        # install all dependencies
uv add <package>               # add a runtime dependency
uv add --dev <package>         # add a dev dependency
uv run <command>               # run a command in the project environment
```

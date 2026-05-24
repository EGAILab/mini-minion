# mini-minion

A minimal multi-agent CLI assistant with pluggable LLM providers, tool execution, and persistent memory. Two agents — **Ada** (general assistant) and **Elizabeth** (research specialist) — run a Think-Act-Observe loop, calling tools until they reach a final answer, then persisting conversation history across sessions.

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
  - [memory](#memory)
  - [session](#session)
- [Adding a Provider](#adding-a-provider)
- [Adding a Tool](#adding-a-tool)
- [Adding an Agent](#adding-an-agent)
- [Running Tests](#running-tests)

---

## Quick Start

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/)

```bash
# Install dependencies
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
├── config.json                  # Provider, model, agent, and workspace config
├── .env                         # API keys (never commit)
├── pyproject.toml               # Package metadata and dependencies
├── src/mini_minion/
│   ├── minion.py                # Entry point — interactive REPL
│   ├── config.py                # Config loader (config.json + .env)
│   ├── providers/               # LLM API adapters
│   │   ├── base.py              # Protocol, ToolCall, LLMResponse types
│   │   ├── openai_compatible.py # OpenAI Chat Completions adapter
│   │   ├── anthropic.py         # Anthropic Claude adapter
│   │   ├── lmstudio.py          # LM Studio alias
│   │   └── __init__.py          # create_provider() factory
│   ├── agents/                  # Agent definitions and execution
│   │   ├── definitions.py       # AgentConfig, AGENTS dict
│   │   ├── router.py            # Message → agent routing
│   │   ├── runner.py            # TAO loop (run_turn)
│   │   └── __init__.py
│   ├── tools/                   # Executable tools for agents
│   │   ├── base.py              # Tool ABC, ToolSchema
│   │   ├── registry.py          # ToolRegistry
│   │   ├── read.py              # ReadTool — file/directory reading
│   │   ├── write.py             # WriteTool — file writing
│   │   ├── glob.py              # GlobTool — file pattern search
│   │   ├── bash.py              # BashTool — shell commands
│   │   ├── memory.py            # SaveMemoryTool, SearchMemoryTool
│   │   └── __init__.py          # default_registry() factory
│   ├── memory/                  # Persistent memory storage
│   │   ├── short_term.py        # JSONL conversation history
│   │   ├── long_term.py         # Markdown notes store
│   │   └── __init__.py
│   └── session/                 # Session metadata tracking
│       ├── store.py             # JSON session store
│       └── __init__.py
└── tests/                       # pytest test suite (110 tests)
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
          {"id": "qwen-qwen3.5-9b", "name": "Qwen 3.5 9B", "maxTokens": 4096}
        ]
      },
      "aliyuncs": {
        "baseUrl": "https://coding.dashscope.aliyuncs.com/v1",
        "api": "openai-completions",
        "models": [
          {"id": "glm-5", "name": "glm-5", "maxTokens": 128000}
        ]
      }
    }
  },
  "agents": {
    "main":       {"model": "lmstudio/qwen-qwen3.5-9b"},
    "researcher": {"model": "aliyuncs/glm-5"}
  },
  "workspace": {
    "path": "~/.mini-minion"
  }
}
```

**Fields:**

| Field | Description |
|---|---|
| `models.providers.<name>.baseUrl` | API base URL for the provider |
| `models.providers.<name>.api` | Adapter type: `openai-completions`, `lmstudio`, or `anthropic` |
| `models.providers.<name>.models` | List of available models with `id` and `maxTokens` |
| `agents.<id>.model` | `"<provider>/<model-id>"` — which model each agent uses |
| `workspace.path` | Root directory for persisted history and memory (tilde-expanded) |

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
run_turn(provider, agent_name, system_prompt, max_tokens, tools, messages)
    │
    │  ┌─────────────────────────────────────────────┐
    │  │           Think-Act-Observe loop             │
    │  │                                              │
    │  │  provider.chat(system, messages, tools, n)   │
    │  │       │                                      │
    │  │       ▼                                      │
    │  │  LLMResponse(text, tool_calls, finish_reason)│
    │  │       │                                      │
    │  │  finish_reason == "tool_calls"?              │
    │  │    yes → execute each tool → append results  │
    │  │          → loop                              │
    │  │    no  → print response → return             │
    │  └─────────────────────────────────────────────┘
    │
    ▼
short_term.save(agent_id, messages)   ← persists history to JSONL
session_store.touch(agent_id, ...)    ← updates turn count and timestamp
```

**Message format** is the OpenAI Chat Completions wire format throughout — `{"role": "user"|"assistant"|"tool", "content": "..."}`. Providers that use a different format (Anthropic) convert internally.

**Workspace layout** (default `~/.mini-minion/`):

```
~/.mini-minion/
├── sessions/
│   ├── main.jsonl        ← Ada's conversation history
│   └── researcher.jsonl  ← Elizabeth's conversation history
├── memory/
│   ├── project-goals.md  ← long-term notes saved by agents
│   └── api-research.md
└── sessions.json         ← session metadata (turn counts, timestamps)
```

---

## Module Reference

### `config`

**File:** `src/mini_minion/config.py`

Loads `config.json` and `.env` at import time. Exposes two module-level values:

```python
from mini_minion.config import agents, workspace

agents    # dict[str, AgentModelConfig] — one entry per agent in config.json
workspace # Path — resolved workspace directory
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
    id: str         # e.g. "qwen-qwen3.5-9b"
    max_tokens: int

@dataclass(frozen=True)
class AgentModelConfig:
    provider: ProviderConfig
    model: ModelConfig
```

API key resolution order: `config.json` `apiKey` field → `{PROVIDER_NAME}_API_KEY` environment variable.

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
    ) -> LLMResponse: ...
```

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
```

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

Rules:
- `/research ` (with trailing space) → `"researcher"`, payload is everything after the prefix
- Everything else → `"main"`

#### Runner (`runner.py`)

```python
from mini_minion.agents.runner import run_turn

run_turn(
    provider,     # LLMProvider
    agent_name,   # str — used for console output prefix
    system,       # str — system prompt / soul
    max_tokens,   # int
    tools,        # ToolRegistry
    messages,     # list[dict] — mutated in place
)
```

Implements the TAO loop. Appends all messages (assistant turns, tool calls, tool results) directly to the `messages` list. Prints `"{agent_name}: {text}"` when the agent produces a text response. Tool calls are printed as `"  [tool: name({args})]"` for observability.

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
| `ReadTool` | `read` | Read a file (with optional `offset`/`limit`) or list a directory |
| `WriteTool` | `write` | Write content to a file, creating parent directories as needed |
| `GlobTool` | `glob` | Find files matching a glob pattern, sorted by modification time |
| `BashTool` | `bash` | Run a shell command — PowerShell on Windows, bash on Unix |
| `SaveMemoryTool` | `save_memory` | Save a Markdown note to long-term memory |
| `SearchMemoryTool` | `search_memory` | Case-insensitive search across all long-term memory notes |

**`ReadTool` parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | string | required | File or directory path |
| `offset` | integer | 1 | First line to return (1-indexed) |
| `limit` | integer | 200 | Maximum lines to return |

**`GlobTool` parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pattern` | string | required | Glob pattern (e.g. `**/*.py`) |
| `path` | string | cwd | Root directory to search in |

**`BashTool` parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `command` | string | required | Shell command to execute |
| `timeout` | integer | 30 | Timeout in seconds |

#### `default_registry()`

```python
from mini_minion.tools import default_registry
from mini_minion.memory import LongTermMemory

# Without memory tools (4 tools: read, write, glob, bash)
reg = default_registry()

# With memory tools (6 tools: + save_memory, search_memory)
reg = default_registry(long_term=LongTermMemory(some_path))
```

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

`search()` is a simple substring scan across all `.md` files. It returns all matching `(key, content)` pairs sorted alphabetically by key.

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

---

## Adding a Provider

1. Create `src/mini_minion/providers/myprovider.py` implementing `chat()`:

```python
from .base import LLMResponse, ToolCall

class MyProvider:
    def __init__(self, api_key: str, model: str) -> None:
        ...

    def chat(self, system, messages, tools, max_tokens) -> LLMResponse:
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

2. Add a model assignment in `config.json`:

```json
"agents": {
  "coder": {"model": "lmstudio/qwen-qwen3.5-9b"}
}
```

3. Add a routing rule in `agents/router.py`:

```python
def resolve(message: str) -> tuple[str, str]:
    if message.startswith("/research "):
        return "researcher", message[len("/research "):]
    if message.startswith("/code "):
        return "coder", message[len("/code "):]
    return "main", message
```

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

The test suite covers 110 cases across all modules. One test (`test_create_provider_anthropic`) is skipped unless the `anthropic` package is installed:

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

"""AgentSession — reusable, headless agent execution unit.

This module defines :class:`AgentSession`, which encapsulates everything needed
to run one agent's conversation: provider, history, tools, compactor, memory
persistence, and session tracking.  It exposes a single public method,
:meth:`AgentSession.send`, that non-interactive callers (tests, scripts, web
APIs, background jobs) can use without depending on ``print()``, ``input()``,
or any CLI state.

Why this class exists
----------------------
Before this class existed, all of the per-turn wiring lived in ``minion.py``
alongside the REPL loop.  That made the agent logic impossible to call without
a terminal.  ``AgentSession`` lifts that wiring into a reusable object so the
same turn logic runs identically in:

- **CLI** (``minion.py``): creates one session per agent, drives them from
  ``input()``, renders events with a console handler.
- **Tests** (``test_agents_session.py``): creates a session with a mock
  provider, calls ``send()``, asserts on the returned text or collected events.
- **Web APIs**: creates a session per user, streams events over SSE.
- **Scripts**: creates a session, loops ``send()`` calls programmatically.

How a turn works
----------------
:meth:`AgentSession.send` follows the same sequence as the old ``minion.py``
per-turn block:

1. Append the user message to history.
2. Compact the history if it's approaching the model's context limit.
3. Persist the user message to disk (so it's saved even if the provider crashes).
4. Snapshot the history length for rollback.
5. Call :func:`run_turn` — the TAO loop.
6. Increment the turn counter in the session store.
7. Return the last assistant text from history.

On exception:
- Roll back partial messages appended by :func:`run_turn` before the crash.
- Append an error message to history so the model has context next turn.
- Re-raise so the caller can display the error.
- ``finally``: always persist (user message + any error record survive a crash).

Talks to
--------
- ``agents/runner.py`` — :func:`run_turn` is called from :meth:`send`.
- ``agents/events.py`` — :class:`CompactionStarted` is translated from the
  compactor's ``on_compaction`` callback into a structured event.
- ``context.py`` — :class:`Compactor` is called before every :func:`run_turn`.
- ``memory/short_term.py`` — history is loaded at construction and saved after
  every turn.
- ``session/store.py`` — turn count is incremented on successful turns.
- ``tools/`` — :class:`ToolRegistry` is passed into :func:`run_turn`.
"""

from __future__ import annotations

from collections.abc import Callable

from ..context import Compactor
from ..memory.short_term import ShortTermMemory
from ..providers.base import LLMProvider
from ..session import SessionStore
from ..tools import ToolRegistry
from .definitions import AgentConfig
from .events import CompactionStarted
from .runner import run_turn


class AgentSession:
    """Encapsulates all state and logic for a single agent's conversation.

    Create one instance per agent at startup (as ``minion.py`` does), then
    call :meth:`send` for each user message.  All state (history, compaction
    budget) is maintained on the instance across turns.

    Args:
        agent_id (str): The agent's registry key, e.g. ``"main"`` or
            ``"researcher"``.  Used as the filename for JSONL history.
        agent (AgentConfig): The agent's name and system prompt (soul).
        provider (LLMProvider): The LLM API client for this agent.
        max_output_tokens (int): Maximum tokens the model may generate per turn.
        tools (ToolRegistry): The pre-built tool registry for this agent.
            Wired to this agent's own :class:`LongTermMemory` so agents cannot
            read or overwrite each other's notes.
        compactor (Compactor): Pre-built compactor sized to this model's context
            window.  Called before every turn to summarise old history if needed.
        short_term (ShortTermMemory): Shared short-term memory backend.
            Loads conversation history at construction and saves after every turn.
        session_store (SessionStore): Shared session metadata store.
            Records turn counts and timestamps for this agent.
        soul_suffix (str): Optional text appended to the agent's soul (system
            prompt) on every turn.  Used by ``minion.py`` to inject the
            ``<available_skills>`` block without modifying the static soul
            definition.  Empty string (default) leaves the soul unchanged.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        agent: AgentConfig,
        provider: LLMProvider,
        max_output_tokens: int,
        tools: ToolRegistry,
        compactor: Compactor,
        short_term: ShortTermMemory,
        session_store: SessionStore,
        soul_suffix: str = "",
    ) -> None:
        self._agent_id = agent_id
        self._agent = agent
        self._provider = provider
        self._max_output_tokens = max_output_tokens
        self._tools = tools
        self._compactor = compactor
        self._short_term = short_term
        self._session_store = session_store
        self._soul_suffix = soul_suffix
        # Load prior history from disk so conversation survives restarts.
        self._history: list[dict] = short_term.load(agent_id)
        # Create session record if this is the agent's first run.
        session_store.get_or_create(agent_id)

    @property
    def history(self) -> list[dict]:
        """Snapshot of the current conversation history (defensive copy)."""
        return list(self._history)

    def send(
        self,
        message: str,
        on_event: Callable[[object], None] | None = None,
        stream: bool = False,
    ) -> str | None:
        """Send a user message and return the agent's text response.

        This is the primary public method.  Non-interactive callers call this
        directly; ``minion.py`` calls it from the REPL loop.

        Args:
            message (str): The user's message text.
            on_event (Callable | None): Optional callback for structured events.
                Receives :class:`TokenStreamed`, :class:`ToolCalled`,
                :class:`FinalAnswer`, :class:`CompactionStarted`, etc.
                Pass ``None`` for silent/headless use.
            stream (bool): If ``True`` *and* ``on_event`` is provided, the
                provider is called with a token callback so the caller receives
                streaming events token-by-token.  Defaults to ``False``.

        Returns:
            str | None: The agent's final response text, or ``None`` if the
                model produced no text (only tool calls).

        Raises:
            Exception: Re-raises any provider exception after rolling back
                partial history and persisting the user message + error record.
                The caller is responsible for displaying the error.
        """
        self._history.append({"role": "user", "content": message})

        # Compact before running the turn. The on_compaction callback translates
        # to a CompactionStarted event so the caller sees a status notification.
        _on_compaction = (lambda: on_event(CompactionStarted())) if on_event else None
        self._history = self._compactor.compact(
            self._history, self._provider, on_compaction=_on_compaction
        )

        # Persist the user message now — it's safe on disk even if the
        # provider crashes before the finally block.
        self._short_term.save(self._agent_id, self._history)

        # Snapshot history length so we can roll back partial assistant/tool
        # messages that run_turn appends before a mid-turn crash.
        snapshot_len = len(self._history)

        # Build the effective system prompt: base soul + optional suffix.
        # The suffix carries the <available_skills> block injected by minion.py.
        system = self._agent.soul
        if self._soul_suffix:
            system = system + "\n\n" + self._soul_suffix

        try:
            run_turn(
                self._provider,
                self._agent.name,
                system,
                self._max_output_tokens,
                self._tools,
                self._history,
                on_event=on_event,
                stream=stream,
            )
            # Only increment on success — failed turns don't count.
            self._session_store.touch(self._agent_id, increment_turns=True)
        except Exception as exc:
            # Roll back any partial messages appended before the crash so the
            # history ends at a clean turn boundary.
            del self._history[snapshot_len:]
            error_text = f"[Provider error: {exc.__class__.__name__}: {exc}]"
            self._history.append({"role": "assistant", "content": error_text})
            raise
        finally:
            # Always persist — user message and any error record survive crashes.
            self._short_term.save(self._agent_id, self._history)

        # Return the last non-empty assistant message as the response text.
        for msg in reversed(self._history):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"]
        return None

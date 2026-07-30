"""Structured event types emitted by the agent runtime.

Instead of calling ``print()`` directly, the runner, compactor, and bash tool
emit instances of these dataclasses to an ``on_event`` callback supplied by the
caller.  This decouples *what happened* (a model token arrived, a tool was
called) from *how it is displayed* (terminal colours, JSON logs, SSE streams).

How the event flow works
------------------------
Every agent turn produces a stream of events in roughly this order:

1. **Before the turn** — if compaction was triggered:
   ``CompactionStarted`` (about to summarise), or ``CompactionFailed``
   (summarisation provider call failed; original history retained).
2. **During the turn** — if streaming is enabled:
   one ``StreamingStarted`` followed by N ``TokenStreamed`` events (one per
   token).  In non-streaming mode, any preamble text the model emits *before*
   calling tools arrives as a single ``ThoughtEmitted`` event.
3. Zero or more ``ToolCalled`` events, one per tool the model invokes.
4. Exactly one ``FinalAnswer`` when the turn ends successfully, or
   ``MaxRoundsReached`` when the tool-round cap is hit.

Who emits what
--------------
- ``agents/runner.py`` emits :class:`StreamingStarted`, :class:`TokenStreamed`,
  :class:`ThoughtEmitted`, :class:`ToolCalled`, :class:`FinalAnswer`,
  :class:`MaxRoundsReached`.
- ``agents/session.py`` translates the compactor's plain callbacks into
  :class:`CompactionStarted` and :class:`CompactionFailed` events before
  forwarding them to the compactor.
- ``minion.py`` receives all events via its ``_on_event`` handler, which renders
  them as terminal output.  Tests and other callers can use a custom handler
  (e.g. ``events.append``) to collect events for assertions without any I/O.

Talks to
--------
- ``agents/runner.py`` — imports and instantiates these dataclasses.
- ``agents/session.py`` — imports :class:`CompactionStarted`.
- ``minion.py`` — imports all event types for the console ``isinstance`` checks.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StreamingStarted:
    """Emitted just before the first streamed token of a new LLM response.

    The caller should use this to print an agent-name prefix so that
    subsequent :class:`TokenStreamed` events appear inline after it.

    Attributes:
        agent_name: The agent's display name, e.g. ``"Ada"``.  Used by the
            CLI handler to print ``"\\nAda: "`` before the first token.
    """
    agent_name: str


@dataclass
class TokenStreamed:
    """One text fragment emitted by the model during streaming.

    The caller should render this inline (no newline), building up the
    full response character-by-character as the model generates it.

    Attributes:
        token: A single text fragment, e.g. ``"Hello"`` or ``","`` or
            ``" world"``.  Whitespace is part of the token.
    """
    token: str


@dataclass
class ToolCalled:
    """The model requested a tool call; emitted before the tool executes.

    Attributes:
        name: Tool name, e.g. ``"read"`` or ``"bash"``.
        args: Parsed argument dict the tool will be called with.
    """
    name: str
    args: dict


@dataclass
class ThoughtEmitted:
    """Preamble text the model produced before requesting tool calls (non-streaming only).

    In streaming mode this text was already rendered token-by-token via
    :class:`TokenStreamed` events, so no separate event is needed.  In
    non-streaming mode the text would otherwise be silently discarded, so the
    runner emits this event before the first :class:`ToolCalled` event of the
    same round.

    Attributes:
        agent_name: The agent's display name, e.g. ``"Ada"``.
        text: The full preamble text the model generated before its tool call.
    """
    agent_name: str
    text: str


@dataclass
class FinalAnswer:
    """The model produced its final text answer for the current turn.

    Always emitted at the end of a successful turn.  When streaming was
    active the caller should check its own streaming state: the text was
    already rendered token-by-token, so re-printing it would duplicate output.
    When streaming was not active, ``text`` contains the full response and
    should be displayed now.

    Attributes:
        agent_name: The agent's display name, e.g. ``"Ada"``.
        text: The full response text.  Empty string when the model only
              called tools and produced no prose.
    """
    agent_name: str
    text: str


@dataclass
class MaxRoundsReached:
    """The TAO loop hit the tool-round cap without producing a final answer.

    Attributes:
        agent_name: The agent's display name.
        message: The cap-notification message that was appended to history.
    """
    agent_name: str
    message: str


@dataclass
class CompactionStarted:
    """Emitted when the compactor decides to summarise conversation history.

    No attributes — the event is a pure notification that compaction is
    about to happen.  The caller can use it to show a status indicator.
    """


@dataclass
class ToolCompleted:
    """Emitted after a tool finishes executing, regardless of success or failure.

    Complements :class:`ToolCalled` (emitted before execution) by providing
    the outcome and timing.

    Attributes:
        name:         Tool name, e.g. ``"read"`` or ``"bash"``.
        elapsed_ms:   Wall-clock milliseconds from execute() call to return.
        output_chars: Character length of the tool's output string.
                      Useful for detecting unexpectedly large tool outputs.
    """
    name: str
    elapsed_ms: int
    output_chars: int


@dataclass
class TurnCompleted:
    """Emitted after every successful turn.

    Ignored by the interactive CLI — intended for structured log handlers,
    monitoring dashboards, and cost-tracking systems.

    Attributes:
        agent_name:      The agent's display name.
        trace_id:        UUID4 string unique to this turn. Correlates all
                         events within the same send() call.
        turn_number:     The session's cumulative turn count after this turn.
        tool_calls_made: Number of tool-result messages appended during this turn.
        input_tokens:    Total input tokens across all provider.chat() calls
                         in the turn. 0 when the provider does not return usage.
        output_tokens:   Total output tokens across all provider.chat() calls.
        elapsed_ms:      Wall-clock milliseconds from send() entry to return.
        compacted:       True if context compaction ran before this turn.
    """
    agent_name: str
    trace_id: str
    turn_number: int
    tool_calls_made: int
    input_tokens: int
    output_tokens: int
    elapsed_ms: int
    compacted: bool


@dataclass
class CompactionFailed:
    """Emitted when the summarisation LLM call during compaction fails.

    The compactor falls back to the original history when this happens —
    the conversation continues, but the context window is not reclaimed.
    Repeated failures will eventually cause the context window to overflow.

    Attributes:
        error: Short description of the failure, e.g.
               ``"TimeoutError: provider did not respond"``.
    """
    error: str


@dataclass
class MemoryFlushed:
    """Emitted after a pre-compaction flush attempt (Stage One Phase 2, slice B).

    Fired right before :class:`CompactionStarted`/compaction actually runs,
    whenever a memory backend is configured and compaction is about to
    summarize away part of history. See
    :meth:`~minion_assist.memory.service.MemoryService.flush_head`.

    Attributes:
        status: One of ``"flushed"``, ``"empty"``, or ``"failed"`` — see
            :class:`~minion_assist.memory.models.FlushOutcome`.
        detail: Exception description when ``status == "failed"``; empty
            otherwise.
    """
    status: str
    detail: str = ""


@dataclass
class MemoryInjected:
    """Emitted when ``build_prompt_section()`` injects memory (Stage One Phase 4, slice D).

    Purely observational — a record of what was actually placed in this
    turn's system prompt, not a mechanism that changes what gets injected.
    OpenClaw's own memory prompt builder (verified by reading
    ``plugins/memory-state.ts``) has no cross-turn dedup/suppression
    either: it rebuilds the section fresh on every call. The injected block
    here is never stored in ``AgentSession._history``, so a past turn's
    injection is not visible to the model on a later turn regardless —
    skipping a still-relevant re-injection would only make the model lose
    context it needs *now*, not save it something it already has. This
    event exists so *what* was injected, and in which context generation,
    is inspectable (e.g. in tests or structured logs) without changing
    that behavior.

    Attributes:
        keys: The injected notes' keys, in injection order.
        context_generation: :attr:`AgentSession._context_generation` at
            injection time — increments on :meth:`AgentSession.reset` and
            after a successful compaction, so this event's consumer can
            tell "these were injected in the same stretch of history" from
            "context has since been reset/compacted." A forked session
            starts its own count at 0 rather than inheriting the parent's —
            forking begins an independently tracked branch (the parent/child
            relationship remains visible via ``SessionInfo.parent_id``, not
            via this counter).
        token_count: Approximate token cost of the injected block (the
            real token budget check that replaced the old 4-chars-per-token
            heuristic).
    """
    keys: tuple[str, ...]
    context_generation: int
    token_count: int


# ---------------------------------------------------------------------------
# Hook event objects — used by ToolRegistry's plugin-facing hook system.
# ---------------------------------------------------------------------------
# These are distinct from ToolCalled/ToolCompleted (which are emitted through
# the on_event stream to callers like minion.py). Hook events are passed
# directly to registered hook callables in the registry, giving plugins a
# structured object to inspect rather than a positional argument list.

@dataclass
class ToolPreExecuteHookEvent:
    """Passed to before-execute hooks registered on the ToolRegistry.

    Hooks receive this before the tool's execute() is called.  The hook
    cannot cancel or modify execution in the current implementation — it is
    purely observational (logging, metrics, audit trails).

    Attributes:
        name:      Tool name, e.g. ``"edit"`` or ``"bash"``.
        arguments: The argument dict the tool will be called with.
    """
    name: str
    arguments: dict


@dataclass
class ToolPostExecuteHookEvent:
    """Passed to after-execute hooks registered on the ToolRegistry.

    Hooks receive this after the tool's execute() returns.

    Attributes:
        name:       Tool name, e.g. ``"edit"`` or ``"bash"``.
        arguments:  The argument dict the tool was called with.
        output:     The string the tool returned (may be an error message).
        elapsed_ms: Wall-clock time the tool took to execute.
    """
    name: str
    arguments: dict
    output: str
    elapsed_ms: int

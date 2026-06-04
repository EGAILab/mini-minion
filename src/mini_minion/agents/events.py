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

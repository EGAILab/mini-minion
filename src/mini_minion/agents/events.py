"""Structured event types emitted by the agent runtime.

Instead of calling ``print()`` directly, the runner, compactor, and bash tool
emit instances of these dataclasses to an ``on_event`` callback supplied by the
caller.  This decouples *what happened* (a model token arrived, a tool was
called) from *how it is displayed* (terminal colours, JSON logs, SSE streams).

How the event flow works
------------------------
Every agent turn produces a stream of events in roughly this order:

1. If streaming is enabled, one :class:`StreamingStarted` followed by N
   :class:`TokenStreamed` events (one per token).
2. Zero or more :class:`ToolCalled` events, one per tool the model invokes.
3. Exactly one :class:`FinalAnswer` when the turn ends.

Additionally, :class:`CompactionStarted` may appear *before* the turn if the
conversation history was too long and needed to be summarised first.
:class:`MaxRoundsReached` replaces :class:`FinalAnswer` in the rare case where
the model never stops calling tools.

Who emits what
--------------
- ``agents/runner.py`` emits :class:`StreamingStarted`, :class:`TokenStreamed`,
  :class:`ToolCalled`, :class:`FinalAnswer`, :class:`MaxRoundsReached`.
- ``agents/session.py`` translates the compactor's plain callback into
  :class:`CompactionStarted` before forwarding it to the compactor.
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

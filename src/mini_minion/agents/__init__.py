"""Public API for the agents subsystem.

Re-exports the key symbols from the agent sub-modules so callers can do::

    from mini_minion.agents import AGENTS, AgentSession, resolve, run_turn

instead of importing from each sub-module directly.

Sub-modules
-----------
- ``definitions`` — :class:`AgentConfig` and the :data:`AGENTS` registry dict.
- ``events``      — Structured event dataclasses emitted during agent turns.
- ``router``      — :func:`resolve` for routing user input to an agent.
- ``runner``      — :func:`run_turn` for executing the Think–Act–Observe loop.
- ``session``     — :class:`AgentSession`, the reusable headless agent unit.
"""

from .definitions import AGENTS, AgentConfig
from .events import (
    CompactionStarted,
    FinalAnswer,
    MaxRoundsReached,
    StreamingStarted,
    TokenStreamed,
    ToolCalled,
)
from .router import resolve
from .runner import run_turn
from .session import AgentSession

__all__ = [
    "AgentConfig",
    "AGENTS",
    "AgentSession",
    "CompactionStarted",
    "FinalAnswer",
    "MaxRoundsReached",
    "resolve",
    "run_turn",
    "StreamingStarted",
    "TokenStreamed",
    "ToolCalled",
]

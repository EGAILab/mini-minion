"""Public API for the agents subsystem.

Re-exports the key symbols from the three agent sub-modules so callers can do:

    from mini_minion.agents import AGENTS, resolve, run_turn

instead of importing from each sub-module directly.

Sub-modules
-----------
- ``definitions`` — :class:`AgentConfig` and the :data:`AGENTS` registry dict.
- ``router``      — :func:`resolve` for routing user input to an agent.
- ``runner``      — :func:`run_turn` for executing the Think–Act–Observe loop.
"""

from .definitions import AGENTS, AgentConfig
from .router import resolve
from .runner import run_turn

__all__ = ["AgentConfig", "AGENTS", "resolve", "run_turn"]

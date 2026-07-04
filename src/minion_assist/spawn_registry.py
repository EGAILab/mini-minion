"""Spawn depth and child-count tracking for multi-agent session hierarchies.

Prevents unbounded recursion and runaway subagent creation by enforcing two limits:

- :data:`MAX_SPAWN_DEPTH` (4): maximum nesting depth of parent → child → grandchild
  session chains.  A root agent has depth 0; each spawned child adds 1.
- :data:`MAX_CHILDREN_PER_AGENT` (5): maximum number of child sessions one parent
  session may create in its lifetime.

Both functions query the :class:`~minion_assist.session.store.SessionStore`'s
``parent_id`` chain at call time — no state is maintained in this module itself.
This means limits are enforced even across restarts as long as the sessions file
persists.

Public API
----------
- :func:`get_spawn_depth`        — depth of a session in the parent chain.
- :func:`count_active_children`  — number of children spawned by a session.
- :data:`MAX_SPAWN_DEPTH`        — default maximum nesting depth.
- :data:`MAX_CHILDREN_PER_AGENT` — default maximum children per parent.

Talks to
--------
- ``tools/spawn_subagent.py`` — calls both functions before spawning a new
  child session to enforce the configured limits.
- ``config.py``               — :class:`MultiAgentConfig` may override the
  constants; callers should use ``multi_agent_cfg.max_spawn_depth`` and
  ``multi_agent_cfg.max_children_per_agent`` rather than these constants directly.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Constants — default limits used when config.json has no "multi_agent" section.
# ---------------------------------------------------------------------------

MAX_SPAWN_DEPTH = 4
"""Maximum parent→child nesting depth.  Root = 0.  Subagent of root = 1, etc."""

MAX_CHILDREN_PER_AGENT = 5
"""Maximum child sessions one parent session may spawn in its lifetime."""


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def get_spawn_depth(session_id: str, store: object) -> int:
    """Walk the ``parent_id`` chain to compute this session's depth.

    Root agents (no parent) have depth 0.  Each spawned child adds 1.  The
    loop is capped at ``MAX_SPAWN_DEPTH + 2`` to break pathological parent
    cycles that should never occur in practice but could arise from corrupt
    session files.

    Args:
        session_id: The session whose depth to compute.
        store: A :class:`~minion_assist.session.store.SessionStore` instance.
            Duck-typed — only ``list_sessions()`` is called.

    Returns:
        int: Depth in the session hierarchy (0 = root agent).
    """
    depth = 0
    current = session_id
    # Cap avoids infinite loops on corrupt parent-id cycles.
    # Using strict less-than ensures the returned value never exceeds the cap.
    cap = MAX_SPAWN_DEPTH + 2
    while depth < cap:
        sessions = store.list_sessions()
        entry = next((s for s in sessions if s.agent_id == current), None)
        if entry is None or not entry.parent_id:
            return depth
        depth += 1
        current = entry.parent_id
    return depth


def count_active_children(session_id: str, store: object) -> int:
    """Count the number of child sessions spawned by this parent.

    Scans ``store.list_sessions()`` for records whose ``parent_id`` equals
    ``session_id``.  All such sessions are counted regardless of whether they
    have finished — this is a lifetime limit, not a concurrent-session limit.

    Args:
        session_id: The parent session whose children to count.
        store: A :class:`~minion_assist.session.store.SessionStore` instance.
            Duck-typed — only ``list_sessions()`` is called.

    Returns:
        int: Number of sessions whose ``parent_id`` equals ``session_id``.
    """
    return sum(1 for s in store.list_sessions() if s.parent_id == session_id)

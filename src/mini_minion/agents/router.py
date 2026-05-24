"""Message routing to agents."""

from __future__ import annotations


def resolve(message: str) -> tuple[str, str]:
    """Route a message to the right agent. Returns (agent_id, stripped_message)."""
    if message.startswith("/research "):
        return "researcher", message[len("/research "):]
    return "main", message

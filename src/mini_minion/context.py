"""Context window management — detects overflow and compacts conversation history.

When a conversation grows long enough to approach the model's context limit,
``Compactor.compact()`` summarises the older portion of the history and prunes
large tool outputs from the preserved tail.

Strategy (mirrors open-minds ``SessionCompaction``)
----------------------------------------------------
1. Estimate total tokens in the current history (4 chars ≈ 1 token).
2. Compare against the *usable* budget: ``context_window - preserve_tokens``.
3. If over budget, scan from the start, accumulating tokens until the budget
   is exceeded.  Everything before that point is the ``head`` (to summarise);
   everything from that point onward is the ``tail`` (to keep).
4. Call the LLM with a structured prompt to summarise the head.
5. Truncate oversized tool outputs in the tail to avoid immediate re-overflow.
6. Return ``[summary_user_msg] + pruned_tail`` as the new history.

The caller (``minion.py``) is responsible for persisting the compacted history
to disk via ``ShortTermMemory.save()`` — this module only manipulates the list.

Talks to
--------
- ``minion.py`` calls :meth:`Compactor.compact` before each ``run_turn()``.
- ``providers.base.LLMProvider`` is used for the summarisation LLM call.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from mini_minion.providers.base import LLMProvider

# ── Token-budget constants ────────────────────────────────────────────────────

# Floor and ceiling for the preserve_tokens setting.  Prevents accidental
# misconfiguration from making the usable budget negative or trivially small.
_MIN_PRESERVE = 2_000
_MAX_PRESERVE = 8_000

# How many tokens to reserve by default for the model's response + overhead.
_DEFAULT_PRESERVE = 4_000

# Tool outputs in the preserved tail are truncated to this character count.
# Prevents a single large file-read result from immediately re-overflowing
# the window right after compaction.
_MAX_TOOL_OUTPUT = 2_000

# Per-message character cap for all roles in the summary prompt.
# Prevents a single very long user or assistant message from making the
# summarisation request itself overflow the context window.
_MAX_HEAD_CONTENT = 2_000

# Structured summarisation prompt.
# IMPORTANT: do NOT include a "Next Steps" section.  Describing future actions
# in the summary causes the model to continue the old task on the next turn
# instead of addressing the user's new message.
_SUMMARY_PROMPT = """\
You are a conversation summarizer.
Produce a concise, structured summary of what was discussed and accomplished.
Focus only on what has already happened — do NOT describe plans or next steps.

Structure your summary as:
## Goal
What the user was trying to accomplish.

## What Was Done
Actions taken and results obtained (past tense only).

## Key Decisions
Important choices made, with rationale.

## Relevant Files
Files created, modified, or discussed (if any).
"""


def _estimate_tokens(msg: dict) -> int:
    """Rough token estimate for one message dict: 4 chars ≈ 1 token."""
    return max(1, len(json.dumps(msg)) // 4)


class Compactor:
    """Detects context window overflow and compacts conversation history.

    Instantiate once in ``minion.py`` and call :meth:`compact` before every
    ``run_turn()`` call.  The method is a no-op when the history is well
    within budget.

    Args:
        context_window (int): Total token capacity of the model in use.
            Set this to match the model's actual context window size (e.g.
            32 768 for Qwen 3.5 9B, 128 000 for GPT-4o).
        preserve_tokens (int): Tokens to reserve for the model's response and
            protocol overhead.  Clamped to [2 000, 8 000].  Defaults to 4 000.
    """

    def __init__(
        self,
        context_window: int,
        preserve_tokens: int = _DEFAULT_PRESERVE,
    ) -> None:
        self._context_window = context_window
        # Clamp preserve_tokens so extreme config values don't produce a
        # negative or trivially small usable budget.
        self._preserve_tokens = max(_MIN_PRESERVE, min(_MAX_PRESERVE, preserve_tokens))

    @property
    def _usable_tokens(self) -> int:
        """Token budget available for conversation history (context minus reserve)."""
        return self._context_window - self._preserve_tokens

    def needs_compaction(self, messages: list[dict]) -> bool:
        """Return ``True`` if the estimated token total exceeds the usable budget."""
        total = sum(_estimate_tokens(m) for m in messages)
        return total > self._usable_tokens

    def compact(
        self,
        messages: list[dict],
        provider: LLMProvider,
        on_compaction: "Callable[[], None] | None" = None,
    ) -> list[dict]:
        """Return a compacted version of ``messages``, or the original list unchanged.

        When compaction is triggered:

        1. Calls ``on_compaction()`` (if provided) so the caller can notify
           the user before the summarisation LLM call is made.
        2. Splits the history into an older ``head`` (to summarise) and a
           recent ``tail`` (to preserve).
        3. Calls ``provider.chat()`` to produce a structured summary of the head.
        4. Prunes oversized tool outputs in the tail.
        5. Returns ``[summary_msg] + pruned_tail``.

        Falls back silently to the original list if the summarisation call
        fails, so a transient provider error never crashes the session.

        Args:
            messages: The full in-memory conversation history for one agent.
            provider: The agent's configured LLMProvider, used for summarisation.
            on_compaction: Optional zero-argument callback invoked when compaction
                is about to happen.  The caller uses this to emit a status event
                (e.g. :class:`CompactionStarted`) without ``compact()`` having any
                knowledge of event types.  ``None`` for silent/headless callers.

        Returns:
            Compacted message list, or the original list if compaction was not
            needed or summarisation failed.
        """
        if not self.needs_compaction(messages):
            return messages

        head, tail = self._select(messages)
        if not head:
            # History is too short to split — nothing to summarise.
            return messages

        if on_compaction is not None:
            on_compaction()
        try:
            summary = self._summarise(head, provider)
        except Exception:
            # Provider call failed — keep the original history intact.
            return messages

        tail = self._prune(tail)

        # The summary is injected as a synthetic user/assistant exchange so the
        # model treats it as closed, completed history — not an ongoing task.
        summary_msg = {"role": "user", "content": f"[Previous conversation summary — already completed]\n{summary}"}
        ack_msg = {"role": "assistant", "content": "Understood. I'm ready for your next question."}
        return [summary_msg, ack_msg] + tail

    # ── Private helpers ───────────────────────────────────────────────────────

    def _select(self, messages: list[dict]) -> tuple[list[dict], list[dict]]:
        """Split messages into ``(head, tail)`` using a token-accumulating scan.

        Iterates from the start of the history, adding message tokens to a
        running total.  The moment adding the next message would exceed the
        usable budget, that message becomes the first element of the tail;
        everything before it is the head.

        Both halves are guaranteed to be non-empty (at least one message each).
        If the history has fewer than two messages there is nothing to split.
        """
        if len(messages) < 2:
            # Can't produce a non-empty head and a non-empty tail.
            return [], messages

        accumulated = 0
        split_idx = len(messages)  # default: everything in head if no overflow found

        for i, msg in enumerate(messages):
            tok = _estimate_tokens(msg)
            if accumulated + tok > self._usable_tokens:
                split_idx = i
                break
            accumulated += tok

        # Clamp so both slices are non-empty.
        split_idx = max(1, min(split_idx, len(messages) - 1))
        return messages[:split_idx], messages[split_idx:]

    def _summarise(self, head: list[dict], provider: LLMProvider) -> str:
        """Call the LLM to produce a structured summary of the head messages."""
        conversation_text = self._format_head(head)
        response = provider.chat(
            system=_SUMMARY_PROMPT,
            messages=[{"role": "user", "content": conversation_text}],
            tools=[],    # no tools needed for summarisation
            max_tokens=2_000,
        )
        return response.text

    def _format_head(self, head: list[dict]) -> str:
        """Render the head message list as plain text for the summary prompt.

        Applies per-role character caps so the summary prompt itself cannot
        exceed the model's context window:

        - **Tool results**: truncated to 500 chars — the summary only needs
          the gist of what a file contained, not the full content.
        - **User and assistant text**: truncated to ``_MAX_HEAD_CONTENT`` chars.
          Guards against a single very long message making the summarisation
          request overflow the context window.
        - **Tool calls**: the argument preview is capped at 200 chars.
        """
        lines: list[str] = []
        for msg in head:
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""

            if role == "tool":
                # Keep only the first 500 chars of tool output for the prompt.
                lines.append(f"[tool result]: {content[:500]}")
            elif role == "assistant":
                if content:
                    lines.append(f"[assistant]: {content[:_MAX_HEAD_CONTENT]}")
                # Show which tools were called so the summary captures actions taken.
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    args_preview = fn.get("arguments", "")[:200]
                    lines.append(f"[tool call]: {fn.get('name')}({args_preview})")
            else:
                lines.append(f"[{role}]: {content[:_MAX_HEAD_CONTENT]}")

        return "\n".join(lines)

    def _prune(self, tail: list[dict]) -> list[dict]:
        """Truncate oversized tool outputs in the tail to prevent immediate re-overflow.

        Only ``role: "tool"`` messages are affected.  All other messages pass
        through unchanged.  Original dicts are never mutated — truncated
        messages are replaced with a shallow copy.
        """
        result: list[dict] = []
        for msg in tail:
            if msg.get("role") == "tool":
                content = msg.get("content") or ""
                if len(content) > _MAX_TOOL_OUTPUT:
                    # Shallow copy with truncated content — don't mutate the original.
                    msg = {**msg, "content": content[:_MAX_TOOL_OUTPUT] + "\n[truncated during compaction]"}
            result.append(msg)
        return result

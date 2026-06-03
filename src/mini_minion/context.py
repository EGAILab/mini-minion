"""Context window management — detects overflow and compacts conversation history.

When a conversation grows long enough to approach the model's context limit,
``Compactor.compact()`` summarises the older portion of the history and prunes
large tool outputs from the preserved tail.

Strategy (mirrors open-minds ``SessionCompaction``)
----------------------------------------------------
1. **Estimate tokens** — uses ``tiktoken`` (cl100k_base encoding) when the
   optional dependency is installed; otherwise falls back to the
   ``len(json.dumps) // 4`` heuristic.  Install with
   ``uv add --optional tiktoken tiktoken``.
2. **Check budget** — compare the estimated total against the usable budget:
   ``context_window - preserve_tokens``.  No-op if under budget.
3. **Split** — scan from the start, accumulating token estimates; everything
   before the overflow point is the ``head`` (to summarise); everything from
   that point onward is the ``tail`` (to keep).
4. **Summarise** — call the LLM with a structured prompt to produce a concise
   summary of the ``head``.  On failure, ``on_compaction_failed`` is called
   (if provided) and the original history is returned unchanged.
5. **Prune tail** — microcompact tool-result messages in the tail:

   - The most recent ``tail_keep_full_results`` (default 4) tool results are
     kept in full (capped at ``_MAX_TOOL_OUTPUT`` chars each).
   - Older tool results are replaced with a one-liner such as
     ``[result: 1 842 chars — use tools to re-read if needed]``, costing ~15
     tokens instead of ~300+.  This preserves the action trace without wasting
     context on stale data.

6. **Return** ``[summary_msg, ack_msg] + pruned_tail`` as the new history.

The caller (``session.py``) is responsible for persisting the compacted history
to disk via ``ShortTermMemory.save()`` — this module only manipulates the list.

Talks to
--------
- ``session.py``          — calls :meth:`Compactor.compact` before each turn.
- ``providers.base``      — used for the summarisation LLM call.
- ``agents/events.py``    — :class:`CompactionFailed` is referenced by callers.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

from mini_minion.providers.base import LLMProvider

_log = logging.getLogger("mini_minion.compactor")

# ── Token-budget constants ────────────────────────────────────────────────────

_MIN_PRESERVE = 2_000
_MAX_PRESERVE = 40_000
_DEFAULT_PRESERVE = 4_000

# Hard cap on tool output kept in full (most-recent N results).
_MAX_TOOL_OUTPUT = 2_000

# Number of recent tool-result messages kept at full content in the tail.
# Older tool results are microcompacted to one-liners (~15 tokens each).
_TAIL_KEEP_FULL_RESULTS = 4

_MAX_HEAD_CONTENT = 2_000

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


# ── Token estimator factory ───────────────────────────────────────────────────

def _make_token_estimator():
    """Return the best available per-message token estimator.

    Tries tiktoken (cl100k_base — exact for GPT-4 family, good proxy for Qwen).
    Falls back to a 4-char-per-token heuristic when tiktoken is not installed.
    ``ensure_ascii=False`` keeps CJK characters as single chars rather than
    ``\\uXXXX`` escape sequences, which would inflate the char count 7×.
    """
    try:
        import tiktoken
        _enc = tiktoken.get_encoding("cl100k_base")

        def _estimate(msg: dict) -> int:
            return max(1, len(_enc.encode(json.dumps(msg, ensure_ascii=False))))

        return _estimate
    except ImportError:
        def _estimate(msg: dict) -> int:
            return max(1, len(json.dumps(msg, ensure_ascii=False)) // 4)

        return _estimate


_estimate_tokens = _make_token_estimator()


# ── Compactor ─────────────────────────────────────────────────────────────────

class Compactor:
    """Detects context window overflow and compacts conversation history.

    Instantiate once and call :meth:`compact` before every turn.
    The method is a no-op when the history is well within budget.

    Args:
        context_window:        Total token capacity of the model in use.
        preserve_tokens:       Tokens to reserve for the model's response and
                               protocol overhead.  Clamped to
                               [``_MIN_PRESERVE``, ``_MAX_PRESERVE``].
                               Defaults to 4 000.
        tail_keep_full_results: Number of recent tool-result messages to keep
                               at full content in the tail.  Older tool results
                               are microcompacted to one-liners.  Defaults to
                               :data:`_TAIL_KEEP_FULL_RESULTS` (4).
    """

    def __init__(
        self,
        context_window: int,
        preserve_tokens: int = _DEFAULT_PRESERVE,
        tail_keep_full_results: int = _TAIL_KEEP_FULL_RESULTS,
    ) -> None:
        self._context_window = context_window
        self._preserve_tokens = max(_MIN_PRESERVE, min(_MAX_PRESERVE, preserve_tokens))
        self._tail_keep_full = tail_keep_full_results

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
        on_compaction_failed: "Callable[[str], None] | None" = None,
    ) -> list[dict]:
        """Return a compacted version of ``messages``, or the original list unchanged.

        When compaction is triggered:

        1. Calls ``on_compaction()`` so the caller can notify the user.
        2. Splits history into an older ``head`` (to summarise) and a recent
           ``tail`` (to preserve).
        3. Calls ``provider.chat()`` to produce a structured summary of the head.
        4. Microcompacts tool outputs in the tail.
        5. Returns ``[summary_msg, ack_msg] + pruned_tail``.

        On summarisation failure falls back to the original list and calls
        ``on_compaction_failed(error_description)`` if provided.

        Args:
            messages:             Full in-memory conversation history.
            provider:             The agent's LLMProvider, used for summarisation.
            on_compaction:        Zero-argument callback invoked when compaction
                                  is about to happen.
            on_compaction_failed: One-argument callback invoked with an error
                                  description string when summarisation fails.
                                  ``None`` for silent fallback.

        Returns:
            Compacted message list, or the original list if compaction was not
            needed or summarisation failed.
        """
        if not self.needs_compaction(messages):
            return messages

        head, tail = self._select(messages)
        if not head:
            return messages

        if on_compaction is not None:
            on_compaction()

        try:
            summary = self._summarise(head, provider)
        except Exception as exc:
            error_desc = f"{type(exc).__name__}: {exc}"
            _log.warning(
                "Compaction summarization failed (%s) — using original history.",
                error_desc,
            )
            if on_compaction_failed is not None:
                on_compaction_failed(error_desc)
            return messages

        tail = self._prune(tail)

        summary_msg = {
            "role": "user",
            "content": (
                f"[Previous conversation summary — already completed]\n{summary}"
            ),
        }
        ack_msg = {
            "role": "assistant",
            "content": "Understood. I'm ready for your next question.",
        }
        return [summary_msg, ack_msg] + tail

    # ── Private helpers ───────────────────────────────────────────────────────

    def _select(self, messages: list[dict]) -> tuple[list[dict], list[dict]]:
        """Split messages into ``(head, tail)`` using a token-accumulating scan."""
        if len(messages) < 2:
            return [], messages

        accumulated = 0
        split_idx = len(messages)

        for i, msg in enumerate(messages):
            tok = _estimate_tokens(msg)
            if accumulated + tok > self._usable_tokens:
                split_idx = i
                break
            accumulated += tok

        split_idx = max(1, min(split_idx, len(messages) - 1))
        return messages[:split_idx], messages[split_idx:]

    def _summarise(self, head: list[dict], provider: LLMProvider) -> str:
        """Call the LLM to produce a structured summary of the head messages."""
        conversation_text = self._format_head(head)
        response = provider.chat(
            system=_SUMMARY_PROMPT,
            messages=[{"role": "user", "content": conversation_text}],
            tools=[],
            max_tokens=2_000,
        )
        return response.text

    def _format_head(self, head: list[dict]) -> str:
        """Render the head message list as plain text for the summary prompt."""
        lines: list[str] = []
        for msg in head:
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""

            if role == "tool":
                lines.append(f"[tool result]: {content[:500]}")
            elif role == "assistant":
                if content:
                    lines.append(f"[assistant]: {content[:_MAX_HEAD_CONTENT]}")
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    args_preview = fn.get("arguments", "")[:200]
                    lines.append(f"[tool call]: {fn.get('name')}({args_preview})")
            else:
                lines.append(f"[{role}]: {content[:_MAX_HEAD_CONTENT]}")

        return "\n".join(lines)

    def _prune(self, tail: list[dict]) -> list[dict]:
        """Microcompact older tool outputs; truncate recent ones to _MAX_TOOL_OUTPUT.

        Two-tier approach:
        - The most recent ``_tail_keep_full`` tool-result messages are kept in full
          (up to ``_MAX_TOOL_OUTPUT`` chars).
        - All older tool-result messages are replaced with a one-liner summary
          (~15 tokens vs ~300+ for a truncated result).

        This preserves the action trace — the model sees what was done — at
        near-zero token cost for the historical record.
        Original dicts are never mutated; truncated messages use shallow copies.
        """
        tool_result_indices = [
            i for i, m in enumerate(tail) if m.get("role") == "tool"
        ]
        keep_full = set(tool_result_indices[-self._tail_keep_full:])

        result: list[dict] = []
        for i, msg in enumerate(tail):
            if msg.get("role") == "tool":
                content = msg.get("content") or ""
                if i in keep_full:
                    # Recent result — keep in full, apply hard cap if needed.
                    if len(content) > _MAX_TOOL_OUTPUT:
                        msg = {
                            **msg,
                            "content": content[:_MAX_TOOL_OUTPUT]
                            + "\n[truncated during compaction]",
                        }
                else:
                    # Older result — microcompact to a one-liner.
                    msg = {
                        **msg,
                        "content": (
                            f"[result: {len(content):,} chars"
                            " — use tools to re-read if needed]"
                        ),
                    }
            result.append(msg)
        return result

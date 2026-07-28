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

from minion_assist.providers.base import LLMProvider
from minion_assist.messages import content_to_summary_text

_log = logging.getLogger("minion_assist.compactor")

# ── Token-budget constants ────────────────────────────────────────────────────

# Absolute floor: any model needs at least 2 000 tokens of output headroom.
_MIN_PRESERVE = 2_000

# Fallback preserve budget used only when a Compactor is constructed in tests
# without a preserve_tokens argument.  Production code in minion.py always
# passes preserve_tokens = model.max_output_tokens + _SNIP_SAFETY_BUFFER,
# so this default never applies at runtime.
_DEFAULT_PRESERVE = 4_000

# Extra tokens added on top of max_output_tokens when auto-computing the
# preserve budget (pattern: nanobot runner.py _SNIP_SAFETY_BUFFER = 1024).
# Accounts for system prompt tokens, tool-definition JSON overhead, and
# token-count estimation inaccuracies that the raw max_output_tokens value
# does not cover.
_SNIP_SAFETY_BUFFER = 1_024

# Practical maximum for a single tool result kept in the tail after compaction.
# Both nanobot (max_tool_result_chars=16_000) and OpenHarness
# (DEFAULT_TOOL_OUTPUT_INLINE_CHARS=16_000) independently converged on 16 000 chars
# as the upper limit of useful information in one tool response — beyond that,
# the model gains little from additional tokens.  We keep a proportional floor
# for small-context models and cap at this reference value.
_TOOL_OUTPUT_REFERENCE_CAP = 16_000

# Number of recent tool-result messages kept at full content in the tail.
_TAIL_KEEP_FULL_RESULTS = 4

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

    Multimodal safety:
    ------------------
    When a message has list content containing image blocks, estimating by
    json.dumps() would count the "path" and metadata strings rather than the
    actual image cost.  Instead we:
    - Count chars across all text blocks (same 4-char heuristic).
    - Add _IMAGE_TOKENS_ESTIMATE per image block (OpenAI's ~85 "base" tiles for
      a low-detail image — a reasonable worst-case for token budgeting).
    - Never attempt to base64-encode the file here; that would be extremely slow
      and would make the compactor load image data just to count tokens.
    """
    # Approximate token cost for one image attachment in the context window.
    # OpenAI vision pricing: 85 base tokens per image (low-detail).
    # Anthropic is similar.  We use this as a conservative estimate.
    _IMAGE_TOKENS_ESTIMATE = 85

    def _multimodal_estimate(content) -> int:
        """Count tokens for a content value that may be a list of blocks."""
        if isinstance(content, list):
            total = 0
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    # 4 chars ≈ 1 token (standard LLM heuristic).
                    total += max(1, len(block.get("text", "")) // 4)
                elif block.get("type") == "image":
                    # Fixed cost per image — don't read bytes from disk.
                    total += _IMAGE_TOKENS_ESTIMATE
            return max(1, total)
        # Plain string or other content: fall through to caller's method.
        return None  # type: ignore[return-value]

    try:
        import tiktoken
        _enc = tiktoken.get_encoding("cl100k_base")

        def _estimate(msg: dict) -> int:
            content = msg.get("content", "")
            multimodal = _multimodal_estimate(content)
            if multimodal is not None:
                # Account for role overhead (~4 tokens) same as tiktoken would.
                return multimodal + 4
            return max(1, len(_enc.encode(json.dumps(msg, ensure_ascii=False))))

        return _estimate
    except ImportError:
        def _estimate(msg: dict) -> int:
            content = msg.get("content", "")
            multimodal = _multimodal_estimate(content)
            if multimodal is not None:
                return multimodal + 4
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
                               protocol overhead.  In production, pass
                               ``model.max_output_tokens`` here so the budget
                               scales automatically when the model changes.
                               Clamped to ``[_MIN_PRESERVE, context_window // 2]``
                               so the usable window is always positive.
                               Defaults to :data:`_DEFAULT_PRESERVE` (4 000,
                               used by tests only).
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

        # ── Preserve-token clamping ───────────────────────────────────────────
        # The preserve budget is the slice of the context window reserved for the
        # model's response + protocol overhead.  It must satisfy two constraints:
        #
        #   (1) At least _MIN_PRESERVE tokens — any model needs some output room.
        #   (2) At most context_window // 2 — so at least HALF the window stays
        #       usable for conversation history.  A fixed cap like the old 40 000
        #       broke on small models: a 32K-window model with 40K preserve would
        #       have NEGATIVE usable budget.  Dynamic ceiling solves this.
        #
        # In production minion.py always passes preserve = max_output_tokens + 1024,
        # so the clamp is a safety net for misconfiguration, not the normal path.
        _max_preserve = max(_MIN_PRESERVE, context_window // 2)
        self._preserve_tokens = max(_MIN_PRESERVE, min(_max_preserve, preserve_tokens))
        self._tail_keep_full = tail_keep_full_results

        # ── Proportional derived limits ───────────────────────────────────────
        # All four limits below are computed FROM context_window (and, for the
        # summary budget, from preserve_tokens).  This means they automatically
        # adjust when the model is switched — no manual config changes needed.
        #
        # Design principle:
        #   floor  = absolute minimum that makes the feature useful at all
        #   cap    = practical maximum beyond which there is diminishing return
        #   ratio  = the "natural" value when context_window is in a normal range
        #
        # ── (A) Tool output cap ───────────────────────────────────────────────
        # How much of a single tool result to keep in full during tail pruning.
        #
        # Formula:  context_window × 4 ÷ 50  = context_window ÷ 12.5
        #         ≈ 2 % of context window in CHARS  (4 chars ≈ 1 token)
        #
        # Why 2 %?  A single tool result > 2 % of the window is unusual and
        # would dominate the tail.  Reference: both nanobot and OpenHarness
        # independently chose 16 000 chars as the practical maximum (floor
        # ensures tiny models still get a sensible minimum).
        #
        # Example outputs:  1M → 16 000 (cap); 262K → 16 000 (cap);
        #                   32K → 2 621; 8K → 2 000 (floor).
        self._max_tool_output: int = max(
            2_000, min(_TOOL_OUTPUT_REFERENCE_CAP, context_window * 4 // 50)
        )
        #
        # ── (B) Head content cap ─────────────────────────────────────────────
        # How much of any single message is included when rendering the "head"
        # (the old messages) for the summarisation prompt.
        #
        # Formula:  context_window × 4 ÷ 100  = context_window ÷ 25
        #         ≈ 1 % of context window in CHARS
        #
        # Why cap per-message?  Unlike nanobot/Pi (which keep whole messages),
        # we truncate per message to prevent a single very long user message
        # from making the summarisation prompt itself overflow the context.
        #
        # Example outputs:  1M → 20 000 (cap); 262K → 10 485;
        #                   32K → 1 310; 8K → 500 (floor).
        self._max_head_content: int = max(500, min(20_000, context_window * 4 // 100))
        #
        # ── (C) Summarisation LLM call budget ────────────────────────────────
        # How many tokens the model may use when writing the compaction summary.
        #
        # Formula (from Pi harness/compaction.ts):
        #   maxTokens = min(0.8 × reserveTokens, model.maxTokens)
        #
        # Why 0.8 × preserve?  The summary is "setup for the NEXT response", so
        # its budget should scale with the response budget (preserve_tokens),
        # not the entire context window.  Using 0.8 leaves a 20 % headroom for
        # the model's actual response after reading the summary.
        #
        # Why cap at 16 000?  Without a cap, a 128K-output model would get
        # 0.8 × 128 000 = 102 400 tokens for a summary — wasteful and slow.
        # 16 000 tokens is sufficient to capture any practical conversation.
        #
        # Example outputs:  128K preserve → 102 400 → capped 16 000;
        #                   32K preserve → 26 214 → capped 16 000;
        #                   4K preserve → 3 276; 2K preserve → 1 638.
        self._summarise_max_tokens: int = max(
            500, min(16_000, int(self._preserve_tokens * 0.8))
        )

    @property
    def _usable_tokens(self) -> int:
        """Token budget available for conversation history (context minus reserve)."""
        return self._context_window - self._preserve_tokens

    def needs_compaction(self, messages: list[dict]) -> bool:
        """Return ``True`` if the estimated token total exceeds the usable budget."""
        total = sum(_estimate_tokens(m) for m in messages)
        return total > self._usable_tokens

    def peek_compaction_head(self, messages: list[dict]) -> list[dict] | None:
        """Return the messages :meth:`compact` would summarize away right now, without compacting.

        Stage One Phase 2, slice B: lets a caller flush the head's content to
        a durable note *before* calling :meth:`compact`, so a failed or lossy
        summarization call can never be the only place that content existed.
        Read-only — never mutates ``messages`` and never calls the provider.

        Args:
            messages: Full in-memory conversation history.

        Returns:
            list[dict] | None: The head messages that would be summarized,
                or ``None`` if compaction is not needed (or would have
                nothing to summarize) right now.
        """
        if not self.needs_compaction(messages):
            return None
        head, _tail = self._select(messages)
        return head or None

    def compact(
        self,
        messages: list[dict],
        provider: LLMProvider,
        on_compaction: "Callable[[], None] | None" = None,
        on_compaction_failed: "Callable[[str], None] | None" = None,
        force: bool = False,
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
            force:                When ``True``, skip the ``needs_compaction()``
                                  budget check and compact immediately. Used by
                                  the ``/compact`` REPL command to let the user
                                  manually trigger compaction. Still requires at
                                  least 2 messages (enforced by ``_select``).

        Returns:
            Compacted message list, or the original list if compaction was not
            needed or summarisation failed.
        """
        if not force and not self.needs_compaction(messages):
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
            max_tokens=self._summarise_max_tokens,
        )
        return response.text

    def _format_head(self, head: list[dict]) -> str:
        """Render the head message list as plain text for the summary prompt.

        Uses content_to_summary_text() to handle multimodal content blocks
        (images are represented as metadata labels, never as base64 data).
        This keeps the summarization prompt small even when images were sent.
        """
        lines: list[str] = []
        for msg in head:
            role = msg.get("role", "unknown")
            # content_to_summary_text handles both strings and block lists safely.
            content_raw = msg.get("content") or ""
            content = content_to_summary_text(content_raw)

            if role == "tool":
                lines.append(f"[tool result]: {content[:500]}")
            elif role == "assistant":
                if content:
                    lines.append(f"[assistant]: {content[:self._max_head_content]}")
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    args_preview = fn.get("arguments", "")[:200]
                    lines.append(f"[tool call]: {fn.get('name')}({args_preview})")
            else:
                lines.append(f"[{role}]: {content[:self._max_head_content]}")

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
                # Tool results are always string content in practice, but guard
                # against list content to be multimodal-safe.
                raw_content = msg.get("content") or ""
                content = raw_content if isinstance(raw_content, str) else str(raw_content)
                if i in keep_full:
                    # Recent result — keep in full, apply hard cap if needed.
                    if len(content) > self._max_tool_output:
                        msg = {
                            **msg,
                            "content": content[:self._max_tool_output]
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

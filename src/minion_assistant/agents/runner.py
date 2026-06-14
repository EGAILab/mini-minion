"""Agent turn execution — the Think–Act–Observe (TAO) loop.

This is the heart of how agents actually do work. When the user sends a
message, we don't just send it to the LLM and print one reply. Instead, we
run a loop:

  1. **Think**: call the LLM with the current conversation history.
  2. **Act**: if the LLM wants to use a tool, run the tool and collect the result.
  3. **Observe**: append the tool result to the conversation, then loop back.
  4. **Stop**: when the LLM produces a final answer (no tool calls), emit a
     :class:`FinalAnswer` event and return.

This loop is what makes an agent more than a chatbot — it can read files,
run commands, search memory, and use what it finds before responding.

Reliability features
--------------------
- **Retry with backoff** — every ``provider.chat()`` call is wrapped in
  :func:`_call_with_retry`, which retries transient HTTP errors (429, 5xx,
  timeouts, connection failures) up to three times with exponential backoff
  (2 s → 4 s → 8 s) plus ±0.5 s jitter.  Permanent errors (400, 401) are
  re-raised immediately without retrying.
- **Empty response recovery** — if the model returns no text and no tool calls,
  a recovery nudge is injected and the loop continues.  Only attempted on
  non-final rounds so the last round still emits a ``FinalAnswer``.
- **Length recovery** — if ``finish_reason == "length"`` (model hit
  ``max_tokens`` mid-sentence), a continuation prompt is injected and the loop
  continues once.

No I/O in this module
---------------------
``run_turn`` never calls ``print()`` or ``input()`` directly.  All output is
emitted as structured events to the optional ``on_event`` callback.  The CLI
handler in ``minion.py`` renders them as terminal text; tests collect them for
assertions; web APIs can forward them over SSE.  This keeps the execution logic
testable and reusable without any terminal dependency.

Streaming support
-----------------
When ``stream=True`` *and* ``on_event`` is provided, a closure (``_on_token``)
is created and forwarded to the provider's ``chat()`` method.  The provider
calls this closure once per text token; the closure emits one
:class:`StreamingStarted` before the first token (so the caller knows to print
a name prefix) and then one :class:`TokenStreamed` per subsequent token.

If ``stream=False`` or ``on_event`` is ``None``, no ``on_token`` callback is
passed to the provider — a plain blocking call is made instead, and the full
text arrives at once in the :class:`FinalAnswer` event.

Key concept: message mutation
-----------------------------
The ``messages`` list passed to :func:`run_turn` is mutated *in place* —
assistant responses and tool results are appended directly to it. This keeps
the full conversation context alive across turns so the LLM always has history.

Message format
--------------
All messages follow the OpenAI Chat Completions wire format:
  - User message:    ``{"role": "user", "content": "..."}``
  - Assistant reply: ``{"role": "assistant", "content": "..."}``
  - Tool invocation: ``{"role": "assistant", "content": "...", "tool_calls": [...]}``
  - Tool result:     ``{"role": "tool", "tool_call_id": "...", "content": "..."}``

Talks to
--------
- ``providers`` — via the ``LLMProvider`` protocol to call the model API.
- ``tools`` — via :class:`ToolRegistry` to look up and run tools.
- ``agents/session.py`` — :func:`run_turn` is called from :meth:`AgentSession.send`.
- ``agents/events.py`` — event dataclasses are imported and emitted here.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..providers import LLMProvider, LLMResponse
from ..providers.base import TokenUsage
from ..tools import ToolRegistry
from .events import (
    FinalAnswer,
    MaxRoundsReached,
    StreamingStarted,
    ThoughtEmitted,
    TokenStreamed,
    ToolCalled,
    ToolCompleted,
)

_log = logging.getLogger("minion_assistant.runner")

# Default maximum LLM calls per turn.  Kept as a module-level constant so
# existing tests can import it; :func:`run_turn` accepts it as a parameter
# so per-agent overrides are possible without touching this value.
_MAX_TOOL_ROUNDS = 10

# Retry configuration for transient provider errors.
_RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES = 3
_RETRY_BASE_SECONDS = 2.0   # doubles each attempt: 2 s → 4 s → 8 s


def _call_with_retry(fn: Callable[[], LLMResponse]) -> LLMResponse:
    """Call ``fn()``, retrying on transient provider errors with exponential backoff.

    Distinguishes retryable errors (network problems, rate limits, server
    errors) from permanent failures (bad request, auth error).  Only retryable
    errors trigger a retry; permanent failures propagate immediately.

    Jitter (random 0–0.5 s per attempt) prevents thundering-herd when multiple
    agents hit a rate limit at the same time.

    Args:
        fn: Zero-argument callable that calls ``provider.chat()``.

    Returns:
        :class:`LLMResponse` from the first successful call.

    Raises:
        The original exception after ``_MAX_RETRIES`` retryable attempts, or
        immediately for non-retryable errors.
    """
    last_exc: Exception | None = None

    # range(_MAX_RETRIES + 1) gives attempts 0, 1, 2, 3 — four chances total
    # (one original attempt plus up to _MAX_RETRIES retries).
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return fn()  # success — exit immediately, no retry needed
        except Exception as exc:
            last_exc = exc  # save so we can re-raise after all attempts exhausted

            # --- Determine if this error is worth retrying ---
            # HTTP errors carry a .response.status_code attribute; network/OS
            # errors do not, so we use getattr with a None default.
            response_obj = getattr(exc, "response", None)
            status = getattr(response_obj, "status_code", None)

            is_retryable = (
                # 429 = rate limited; 5xx = server-side problem.  Both are
                # temporary — the provider will recover on its own.
                (status is not None and status in _RETRYABLE_HTTP_STATUS)
                # Network-level errors (DNS failure, dropped connection, OS socket error).
                or isinstance(exc, (TimeoutError, ConnectionError, OSError))
                # Some providers raise plain Exception with a description string
                # instead of a typed error class, so also check the message text.
                or any(
                    kw in str(exc).lower()
                    for kw in ("timeout", "connection", "rate limit", "timed out")
                )
            )

            # Stop immediately if the error is permanent (e.g. 400 Bad Request,
            # 401 Unauthorized) or we have exhausted all retry attempts.
            if not is_retryable or attempt == _MAX_RETRIES:
                raise

            # --- Exponential backoff: 2s → 4s → 8s ---
            # 2 ** attempt gives 1, 2, 4 for attempts 0, 1, 2.
            # Multiplying by _RETRY_BASE_SECONDS (2.0) gives 2, 4, 8 seconds.
            # Adding random jitter (0–0.5 s) spreads out simultaneous retries
            # from multiple agents so they don't all hammer the provider at once.
            delay = _RETRY_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
            _log.warning(
                "Provider call failed (attempt %d/%d, %s). Retrying in %.1f s.",
                attempt + 1,
                _MAX_RETRIES,
                type(exc).__name__,
                delay,
            )
            time.sleep(delay)

    # This line is unreachable in practice: the loop either returns on success
    # or raises inside the except block.  It exists only to satisfy the type
    # checker, which cannot prove that raise always runs before the loop ends.
    raise last_exc  # type: ignore[misc]


def _timed_execute(registry: ToolRegistry, name: str, arguments: dict) -> tuple[str, int]:
    """Execute a tool and return ``(output, elapsed_ms)``.

    Wrapped into its own function so ``ThreadPoolExecutor.submit()`` can call it
    as a picklable callable.  Returns the elapsed time in milliseconds so the
    caller can emit a :class:`ToolCompleted` event with timing information.
    ``time.monotonic()`` is used instead of ``time.time()`` because it is
    unaffected by system clock adjustments (NTP, daylight saving, etc.).
    """
    start = time.monotonic()
    output = registry.execute(name, arguments)
    elapsed = int((time.monotonic() - start) * 1000)
    return output, elapsed


def run_turn(
    provider: LLMProvider,
    agent_name: str,
    system: str,
    max_tokens: int,
    tools: ToolRegistry,
    messages: list[dict],
    on_event: Callable[[object], None] | None = None,
    stream: bool = False,
    max_tool_rounds: int = _MAX_TOOL_ROUNDS,
) -> TokenUsage | None:
    """Drive a single user turn to completion, executing tools as needed.

    Sends the current conversation to the LLM, handles any tool calls the
    model requests, and loops until the model produces a final answer.
    Mutates ``messages`` in place by appending all assistant and tool-result
    messages generated during this turn.  All output is routed through the
    optional ``on_event`` callback — this function never calls ``print()``.

    Args:
        provider (LLMProvider): The LLM API client for this agent.
        agent_name (str): Display name forwarded in events, e.g. ``"Ada"``.
        system (str): The system prompt / agent soul.
        max_tokens (int): Maximum tokens the model may generate per response.
        tools (ToolRegistry): Registry of available tools.
        messages (list[dict]): Full conversation history (mutated in place).
        on_event (Callable | None): Optional structured-event callback.
        stream (bool): If ``True`` and ``on_event`` is set, stream tokens.
        max_tool_rounds (int): Maximum LLM→tool iterations before stopping.
            Defaults to :data:`_MAX_TOOL_ROUNDS`.  Pass a higher value for
            task-focused agents that need more sequential tool calls.

    Returns:
        TokenUsage | None: Accumulated token usage across all provider.chat()
            calls within this turn, or None if no provider returned usage data.
            Side effects: appends to ``messages`` and calls ``on_event``.
            If the model requests tools on every round without producing a final
            answer, a :class:`MaxRoundsReached` event is emitted and the function
            returns normally — no exception is raised.
    """
    _header_printed = [False]

    def _on_token(token: str) -> None:
        if on_event is None:
            return
        if not _header_printed[0]:
            on_event(StreamingStarted(agent_name=agent_name))
            _header_printed[0] = True
        on_event(TokenStreamed(token=token))

    _token_callback = _on_token if (on_event is not None and stream) else None

    # Token usage accumulators — summed across all provider.chat() calls in this turn.
    # A single user turn can involve multiple LLM calls (one per tool-call round),
    # so we add up usage from each call and report the total at the end.
    # _has_usage stays False when the provider doesn't report usage (e.g. local models
    # via LM Studio often omit it), so we can return None instead of a misleading zero.
    _total_input = 0
    _total_output = 0
    _has_usage = False

    # Track whether the PREVIOUS round executed tool calls.  When Recovery A
    # fires (empty text after tools ran), we use a targeted nudge that asks the
    # model to acknowledge what it just did — rather than the generic "please
    # respond or call a tool" which leaves open the choice to do nothing.
    _prev_round_had_tools = False

    # --- TAO loop ---
    for _round in range(max_tool_rounds):
        _header_printed[0] = False

        # --- THINK: ask the model what to do next (with retry on transient errors) ---
        response: LLMResponse = _call_with_retry(
            lambda: provider.chat(
                system,
                messages,
                tools.definitions,
                max_tokens,
                on_token=_token_callback,
            )
        )

        # Accumulate token usage from this provider.chat() call.
        if response.usage is not None:
            _total_input += response.usage.input_tokens
            _total_output += response.usage.output_tokens
            _has_usage = True

        assistant_msg: dict = {"role": "assistant", "content": response.text}
        if response.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in response.tool_calls
            ]

        # --- Recovery A: empty response ---
        # Some models occasionally return no text and no tool calls — this can
        # happen when a thinking model produces only internal reasoning with no
        # visible output, when the model considers a tool result "enough of an
        # answer", or when it gets confused by the context.
        #
        # The nudge is two messages: an empty assistant message (to record that
        # the model said nothing this round) followed by a system-style user
        # message.  We choose WHICH nudge based on context:
        #
        # - After tool calls  → targeted nudge: "tell the user what you just did"
        #   This closes the most common failure mode where the model writes a file
        #   or saves to memory, considers the task complete, and returns empty text.
        #   The generic nudge ("please respond or call a tool") leaves open the
        #   option to do nothing again; the targeted nudge removes that option.
        #
        # - No prior tools    → generic nudge: "please respond or call a tool"
        #   The model simply didn't say anything on its first response — the
        #   prompt may have been ambiguous.  Ask it to try again.
        #
        # We only nudge on non-final rounds; on the last round we fall through
        # to MaxRoundsReached.
        if not response.text and not response.tool_calls:
            if _round < max_tool_rounds - 1:
                _log.debug(
                    "Empty response on round %d (prev_had_tools=%s) — injecting recovery nudge.",
                    _round,
                    _prev_round_had_tools,
                )
                messages.append({"role": "assistant", "content": ""})
                if _prev_round_had_tools:
                    # Targeted nudge: the model ran tools but didn't tell the user anything.
                    # Ask it to acknowledge what was done and provide details (e.g. file path).
                    nudge = (
                        "[System: You completed the tool calls but did not respond to "
                        "the user. Please tell the user what was accomplished — include "
                        "the file path if a file was written, or a brief summary of the "
                        "result.]"
                    )
                else:
                    # Generic nudge: no tools ran, model just said nothing.
                    nudge = (
                        "[System: No response received. "
                        "Please respond to the user's message or call a tool to proceed.]"
                    )
                messages.append({"role": "user", "content": nudge})
                _prev_round_had_tools = False  # this round executed no tools
                continue  # re-enter THINK with the nudge in history
            # Last round with still-empty response — fall through to MaxRoundsReached.

        # --- Recovery B: truncated response (model hit max_tokens mid-sentence) ---
        # When `finish_reason == "length"` the model ran out of its token budget
        # before finishing its sentence.  The response ends abruptly mid-thought.
        # We save the partial text into history and inject a continuation prompt
        # so the model can pick up exactly where it stopped.
        #
        # We only attempt this once (checked by _round < max_tool_rounds - 1) to
        # avoid infinite continuation loops if the model keeps truncating.  A
        # second truncation is accepted as-is and emitted as a FinalAnswer.
        #
        # Note: this path only applies when there are no tool calls.  A truncated
        # response that also has tool calls is treated normally — the tool calls
        # are executed and the loop continues as usual.
        if response.finish_reason == "length" and not response.tool_calls:
            messages.append(assistant_msg)
            if _round < max_tool_rounds - 1:
                _log.debug(
                    "Truncated response on round %d — injecting continuation.", _round
                )
                messages.append({
                    "role": "user",
                    "content": (
                        "[System: Your response was cut off. "
                        "Please continue from where you stopped.]"
                    ),
                })
                continue
            # Last round and still truncated — emit what we have.
            if on_event:
                on_event(FinalAnswer(agent_name=agent_name, text=response.text or ""))
            return TokenUsage(_total_input, _total_output) if _has_usage else None

        messages.append(assistant_msg)

        # --- CHECK: final answer or more tool calls? ---
        if response.finish_reason != "tool_calls":
            if on_event:
                on_event(FinalAnswer(agent_name=agent_name, text=response.text or ""))
            _prev_round_had_tools = False  # not needed after return, but keeps state consistent
            return TokenUsage(_total_input, _total_output) if _has_usage else None

        # --- Show preamble text that preceded the tool call (non-streaming only) ---
        # When the model writes "Let me check that for you." before calling a tool,
        # that text is stored in the assistant message (for conversation history)
        # but would never be displayed to the user — the runner only emits
        # FinalAnswer for text that ends the turn, and tool-call rounds don't end.
        #
        # ThoughtEmitted fixes this gap: emit the preamble now so the user can
        # see what the model said before the tool status lines appear.
        #
        # We skip this in streaming mode (`was_streamed=True`) because the text
        # was already printed token-by-token via the on_token callback while the
        # model was generating.  Emitting it again would duplicate the output.
        if response.text and not response.was_streamed and on_event:
            on_event(ThoughtEmitted(agent_name=agent_name, text=response.text))

        # --- ACT: execute each requested tool ---
        # The model may request multiple tools in a single response (e.g. read
        # three files at once).  When all of them are read-only and parse-error-
        # free we can run them in parallel threads — they don't write anything, so
        # there's no risk of one call interfering with another.
        #
        # When any tool is write-capable (bash, write, save_memory, update_task)
        # or has a parse error, we fall back to serial execution so the model's
        # intended ordering is respected.
        _all_read_only = (
            len(response.tool_calls) > 1          # only worth parallelising if there are 2+
            and all(not tc.error for tc in response.tool_calls)   # skip if any args failed to parse
            and all(tools.is_read_only(tc.name) for tc in response.tool_calls)  # all must be safe
        )

        if _all_read_only:
            # Emit all ToolCalled events BEFORE any tool executes.
            # This lets the UI show "[tool: read(...)]" for all pending calls up
            # front, rather than interleaving "started" and "finished" lines.
            for tc in response.tool_calls:
                if on_event:
                    on_event(ToolCalled(name=tc.name, args=tc.arguments))

            # Run all tools concurrently using a thread pool.
            # ThreadPoolExecutor(max_workers=N) creates at most N threads — one
            # per tool call.  executor.submit() schedules each call and returns a
            # Future.  We key _futures by Future so we can look up which tool call
            # each Future belongs to when it completes.
            _results: dict[str, tuple[str, int]] = {}
            with ThreadPoolExecutor(max_workers=len(response.tool_calls)) as executor:
                _futures = {
                    executor.submit(_timed_execute, tools, tc.name, tc.arguments): tc
                    for tc in response.tool_calls
                }
                # as_completed() yields Futures in completion order (fastest first),
                # which may differ from the original call order.  That's fine here
                # because we store results by tool_call_id and re-order below.
                for future in as_completed(_futures):
                    _tc = _futures[future]
                    _out, _elapsed = future.result()
                    _results[_tc.id] = (_out, _elapsed)

            # --- OBSERVE: append tool-result messages in the ORIGINAL call order ---
            # IMPORTANT: the OpenAI API requires that tool-result messages appear
            # in the same order as the tool calls in the preceding assistant message.
            # If we appended in completion order (fastest-first), the provider would
            # reject the request with a validation error.  We restore original order
            # by iterating response.tool_calls (unchanged) and looking up results
            # by tool_call_id from the dict we built above.
            for tc in response.tool_calls:
                _output, _elapsed_ms = _results[tc.id]
                if on_event:
                    on_event(ToolCompleted(name=tc.name, elapsed_ms=_elapsed_ms, output_chars=len(_output)))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": _output})

            # Mark that this round executed tools — Recovery A will use a targeted nudge
            # next round if the model returns empty text without responding to the user.
            _prev_round_had_tools = True

        else:
            # --- Serial execution (write tools, single tool, or parse errors) ---
            # Execute each tool call one at a time in the order the model requested.
            for tc in response.tool_calls:
                if on_event:
                    on_event(ToolCalled(name=tc.name, args=tc.arguments))

                if tc.error:
                    # The provider couldn't parse the model's JSON arguments.
                    # Feed the parse error back as the tool's output so the model
                    # can see what went wrong and retry with corrected arguments.
                    output = tc.error
                    elapsed_ms = 0
                else:
                    _tool_start = time.monotonic()
                    output = tools.execute(tc.name, tc.arguments)
                    elapsed_ms = int((time.monotonic() - _tool_start) * 1000)

                if on_event:
                    on_event(ToolCompleted(name=tc.name, elapsed_ms=elapsed_ms, output_chars=len(output)))

                # --- OBSERVE: append this tool's result before moving to the next ---
                # The result becomes part of the conversation history so the model
                # can see what the tool returned on the next Think step.
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})

            # Mark that this round executed tools — Recovery A will use a targeted nudge
            # next round if the model returns empty text without responding to the user.
            _prev_round_had_tools = True

    # Cap reached without a final answer.
    _limit_msg = (
        f"[Stopped after {max_tool_rounds} tool rounds without a final answer. "
        "Your progress has been saved. Start a new message to continue.]"
    )
    messages.append({"role": "assistant", "content": _limit_msg})
    if on_event:
        on_event(MaxRoundsReached(agent_name=agent_name, message=_limit_msg))
    return TokenUsage(_total_input, _total_output) if _has_usage else None

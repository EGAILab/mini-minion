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

from ..providers import LLMProvider, LLMResponse
from ..tools import ToolRegistry
from .events import (
    FinalAnswer,
    MaxRoundsReached,
    StreamingStarted,
    ThoughtEmitted,
    TokenStreamed,
    ToolCalled,
)

_log = logging.getLogger("mini_minion.runner")

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
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            response_obj = getattr(exc, "response", None)
            status = getattr(response_obj, "status_code", None)
            is_retryable = (
                (status is not None and status in _RETRYABLE_HTTP_STATUS)
                or isinstance(exc, (TimeoutError, ConnectionError, OSError))
                or any(
                    kw in str(exc).lower()
                    for kw in ("timeout", "connection", "rate limit", "timed out")
                )
            )
            if not is_retryable or attempt == _MAX_RETRIES:
                raise
            delay = _RETRY_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
            _log.warning(
                "Provider call failed (attempt %d/%d, %s). Retrying in %.1f s.",
                attempt + 1,
                _MAX_RETRIES,
                type(exc).__name__,
                delay,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]  — unreachable; satisfies mypy


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
) -> None:
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
        None.  Side effects: appends to ``messages`` and calls ``on_event``.
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
        # Some models return nothing on edge cases (unusual prompt characters,
        # uncertainty).  Inject a nudge and loop — don't burn a round on silence.
        if not response.text and not response.tool_calls:
            if _round < max_tool_rounds - 1:
                _log.debug("Empty response on round %d — injecting recovery nudge.", _round)
                messages.append({"role": "assistant", "content": ""})
                messages.append({
                    "role": "user",
                    "content": (
                        "[System: No response received. "
                        "Please respond to the user's message or call a tool to proceed.]"
                    ),
                })
                continue
            # Last round — fall through to MaxRoundsReached.

        # --- Recovery B: truncated response (model hit max_tokens mid-sentence) ---
        # Inject a continuation prompt so the model picks up where it stopped.
        # Only attempt once — accept a second truncation.
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
            return

        messages.append(assistant_msg)

        # --- CHECK: final answer or more tool calls? ---
        if response.finish_reason != "tool_calls":
            if on_event:
                on_event(FinalAnswer(agent_name=agent_name, text=response.text or ""))
            return

        # In non-streaming mode the model may narrate before calling tools.
        # That text was stored in the assistant message but never shown.
        # Emit it now so the caller can display it before tool status lines.
        # Streaming mode skips this — tokens were already sent via on_token.
        if response.text and not response.was_streamed and on_event:
            on_event(ThoughtEmitted(agent_name=agent_name, text=response.text))

        # --- ACT: execute each requested tool ---
        for tc in response.tool_calls:
            if on_event:
                on_event(ToolCalled(name=tc.name, args=tc.arguments))

            if tc.error:
                output = tc.error
            else:
                output = tools.execute(tc.name, tc.arguments)

            # --- OBSERVE: append the tool result ---
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})

    # Cap reached without a final answer.
    _limit_msg = (
        f"[Stopped after {max_tool_rounds} tool rounds without a final answer. "
        "Your progress has been saved. Start a new message to continue.]"
    )
    messages.append({"role": "assistant", "content": _limit_msg})
    if on_event:
        on_event(MaxRoundsReached(agent_name=agent_name, message=_limit_msg))

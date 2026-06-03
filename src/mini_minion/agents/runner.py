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
from collections.abc import Callable

from ..providers import LLMProvider, LLMResponse
from ..tools import ToolRegistry
from .events import FinalAnswer, MaxRoundsReached, StreamingStarted, ThoughtEmitted, TokenStreamed, ToolCalled

# Maximum LLM calls allowed per turn. Prevents indefinite loops when a model
# repeatedly requests tools without producing a final answer.
_MAX_TOOL_ROUNDS = 10


def run_turn(
    provider: LLMProvider,
    agent_name: str,
    system: str,
    max_tokens: int,
    tools: ToolRegistry,
    messages: list[dict],
    on_event: Callable[[object], None] | None = None,
    stream: bool = False,
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
        system (str): The system prompt / agent soul that defines personality
            and behavioral rules.  Prepended to every API call.
        max_tokens (int): Maximum tokens the model may generate per response.
        tools (ToolRegistry): The registry of available tools.  Provides both
            the schema definitions the LLM sees and the execution logic.
        messages (list[dict]): The full conversation history (mutated in place).
        on_event (Callable | None): Optional callback that receives structured
            event objects (:class:`TokenStreamed`, :class:`ToolCalled`,
            :class:`FinalAnswer`, etc.).  Pass ``None`` for silent/headless use.
        stream (bool): If ``True`` *and* ``on_event`` is provided, the provider
            is called with a token callback so the caller receives
            :class:`StreamingStarted` + N×:class:`TokenStreamed` events during
            generation.  Has no effect when ``on_event`` is ``None``.
            Defaults to ``False``.

    Returns:
        None.  Side effects: appends to ``messages`` and calls ``on_event``.
        If the model requests tools on every round without producing a final
        answer, a :class:`MaxRoundsReached` event is emitted and the function
        returns normally — no exception is raised.
    """
    # _header_printed is a mutable cell so the closure below can flip it.
    # Resets to False at the start of each provider call so every new
    # streaming response gets its own StreamingStarted event.
    _header_printed = [False]

    def _on_token(token: str) -> None:
        """Called by the provider for each text token when streaming is active.

        Emits :class:`StreamingStarted` before the very first token so the
        caller knows to print the agent-name prefix, then :class:`TokenStreamed`
        for every subsequent token.
        """
        if on_event is None:
            return
        if not _header_printed[0]:
            on_event(StreamingStarted(agent_name=agent_name))
            _header_printed[0] = True
        on_event(TokenStreamed(token=token))

    # Only pass the streaming callback when the caller wants streaming output.
    # If on_event is None there's nowhere to send tokens, so we always fall
    # back to a plain blocking call in that case.
    _token_callback = _on_token if (on_event is not None and stream) else None

    # --- TAO loop ---
    for _round in range(_MAX_TOOL_ROUNDS):
        # Reset so every new provider call starts with a fresh StreamingStarted.
        _header_printed[0] = False

        # --- THINK: ask the model what to do next ---
        response: LLMResponse = provider.chat(
            system,
            messages,
            tools.definitions,
            max_tokens,
            on_token=_token_callback,
        )

        # Build the assistant message to record in history.
        assistant_msg: dict = {"role": "assistant", "content": response.text}
        if response.tool_calls:
            # Serialize tool_calls into the OpenAI wire format so they can be
            # stored as plain dicts and later re-sent to any provider.
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in response.tool_calls
            ]
        messages.append(assistant_msg)

        # --- CHECK: did the model ask for tools, or is this the final answer? ---
        if response.finish_reason != "tool_calls":
            if on_event:
                on_event(FinalAnswer(agent_name=agent_name, text=response.text or ""))
            return

        # In non-streaming mode the model may narrate before calling tools
        # (e.g. "Let me search memory for you.").  That text was already stored
        # in the assistant message but never shown.  Emit it now so the caller
        # can display it before the tool status lines appear.
        # Streaming mode skips this — the tokens were already sent via on_token.
        if response.text and not response.was_streamed and on_event:
            on_event(ThoughtEmitted(agent_name=agent_name, text=response.text))

        # --- ACT: execute each requested tool ---
        for tc in response.tool_calls:
            # Notify the caller that a tool is about to run.
            if on_event:
                on_event(ToolCalled(name=tc.name, args=tc.arguments))

            if tc.error:
                # Provider couldn't parse the model's JSON arguments — feed the
                # parse error back as the tool observation so the model can retry.
                output = tc.error
            else:
                output = tools.execute(tc.name, tc.arguments)

            # --- OBSERVE: append the tool result to the conversation ---
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})

        # Loop: go back to THINK with the tool results now in context.

    # The model used tools on every allowed round without producing a final answer.
    _limit_msg = (
        f"[Stopped after {_MAX_TOOL_ROUNDS} tool rounds without a final answer. "
        "Start a new message to continue.]"
    )
    messages.append({"role": "assistant", "content": _limit_msg})
    if on_event:
        on_event(MaxRoundsReached(agent_name=agent_name, message=_limit_msg))

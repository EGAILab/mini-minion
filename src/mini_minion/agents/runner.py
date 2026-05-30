"""Agent turn execution — the Think–Act–Observe (TAO) loop.

This is the heart of how agents actually do work. When the user sends a
message, we don't just send it to the LLM and print one reply. Instead, we
run a loop:

  1. **Think**: call the LLM with the current conversation history.
  2. **Act**: if the LLM wants to use a tool, run the tool and collect the result.
  3. **Observe**: append the tool result to the conversation, then loop back.
  4. **Stop**: when the LLM produces a final text answer (no tool calls), print
     it and return.

This loop is what makes an agent more than a chatbot — it can read files,
run commands, search memory, and use what it finds before responding.

Streaming support
-----------------
When ``stream=True`` is passed, a closure (``_on_token``) is created and
forwarded to the provider's ``chat()`` method. The provider calls this closure
once per text token, which prints the agent-name prefix before the first token
and then prints subsequent tokens immediately. This gives the user progressive
output instead of a long pause followed by the complete reply.

The ``_header_printed`` flag (a one-element list used as a mutable cell) tracks
whether the prefix has been printed for the current provider call. It resets at
the start of each loop iteration so every new response starts fresh.

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
- ``minion.py`` — :func:`run_turn` is called there once per user turn.
"""

from __future__ import annotations

import json

from ..providers import LLMProvider, LLMResponse
from ..tools import ToolRegistry

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
    stream: bool = False,
) -> None:
    """Drive a single user turn to completion, executing tools as needed.

    Sends the current conversation to the LLM, handles any tool calls the
    model requests, and loops until the model produces a final answer.
    Mutates ``messages`` in place by appending all assistant and tool-result
    messages generated during this turn.

    Args:
        provider (LLMProvider): The LLM API client for this agent (e.g. an
            :class:`OpenAICompatibleProvider` or :class:`AnthropicProvider`).
        agent_name (str): Display name for console output, e.g. ``"Ada"``.
        system (str): The system prompt / agent soul that defines personality
            and behavioral rules. Prepended to every API call.
        max_tokens (int): Maximum tokens the model may generate per response.
        tools (ToolRegistry): The registry of available tools. Provides both
            the schema definitions the LLM sees and the execution logic.
        messages (list[dict]): The full conversation history (mutated in place).
            Each dict has at minimum a ``"role"`` and ``"content"`` key.
        stream (bool): If ``True``, stream text tokens to the terminal as the
            model generates them. The agent-name prefix is printed before the
            first token. If ``False`` (default), the complete response is
            printed only after the model finishes generating. Defaults to
            ``False`` so task/programmatic callers get the full text at once.

    Returns:
        None. Side effects: appends to ``messages`` and prints to stdout.
        If the model requests tools on every round without producing a final
        answer, a notification is appended to ``messages`` after
        ``_MAX_TOOL_ROUNDS`` rounds and the function returns normally — no
        exception is raised.
    """
    # --- Streaming callback setup ---
    # _header_printed is a one-element list used as a mutable cell so the
    # closure below can flip it from False to True. (A plain bool variable
    # cannot be rebound inside a closure in Python without 'nonlocal'.)
    _header_printed = [False]

    def _on_token(token: str) -> None:
        """Called by the provider for each text token when streaming is active.

        Prints ``"\\nAgentName: "`` before the very first token so the output
        looks like a normal response prefix, then prints subsequent tokens
        without any separator so they appear inline.

        Args:
            token (str): A single text fragment from the model, e.g. ``"Hello"``
                or ``","`` or ``" world"`` (whitespace is part of the token).
        """
        if not _header_printed[0]:
            # First token of this response — print the agent name prefix.
            print(f"\n{agent_name}: ", end="", flush=True)
            _header_printed[0] = True
        # Print the token immediately without a newline or buffering.
        print(token, end="", flush=True)

    # --- TAO loop ---
    for _round in range(_MAX_TOOL_ROUNDS):
        # Reset the header flag at the start of each provider call so that
        # every new model response (including post-tool-call follow-ups)
        # gets its own agent-name prefix before the first token.
        _header_printed[0] = False

        # --- THINK: ask the model what to do next ---
        # Pass the on_token callback only when streaming is requested.
        # When on_token is None the provider uses a plain blocking call.
        response: LLMResponse = provider.chat(
            system,
            messages,
            tools.definitions,
            max_tokens,
            on_token=_on_token if stream else None,
        )

        # Build the assistant message to record in history.
        # If the model wants to call tools, we include the tool_calls field.
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
            if response.was_streamed:
                # Text was already printed token-by-token; just move to a new line.
                print()
            elif response.text:
                # Non-streaming: print the complete response now.
                print(f"\n{agent_name}: {response.text}")
            return

        # --- Model requested tool calls ---
        # If any text was streamed before the tool calls were announced,
        # end that line before printing the [tool: ...] status lines.
        if response.was_streamed:
            print()

        # --- ACT: execute each requested tool ---
        for tc in response.tool_calls:
            # Show the user what the agent is doing (observability).
            print(f"  [tool: {tc.name}({tc.arguments})]")

            if tc.error:
                # Provider couldn't parse the model's JSON arguments — feed the
                # parse error back as the tool observation so the model can retry
                # with corrected JSON rather than crashing the turn.
                output = tc.error
            else:
                # Run the tool. All tools return a plain string — even errors.
                # This means the model can read the error and decide how to recover.
                output = tools.execute(tc.name, tc.arguments)

            # --- OBSERVE: append the tool result to the conversation ---
            # The "tool_call_id" ties this result back to the specific tool_call
            # the model requested above. Required by the OpenAI protocol.
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})

        # Loop: go back to THINK with the tool results now in context.

    # The model used tools on every allowed round without producing a final answer.
    _limit_msg = (
        f"[Stopped after {_MAX_TOOL_ROUNDS} tool rounds without a final answer. "
        "Start a new message to continue.]"
    )
    messages.append({"role": "assistant", "content": _limit_msg})
    print(f"\n{agent_name}: {_limit_msg}")

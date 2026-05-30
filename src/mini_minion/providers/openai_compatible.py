"""OpenAI-compatible provider (Chat Completions API) with optional streaming.

This provider works with any API that speaks the OpenAI Chat Completions
protocol, which includes:
- OpenAI (api.openai.com)
- LM Studio (local server at http://127.0.0.1:1234/v1)
- Aliyun DashScope (https://coding.dashscope.aliyuncs.com/v1)
- Ollama, vLLM, Together AI, and many others

The :class:`LMStudioProvider` (in ``lmstudio.py``) is just an alias for this
class — they are identical.

Streaming support
-----------------
When an ``on_token`` callback is passed to ``chat()``, the provider switches to
the streaming API (``stream=True`` in the SDK call). Text tokens are delivered
to the callback as they arrive, so the terminal shows output progressively
rather than waiting for the full response.

Tool calls in streaming mode require special handling because the model sends
arguments in fragments across multiple chunks. We accumulate those fragments
into a dict indexed by tool-call position, then assemble complete
:class:`ToolCall` objects once the stream ends.

Talks to
--------
- ``openai`` library — the official OpenAI Python SDK.
- ``base.py`` — imports :class:`LLMResponse` and :class:`ToolCall`.
- ``providers/__init__.py`` — imports this class for the factory.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from openai import OpenAI

from .base import LLMResponse, ToolCall


def _parse_tool_arguments(raw: str, tool_name: str) -> tuple[dict, str | None]:
    """Parse a tool-call arguments JSON string without raising.

    Returns:
        ``(parsed_dict, None)`` on success.
        ``({}, error_message)`` when ``raw`` is not valid JSON or not an object —
        the caller should store the error string on the :class:`ToolCall` so the
        runner can feed it back to the model as a recoverable observation.
    """
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        return {}, f"Invalid JSON arguments for '{tool_name}': {exc.msg}"
    if not isinstance(value, dict):
        return {}, f"Tool arguments for '{tool_name}' must be a JSON object, got {type(value).__name__}."
    return value, None


class OpenAICompatibleProvider:
    """LLM provider for any OpenAI-compatible Chat Completions API.

    Args:
        base_url (str): The base URL of the API endpoint, e.g.
            ``"http://127.0.0.1:1234/v1"`` for LM Studio.
        api_key (str): Authentication token. For local providers like LM Studio,
            any non-empty string is accepted.
        model (str): The model ID to use in API requests, e.g.
            ``"qwen-qwen3.5-9b"`` or ``"gpt-4o"``.
    """

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        # The OpenAI SDK handles HTTP, retries, and auth header injection.
        # Passing a custom base_url redirects it to any compatible endpoint.
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Send a conversation to the API, optionally streaming tokens.

        When ``on_token`` is provided, uses the streaming API and calls
        ``on_token`` for each text token as it arrives. Tool call arguments
        are accumulated silently across chunks — they are not streamed to the
        callback because they arrive as structured JSON fragments, not prose.

        When ``on_token`` is ``None``, falls back to a single blocking request
        (the original behaviour).

        Args:
            system (str): System prompt — defines agent personality and rules.
            messages (list[dict]): Conversation history (user, assistant, tool roles).
            tools (list[dict]): Tool definitions in OpenAI function-calling format.
                Omitted from the API call entirely if the list is empty.
            max_tokens (int): Token budget for the model's response.
            on_token (Callable[[str], None] | None): If provided, called once per
                text token with the token string. Enables streaming mode.

        Returns:
            LLMResponse: Parsed response. ``was_streamed`` is ``True`` if at
                least one text token was delivered via ``on_token``.
        """
        # Build the base request parameters shared by both modes.
        kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            # Prepend the system prompt as the first message in the conversation.
            "messages": [{"role": "system", "content": system}, *messages],
        }
        # Only include the "tools" field if there are tools to offer.
        # Some providers reject an empty tools list, so we omit it entirely.
        if tools:
            kwargs["tools"] = tools

        if on_token is not None:
            return self._chat_streaming(kwargs, on_token)
        return self._chat_blocking(kwargs)

    # ------------------------------------------------------------------
    # Private helpers — one per execution path.
    # ------------------------------------------------------------------

    def _chat_blocking(self, kwargs: dict) -> LLMResponse:
        """Make a single blocking API call and return the complete response.

        Args:
            kwargs (dict): Pre-built request parameters for the SDK call.

        Returns:
            LLMResponse: Complete response with text, tool calls, finish reason.
        """
        response = self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        # Parse tool calls from the SDK's typed objects into our ToolCall dataclass.
        # tc.function.arguments comes back as a JSON string — use the safe parser so
        # a malformed JSON response becomes a recoverable model-facing error rather
        # than a JSONDecodeError that crashes the turn.
        tool_calls = []
        for tc in (msg.tool_calls or []):
            args, err = _parse_tool_arguments(tc.function.arguments or "", tc.function.name)
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args, error=err))
        return LLMResponse(
            text=msg.content or "",
            tool_calls=tool_calls,
            # If the model returned tool calls, override finish_reason to "tool_calls"
            # even if the API returned something else — this ensures the runner loops.
            finish_reason="tool_calls" if tool_calls else (response.choices[0].finish_reason or "stop"),
            was_streamed=False,
        )

    def _chat_streaming(
        self,
        kwargs: dict,
        on_token: Callable[[str], None],
    ) -> LLMResponse:
        """Make a streaming API call, delivering text tokens via ``on_token``.

        Iterates over server-sent event chunks. Text delta fragments are passed
        directly to ``on_token``. Tool-call delta fragments (id, name, arguments)
        are accumulated per tool-call index and assembled into :class:`ToolCall`
        objects once the stream is exhausted.

        Args:
            kwargs (dict): Pre-built request parameters. ``stream=True`` is
                added internally before the SDK call.
            on_token (Callable[[str], None]): Called once per text token chunk.

        Returns:
            LLMResponse: Complete response. ``was_streamed=True`` if at least
                one text token was delivered. ``text`` contains the full
                concatenated text regardless of streaming.
        """
        kwargs = {**kwargs, "stream": True}  # copy to avoid mutating the caller's dict

        text_parts: list[str] = []
        # Accumulate tool-call fragments keyed by their position index.
        # The streaming API sends them in pieces: first the id/name, then the
        # arguments JSON string character-by-character.
        tool_accumulators: dict[int, dict] = {}
        finish_reason = "stop"

        stream = self._client.chat.completions.create(**kwargs)
        for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta

            # --- Text tokens ---
            if delta.content:
                on_token(delta.content)
                text_parts.append(delta.content)

            # --- Tool call fragments ---
            # Each chunk may carry a partial update for one or more tool calls.
            # We merge them into accumulators so we can assemble complete ToolCalls
            # after the stream ends.
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index  # position index, e.g. 0 for the first tool call
                    if idx not in tool_accumulators:
                        # First chunk for this tool call — initialise the accumulator.
                        tool_accumulators[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tool_accumulators[idx]["id"] += tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        tool_accumulators[idx]["name"] += tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        # Arguments arrive as a partial JSON string; concatenate them.
                        tool_accumulators[idx]["arguments"] += tc_delta.function.arguments

            if choice.finish_reason:
                finish_reason = choice.finish_reason

        # Assemble complete ToolCall objects from the accumulated fragments.
        # Use the safe parser — a model that sends truncated argument JSON mid-stream
        # becomes a recoverable error rather than a JSONDecodeError.
        tool_calls = []
        for acc in tool_accumulators.values():
            args, err = _parse_tool_arguments(acc["arguments"], acc["name"])
            tool_calls.append(ToolCall(id=acc["id"], name=acc["name"], arguments=args, error=err))

        text = "".join(text_parts)
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else finish_reason,
            # was_streamed is True only if we actually called on_token at least once,
            # meaning the terminal already shows the text. The runner uses this flag
            # to avoid printing the text a second time.
            was_streamed=len(text_parts) > 0,
        )

"""OpenAI Responses API provider — for Codex subscription models.

This provider targets the OpenAI Responses API (``/v1/responses``), which is
the API surface that backs Codex subscription models (``codex-mini-latest``,
``codex-1``) and o-series reasoning models.

Key differences from :class:`~minion_assist.providers.openai_compatible.OpenAICompatibleProvider`
(Chat Completions):

1. **System prompt** is sent as the ``instructions`` field, not a
   ``role="system"`` message prepended to the input list.
2. **Conversation history** is converted to Responses API input format: user
   messages stay the same, but assistant tool calls become standalone
   ``function_call`` items and tool results become ``function_call_output``
   items rather than ``role="tool"`` messages.
3. **Tool definitions** are flattened — ``name``, ``description``, and
   ``parameters`` live at the top level of each tool dict, not nested under a
   ``"function"`` key.
4. **Response structure** differs: output comes as a typed ``output`` list
   instead of ``choices[0].message``.  Text is in items of
   ``type="message"`` and tool calls are in items of ``type="function_call"``.
5. **Usage fields** use ``input_tokens`` / ``output_tokens`` instead of
   ``prompt_tokens`` / ``completion_tokens``.

Conversation history arrives here in the Chat Completions wire format that
the rest of minion-assist uses internally; conversion happens entirely inside
this module so no other module needs to know which API surface is active.

Requires openai SDK >= 2.0 (``client.responses.create`` / ``stream``).

Talks to
--------
- ``openai`` Python SDK — ``client.responses.create()`` and
  ``client.responses.stream()``.
- ``base.py`` — imports :class:`LLMResponse`, :class:`TokenUsage`,
  :class:`ToolCall`.
- ``providers/__init__.py`` — imported for the factory.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from openai import OpenAI

from .base import LLMResponse, TokenUsage, ToolCall


def _convert_messages(messages: list[dict]) -> list:
    """Convert Chat Completions format messages to Responses API input items.

    Each Chat Completions role maps to a different Responses API item type:

    - ``user`` → ``{"role": "user", "content": "..."}`` (unchanged)
    - ``assistant`` with text → ``{"role": "assistant", "content": [{...}]}``
    - ``assistant`` with tool_calls → one ``{"type": "function_call", ...}``
      per call (separate items, not bundled under the assistant role)
    - ``tool`` → ``{"type": "function_call_output", "call_id": ..., "output": ...}``

    Args:
        messages: Conversation history in minion-assist's internal Chat
            Completions format.

    Returns:
        List of Responses API input items ready for ``client.responses.create()``.
    """
    result = []
    for msg in messages:
        role = msg.get("role")

        if role == "user":
            result.append({"role": "user", "content": msg.get("content", "")})

        elif role == "assistant":
            content = msg.get("content")
            tool_calls = msg.get("tool_calls") or []

            # Text content goes as an assistant message with output_text format.
            if content:
                result.append({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                })

            # Each tool call becomes a standalone function_call item — the
            # Responses API does not bundle them under the assistant role.
            for tc in tool_calls:
                result.append({
                    "type": "function_call",
                    "call_id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", "{}"),
                })

        elif role == "tool":
            # Tool results map to function_call_output items, keyed by call_id.
            result.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", ""),
                "output": msg.get("content", ""),
            })

    return result


def _convert_tools(tools: list[dict]) -> list[dict]:
    """Convert Chat Completions tool definitions to Responses API format.

    Chat Completions wraps tool specs under a nested ``"function"`` key::

        {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

    The Responses API expects them flattened to the top level::

        {"type": "function", "name": ..., "description": ..., "parameters": ...}

    Args:
        tools: Tool definitions in Chat Completions format.

    Returns:
        Tool definitions in Responses API format.
    """
    result = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            result.append({
                "type": "function",
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            })
    return result


def _parse_arguments(raw: str, tool_name: str) -> tuple[dict, str | None]:
    """Parse tool-call arguments JSON string without raising.

    Returns:
        ``(parsed_dict, None)`` on success.
        ``({}, error_message)`` on parse failure — the caller stores the error
        on :class:`ToolCall` so the runner feeds it back as a recoverable
        model-facing observation rather than crashing the turn.
    """
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        return {}, f"Invalid JSON arguments for '{tool_name}': {exc.msg}"
    if not isinstance(value, dict):
        return {}, (
            f"Tool arguments for '{tool_name}' must be a JSON object, "
            f"got {type(value).__name__}."
        )
    return value, None


class CodexProvider:
    """LLM provider using the OpenAI Responses API.

    Supports Codex subscription models (``codex-mini-latest``, ``codex-1``)
    and o-series reasoning models that are served through the Responses API
    rather than Chat Completions.

    All conversation history is received in Chat Completions wire format
    (minion-assist's internal format) and converted to Responses API format
    before each call.  The caller never needs to know which API surface is
    active.

    Args:
        base_url (str): API base URL, e.g. ``"https://api.openai.com/v1"``.
        api_key (str): OpenAI API key.  Loaded from the ``OPENAI_API_KEY``
            environment variable by :func:`~minion_assist.config._resolve_provider`.
        model (str): Model identifier, e.g. ``"codex-mini-latest"``.
    """

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        # Codex models can take longer than typical chat models; use a larger
        # timeout than the default.  Retries are handled externally by
        # runner._call_with_retry, so disable the SDK's own retry loop.
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=120.0,
            max_retries=0,
        )
        self._model = model

    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Send a conversation to the Responses API.

        Converts ``messages`` and ``tools`` from Chat Completions format to
        Responses API format before the call, then normalises the typed output
        items back to :class:`LLMResponse`.

        Args:
            system (str): System prompt sent as the ``instructions`` field.
            messages (list[dict]): Full conversation history in Chat
                Completions format.
            tools (list[dict]): Tool definitions in Chat Completions format.
            max_tokens (int): Maximum output token budget.
            on_token (Callable | None): When provided, enables streaming and
                calls this callback with each text token as it arrives.

        Returns:
            LLMResponse: Normalised response with text, tool calls, finish
                reason, streaming flag, and token usage.
        """
        input_items = _convert_messages(messages)
        converted_tools = _convert_tools(tools) if tools else []

        kwargs: dict = {
            "model": self._model,
            "instructions": system,
            "input": input_items,
            "max_output_tokens": max_tokens,
        }
        if converted_tools:
            kwargs["tools"] = converted_tools

        if on_token is not None:
            return self._chat_streaming(kwargs, on_token)
        return self._chat_blocking(kwargs)

    # ------------------------------------------------------------------
    # Private helpers — one per execution path.
    # ------------------------------------------------------------------

    def _chat_blocking(self, kwargs: dict) -> LLMResponse:
        """Make a single blocking Responses API call and return the response."""
        response = self._client.responses.create(**kwargs)
        return self._parse_response(response, was_streamed=False)

    def _chat_streaming(
        self,
        kwargs: dict,
        on_token: Callable[[str], None],
    ) -> LLMResponse:
        """Make a streaming Responses API call, delivering text via ``on_token``.

        Text tokens are delivered to ``on_token`` as they arrive via the
        ``.text_stream`` iterator.  Tool calls are not streamed — they are
        extracted from the final response object after the stream closes.

        Args:
            kwargs (dict): Pre-built request parameters.
            on_token (Callable[[str], None]): Called once per text token.

        Returns:
            LLMResponse: Complete response. ``was_streamed=True`` when at
                least one text token was delivered.
        """
        text_parts: list[str] = []

        with self._client.responses.stream(**kwargs) as stream:
            for text in stream.text_stream:
                on_token(text)
                text_parts.append(text)
            response = stream.get_final_response()

        # Extract tool calls and usage from the final response.  Text is taken
        # from text_parts (already captured) to avoid re-parsing.
        tool_calls: list[ToolCall] = []
        for item in response.output:
            if getattr(item, "type", None) == "function_call":
                args, err = _parse_arguments(item.arguments or "{}", item.name)
                tool_calls.append(ToolCall(
                    id=item.call_id,
                    name=item.name,
                    arguments=args,
                    error=err,
                ))

        usage = _extract_usage(response)
        text = "".join(text_parts)
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            was_streamed=len(text_parts) > 0,
            usage=usage,
        )

    def _parse_response(self, response, was_streamed: bool) -> LLMResponse:
        """Extract text, tool calls, and usage from a Responses API response object.

        Iterates over ``response.output`` and dispatches on item type:
        - ``"message"``     → collects ``output_text`` parts into the text string.
        - ``"function_call"``→ builds a :class:`ToolCall` from call_id / name /
          arguments.
        - ``"reasoning"``   → silently skipped (thinking trace, not for callers).

        Args:
            response: Raw response from ``client.responses.create()``.
            was_streamed (bool): Pass ``True`` when called after streaming so
                ``LLMResponse.was_streamed`` is set correctly.

        Returns:
            LLMResponse: Parsed and normalised response.
        """
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for item in response.output:
            item_type = getattr(item, "type", None)

            if item_type == "message":
                for part in (getattr(item, "content", None) or []):
                    if getattr(part, "type", None) == "output_text":
                        text_parts.append(part.text)

            elif item_type == "function_call":
                args, err = _parse_arguments(
                    getattr(item, "arguments", None) or "{}",
                    getattr(item, "name", ""),
                )
                tool_calls.append(ToolCall(
                    id=getattr(item, "call_id", ""),
                    name=getattr(item, "name", ""),
                    arguments=args,
                    error=err,
                ))
            # "reasoning" items (thinking traces) are intentionally ignored.

        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            was_streamed=was_streamed,
            usage=_extract_usage(response),
        )


def _extract_usage(response) -> TokenUsage | None:
    """Read token usage from a Responses API response object.

    The Responses API uses ``input_tokens`` / ``output_tokens`` field names,
    unlike Chat Completions which uses ``prompt_tokens`` / ``completion_tokens``.

    Returns:
        :class:`TokenUsage` when the response includes usage data, else ``None``.
    """
    raw_usage = getattr(response, "usage", None)
    if raw_usage is None:
        return None
    return TokenUsage(
        input_tokens=getattr(raw_usage, "input_tokens", 0),
        output_tokens=getattr(raw_usage, "output_tokens", 0),
    )

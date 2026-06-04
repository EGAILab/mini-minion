"""Anthropic provider (Claude models) with optional streaming.

Anthropic's API (used by Claude models) is *not* OpenAI-compatible. It uses a
different message structure, different tool call format, and has a different
SDK. This provider adapts between the two formats.

Format differences handled here
---------------------------------
1. **Tool calls in assistant messages**
   - OpenAI: ``{"role": "assistant", "tool_calls": [{"id": "...", "function": {...}}]}``
   - Anthropic: ``{"role": "assistant", "content": [{"type": "tool_use", "id": "...", ...}]}``

2. **Tool results**
   - OpenAI: ``{"role": "tool", "tool_call_id": "...", "content": "..."}``
   - Anthropic: ``{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "..."}]}``
     (Tool results must be wrapped inside a *user* message, not their own role.)
     When consecutive tool results appear, they are merged into a single user
     message with multiple ``tool_result`` blocks.

3. **Tool definitions**
   - OpenAI: ``{"type": "function", "function": {"name": "...", "parameters": {...}}}``
   - Anthropic: ``{"name": "...", "description": "...", "input_schema": {...}}``

4. **Response content**
   - Anthropic returns a list of content blocks (``{"type": "text"}`` or
     ``{"type": "tool_use"}``). We concatenate text blocks and collect tool_use
     blocks into our :class:`ToolCall` list.

Streaming support
-----------------
When ``on_token`` is provided, the provider uses ``client.messages.stream()``
(a context manager in the Anthropic SDK). Text tokens are delivered via
``stream.text_stream``. Tool-use blocks are not part of the text stream —
they appear in the final assembled message retrieved via
``stream.get_final_message()`` after the stream completes.

Lazy import
-----------
The ``anthropic`` package is optional (listed under ``[anthropic]`` extras in
``pyproject.toml``). We only import it inside ``__init__`` so that users without
the package can still use OpenAI-compatible providers.

Talks to
--------
- ``anthropic`` library (optional extra dependency).
- ``base.py`` — imports :class:`LLMResponse` and :class:`ToolCall`.
- ``providers/__init__.py`` — imports this class for the factory.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from .base import LLMResponse, TokenUsage, ToolCall


class AnthropicProvider:
    """LLM provider for Anthropic Claude models.

    Args:
        api_key (str): Your Anthropic API key (``sk-ant-...``).
        model (str): The Claude model ID, e.g.
            ``"claude-3-5-sonnet-20241022"`` or ``"claude-3-haiku-20240307"``.

    Raises:
        ImportError: If the ``anthropic`` package is not installed.
            Install it with ``uv add anthropic`` or
            ``pip install mini-minion[anthropic]``.
    """

    def __init__(self, api_key: str, model: str) -> None:
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install mini-minion[anthropic]")
        self._client = _anthropic.Anthropic(api_key=api_key)
        self._model = model

    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Send a conversation to the Anthropic API, optionally streaming tokens.

        When ``on_token`` is provided, uses ``client.messages.stream()`` to
        receive text tokens as they are generated. Tool-use blocks (if any) are
        not streamed — they are read from the final assembled message after the
        stream completes.

        When ``on_token`` is ``None``, falls back to a single blocking request
        (the original behaviour).

        Args:
            system (str): System prompt sent as Anthropic's top-level ``system`` field.
            messages (list[dict]): Conversation history in OpenAI format.
                Converted by :func:`_to_anthropic_messages`.
            tools (list[dict]): Tool definitions in OpenAI format.
                Converted by :func:`_format_tools`.
            max_tokens (int): Maximum tokens to generate.
            on_token (Callable[[str], None] | None): If provided, called once
                per text token with the token string. Enables streaming mode.

        Returns:
            LLMResponse: Normalized response. ``was_streamed=True`` if at least
                one text token was delivered via ``on_token``.
        """
        # Build the base request parameters shared by both modes.
        kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            # Anthropic takes the system prompt as a top-level field, not inside
            # the messages list — unlike OpenAI which uses {"role": "system"}.
            "system": system,
            "messages": _to_anthropic_messages(messages),
        }
        if tools:
            kwargs["tools"] = _format_tools(tools)

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
        response = self._client.messages.create(**kwargs)

        # Anthropic returns a list of content blocks rather than a single string.
        # We separate them into text parts and tool calls.
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                # block.input is already a dict (not a JSON string) in the Anthropic SDK.
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

        _usage = None
        if hasattr(response, "usage") and response.usage is not None:
            _usage = TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        return LLMResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else (response.stop_reason or "stop"),
            was_streamed=False,
            usage=_usage,
        )

    def _chat_streaming(
        self,
        kwargs: dict,
        on_token: Callable[[str], None],
    ) -> LLMResponse:
        """Make a streaming API call, delivering text tokens via ``on_token``.

        Uses the Anthropic SDK's ``messages.stream()`` context manager.
        ``stream.text_stream`` yields text tokens only — tool-use blocks do
        not appear here. After the stream completes, ``get_final_message()``
        returns the assembled message that includes any tool-use blocks.

        Args:
            kwargs (dict): Pre-built request parameters for the SDK call.
            on_token (Callable[[str], None]): Called once per text token chunk.

        Returns:
            LLMResponse: Complete response. ``was_streamed=True`` if at least
                one text token was delivered.
        """
        text_emitted = False  # track whether on_token was called at least once

        with self._client.messages.stream(**kwargs) as stream:
            # stream.text_stream yields only the text portions of the response.
            # Tool-use blocks are not delivered here; they appear in final_message.
            for text_chunk in stream.text_stream:
                on_token(text_chunk)
                text_emitted = True

            # Retrieve the fully assembled message after the stream completes.
            # This is the canonical source for both text content and tool-use blocks.
            final_message = stream.get_final_message()

        # Extract text and tool calls from the final message content blocks.
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in final_message.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

        _usage = None
        if hasattr(final_message, "usage") and final_message.usage is not None:
            _usage = TokenUsage(
                input_tokens=final_message.usage.input_tokens,
                output_tokens=final_message.usage.output_tokens,
            )
        return LLMResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else (final_message.stop_reason or "stop"),
            # was_streamed is True only if we actually called on_token at least once.
            was_streamed=text_emitted,
            usage=_usage,
        )


def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-format messages to Anthropic's messages format.

    Anthropic has strict requirements for message alternation and tool results.
    This function handles three cases:
    - Regular user/assistant messages: straightforward conversion.
    - Assistant messages with tool calls: converted to content blocks.
    - Tool result messages: must be wrapped in a user role with tool_result blocks.

    Args:
        messages (list[dict]): Messages in OpenAI chat completions format.

    Returns:
        list[dict]: Messages in Anthropic messages API format.
    """
    result: list[dict] = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        if role == "user":
            # Plain user message: direct conversion.
            result.append({"role": "user", "content": content})

        elif role == "assistant":
            if tool_calls:
                # Assistant message WITH tool calls: must use content blocks.
                # Anthropic represents tool calls as "tool_use" blocks inside content.
                blocks: list[dict] = []
                if content:
                    # Include any text the assistant generated before calling a tool.
                    blocks.append({"type": "text", "text": content})
                for tc in tool_calls:
                    fn = tc["function"]
                    try:
                        # Arguments are stored as a JSON string in OpenAI format;
                        # Anthropic wants a parsed dict ("input").
                        input_args = json.loads(fn["arguments"])
                    except (json.JSONDecodeError, KeyError):
                        input_args = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": fn["name"],
                        "input": input_args,
                    })
                result.append({"role": "assistant", "content": blocks})
            else:
                # Regular assistant message: direct conversion.
                result.append({"role": "assistant", "content": content or ""})

        elif role == "tool":
            # Tool result: Anthropic requires this inside a "user" turn as a
            # "tool_result" block — not as a standalone message.
            block: dict = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": content or "",
            }

            # If the previous message is already a user turn with tool_result blocks,
            # append this block to it rather than creating a new user message.
            # This handles the case where multiple tools were called in parallel.
            if (
                result
                and result[-1]["role"] == "user"
                and isinstance(result[-1]["content"], list)
                and result[-1]["content"]
                and result[-1]["content"][-1].get("type") == "tool_result"
            ):
                result[-1]["content"].append(block)
            else:
                result.append({"role": "user", "content": [block]})

    return result


def _format_tools(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-format tool definitions to Anthropic's tool format.

    OpenAI uses ``"parameters"`` for the JSON schema; Anthropic calls it
    ``"input_schema"``. The wrapping structure is also different.

    Args:
        tools (list[dict]): Tool definitions in OpenAI function-calling format.

    Returns:
        list[dict]: Tool definitions in Anthropic's format.
    """
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "input_schema": t["function"]["parameters"],  # renamed field
        }
        for t in tools
    ]

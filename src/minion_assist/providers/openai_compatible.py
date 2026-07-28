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
from pathlib import Path

from openai import OpenAI

from .base import LLMResponse, TokenUsage, ToolCall
from ..messages import EVENT_ID_KEY, content_has_images, materialize_image_data
from ..llm_logger import log_request, log_response


def _convert_content_for_openai(content: str | list) -> str | list:
    """Convert internal content blocks to OpenAI Chat Completions wire format.

    Internal image blocks become {"type": "image_url", "image_url": {"url": "data:..."}}.
    Text blocks become {"type": "text", "text": "..."}.
    String content is returned unchanged (preserves text-only behavior exactly).

    Base64 is materialized here — just before the API call — so JSONL history
    files never contain raw bytes.

    Args:
        content: Either a plain string or a list of internal content blocks.

    Returns:
        The input string unchanged, or a list of OpenAI-format content blocks.
    """
    if isinstance(content, str):
        return content

    if not content_has_images(content):
        # Text-only block list — flatten to a single string for compatibility
        # with providers that don't accept a list when there are no images.
        texts = [b.get("text", "") for b in content if b.get("type") == "text"]
        return " ".join(t for t in texts if t) or ""

    # Mixed content: convert each block to OpenAI format.
    result = []
    for block in content:
        if block.get("type") == "text":
            result.append({"type": "text", "text": block.get("text", "")})
        elif block.get("type") == "image":
            # Read bytes from disk and encode as base64 data URL.
            data = materialize_image_data(block)
            mime = block.get("media_type", "image/png")
            result.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
            })
    return result


def _prepare_messages_for_openai(messages: list[dict]) -> list[dict]:
    """Convert internal content blocks to OpenAI wire format for all messages.

    WHY WE CONVERT HERE (not in session.py or messages.py):
    --------------------------------------------------------
    The internal history stores images as {"type": "image", "path": "...", ...}.
    Base64 is materialized from disk RIGHT HERE, just before the API call, because:
    1. session.py persists history to JSONL — we don't want raw base64 in those files.
    2. Different providers need different wire formats (OpenAI vs Anthropic).
    3. Image data only needs to exist in memory for the duration of one API call.

    Only messages whose content is a list (multimodal) are touched.  Plain
    string content (all existing text-only turns) passes through unchanged —
    this guarantees zero behavior change for text-only conversations.

    Also strips ``EVENT_ID_KEY`` (``messages.py``'s internal mirroring
    metadata — see Stage One Phase 2, slice A): this is the only provider
    conversion in the codebase that rebuilds a message via dict-spread
    (``{**msg, ...}``), so it is the one place an internal-only key could
    otherwise leak into the API request. Anthropic's and Codex's converters
    already extract named fields one at a time and drop it naturally.

    Args:
        messages: Conversation history in minion-assist's internal format.

    Returns:
        New list of messages ready to pass to the OpenAI SDK.
    """
    prepared = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            # Only multimodal messages need conversion — string content is unchanged.
            content = _convert_content_for_openai(content)
        cleaned = {k: v for k, v in msg.items() if k != EVENT_ID_KEY}
        prepared.append({**cleaned, "content": content})
    return prepared


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


def _blocking_response_to_dict(response) -> dict:
    """Serialize an OpenAI SDK ChatCompletion object to a plain dict for logging."""
    choices = []
    for ch in response.choices:
        msg: dict = {"role": "assistant"}
        if ch.message.content is not None:
            msg["content"] = ch.message.content
        if ch.message.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in ch.message.tool_calls
            ]
        choices.append({"index": ch.index, "message": msg, "finish_reason": ch.finish_reason})
    out: dict = {
        "id": response.id,
        "object": response.object,
        "created": response.created,
        "model": response.model,
        "choices": choices,
    }
    if response.usage:
        out["usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return out


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

    def __init__(self, base_url: str, api_key: str, model: str, log_dir: Path | None = None) -> None:
        # The OpenAI SDK handles HTTP, retries, and auth header injection.
        # Passing a custom base_url redirects it to any compatible endpoint.
        #
        # Keep the SDK timeout finite so a provider stream that stops sending
        # chunks returns control to the REPL instead of leaving the CLI wedged.
        # Retries are handled by agents.runner._call_with_retry, so disable the
        # SDK's own retry loop to avoid multiplying wait time.
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=60.0,
            max_retries=0,
        )
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._log_dir = log_dir

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
        # Convert any multimodal content blocks to OpenAI wire format before
        # building the request.  Plain string content is unchanged by this call,
        # so text-only conversations are completely unaffected in performance.
        prepared = _prepare_messages_for_openai(messages)

        # Build the base request parameters shared by both modes.
        kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            # Prepend the system prompt as the first message in the conversation.
            "messages": [{"role": "system", "content": system}, *prepared],
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
        if self._log_dir is not None:
            log_request(self._log_dir, f"{self._base_url}/chat/completions", kwargs)
        response = self._client.chat.completions.create(**kwargs)
        if self._log_dir is not None:
            log_response(self._log_dir, self._model, _blocking_response_to_dict(response))
        msg = response.choices[0].message

        # Parse tool calls from the SDK's typed objects into our ToolCall dataclass.
        # tc.function.arguments comes back as a JSON string — use the safe parser so
        # a malformed JSON response becomes a recoverable model-facing error rather
        # than a JSONDecodeError that crashes the turn.
        tool_calls = []
        for tc in (msg.tool_calls or []):
            args, err = _parse_tool_arguments(tc.function.arguments or "", tc.function.name)
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args, error=err))
        _usage = None
        if response.usage is not None:
            _usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
        return LLMResponse(
            text=msg.content or "",
            tool_calls=tool_calls,
            # If the model returned tool calls, override finish_reason to "tool_calls"
            # even if the API returned something else — this ensures the runner loops.
            finish_reason="tool_calls" if tool_calls else (response.choices[0].finish_reason or "stop"),
            was_streamed=False,
            usage=_usage,
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

        Usage tracking: ``stream_options={"include_usage": True}`` asks the API
        to append a final empty chunk (``choices=[]``) containing the token counts.
        Older or non-OpenAI-compatible providers that don't support this field
        simply ignore it — usage remains ``None`` rather than raising an error.

        Args:
            kwargs (dict): Pre-built request parameters. ``stream=True`` and
                ``stream_options`` are added internally before the SDK call.
            on_token (Callable[[str], None]): Called once per text token chunk.

        Returns:
            LLMResponse: Complete response. ``was_streamed=True`` if at least
                one text token was delivered. ``text`` contains the full
                concatenated text regardless of streaming. ``usage`` is
                populated when the provider supports ``stream_options``.
        """
        # Copy to avoid mutating the caller's dict.
        # stream_options asks the API to include a final usage-only chunk after
        # all content chunks have been delivered.  Not all providers honour this,
        # but those that don't will simply send no final chunk — usage stays None.
        kwargs = {**kwargs, "stream": True, "stream_options": {"include_usage": True}}
        if self._log_dir is not None:
            log_request(self._log_dir, f"{self._base_url}/chat/completions", kwargs)

        text_parts: list[str] = []
        # Accumulate tool-call fragments keyed by their position index.
        # The streaming API sends them in pieces: first the id/name, then the
        # arguments JSON string character-by-character.
        tool_accumulators: dict[int, dict] = {}
        finish_reason = "stop"
        # Captured from the final empty chunk when stream_options.include_usage is set.
        _raw_usage = None

        stream = self._client.chat.completions.create(**kwargs)
        for chunk in stream:
            # When include_usage is set, the API appends one final chunk where
            # choices is empty and usage is populated.  Skip normal delta processing
            # for that chunk, but capture the usage object.
            if not chunk.choices:
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    _raw_usage = chunk.usage
                continue

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

        # Build usage from the final chunk when available.
        _usage = None
        if _raw_usage is not None:
            _usage = TokenUsage(
                input_tokens=_raw_usage.prompt_tokens,
                output_tokens=_raw_usage.completion_tokens,
            )

        text = "".join(text_parts)
        if self._log_dir is not None:
            _tool_calls_log = [
                {"id": a["id"], "type": "function", "function": {"name": a["name"], "arguments": a["arguments"]}}
                for a in tool_accumulators.values()
            ]
            _msg: dict = {"role": "assistant", "content": text or None}
            if _tool_calls_log:
                _msg["tool_calls"] = _tool_calls_log
            _resp_dict: dict = {
                "model": self._model,
                "choices": [{"message": _msg, "finish_reason": "tool_calls" if _tool_calls_log else finish_reason}],
            }
            if _raw_usage is not None:
                _resp_dict["usage"] = {
                    "prompt_tokens": _raw_usage.prompt_tokens,
                    "completion_tokens": _raw_usage.completion_tokens,
                }
            log_response(self._log_dir, self._model, _resp_dict)
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else finish_reason,
            # was_streamed is True only if we actually called on_token at least once,
            # meaning the terminal already shows the text. The runner uses this flag
            # to avoid printing the text a second time.
            was_streamed=len(text_parts) > 0,
            usage=_usage,
        )

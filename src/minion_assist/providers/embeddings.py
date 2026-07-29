"""``EmbeddingProvider`` — turns text into vectors for the memory system's vector lane.

Stage One Phase 4, slice A.

Why a separate class from :class:`~minion_assist.providers.base.LLMProvider`?
--------------------------------------------------------------------------------
``LLMProvider.chat()`` and an embeddings call are fundamentally different
operations (conversation in, text out vs. text in, vector out) with no
shared request/response shape — folding both into one protocol would force
every chat provider to also implement embeddings (most can't; Anthropic
has no embeddings API at all) or grow ``chat()``'s signature with
embedding-only parameters that make no sense for a conversation. Keeping
them separate means a provider that only does chat is unaffected, and this
class only needs to exist at all when ``config.embeddings`` is set.

Why reuse a chat provider's credentials instead of a dedicated API key?
--------------------------------------------------------------------------
Per the plan's Task 1 ("...using existing provider credentials without
coupling memory to chat completions"): most self-hosted / OpenAI-compatible
endpoints (LM Studio, and many hosted gateways) serve `/embeddings`
alongside `/chat/completions` under the same base URL and auth. Requiring a
second, separately configured API key for embeddings would be unnecessary
friction for the common case — see ``config.py``'s ``EmbeddingConfig``,
which resolves ``base_url``/``api_key`` from an *existing*
``models.providers`` entry named in ``config.json``'s ``"embeddings"``
section.

Why the ``openai`` SDK, not raw HTTP?
----------------------------------------
Same reasoning as ``providers/openai_compatible.py``: the SDK already
handles auth headers, timeouts, and request/response parsing for any
OpenAI-compatible endpoint (which is what an embeddings-capable LM Studio
or hosted gateway exposes), so reimplementing that over raw ``httpx`` would
just duplicate working code for no benefit.

Talks to
--------
- ``config.py`` — :class:`~minion_assist.config.EmbeddingConfig` supplies
  the ``base_url``/``api_key``/``model``/``dimensions`` this class is
  constructed from.
- ``memory/postgres_index.py`` — the only caller (Stage One Phase 4, slice
  C): embeds new/changed chunks and stores the result in
  ``memory_chunk_embeddings``.
"""

from __future__ import annotations

from openai import OpenAI


class EmbeddingProvider:
    """Turns text into embedding vectors via an OpenAI-compatible ``/embeddings`` endpoint.

    Args:
        base_url: The API endpoint, e.g. ``"http://127.0.0.1:1234/v1"`` for
            LM Studio. Reused from an existing chat provider's config —
            see the module docstring.
        api_key: Authentication token for that same endpoint.
        model: The embedding model id to request, e.g.
            ``"nomic-embed-text-v1.5"``. Does not need to be one of that
            provider's *chat* models.
        dimensions: The embedding vector's expected length. Not sent in the
            request (not every embeddings API accepts a ``dimensions``
            parameter) — used only by callers that need to know the vector
            size up front, e.g. to size a ``pgvector`` column.
    """

    def __init__(self, base_url: str, api_key: str, model: str, dimensions: int) -> None:
        # Same timeout/no-retry rationale as OpenAICompatibleProvider: keep
        # a finite timeout so an unresponsive endpoint doesn't hang the
        # caller indefinitely, and let the caller's own error handling
        # (memory/postgres_index.py wraps embedding calls in a
        # never-block-the-write try/except) decide whether to retry.
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0, max_retries=0)
        self.model = model
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, one vector per input, in the same order.

        Args:
            texts: The strings to embed. An empty list returns an empty
                list without making a request.

        Returns:
            list[list[float]]: One embedding vector per input text, in the
                same order as ``texts``.

        Raises:
            Exception: Whatever the underlying HTTP/SDK call raises (e.g. a
                connection error, an unknown model). Deliberately not
                caught here — callers that need a never-fail contract (the
                write path) wrap this themselves, the same split
                responsibility ``memory/extractor.py``'s ``extract_facts``
                uses for provider calls.
        """
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model, input=texts)
        # The API guarantees response.data is ordered to match the input list.
        return [item.embedding for item in response.data]

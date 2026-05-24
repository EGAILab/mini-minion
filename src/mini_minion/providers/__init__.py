from .base import LLMProvider, LLMResponse, ToolCall
from .openai_compatible import OpenAICompatibleProvider
from .anthropic import AnthropicProvider
from .lmstudio import LMStudioProvider


def create_provider(api: str, base_url: str, api_key: str, model: str) -> LLMProvider:
    match api:
        case "anthropic":
            return AnthropicProvider(api_key=api_key, model=model)
        case "lmstudio":
            return LMStudioProvider(base_url=base_url, api_key=api_key, model=model)
        case _:  # openai-completions, openai-responses, openai, etc.
            return OpenAICompatibleProvider(base_url=base_url, api_key=api_key, model=model)


__all__ = [
    "LLMProvider",
    "LLMResponse",
    "ToolCall",
    "OpenAICompatibleProvider",
    "AnthropicProvider",
    "LMStudioProvider",
    "create_provider",
]

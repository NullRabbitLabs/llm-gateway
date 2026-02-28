"""LLM Service with provider selection and fallback logic."""

import logging
from typing import Optional

from config import Config
from providers.base import Provider, ProviderError, LLMResponse
from providers.deepseek import DeepSeekProvider
from providers.gemini import GeminiProvider
from providers.openai_provider import OpenAIProvider
from providers.anthropic_provider import AnthropicProvider
from providers.ollama import OllamaProvider

log = logging.getLogger("llm-gateway.service")


class AllProvidersFailedError(Exception):
    """Raised when all LLM providers fail."""

    pass


class LLMService:
    """Service for making LLM calls with auto-fallback."""

    def __init__(self, config: Config):
        """Initialize LLM service with configured providers.

        Args:
            config: Configuration with API keys and provider settings
        """
        self.config = config
        self.providers: list[Provider] = self._init_providers(config)

        if not self.providers:
            raise ValueError("No LLM providers configured")

        log.info(f"LLM Service initialized with {len(self.providers)} providers: "
                 f"{[p.name for p in self.providers]}")

    def _init_providers(self, config: Config) -> list[Provider]:
        """Initialize providers based on configuration.

        In auto mode, providers are ordered by cost-effectiveness:
        0. Ollama (free / local)
        1. DeepSeek (cheapest)
        2. Gemini (very cheap)
        3. OpenAI (mid-tier)
        4. Anthropic (premium)
        """
        providers = []

        if config.provider == "auto":
            # Auto mode: add all configured providers in cost order
            if config.ollama_host and config.ollama_model:
                providers.append(OllamaProvider(config.ollama_host, config.ollama_model))
            if config.deepseek_api_key and config.deepseek_model:
                providers.append(DeepSeekProvider(config.deepseek_api_key, config.deepseek_model))
            if config.gemini_api_key and config.gemini_model:
                providers.append(GeminiProvider(config.gemini_api_key, config.gemini_model))
            if config.openai_api_key and config.openai_model:
                providers.append(OpenAIProvider(config.openai_api_key, config.openai_model))
            if config.anthropic_api_key and config.anthropic_model:
                providers.append(AnthropicProvider(config.anthropic_api_key, config.anthropic_model))
        else:
            # Explicit provider selection
            if config.provider == "ollama":
                providers.append(OllamaProvider(config.ollama_host, config.ollama_model))
            elif config.provider == "deepseek":
                providers.append(DeepSeekProvider(config.deepseek_api_key, config.deepseek_model))
            elif config.provider == "gemini":
                providers.append(GeminiProvider(config.gemini_api_key, config.gemini_model))
            elif config.provider == "openai":
                providers.append(OpenAIProvider(config.openai_api_key, config.openai_model))
            elif config.provider == "anthropic":
                providers.append(AnthropicProvider(config.anthropic_api_key, config.anthropic_model))

        return providers

    def call(self, prompt: str, system_prompt: Optional[str] = None, model_override: str | None = None) -> LLMResponse:
        """Call LLM with auto-fallback on failure.

        Args:
            prompt: User prompt/context
            system_prompt: Optional system prompt

        Returns:
            LLMResponse with text and metadata

        Raises:
            AllProvidersFailedError: If all providers fail
        """
        errors = []

        for provider in self.providers:
            try:
                log.info(f"Calling provider: {provider.name}")
                response = provider.call(prompt, system_prompt, model_override=model_override)
                log.info(f"Provider {provider.name} succeeded: {response.prompt_tokens} prompt tokens, "
                         f"{response.completion_tokens} completion tokens, {response.latency_ms}ms")
                return response
            except ProviderError as e:
                log.warning(f"Provider {provider.name} failed: {e}")
                errors.append(f"{provider.name}: {e}")
                continue

        error_msg = "All providers failed:\n" + "\n".join(errors)
        log.error(error_msg)
        raise AllProvidersFailedError(error_msg)

    def call_with_tools(
        self,
        messages: list,
        tools: list | None = None,
        model_override: str | None = None,
    ) -> LLMResponse:
        """Call LLM with full message history and optional tools, with auto-fallback.

        Args:
            messages: Full conversation history as list of dicts (OpenAI format)
            tools: Optional list of OpenAI function schema dicts

        Returns:
            LLMResponse with text, tool_calls, and finish_reason

        Raises:
            AllProvidersFailedError: If all providers fail
        """
        errors = []

        for provider in self.providers:
            try:
                log.info(f"Calling provider with tools: {provider.name}" + (f" (model override: {model_override})" if model_override else ""))
                response = provider.call_with_tools(messages, tools, model_override=model_override)
                log.info(f"Provider {provider.name} succeeded: {response.prompt_tokens} prompt tokens, "
                         f"{response.completion_tokens} completion tokens, {response.latency_ms}ms")
                return response
            except ProviderError as e:
                log.warning(f"Provider {provider.name} failed: {e}")
                errors.append(f"{provider.name}: {e}")
                continue

        error_msg = "All providers failed:\n" + "\n".join(errors)
        log.error(error_msg)
        raise AllProvidersFailedError(error_msg)

    def get_provider_info(self) -> list[dict]:
        """Get information about configured providers."""
        return [
            {"name": p.name, "model": p.model}
            for p in self.providers
        ]

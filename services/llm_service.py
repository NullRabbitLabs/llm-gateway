"""LLM Service with provider selection and fallback logic."""

import logging
from typing import Optional

from config import Config
from providers.base import Provider, ProviderError, LLMResponse
from providers.registry import create_providers_for_mode

log = logging.getLogger("llm-gateway.service")


class AllProvidersFailedError(Exception):
    """Raised when all LLM providers fail."""

    pass


class LLMService:
    """Service for making LLM calls with auto-fallback."""

    def __init__(self, config: Config):
        """Initialize LLM service with configured providers.

        Args:
            config: Configuration with provider settings
        """
        self.config = config
        self.providers: list[Provider] = create_providers_for_mode(
            config.provider, config.auto_priority, config.provider_configs,
        )

        if not self.providers:
            raise ValueError("No LLM providers configured")

        log.info(f"LLM Service initialized with {len(self.providers)} providers: "
                 f"{[p.name for p in self.providers]}")

    def call(self, prompt: str, system_prompt: Optional[str] = None, model_override: str | None = None) -> LLMResponse:
        """Call LLM with auto-fallback on failure.

        Args:
            prompt: User prompt/context
            system_prompt: Optional system prompt
            model_override: Optional model name override

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
            model_override: Optional model name override

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

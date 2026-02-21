"""Base provider interface and common utilities."""

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("llm-gateway.providers")


class ProviderError(Exception):
    """Base exception for provider errors."""

    pass


class RateLimitError(ProviderError):
    """Raised when provider rate limit is hit."""

    pass


@dataclass
class ToolCall:
    """A single tool invocation requested by the LLM."""

    id: str
    name: str
    arguments: dict  # already parsed from JSON


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    text: str | None
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_microcents: int
    latency_ms: int
    tool_calls: list = field(default_factory=list)
    finish_reason: str = "stop"
    reasoning_content: str | None = None


def _parse_tool_arguments(args_str: str) -> dict:
    """Parse tool call arguments JSON with resilience for common LLM formatting errors."""
    try:
        return json.loads(args_str)
    except (json.JSONDecodeError, TypeError):
        pass

    # Strip markdown fences
    stripped = re.sub(r"^```(?:json)?\s*\n?(.*?)\n?```$", r"\1", args_str.strip(), flags=re.DOTALL)
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        pass

    # Strip trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", stripped)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass

    log.warning("Failed to parse tool arguments: %r", args_str)
    return {"_parse_error": args_str}


class Provider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @property
    @abstractmethod
    def name(self) -> str:
        """Return provider name."""
        pass

    @abstractmethod
    def _call_api(self, prompt: str, system_prompt: Optional[str]) -> LLMResponse:
        """Make the actual API call. Subclasses implement this."""
        pass

    def _call_api_with_tools(
        self,
        messages: list,
        tools: list | None,
        model_override: str | None = None,
    ) -> LLMResponse:
        """Make API call with full message history and tools.

        Default implementation raises ProviderError. Override in providers
        that support tool calls.
        """
        raise ProviderError(f"{self.name} does not support tool calls")

    def call(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Call the provider with retry logic.

        Args:
            prompt: User prompt/context
            system_prompt: Optional system prompt

        Returns:
            LLMResponse with text and metadata

        Raises:
            ProviderError: If call fails after retries
        """
        max_retries = 2
        delay = 2.0

        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                response = self._call_api(prompt, system_prompt)
                response.latency_ms = int((time.time() - start_time) * 1000)
                return response
            except RateLimitError as e:
                if attempt < max_retries:
                    log.warning(f"{self.name}: Rate limit hit, retrying in {delay}s (attempt {attempt + 1})")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise ProviderError(f"{self.name} rate limited after {max_retries + 1} attempts") from e
            except Exception as e:
                error_str = str(e).lower()
                is_client_error = any(
                    f"error code: {code}" in error_str or f" {code} " in error_str
                    for code in ["400", "401", "403", "404", "422"]
                )
                if is_client_error:
                    raise ProviderError(f"{self.name} call failed (non-retryable 4xx): {e}") from e
                is_retryable = any(
                    keyword in error_str
                    for keyword in ["timeout", "connection", "rate limit", "429", "502", "503", "504"]
                )
                if is_retryable and attempt < max_retries:
                    log.warning(f"{self.name}: Retryable error, retrying in {delay}s: {e}")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise ProviderError(f"{self.name} call failed: {e}") from e

        raise ProviderError(f"{self.name} call failed after all retries")

    def call_with_tools(
        self,
        messages: list,
        tools: list | None = None,
        model_override: str | None = None,
    ) -> LLMResponse:
        """Call the provider with full message history and optional tools.

        Args:
            messages: Full conversation history as list of dicts (OpenAI format)
            tools: Optional list of OpenAI function schema dicts

        Returns:
            LLMResponse with text, tool_calls, and finish_reason

        Raises:
            ProviderError: If call fails after retries or provider doesn't support tools
        """
        max_retries = 2
        delay = 2.0

        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                response = self._call_api_with_tools(messages, tools, model_override=model_override)
                response.latency_ms = int((time.time() - start_time) * 1000)
                return response
            except RateLimitError as e:
                if attempt < max_retries:
                    log.warning(f"{self.name}: Rate limit hit, retrying in {delay}s (attempt {attempt + 1})")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise ProviderError(f"{self.name} rate limited after {max_retries + 1} attempts") from e
            except ProviderError:
                raise
            except Exception as e:
                error_str = str(e).lower()
                is_client_error = any(
                    f"error code: {code}" in error_str or f" {code} " in error_str
                    for code in ["400", "401", "403", "404", "422"]
                )
                if is_client_error:
                    raise ProviderError(f"{self.name} call failed (non-retryable 4xx): {e}") from e
                is_retryable = any(
                    keyword in error_str
                    for keyword in ["timeout", "connection", "rate limit", "429", "502", "503", "504"]
                )
                if is_retryable and attempt < max_retries:
                    log.warning(f"{self.name}: Retryable error, retrying in {delay}s: {e}")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise ProviderError(f"{self.name} call failed: {e}") from e

        raise ProviderError(f"{self.name} call failed after all retries")


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM response.

    Args:
        text: Response text potentially wrapped in markdown

    Returns:
        Cleaned text without markdown fences
    """
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()

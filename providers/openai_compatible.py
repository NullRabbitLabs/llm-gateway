"""Unified OpenAI-compatible provider (covers DeepSeek, OpenAI, Groq, Together, etc.)."""

import logging
from typing import Any, Optional

from openai import OpenAI

from .base import (
    Provider,
    LLMResponse,
    ToolCall,
    RateLimitError,
    _parse_tool_arguments,
    strip_markdown_fences,
)

log = logging.getLogger("llm-gateway.providers.openai_compatible")


class OpenAICompatibleProvider(Provider):
    """Provider for any OpenAI-compatible API, parameterized by ProviderConfig."""

    def __init__(self, api_key: str, model: str, config=None):
        super().__init__(api_key, model)
        self._config = config

        # Extract settings from config (or sensible defaults)
        base_url = config.base_url if config else None
        timeout = config.timeout if config else 300
        self._api_params: dict[str, Any] = dict(config.api_params) if config else {}
        features = config.features if config else {}
        pricing = config.pricing if config else {}

        self._provider_name: str = config.name if config else "openai"
        self._json_mode: bool = features.get("json_mode", False)
        self._reasoning_content: bool = features.get("reasoning_content", False)
        self._input_cost_per_1k: float = pricing.get("input_per_1k_microcents", 0)
        self._output_cost_per_1k: float = pricing.get("output_per_1k_microcents", 0)

        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    @property
    def name(self) -> str:
        return self._provider_name

    def _compute_cost(self, prompt_tokens: int, completion_tokens: int) -> int:
        input_cost = (prompt_tokens / 1000) * self._input_cost_per_1k
        output_cost = (completion_tokens / 1000) * self._output_cost_per_1k
        return round(input_cost + output_cost)

    def _call_api(self, prompt: str, system_prompt: Optional[str], model_override: str | None = None) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": model_override or self.model,
            "messages": messages,
            **self._api_params,
        }
        if self._json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                raise RateLimitError(str(e)) from e
            raise

        text = response.choices[0].message.content if response.choices else None
        if text is not None:
            text = strip_markdown_fences(text)

        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0

        return LLMResponse(
            text=text,
            provider=self.name,
            model=model_override or self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_microcents=self._compute_cost(prompt_tokens, completion_tokens),
            latency_ms=0,
        )

    def _call_api_with_tools(self, messages: list, tools: list | None, model_override: str | None = None) -> LLMResponse:
        model = model_override or self.model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                raise RateLimitError(str(e)) from e
            raise

        if not response.choices:
            return LLMResponse(
                text=None, provider=self.name, model=model,
                prompt_tokens=0, completion_tokens=0, cost_microcents=0, latency_ms=0,
            )

        choice = response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason or "stop"
        resp_model = getattr(response, "model", None) or model

        text = message.content
        reasoning_content = getattr(message, "reasoning_content", None) if self._reasoning_content else None

        raw_tool_calls = message.tool_calls or []
        parsed_tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=_parse_tool_arguments(tc.function.arguments or "{}"),
            )
            for tc in raw_tool_calls
        ]

        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0

        return LLMResponse(
            text=text,
            provider=self.name,
            model=resp_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_microcents=self._compute_cost(prompt_tokens, completion_tokens),
            latency_ms=0,
            tool_calls=parsed_tool_calls,
            finish_reason=finish_reason,
            reasoning_content=reasoning_content,
        )

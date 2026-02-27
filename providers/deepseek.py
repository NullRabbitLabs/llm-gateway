"""DeepSeek provider implementation (OpenAI-compatible API)."""

import logging
from typing import Optional

from openai import OpenAI

from .base import Provider, LLMResponse, ToolCall, ProviderError, RateLimitError, _parse_tool_arguments, strip_markdown_fences

log = logging.getLogger("llm-gateway.providers.deepseek")

# DeepSeek pricing: $0.12/1M input, $0.20/1M output
# In microcents (1 microcent = $0.000001)
DEEPSEEK_INPUT_COST_PER_1K = 0.12  # microcents
DEEPSEEK_OUTPUT_COST_PER_1K = 0.20  # microcents


class DeepSeekProvider(Provider):
    """DeepSeek provider using OpenAI-compatible API."""

    # DeepSeek reasoner models can take 100+ seconds for complex prompts
    # Set timeout to 300s (5 min) to accommodate slow responses
    DEFAULT_TIMEOUT = 300

    def __init__(self, api_key: str, model: str, timeout: int = DEFAULT_TIMEOUT):
        super().__init__(api_key, model)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        return "deepseek"

    def _call_api(self, prompt: str, system_prompt: Optional[str], model_override: str | None = None) -> LLMResponse:
        """Call DeepSeek API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=model_override or self.model,
                messages=messages,
            )
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                raise RateLimitError(str(e)) from e
            raise

        text = response.choices[0].message.content
        text = strip_markdown_fences(text)

        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0

        # Calculate cost in microcents
        input_cost = (prompt_tokens / 1000) * DEEPSEEK_INPUT_COST_PER_1K
        output_cost = (completion_tokens / 1000) * DEEPSEEK_OUTPUT_COST_PER_1K
        cost_microcents = round(input_cost + output_cost)

        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_microcents=cost_microcents,
            latency_ms=0,  # Set by base class
        )

    def _call_api_with_tools(self, messages: list, tools: list | None, model_override: str | None = None) -> LLMResponse:
        """Call DeepSeek API with full message history and optional tools."""
        model = model_override or self.model
        kwargs = {
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

        choice = response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason or "stop"
        resp_model = getattr(response, "model", None) or self.model

        text = message.content
        reasoning_content = getattr(message, "reasoning_content", None)

        raw_tool_calls = message.tool_calls or []
        parsed_tool_calls = []
        for tc in raw_tool_calls:
            fn = tc.function
            arguments = _parse_tool_arguments(fn.arguments or "{}")
            parsed_tool_calls.append(
                ToolCall(id=tc.id, name=fn.name, arguments=arguments)
            )

        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        input_cost = (prompt_tokens / 1000) * DEEPSEEK_INPUT_COST_PER_1K
        output_cost = (completion_tokens / 1000) * DEEPSEEK_OUTPUT_COST_PER_1K
        cost_microcents = round(input_cost + output_cost)

        return LLMResponse(
            text=text,
            provider=self.name,
            model=resp_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_microcents=cost_microcents,
            latency_ms=0,  # Set by base class
            tool_calls=parsed_tool_calls,
            finish_reason=finish_reason,
            reasoning_content=reasoning_content,
        )

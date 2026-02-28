"""Anthropic Claude provider implementation."""

import logging
from typing import Any, Optional

import anthropic

from .base import Provider, LLMResponse, ToolCall, ProviderError, RateLimitError, _parse_tool_arguments, strip_markdown_fences

log = logging.getLogger("llm-gateway.providers.anthropic")

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
}


def _convert_messages_to_anthropic(messages: list) -> tuple[str | None, list]:
    """Convert OpenAI-format messages to Anthropic format.

    Returns (system_text, converted_messages).
    """
    system_text = None
    converted = []
    tool_result_buffer: list[dict] = []

    def flush_tool_results():
        if tool_result_buffer:
            converted.append({"role": "user", "content": list(tool_result_buffer)})
            tool_result_buffer.clear()

    for msg in messages:
        role = msg.get("role")

        if role == "system":
            system_text = msg.get("content", "")
            continue

        if role == "tool":
            tool_result_buffer.append({
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            })
            continue

        flush_tool_results()

        if role == "assistant" and msg.get("tool_calls"):
            content_blocks: list[dict] = []
            text = msg.get("content")
            if text:
                content_blocks.append({"type": "text", "text": text})
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                args_str = fn.get("arguments", "{}")
                input_dict = _parse_tool_arguments(args_str)
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": input_dict,
                })
            converted.append({"role": "assistant", "content": content_blocks})
        else:
            converted.append(msg)

    flush_tool_results()
    return system_text, converted


def _convert_tools_to_anthropic(tools: list) -> list:
    """Convert OpenAI function schemas to Anthropic tool format."""
    result = []
    for tool in tools:
        fn = tool.get("function", {})
        result.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {}),
        })
    return result


class AnthropicProvider(Provider):
    """Anthropic Claude provider."""

    def __init__(self, api_key: str, model: str, config=None):
        super().__init__(api_key, model)
        self._config = config

        timeout = config.timeout if config else 300
        api_params = config.api_params if config else {}
        pricing = config.pricing if config else {}

        self._max_tokens: int = api_params.get("max_tokens", 16000)
        self._temperature: float = api_params.get("temperature", 0.1)
        self._input_cost_per_1k: float = pricing.get("input_per_1k_microcents", 300.0)
        self._output_cost_per_1k: float = pricing.get("output_per_1k_microcents", 1500.0)

        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout)

    @property
    def name(self) -> str:
        return "anthropic"

    def _compute_cost(self, prompt_tokens: int, completion_tokens: int) -> int:
        input_cost = (prompt_tokens / 1000) * self._input_cost_per_1k
        output_cost = (completion_tokens / 1000) * self._output_cost_per_1k
        return round(input_cost + output_cost)

    def _call_api(self, prompt: str, system_prompt: Optional[str], model_override: str | None = None) -> LLMResponse:
        """Call Anthropic API."""
        try:
            kwargs: dict[str, Any] = {
                "model": model_override or self.model,
                "max_tokens": self._max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self._temperature,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = self.client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                raise RateLimitError(str(e)) from e
            raise

        # Extract text from response
        text = ""
        for block in response.content:
            if getattr(block, "type", "") == "text":
                text += block.text

        text = strip_markdown_fences(text)

        prompt_tokens = response.usage.input_tokens if response.usage else 0
        completion_tokens = response.usage.output_tokens if response.usage else 0

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
        """Call Anthropic API with full message history and optional tools."""
        system_text, anthropic_messages = _convert_messages_to_anthropic(messages)
        model = model_override or self.model

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "messages": anthropic_messages,
        }
        if system_text is not None:
            kwargs["system"] = system_text
        if tools:
            kwargs["tools"] = _convert_tools_to_anthropic(tools)

        try:
            response = self.client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                raise RateLimitError(str(e)) from e
            raise

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                text_parts.append(block.text)
            elif block_type == "tool_use":
                input_data = block.input if isinstance(block.input, dict) else {}
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=input_data)
                )

        text = text_parts[0] if text_parts else None
        stop_reason = getattr(response, "stop_reason", "end_turn") or "end_turn"
        finish_reason = _STOP_REASON_MAP.get(stop_reason, stop_reason)
        resp_model = getattr(response, "model", None) or model

        prompt_tokens = response.usage.input_tokens if response.usage else 0
        completion_tokens = response.usage.output_tokens if response.usage else 0

        return LLMResponse(
            text=text,
            provider=self.name,
            model=resp_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_microcents=self._compute_cost(prompt_tokens, completion_tokens),
            latency_ms=0,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

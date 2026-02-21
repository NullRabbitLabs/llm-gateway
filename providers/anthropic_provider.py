"""Anthropic Claude provider implementation."""

import logging
from typing import Optional

import anthropic

from .base import Provider, LLMResponse, ToolCall, ProviderError, RateLimitError, _parse_tool_arguments, strip_markdown_fences

log = logging.getLogger("llm-gateway.providers.anthropic")

# Anthropic Claude Sonnet pricing: $3/1M input, $15/1M output
# In microcents (1 microcent = $0.000001)
ANTHROPIC_INPUT_COST_PER_1K = 300.0  # microcents
ANTHROPIC_OUTPUT_COST_PER_1K = 1500.0  # microcents

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
    tool_result_buffer = []

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
            content_blocks = []
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

    # Set explicit timeout to handle slow responses
    DEFAULT_TIMEOUT = 300

    def __init__(self, api_key: str, model: str, timeout: int = DEFAULT_TIMEOUT):
        super().__init__(api_key, model)
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout)

    @property
    def name(self) -> str:
        return "anthropic"

    def _call_api(self, prompt: str, system_prompt: Optional[str]) -> LLMResponse:
        """Call Anthropic API."""
        try:
            kwargs = {
                "model": self.model,
                "max_tokens": 16000,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
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

        # Calculate cost in microcents
        input_cost = (prompt_tokens / 1000) * ANTHROPIC_INPUT_COST_PER_1K
        output_cost = (completion_tokens / 1000) * ANTHROPIC_OUTPUT_COST_PER_1K
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
        """Call Anthropic API with full message history and optional tools.

        Translates OpenAI-format messages and tools to Anthropic format on the
        way in, and translates the response back to LLMResponse with ToolCall objects.
        """
        system_text, anthropic_messages = _convert_messages_to_anthropic(messages)
        model = model_override or self.model

        kwargs = {
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

        text_parts = []
        tool_calls = []
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
        resp_model = getattr(response, "model", None) or self.model

        prompt_tokens = response.usage.input_tokens if response.usage else 0
        completion_tokens = response.usage.output_tokens if response.usage else 0
        input_cost = (prompt_tokens / 1000) * ANTHROPIC_INPUT_COST_PER_1K
        output_cost = (completion_tokens / 1000) * ANTHROPIC_OUTPUT_COST_PER_1K
        cost_microcents = round(input_cost + output_cost)

        return LLMResponse(
            text=text,
            provider=self.name,
            model=resp_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_microcents=cost_microcents,
            latency_ms=0,  # Set by base class
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

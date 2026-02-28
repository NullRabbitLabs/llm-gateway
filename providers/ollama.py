"""Ollama provider implementation (local/remote HTTP API)."""

import logging
import uuid
from typing import Optional

import httpx

from .base import (
    Provider,
    LLMResponse,
    ToolCall,
    ProviderError,
    RateLimitError,
    _parse_tool_arguments,
    strip_markdown_fences,
)

log = logging.getLogger("llm-gateway.providers.ollama")


class OllamaProvider(Provider):
    """Ollama provider using its native HTTP API."""

    def __init__(self, host: str, model: str, config=None):
        super().__init__(api_key="", model=model)
        self._config = config
        self.host = host.rstrip("/")
        self.timeout = config.timeout if config else 120

    @property
    def name(self) -> str:
        return "ollama"

    def _post(self, payload: dict) -> dict:
        """POST to Ollama /api/chat and return parsed JSON."""
        url = f"{self.host}/api/chat"
        try:
            resp = httpx.post(url, json=payload, timeout=self.timeout)
        except httpx.ConnectError as e:
            raise ProviderError(f"ollama: cannot connect to {self.host} — is Ollama running?") from e
        except httpx.TimeoutException as e:
            raise ProviderError(f"ollama: timeout after {self.timeout}s") from e

        if resp.status_code == 429:
            raise RateLimitError("ollama: rate limited")
        if resp.status_code >= 400:
            raise ProviderError(f"ollama: HTTP {resp.status_code} — {resp.text}")

        return resp.json()

    def _call_api(self, prompt: str, system_prompt: Optional[str], model_override: str | None = None) -> LLMResponse:
        """Call Ollama /api/chat."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = self._post({
            "model": model_override or self.model,
            "messages": messages,
            "stream": False,
        })

        text = data.get("message", {}).get("content", "")
        text = strip_markdown_fences(text)

        prompt_tokens = data.get("prompt_eval_count", 0) or 0
        completion_tokens = data.get("eval_count", 0) or 0

        return LLMResponse(
            text=text,
            provider=self.name,
            model=model_override or self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_microcents=0,
            latency_ms=0,
        )

    def _call_api_with_tools(
        self,
        messages: list,
        tools: list | None,
        model_override: str | None = None,
    ) -> LLMResponse:
        """Call Ollama /api/chat with full message history and optional tools."""
        payload: dict = {
            "model": model_override or self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        data = self._post(payload)

        msg = data.get("message", {})
        text = msg.get("content") or None
        raw_tool_calls = msg.get("tool_calls") or []

        parsed_tool_calls = []
        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                args = _parse_tool_arguments(args)
            parsed_tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:24]}",
                    name=fn.get("name", ""),
                    arguments=args,
                )
            )

        finish_reason = "tool_calls" if parsed_tool_calls else "stop"
        prompt_tokens = data.get("prompt_eval_count", 0) or 0
        completion_tokens = data.get("eval_count", 0) or 0

        return LLMResponse(
            text=text,
            provider=self.name,
            model=model_override or self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_microcents=0,
            latency_ms=0,
            tool_calls=parsed_tool_calls,
            finish_reason=finish_reason,
        )

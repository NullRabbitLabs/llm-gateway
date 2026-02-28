"""Google Gemini provider implementation."""

import logging
from typing import Optional

from google import genai
from google.genai import types as genai_types

from .base import Provider, LLMResponse, ProviderError, RateLimitError, strip_markdown_fences

log = logging.getLogger("llm-gateway.providers.gemini")


class GeminiProvider(Provider):
    """Google Gemini provider."""

    def __init__(self, api_key: str, model: str, config=None):
        super().__init__(api_key, model)
        self._config = config

        api_params = config.api_params if config else {}
        pricing = config.pricing if config else {}

        self._temperature: float = api_params.get("temperature", 0.1)
        self._json_mode: bool = (config.features.get("json_mode", True) if config else True)
        self._input_cost_per_1k: float = pricing.get("input_per_1k_microcents", 0.10)
        self._output_cost_per_1k: float = pricing.get("output_per_1k_microcents", 0.40)

        self.client = genai.Client(api_key=api_key)

    @property
    def name(self) -> str:
        return "gemini"

    def _compute_cost(self, prompt_tokens: int, completion_tokens: int) -> int:
        input_cost = (prompt_tokens / 1000) * self._input_cost_per_1k
        output_cost = (completion_tokens / 1000) * self._output_cost_per_1k
        return round(input_cost + output_cost)

    def _call_api(self, prompt: str, system_prompt: Optional[str], model_override: str | None = None) -> LLMResponse:
        """Call Gemini API."""
        config_kwargs = {
            "system_instruction": system_prompt if system_prompt else None,
            "temperature": self._temperature,
        }
        if self._json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        try:
            response = self.client.models.generate_content(
                model=model_override or self.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
                raise RateLimitError(str(e)) from e
            raise

        text = response.text
        if text is not None:
            text = strip_markdown_fences(text)

        prompt_tokens = 0
        completion_tokens = 0
        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count or 0
            completion_tokens = response.usage_metadata.candidates_token_count or 0

        return LLMResponse(
            text=text,
            provider=self.name,
            model=model_override or self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_microcents=self._compute_cost(prompt_tokens, completion_tokens),
            latency_ms=0,
        )

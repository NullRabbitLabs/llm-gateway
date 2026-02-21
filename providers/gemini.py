"""Google Gemini provider implementation."""

import logging
from typing import Optional

from google import genai
from google.genai import types as genai_types

from .base import Provider, LLMResponse, ProviderError, RateLimitError, strip_markdown_fences

log = logging.getLogger("llm-gateway.providers.gemini")

# Gemini Flash pricing: $0.10/1M input, $0.40/1M output
# In microcents (1 microcent = $0.000001)
GEMINI_INPUT_COST_PER_1K = 0.10  # microcents
GEMINI_OUTPUT_COST_PER_1K = 0.40  # microcents


class GeminiProvider(Provider):
    """Google Gemini provider."""

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        self.client = genai.Client(api_key=api_key)

    @property
    def name(self) -> str:
        return "gemini"

    def _call_api(self, prompt: str, system_prompt: Optional[str]) -> LLMResponse:
        """Call Gemini API."""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt if system_prompt else None,
                    response_mime_type="application/json",
                    temperature=0.1,
                )
            )
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
                raise RateLimitError(str(e)) from e
            raise

        text = response.text
        text = strip_markdown_fences(text)

        prompt_tokens = 0
        completion_tokens = 0
        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count or 0
            completion_tokens = response.usage_metadata.candidates_token_count or 0

        # Calculate cost in microcents
        input_cost = (prompt_tokens / 1000) * GEMINI_INPUT_COST_PER_1K
        output_cost = (completion_tokens / 1000) * GEMINI_OUTPUT_COST_PER_1K
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

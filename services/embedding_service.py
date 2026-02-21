"""Embedding service for generating text embeddings via OpenAI API."""

import logging
import time
from dataclasses import dataclass
from openai import OpenAI

log = logging.getLogger("llm-gateway.embedding")

# OpenAI embedding model pricing (per 1M tokens)
EMBEDDING_COSTS = {
    "text-embedding-ada-002": 0.10,
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}

DEFAULT_MODEL = "text-embedding-ada-002"
DEFAULT_DIMENSIONS = 1536


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""

    embeddings: list[list[float]]
    model: str
    dimensions: int
    prompt_tokens: int
    cost_microcents: int
    latency_ms: int


class EmbeddingService:
    """Service for generating text embeddings using OpenAI API."""

    def __init__(self, api_key: str):
        """Initialize the embedding service.

        Args:
            api_key: OpenAI API key.
        """
        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        texts: list[str],
        model: str = DEFAULT_MODEL,
    ) -> EmbeddingResult:
        """Generate embeddings for the given texts.

        Args:
            texts: List of texts to embed.
            model: Embedding model to use.

        Returns:
            EmbeddingResult with vectors and metadata.
        """
        start_time = time.time()

        response = self.client.embeddings.create(
            input=texts,
            model=model,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        embeddings = [item.embedding for item in response.data]
        prompt_tokens = response.usage.prompt_tokens

        cost_per_million = EMBEDDING_COSTS.get(model, 0.10)
        cost_microcents = int((prompt_tokens / 1_000_000) * cost_per_million * 100_000_000)

        dimensions = len(embeddings[0]) if embeddings else 0

        log.info(
            f"Generated {len(embeddings)} embeddings using {model}, "
            f"tokens={prompt_tokens}, cost_microcents={cost_microcents}, latency={latency_ms}ms"
        )

        return EmbeddingResult(
            embeddings=embeddings,
            model=model,
            dimensions=dimensions,
            prompt_tokens=prompt_tokens,
            cost_microcents=cost_microcents,
            latency_ms=latency_ms,
        )

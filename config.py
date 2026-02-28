"""Configuration management for llm-gateway."""

import os
import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("llm-gateway.config")


@dataclass
class Config:
    """Configuration for LLM Gateway service."""

    provider: str  # "ollama", "deepseek", "gemini", "openai", "anthropic", or "auto"

    # Ollama (local)
    ollama_host: Optional[str] = None
    ollama_model: Optional[str] = None

    # DeepSeek
    deepseek_api_key: Optional[str] = None
    deepseek_model: Optional[str] = None

    # Gemini
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None

    # Anthropic
    anthropic_api_key: Optional[str] = None
    anthropic_model: Optional[str] = None

    # Service settings
    port: int = 8090
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables.

        Raises:
            ValueError: If required configuration is missing for the selected provider.
        """
        provider = os.getenv("LLM_PROVIDER", "auto").lower()

        config = cls(
            provider=provider,
            ollama_host=os.getenv("OLLAMA_HOST"),
            ollama_model=os.getenv("OLLAMA_MODEL"),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            deepseek_model=os.getenv("DEEPSEEK_MODEL"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            gemini_model=os.getenv("GEMINI_MODEL"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            anthropic_model=os.getenv("ANTHROPIC_MODEL"),
            port=int(os.getenv("PORT", "8090")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

        config._validate()
        return config

    def _validate(self) -> None:
        """Validate configuration based on provider setting."""
        if self.provider == "ollama":
            self._validate_ollama()
        elif self.provider == "deepseek":
            self._validate_provider("deepseek", self.deepseek_api_key, self.deepseek_model)
        elif self.provider == "gemini":
            self._validate_provider("gemini", self.gemini_api_key, self.gemini_model)
        elif self.provider == "openai":
            self._validate_provider("openai", self.openai_api_key, self.openai_model)
        elif self.provider == "anthropic":
            self._validate_provider("anthropic", self.anthropic_api_key, self.anthropic_model)
        elif self.provider == "auto":
            self._validate_auto_mode()
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")

    def _validate_ollama(self) -> None:
        """Validate Ollama provider has required configuration (no API key needed)."""
        if not self.ollama_host:
            raise ValueError(
                "LLM_PROVIDER is set to 'ollama' but OLLAMA_HOST is not configured."
            )
        if not self.ollama_model:
            raise ValueError(
                "LLM_PROVIDER is set to 'ollama' but OLLAMA_MODEL is not configured."
            )

    def _validate_provider(self, name: str, api_key: Optional[str], model: Optional[str]) -> None:
        """Validate a specific provider has required configuration."""
        name_upper = name.upper()
        if not api_key:
            raise ValueError(
                f"LLM_PROVIDER is set to '{name}' but {name_upper}_API_KEY is not configured."
            )
        if not model:
            raise ValueError(
                f"LLM_PROVIDER is set to '{name}' but {name_upper}_MODEL is not configured."
            )

    def _validate_auto_mode(self) -> None:
        """Validate at least one provider is fully configured in auto mode."""
        providers_ready = []

        if self.ollama_host and self.ollama_model:
            providers_ready.append("ollama")
        if self.deepseek_api_key and self.deepseek_model:
            providers_ready.append("deepseek")
        if self.gemini_api_key and self.gemini_model:
            providers_ready.append("gemini")
        if self.openai_api_key and self.openai_model:
            providers_ready.append("openai")
        if self.anthropic_api_key and self.anthropic_model:
            providers_ready.append("anthropic")

        if not providers_ready:
            raise ValueError(
                "No LLM providers fully configured. Set at least one pair of "
                "<PROVIDER>_API_KEY and <PROVIDER>_MODEL environment variables, "
                "or OLLAMA_HOST and OLLAMA_MODEL for local models."
            )

        log.info(f"Auto mode: {len(providers_ready)} providers ready: {', '.join(providers_ready)}")

    def get_available_providers(self) -> list[str]:
        """Return list of fully configured providers."""
        providers = []
        if self.ollama_host and self.ollama_model:
            providers.append("ollama")
        if self.deepseek_api_key and self.deepseek_model:
            providers.append("deepseek")
        if self.gemini_api_key and self.gemini_model:
            providers.append("gemini")
        if self.openai_api_key and self.openai_model:
            providers.append("openai")
        if self.anthropic_api_key and self.anthropic_model:
            providers.append("anthropic")
        return providers

    def embeddings_available(self) -> bool:
        """Check if embeddings are available (requires OpenAI API key)."""
        return bool(self.openai_api_key)

"""Configuration management for llm-gateway."""

import os
import logging
from dataclasses import dataclass, field

from providers.registry import (
    ProviderConfig,
    load_provider_configs,
    is_provider_configured,
)

log = logging.getLogger("llm-gateway.config")


@dataclass
class Config:
    """Configuration for LLM Gateway service."""

    provider: str  # provider name or "auto"

    # Registry data (loaded from providers.json)
    auto_priority: list[str] = field(default_factory=list)
    provider_configs: dict[str, ProviderConfig] = field(default_factory=dict)

    # Service settings
    port: int = 8090
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables + providers.json.

        Raises:
            ValueError: If required configuration is missing for the selected provider.
        """
        provider = os.getenv("LLM_PROVIDER", "auto").lower()
        auto_priority, provider_configs = load_provider_configs()

        config = cls(
            provider=provider,
            auto_priority=auto_priority,
            provider_configs=provider_configs,
            port=int(os.getenv("PORT", "8090")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

        config._validate()
        return config

    def _validate(self) -> None:
        """Validate configuration based on provider setting."""
        if self.provider == "auto":
            self._validate_auto_mode()
        else:
            pc = self.provider_configs.get(self.provider)
            if pc is None:
                raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")
            if not is_provider_configured(pc):
                if pc.kind == "ollama":
                    raise ValueError(
                        f"LLM_PROVIDER is set to '{self.provider}' but "
                        f"{pc.env_host} is not configured."
                    )
                raise ValueError(
                    f"LLM_PROVIDER is set to '{self.provider}' but "
                    f"{pc.env_key} is not configured."
                )

    def _validate_auto_mode(self) -> None:
        """Validate at least one provider is fully configured in auto mode."""
        providers_ready = [
            name
            for name, pc in self.provider_configs.items()
            if is_provider_configured(pc)
        ]

        if not providers_ready:
            raise ValueError(
                "No LLM providers fully configured. Set at least one pair of "
                "<PROVIDER>_API_KEY and <PROVIDER>_MODEL environment variables, "
                "or OLLAMA_HOST and OLLAMA_MODEL for local models."
            )

        log.info(f"Auto mode: {len(providers_ready)} providers ready: {', '.join(providers_ready)}")

    def get_available_providers(self) -> list[str]:
        """Return list of fully configured providers."""
        return [
            name
            for name in self.auto_priority
            if name in self.provider_configs and is_provider_configured(self.provider_configs[name])
        ]

    def embeddings_available(self) -> bool:
        """Check if embeddings are available (requires OpenAI API key)."""
        return bool(os.getenv("OPENAI_API_KEY"))

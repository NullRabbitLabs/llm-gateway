"""Provider registry — loads providers.json and instantiates provider classes."""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import Provider

log = logging.getLogger("llm-gateway.registry")

_PROVIDERS_JSON = Path(__file__).resolve().parent.parent / "providers.json"


@dataclass
class ProviderConfig:
    """Parsed configuration for a single provider entry."""

    name: str
    kind: str
    env_key: str = ""
    env_host: str = ""
    env_model: str = ""
    default_model: str = ""
    base_url: str | None = None
    timeout: int = 300
    api_params: dict[str, Any] = field(default_factory=dict)
    features: dict[str, bool] = field(default_factory=dict)
    pricing: dict[str, float] = field(default_factory=dict)


def load_provider_configs(
    path: str | Path | None = None,
) -> tuple[list[str], dict[str, ProviderConfig]]:
    """Read providers.json and return (auto_priority, {name: ProviderConfig}).

    Args:
        path: Optional override path to providers.json.

    Returns:
        Tuple of (auto_priority list, dict of name -> ProviderConfig).
    """
    path = Path(path) if path else _PROVIDERS_JSON
    with open(path) as f:
        data = json.load(f)

    auto_priority: list[str] = data.get("auto_priority", [])
    configs: dict[str, ProviderConfig] = {}

    for name, entry in data.get("providers", {}).items():
        configs[name] = ProviderConfig(
            name=name,
            kind=entry.get("kind", ""),
            env_key=entry.get("env_key", ""),
            env_host=entry.get("env_host", ""),
            env_model=entry.get("env_model", ""),
            default_model=entry.get("default_model", ""),
            base_url=entry.get("base_url"),
            timeout=entry.get("timeout", 300),
            api_params=entry.get("api_params", {}),
            features=entry.get("features", {}),
            pricing=entry.get("pricing", {}),
        )

    return auto_priority, configs


def is_provider_configured(pc: ProviderConfig) -> bool:
    """Check whether environment variables are set for a provider."""
    if pc.kind == "ollama":
        return bool(os.getenv(pc.env_host))
    return bool(os.getenv(pc.env_key))


def _resolve_model(pc: ProviderConfig) -> str:
    """Return the model from env or the default."""
    return os.getenv(pc.env_model, "") or pc.default_model


def _resolve_api_key(pc: ProviderConfig) -> str:
    """Return the API key from env."""
    return os.getenv(pc.env_key, "")


def _resolve_host(pc: ProviderConfig) -> str:
    """Return the host from env."""
    return os.getenv(pc.env_host, "")


# kind -> (module_path, class_name) — lazy imports to avoid loading unused SDKs
_KIND_MAP: dict[str, tuple[str, str]] = {
    "openai_compatible": ("providers.openai_compatible", "OpenAICompatibleProvider"),
    "anthropic": ("providers.anthropic_provider", "AnthropicProvider"),
    "gemini": ("providers.gemini", "GeminiProvider"),
    "ollama": ("providers.ollama", "OllamaProvider"),
}


def create_provider(pc: ProviderConfig) -> Provider:
    """Instantiate a provider from its ProviderConfig."""
    import importlib

    entry = _KIND_MAP.get(pc.kind)
    if entry is None:
        raise ValueError(f"Unknown provider kind: {pc.kind!r}")

    module_path, class_name = entry
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    model = _resolve_model(pc)

    if pc.kind == "ollama":
        return cls(host=_resolve_host(pc), model=model, config=pc)
    else:
        return cls(api_key=_resolve_api_key(pc), model=model, config=pc)


def create_providers_for_mode(
    mode: str,
    auto_priority: list[str],
    configs: dict[str, ProviderConfig],
) -> list[Provider]:
    """Build an ordered list of provider instances for the given mode.

    Args:
        mode: "auto" or a specific provider name.
        auto_priority: Priority order for auto mode.
        configs: All provider configs keyed by name.

    Returns:
        List of instantiated Provider objects.
    """
    providers: list[Provider] = []

    if mode == "auto":
        for name in auto_priority:
            pc = configs.get(name)
            if pc and is_provider_configured(pc):
                providers.append(create_provider(pc))
    else:
        pc = configs.get(mode)
        if pc is None:
            raise ValueError(f"Unknown LLM_PROVIDER: {mode}")
        if not is_provider_configured(pc):
            if pc.kind == "ollama":
                raise ValueError(
                    f"LLM_PROVIDER is set to '{mode}' but {pc.env_host} is not configured."
                )
            raise ValueError(
                f"LLM_PROVIDER is set to '{mode}' but {pc.env_key} is not configured."
            )
        providers.append(create_provider(pc))

    return providers

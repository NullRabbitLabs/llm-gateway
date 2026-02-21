"""Test fixtures for llm-gateway tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_env_deepseek(monkeypatch):
    """Set up environment for DeepSeek provider."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")


@pytest.fixture
def mock_env_gemini(monkeypatch):
    """Set up environment for Gemini provider."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")


@pytest.fixture
def mock_env_openai(monkeypatch):
    """Set up environment for OpenAI provider."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LLM_PROVIDER", "openai")


@pytest.fixture
def mock_env_anthropic(monkeypatch):
    """Set up environment for Anthropic provider."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")


@pytest.fixture
def mock_env_auto(monkeypatch):
    """Set up environment for auto provider selection."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("LLM_PROVIDER", "auto")


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for DeepSeek/OpenAI providers."""
    with patch("providers.deepseek.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # Mock successful response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"assessment": "complete", "probes": []}'
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150

        mock_client.chat.completions.create.return_value = mock_response
        yield mock_client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client."""
    with patch("providers.anthropic.anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # Mock successful response
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].type = "text"
        mock_response.content[0].text = '{"assessment": "complete", "probes": []}'
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        mock_client.messages.create.return_value = mock_response
        yield mock_client


@pytest.fixture
def mock_gemini_client():
    """Mock Google Gemini client."""
    with patch("providers.gemini.genai.Client") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # Mock successful response
        mock_response = MagicMock()
        mock_response.text = '{"assessment": "complete", "probes": []}'
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50

        mock_client.models.generate_content.return_value = mock_response
        yield mock_client


@pytest.fixture
def sample_classification_response():
    """Sample classification response from LLM."""
    return {
        "owned": [
            {
                "url": "https://app.example.com",
                "subdomain": "app.example.com",
                "category": "owned",
                "reason": "Main application server",
                "priority": "high",
                "confidence": "high",
                "evidence": ["Direct A record", "No third-party CNAME"]
            }
        ],
        "third_party": [
            {
                "url": "https://cdn.example.com",
                "subdomain": "cdn.example.com",
                "category": "third_party",
                "reason": "Cloudflare CDN",
                "confidence": "high",
                "evidence": ["CNAME to cloudflare.net"]
            }
        ],
        "interesting": [],
        "ignore": []
    }


@pytest.fixture
def sample_plan_response():
    """Sample plan response from LLM."""
    return {
        "assessment": "needs_followup",
        "confidence": "medium",
        "rationale": "Discovered open ports need verification",
        "probes": [
            {
                "cmd": "curl -s -o /dev/null -w '%{http_code}' https://192.168.1.1:443",
                "timeout_s": 30,
                "note": "Verify HTTPS service"
            }
        ],
        "port_status": {
            "443": {"state": "open_confirmed", "service": "https"}
        }
    }

"""Tests for LLM providers - written first following TDD."""

import json
import re
from unittest.mock import MagicMock, patch

import pytest


class TestProviderSelection:
    """Test provider auto-selection logic."""

    def test_select_provider_explicit_deepseek(self, mock_env_deepseek):
        """When LLM_PROVIDER=deepseek, select DeepSeek provider."""
        from config import Config

        config = Config.from_env()
        assert config.provider == "deepseek"
        assert config.deepseek_api_key == "test-deepseek-key"
        assert config.deepseek_model == "deepseek-reasoner"

    def test_select_provider_explicit_gemini(self, mock_env_gemini):
        """When LLM_PROVIDER=gemini, select Gemini provider."""
        from config import Config

        config = Config.from_env()
        assert config.provider == "gemini"
        assert config.gemini_api_key == "test-gemini-key"
        assert config.gemini_model == "gemini-2.0-flash"

    def test_select_provider_explicit_openai(self, mock_env_openai):
        """When LLM_PROVIDER=openai, select OpenAI provider."""
        from config import Config

        config = Config.from_env()
        assert config.provider == "openai"
        assert config.openai_api_key == "test-openai-key"
        assert config.openai_model == "gpt-4o-mini"

    def test_select_provider_explicit_anthropic(self, mock_env_anthropic):
        """When LLM_PROVIDER=anthropic, select Anthropic provider."""
        from config import Config

        config = Config.from_env()
        assert config.provider == "anthropic"
        assert config.anthropic_api_key == "test-anthropic-key"
        assert config.anthropic_model == "claude-3-5-sonnet-20241022"

    def test_select_provider_auto_prefers_deepseek(self, mock_env_auto):
        """In auto mode, DeepSeek should be first choice (cheapest)."""
        from services.llm_service import LLMService
        from config import Config

        config = Config.from_env()
        service = LLMService(config)

        # First provider in the list should be DeepSeek
        assert service.providers[0].name == "deepseek"

    def test_select_provider_missing_key_raises(self, monkeypatch):
        """When provider is set but key is missing, raise error."""
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

        from config import Config

        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            Config.from_env()

    def test_select_provider_missing_model_raises(self, monkeypatch):
        """When provider is set but model is missing, raise error."""
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

        from config import Config

        with pytest.raises(ValueError, match="DEEPSEEK_MODEL"):
            Config.from_env()


class TestDeepSeekProvider:
    """Test DeepSeek provider implementation."""

    def test_deepseek_call_success(self, mock_env_deepseek):
        """DeepSeek provider returns valid response."""
        from providers.deepseek import DeepSeekProvider

        with patch("providers.deepseek.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"assessment": "complete"}'
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 50
            mock_response.usage.total_tokens = 150

            mock_client.chat.completions.create.return_value = mock_response

            provider = DeepSeekProvider("test-key", "deepseek-reasoner")
            result = provider.call("test prompt", "system prompt")

            assert result.text == '{"assessment": "complete"}'
            assert result.prompt_tokens == 100
            assert result.completion_tokens == 50
            assert result.provider == "deepseek"

    def test_deepseek_call_does_not_force_json_mode(self, mock_env_deepseek):
        """DeepSeek text path does not force json_object response_format (breaks non-JSON prompts)."""
        from providers.deepseek import DeepSeekProvider

        with patch("providers.deepseek.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "{}"
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            mock_response.usage.total_tokens = 15

            mock_client.chat.completions.create.return_value = mock_response

            provider = DeepSeekProvider("test-key", "deepseek-reasoner")
            provider.call("test", "system")

            call_args = mock_client.chat.completions.create.call_args
            assert "response_format" not in (call_args.kwargs or {})


class TestGeminiProvider:
    """Test Gemini provider implementation."""

    def test_gemini_call_success(self, mock_env_gemini):
        """Gemini provider returns valid response."""
        from providers.gemini import GeminiProvider

        with patch("providers.gemini.genai.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.text = '{"assessment": "complete"}'
            mock_response.usage_metadata = MagicMock()
            mock_response.usage_metadata.prompt_token_count = 100
            mock_response.usage_metadata.candidates_token_count = 50

            mock_client.models.generate_content.return_value = mock_response

            provider = GeminiProvider("test-key", "gemini-2.0-flash")
            result = provider.call("test prompt", "system prompt")

            assert result.text == '{"assessment": "complete"}'
            assert result.prompt_tokens == 100
            assert result.completion_tokens == 50
            assert result.provider == "gemini"

    def test_gemini_strips_markdown(self, mock_env_gemini):
        """Gemini provider strips markdown code fences from response."""
        from providers.gemini import GeminiProvider

        with patch("providers.gemini.genai.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.text = '```json\n{"assessment": "complete"}\n```'
            mock_response.usage_metadata = MagicMock()
            mock_response.usage_metadata.prompt_token_count = 100
            mock_response.usage_metadata.candidates_token_count = 50

            mock_client.models.generate_content.return_value = mock_response

            provider = GeminiProvider("test-key", "gemini-2.0-flash")
            result = provider.call("test prompt", "system prompt")

            assert result.text == '{"assessment": "complete"}'


class TestOpenAIProvider:
    """Test OpenAI provider implementation."""

    def test_openai_call_success(self, mock_env_openai):
        """OpenAI provider returns valid response."""
        from providers.openai_provider import OpenAIProvider

        with patch("providers.openai_provider.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"assessment": "complete"}'
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 50
            mock_response.usage.total_tokens = 150

            mock_client.chat.completions.create.return_value = mock_response

            provider = OpenAIProvider("test-key", "gpt-4o-mini")
            result = provider.call("test prompt", "system prompt")

            assert result.text == '{"assessment": "complete"}'
            assert result.prompt_tokens == 100
            assert result.completion_tokens == 50
            assert result.provider == "openai"


class TestAnthropicProvider:
    """Test Anthropic provider implementation."""

    def test_anthropic_call_success(self, mock_env_anthropic):
        """Anthropic provider returns valid response."""
        from providers.anthropic_provider import AnthropicProvider

        with patch("providers.anthropic_provider.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].type = "text"
            mock_response.content[0].text = '{"assessment": "complete"}'
            mock_response.usage = MagicMock()
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50

            mock_client.messages.create.return_value = mock_response

            provider = AnthropicProvider("test-key", "claude-3-5-sonnet-20241022")
            result = provider.call("test prompt", "system prompt")

            assert result.text == '{"assessment": "complete"}'
            assert result.prompt_tokens == 100
            assert result.completion_tokens == 50
            assert result.provider == "anthropic"

    def test_anthropic_strips_markdown(self, mock_env_anthropic):
        """Anthropic provider strips markdown code fences from response."""
        from providers.anthropic_provider import AnthropicProvider

        with patch("providers.anthropic_provider.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].type = "text"
            mock_response.content[0].text = '```json\n{"assessment": "complete"}\n```'
            mock_response.usage = MagicMock()
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50

            mock_client.messages.create.return_value = mock_response

            provider = AnthropicProvider("test-key", "claude-3-5-sonnet-20241022")
            result = provider.call("test prompt", "system prompt")

            assert result.text == '{"assessment": "complete"}'


class TestProviderFallback:
    """Test provider fallback behavior."""

    def test_fallback_on_failure(self, mock_env_auto):
        """When first provider fails, fallback to next."""
        from services.llm_service import LLMService
        from config import Config
        from providers.base import ProviderError

        config = Config.from_env()
        service = LLMService(config)

        # Mock first provider to fail, second to succeed
        service.providers[0].call = MagicMock(side_effect=ProviderError("rate limit"))

        mock_result = MagicMock()
        mock_result.text = '{"assessment": "complete"}'
        mock_result.provider = "gemini"
        service.providers[1].call = MagicMock(return_value=mock_result)

        result = service.call("test prompt", "system prompt")

        assert result.provider == "gemini"
        assert service.providers[0].call.called
        assert service.providers[1].call.called

    def test_all_providers_fail_raises(self, mock_env_auto):
        """When all providers fail, raise error."""
        from services.llm_service import LLMService, AllProvidersFailedError
        from config import Config
        from providers.base import ProviderError

        config = Config.from_env()
        service = LLMService(config)

        # Mock all providers to fail
        for provider in service.providers:
            provider.call = MagicMock(side_effect=ProviderError("error"))

        with pytest.raises(AllProvidersFailedError):
            service.call("test prompt", "system prompt")


class TestRetryLogic:
    """Test retry behavior on transient errors."""

    def test_base_provider_does_not_retry_400_errors(self, mock_env_deepseek):
        """400 client errors must NOT be retried — they are deterministic failures."""
        from providers.deepseek import DeepSeekProvider
        from providers.base import ProviderError

        with patch("providers.deepseek.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            call_count = [0]

            def side_effect(*args, **kwargs):
                call_count[0] += 1
                raise Exception("Error code: 400 - invalid_request_error")

            mock_client.chat.completions.create.side_effect = side_effect

            provider = DeepSeekProvider("test-key", "deepseek-reasoner")
            with pytest.raises(ProviderError):
                provider.call("test prompt", "system prompt")

            # Must NOT retry: only one attempt
            assert call_count[0] == 1, "400 errors should not be retried"

    def test_retry_on_rate_limit(self, mock_env_deepseek):
        """Provider retries on rate limit error."""
        from providers.deepseek import DeepSeekProvider
        from providers.base import RateLimitError

        with patch("providers.deepseek.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            # First call fails with rate limit, second succeeds
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"assessment": "complete"}'
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 50
            mock_response.usage.total_tokens = 150

            call_count = [0]

            def side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise Exception("rate limit exceeded")
                return mock_response

            mock_client.chat.completions.create.side_effect = side_effect

            provider = DeepSeekProvider("test-key", "deepseek-reasoner")
            result = provider.call("test prompt", "system prompt")

            assert call_count[0] == 2
            assert result.text == '{"assessment": "complete"}'


class TestProviderTimeouts:
    """Test that providers have appropriate timeouts configured."""

    def test_deepseek_has_timeout_configured(self, mock_env_deepseek):
        """DeepSeek provider should have explicit timeout > 120s for slow reasoning models."""
        from providers.deepseek import DeepSeekProvider

        with patch("providers.deepseek.OpenAI") as mock_cls:
            provider = DeepSeekProvider("test-key", "deepseek-reasoner")

            # Verify OpenAI client was created with timeout
            call_args = mock_cls.call_args
            timeout = call_args.kwargs.get("timeout")

            # Should have a timeout configured (at least 180s for reasoning models)
            assert timeout is not None, "DeepSeek provider should have explicit timeout"
            assert timeout >= 180, f"Timeout should be >= 180s for reasoning models, got {timeout}"

    def test_openai_has_timeout_configured(self, mock_env_openai):
        """OpenAI provider should have explicit timeout configured."""
        from providers.openai_provider import OpenAIProvider

        with patch("providers.openai_provider.OpenAI") as mock_cls:
            provider = OpenAIProvider("test-key", "gpt-4o-mini")

            call_args = mock_cls.call_args
            timeout = call_args.kwargs.get("timeout")

            assert timeout is not None, "OpenAI provider should have explicit timeout"
            assert timeout >= 180, f"Timeout should be >= 180s, got {timeout}"


class TestCostCalculation:
    """Test cost calculation for different providers."""

    def test_deepseek_cost_calculation(self, mock_env_deepseek):
        """DeepSeek cost is calculated correctly."""
        from providers.deepseek import DeepSeekProvider

        with patch("providers.deepseek.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "{}"
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 1000
            mock_response.usage.completion_tokens = 500
            mock_response.usage.total_tokens = 1500

            mock_client.chat.completions.create.return_value = mock_response

            provider = DeepSeekProvider("test-key", "deepseek-reasoner")
            result = provider.call("test", "system")

            # DeepSeek: $0.12/1M input = 0.12 microcents/1K tokens
            # DeepSeek: $0.20/1M output = 0.20 microcents/1K tokens
            # 1000 input * 0.12 / 1000 + 500 output * 0.20 / 1000 = 0.12 + 0.10 = 0.22
            # Rounded to integer microcents
            assert result.cost_microcents is not None

    def test_anthropic_cost_calculation(self, mock_env_anthropic):
        """Anthropic cost is calculated correctly (more expensive)."""
        from providers.anthropic_provider import AnthropicProvider

        with patch("providers.anthropic_provider.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].type = "text"
            mock_response.content[0].text = "{}"
            mock_response.usage = MagicMock()
            mock_response.usage.input_tokens = 1000
            mock_response.usage.output_tokens = 500

            mock_client.messages.create.return_value = mock_response

            provider = AnthropicProvider("test-key", "claude-3-5-sonnet-20241022")
            result = provider.call("test", "system")

            # Anthropic is more expensive than DeepSeek
            assert result.cost_microcents is not None


class TestParseToolArguments:
    """Test resilient argument parsing helper."""

    def test_valid_json_parsed(self):
        from providers.base import _parse_tool_arguments

        result = _parse_tool_arguments('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_markdown_fenced_json_parsed(self):
        from providers.base import _parse_tool_arguments

        result = _parse_tool_arguments('```json\n{"command": "ls"}\n```')
        assert result == {"command": "ls"}

    def test_trailing_comma_json_parsed(self):
        from providers.base import _parse_tool_arguments

        result = _parse_tool_arguments('{"key": "value",}')
        assert result == {"key": "value"}

    def test_garbage_returns_parse_error(self):
        from providers.base import _parse_tool_arguments

        result = _parse_tool_arguments("not valid json at all !@#")
        assert "_parse_error" in result

    def test_empty_braces_returns_empty_dict(self):
        from providers.base import _parse_tool_arguments

        result = _parse_tool_arguments("{}")
        assert result == {}


class TestToolCallDataclass:
    """Test ToolCall and updated LLMResponse dataclasses."""

    def test_tool_call_has_required_fields(self):
        from providers.base import ToolCall

        tc = ToolCall(id="call_001", name="ssh_exec", arguments={"command": "ls"})
        assert tc.id == "call_001"
        assert tc.name == "ssh_exec"
        assert tc.arguments == {"command": "ls"}

    def test_llm_response_defaults_tool_calls_empty(self):
        from providers.base import LLMResponse

        resp = LLMResponse(
            text="hello",
            provider="openai",
            model="gpt-4o-mini",
            prompt_tokens=10,
            completion_tokens=5,
            cost_microcents=1,
            latency_ms=100,
        )
        assert resp.tool_calls == []
        assert resp.finish_reason == "stop"

    def test_llm_response_text_can_be_none(self):
        from providers.base import LLMResponse, ToolCall

        resp = LLMResponse(
            text=None,
            provider="openai",
            model="gpt-4o-mini",
            prompt_tokens=10,
            completion_tokens=5,
            cost_microcents=1,
            latency_ms=100,
            tool_calls=[ToolCall(id="c1", name="fn", arguments={})],
            finish_reason="tool_calls",
        )
        assert resp.text is None
        assert len(resp.tool_calls) == 1
        assert resp.finish_reason == "tool_calls"

    def test_llm_response_reasoning_content_defaults_none(self):
        from providers.base import LLMResponse

        resp = LLMResponse(
            text="hello",
            provider="deepseek",
            model="deepseek-reasoner",
            prompt_tokens=10,
            completion_tokens=5,
            cost_microcents=1,
            latency_ms=100,
        )
        assert resp.reasoning_content is None


class TestOpenAIProviderToolCalls:
    """Test OpenAI provider tool call support."""

    def test_call_with_tools_forwards_tools_in_payload(self, mock_env_openai):
        from providers.openai_provider import OpenAIProvider

        with patch("providers.openai_provider.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "hello"
            mock_response.choices[0].message.tool_calls = None
            mock_response.choices[0].finish_reason = "stop"
            mock_response.model = "gpt-4o-mini"
            mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
            mock_client.chat.completions.create.return_value = mock_response

            tools = [{"type": "function", "function": {"name": "run_cmd"}}]
            provider = OpenAIProvider("test-key", "gpt-4o-mini")
            provider.call_with_tools(
                messages=[{"role": "user", "content": "hi"}],
                tools=tools,
            )

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["tools"] == tools

    def test_call_with_tools_parses_tool_call_response(self, mock_env_openai):
        from providers.openai_provider import OpenAIProvider
        from providers.base import ToolCall

        with patch("providers.openai_provider.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            tc = MagicMock()
            tc.id = "call_001"
            tc.function.name = "ssh_exec"
            tc.function.arguments = '{"command": "ls"}'

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = None
            mock_response.choices[0].message.tool_calls = [tc]
            mock_response.choices[0].finish_reason = "tool_calls"
            mock_response.model = "gpt-4o-mini"
            mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
            mock_client.chat.completions.create.return_value = mock_response

            provider = OpenAIProvider("test-key", "gpt-4o-mini")
            result = provider.call_with_tools(
                messages=[{"role": "user", "content": "go"}],
            )

            assert result.finish_reason == "tool_calls"
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].id == "call_001"
            assert result.tool_calls[0].name == "ssh_exec"
            assert result.tool_calls[0].arguments == {"command": "ls"}

    def test_call_with_tools_no_tools_does_not_send_tools_key(self, mock_env_openai):
        from providers.openai_provider import OpenAIProvider

        with patch("providers.openai_provider.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "ok"
            mock_response.choices[0].message.tool_calls = None
            mock_response.choices[0].finish_reason = "stop"
            mock_response.model = "gpt-4o-mini"
            mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=3)
            mock_client.chat.completions.create.return_value = mock_response

            provider = OpenAIProvider("test-key", "gpt-4o-mini")
            provider.call_with_tools(messages=[{"role": "user", "content": "hi"}], tools=None)

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert "tools" not in call_kwargs


class TestDeepSeekProviderToolCalls:
    """Test DeepSeek provider tool call support (OpenAI-compatible)."""

    def test_call_with_tools_parses_tool_call_response(self, mock_env_deepseek):
        from providers.deepseek import DeepSeekProvider

        with patch("providers.deepseek.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            tc = MagicMock()
            tc.id = "call_ds_1"
            tc.function.name = "run_nmap"
            tc.function.arguments = '{"target": "192.168.1.1"}'

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = None
            mock_response.choices[0].message.tool_calls = [tc]
            mock_response.choices[0].finish_reason = "tool_calls"
            mock_response.model = "deepseek-reasoner"
            mock_response.usage = MagicMock(prompt_tokens=20, completion_tokens=10)
            mock_client.chat.completions.create.return_value = mock_response

            provider = DeepSeekProvider("test-key", "deepseek-reasoner")
            result = provider.call_with_tools(
                messages=[{"role": "user", "content": "scan now"}],
                tools=[{"type": "function", "function": {"name": "run_nmap"}}],
            )

            assert result.finish_reason == "tool_calls"
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].name == "run_nmap"
            assert result.tool_calls[0].arguments == {"target": "192.168.1.1"}

    def test_call_with_tools_preserves_reasoning_content(self, mock_env_deepseek):
        from providers.deepseek import DeepSeekProvider

        with patch("providers.deepseek.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            tc = MagicMock()
            tc.id = "call_ds_2"
            tc.function.name = "run_nmap"
            tc.function.arguments = '{"target": "10.0.0.1"}'

            mock_message = MagicMock()
            mock_message.content = None
            mock_message.tool_calls = [tc]
            mock_message.reasoning_content = "thinking about this..."

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = mock_message
            mock_response.choices[0].finish_reason = "tool_calls"
            mock_response.model = "deepseek-reasoner"
            mock_response.usage = MagicMock(prompt_tokens=20, completion_tokens=10)
            mock_client.chat.completions.create.return_value = mock_response

            provider = DeepSeekProvider("test-key", "deepseek-reasoner")
            result = provider.call_with_tools(
                messages=[{"role": "user", "content": "scan now"}],
                tools=[{"type": "function", "function": {"name": "run_nmap"}}],
            )

            assert result.reasoning_content == "thinking about this..."

    def test_call_with_tools_reasoning_content_none_when_absent(self, mock_env_deepseek):
        from providers.deepseek import DeepSeekProvider

        with patch("providers.deepseek.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            mock_message = MagicMock(spec=["content", "tool_calls"])
            mock_message.content = "hello"
            mock_message.tool_calls = []

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = mock_message
            mock_response.choices[0].finish_reason = "stop"
            mock_response.model = "deepseek-chat"
            mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=3)
            mock_client.chat.completions.create.return_value = mock_response

            provider = DeepSeekProvider("test-key", "deepseek-chat")
            result = provider.call_with_tools(
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
            )

            assert result.reasoning_content is None

    def test_call_with_tools_uses_model_override_in_api_call(self, mock_env_deepseek):
        """When model_override is provided, API call uses the override model, not self.model."""
        from providers.deepseek import DeepSeekProvider

        with patch("providers.deepseek.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            mock_message = MagicMock(spec=["content", "tool_calls"])
            mock_message.content = "hello"
            mock_message.tool_calls = []

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = mock_message
            mock_response.choices[0].finish_reason = "stop"
            mock_response.model = "deepseek-chat"
            mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=3)
            mock_client.chat.completions.create.return_value = mock_response

            provider = DeepSeekProvider("test-key", "deepseek-reasoner")
            provider.call_with_tools(
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
                model_override="deepseek-chat",
            )

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["model"] == "deepseek-chat"

    def test_call_with_tools_no_override_uses_self_model(self, mock_env_deepseek):
        """When model_override is None, API call uses self.model."""
        from providers.deepseek import DeepSeekProvider

        with patch("providers.deepseek.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            mock_message = MagicMock(spec=["content", "tool_calls"])
            mock_message.content = "hello"
            mock_message.tool_calls = []

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = mock_message
            mock_response.choices[0].finish_reason = "stop"
            mock_response.model = "deepseek-reasoner"
            mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=3)
            mock_client.chat.completions.create.return_value = mock_response

            provider = DeepSeekProvider("test-key", "deepseek-reasoner")
            provider.call_with_tools(
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
                model_override=None,
            )

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["model"] == "deepseek-reasoner"


class TestAnthropicProviderToolCalls:
    """Test Anthropic provider tool call support with format translation."""

    def test_call_with_tools_translates_tool_schema(self, mock_env_anthropic):
        from providers.anthropic_provider import AnthropicProvider

        with patch("providers.anthropic_provider.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.content = []
            mock_response.stop_reason = "end_turn"
            mock_response.model = "claude-3-5-sonnet-20241022"
            mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
            mock_client.messages.create.return_value = mock_response

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "ssh_exec",
                        "description": "Execute SSH command",
                        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                    },
                }
            ]
            provider = AnthropicProvider("test-key", "claude-3-5-sonnet-20241022")
            provider.call_with_tools(
                messages=[{"role": "user", "content": "run ls"}],
                tools=tools,
            )

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert "tools" in call_kwargs
            anthropic_tool = call_kwargs["tools"][0]
            assert anthropic_tool["name"] == "ssh_exec"
            assert anthropic_tool["description"] == "Execute SSH command"
            assert "input_schema" in anthropic_tool
            assert "parameters" not in anthropic_tool

    def test_call_with_tools_extracts_system_message(self, mock_env_anthropic):
        from providers.anthropic_provider import AnthropicProvider

        with patch("providers.anthropic_provider.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.content = []
            mock_response.stop_reason = "end_turn"
            mock_response.model = "claude-3-5-sonnet-20241022"
            mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
            mock_client.messages.create.return_value = mock_response

            messages = [
                {"role": "system", "content": "You are a scanner."},
                {"role": "user", "content": "scan"},
            ]
            provider = AnthropicProvider("test-key", "claude-3-5-sonnet-20241022")
            provider.call_with_tools(messages=messages)

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs["system"] == "You are a scanner."
            # System message should not be in messages list
            for msg in call_kwargs["messages"]:
                assert msg.get("role") != "system"

    def test_call_with_tools_parses_tool_use_response(self, mock_env_anthropic):
        from providers.anthropic_provider import AnthropicProvider

        with patch("providers.anthropic_provider.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            tool_use_block = MagicMock()
            tool_use_block.type = "tool_use"
            tool_use_block.id = "tu_001"
            tool_use_block.name = "ssh_exec"
            tool_use_block.input = {"command": "ls -la"}

            mock_response = MagicMock()
            mock_response.content = [tool_use_block]
            mock_response.stop_reason = "tool_use"
            mock_response.model = "claude-3-5-sonnet-20241022"
            mock_response.usage = MagicMock(input_tokens=20, output_tokens=10)
            mock_client.messages.create.return_value = mock_response

            provider = AnthropicProvider("test-key", "claude-3-5-sonnet-20241022")
            result = provider.call_with_tools(messages=[{"role": "user", "content": "go"}])

            assert result.finish_reason == "tool_calls"
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].id == "tu_001"
            assert result.tool_calls[0].name == "ssh_exec"
            assert result.tool_calls[0].arguments == {"command": "ls -la"}
            assert result.text is None

    def test_call_with_tools_maps_end_turn_to_stop(self, mock_env_anthropic):
        from providers.anthropic_provider import AnthropicProvider

        with patch("providers.anthropic_provider.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "Done"

            mock_response = MagicMock()
            mock_response.content = [text_block]
            mock_response.stop_reason = "end_turn"
            mock_response.model = "claude-3-5-sonnet-20241022"
            mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
            mock_client.messages.create.return_value = mock_response

            provider = AnthropicProvider("test-key", "claude-3-5-sonnet-20241022")
            result = provider.call_with_tools(messages=[{"role": "user", "content": "done?"}])

            assert result.finish_reason == "stop"
            assert result.text == "Done"
            assert result.tool_calls == []

    def test_call_with_tools_converts_tool_result_messages(self, mock_env_anthropic):
        from providers.anthropic_provider import AnthropicProvider

        with patch("providers.anthropic_provider.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.content = []
            mock_response.stop_reason = "end_turn"
            mock_response.model = "claude-3-5-sonnet-20241022"
            mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
            mock_client.messages.create.return_value = mock_response

            messages = [
                {"role": "user", "content": "run ls"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c1", "function": {"name": "ssh_exec", "arguments": '{"command": "ls"}'}}],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "file1.txt\nfile2.txt"},
            ]
            provider = AnthropicProvider("test-key", "claude-3-5-sonnet-20241022")
            provider.call_with_tools(messages=messages)

            call_kwargs = mock_client.messages.create.call_args.kwargs
            sent_messages = call_kwargs["messages"]

            # Find the tool result message (should be a user message with tool_result content)
            tool_result_msg = next(
                (m for m in sent_messages if m.get("role") == "user" and isinstance(m.get("content"), list)),
                None,
            )
            assert tool_result_msg is not None
            assert tool_result_msg["content"][0]["type"] == "tool_result"
            assert tool_result_msg["content"][0]["tool_use_id"] == "c1"


class TestLLMServiceCallWithTools:
    """Test LLMService.call_with_tools() method."""

    def test_call_with_tools_routes_to_provider(self, mock_env_openai):
        from services.llm_service import LLMService
        from config import Config
        from providers.base import LLMResponse, ToolCall

        config = Config.from_env()
        service = LLMService(config)

        mock_result = LLMResponse(
            text=None,
            provider="openai",
            model="gpt-4o-mini",
            prompt_tokens=10,
            completion_tokens=5,
            cost_microcents=1,
            latency_ms=100,
            tool_calls=[ToolCall(id="c1", name="fn", arguments={})],
            finish_reason="tool_calls",
        )
        service.providers[0].call_with_tools = MagicMock(return_value=mock_result)

        result = service.call_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "fn"}}],
        )

        assert result.finish_reason == "tool_calls"
        assert len(result.tool_calls) == 1
        service.providers[0].call_with_tools.assert_called_once()

    def test_call_with_tools_fallback_on_failure(self, mock_env_auto):
        from services.llm_service import LLMService
        from config import Config
        from providers.base import ProviderError, LLMResponse

        config = Config.from_env()
        service = LLMService(config)

        service.providers[0].call_with_tools = MagicMock(side_effect=ProviderError("error"))

        mock_result = LLMResponse(
            text="response",
            provider="gemini",
            model="gemini-2.0-flash",
            prompt_tokens=10,
            completion_tokens=5,
            cost_microcents=1,
            latency_ms=100,
        )
        service.providers[1].call_with_tools = MagicMock(return_value=mock_result)

        result = service.call_with_tools(messages=[{"role": "user", "content": "hi"}])

        assert result.provider == "gemini"

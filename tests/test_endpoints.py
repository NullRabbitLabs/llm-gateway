"""Tests for API endpoints - written first following TDD."""

import pytest
import json
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for endpoint tests."""
    with patch("main.llm_service") as mock:
        mock_result = MagicMock()
        mock_result.text = '{"assessment": "complete", "probes": []}'
        mock_result.provider = "deepseek"
        mock_result.model = "deepseek-reasoner"
        mock_result.prompt_tokens = 100
        mock_result.completion_tokens = 50
        mock_result.cost_microcents = 1
        mock_result.latency_ms = 500
        mock.call.return_value = mock_result
        yield mock


@pytest.fixture
def client(mock_env_deepseek, mock_llm_service):
    """Create test client with mocked dependencies."""
    # Need to import after environment is set up
    with patch("main.validate_api_keys"), patch("main.verify_credentials"):
        from main import app
        return TestClient(app)


class TestClassifyEndpoint:
    """Test /classify endpoint."""

    def test_classify_success(self, mock_env_deepseek):
        """Successful classification returns structured response."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            with patch("main.llm_service") as mock_service:
                mock_result = MagicMock()
                mock_result.text = json.dumps({
                    "owned": [{"subdomain": "app.example.com", "category": "owned"}],
                    "third_party": [],
                    "interesting": [],
                    "ignore": []
                })
                mock_result.provider = "deepseek"
                mock_result.model = "deepseek-reasoner"
                mock_result.prompt_tokens = 100
                mock_result.completion_tokens = 50
                mock_result.cost_microcents = 1
                mock_result.latency_ms = 500
                mock_service.call.return_value = mock_result

                from main import app
                client = TestClient(app)

                response = client.post("/classify", json={
                    "prompt": "Classify these hosts: app.example.com"
                })

                assert response.status_code == 200
                data = response.json()
                assert "classification" in data
                assert "ai_call_log" in data
                assert data["classification"]["owned"][0]["subdomain"] == "app.example.com"

    def test_classify_returns_ai_call_log(self, mock_env_deepseek):
        """Classification includes AI call log for auditing."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            with patch("main.llm_service") as mock_service:
                mock_result = MagicMock()
                mock_result.text = '{"owned": [], "third_party": [], "interesting": [], "ignore": []}'
                mock_result.provider = "deepseek"
                mock_result.model = "deepseek-reasoner"
                mock_result.prompt_tokens = 100
                mock_result.completion_tokens = 50
                mock_result.cost_microcents = 1
                mock_result.latency_ms = 500
                mock_service.call.return_value = mock_result

                from main import app
                client = TestClient(app)

                response = client.post("/classify", json={
                    "prompt": "Classify hosts"
                })

                assert response.status_code == 200
                data = response.json()
                ai_log = data["ai_call_log"]
                assert ai_log["provider"] == "deepseek"
                assert ai_log["model"] == "deepseek-reasoner"
                assert ai_log["prompt_tokens"] == 100
                assert ai_log["completion_tokens"] == 50
                assert ai_log["success"] is True

    def test_classify_missing_prompt_returns_422(self, mock_env_deepseek):
        """Missing prompt field returns validation error."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            from main import app
            client = TestClient(app)

            response = client.post("/classify", json={})

            assert response.status_code == 422

    def test_classify_empty_prompt_returns_422(self, mock_env_deepseek):
        """Empty prompt returns validation error."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            from main import app
            client = TestClient(app)

            response = client.post("/classify", json={"prompt": ""})

            assert response.status_code == 422


class TestPlanEndpoint:
    """Test /plan endpoint."""

    def test_plan_success(self, mock_env_deepseek):
        """Successful plan request returns plan response."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            with patch("main.llm_service") as mock_service:
                mock_result = MagicMock()
                mock_result.text = json.dumps({
                    "assessment": "needs_followup",
                    "confidence": "medium",
                    "rationale": "Need to verify ports",
                    "probes": [{"cmd": "nmap -sV 192.168.1.1", "timeout_s": 60}],
                    "port_status": {"22": {"state": "open_confirmed"}}
                })
                mock_result.provider = "deepseek"
                mock_result.model = "deepseek-reasoner"
                mock_result.prompt_tokens = 200
                mock_result.completion_tokens = 100
                mock_result.cost_microcents = 2
                mock_result.latency_ms = 800
                mock_service.call.return_value = mock_result

                from main import app
                client = TestClient(app)

                response = client.post("/plan", json={
                    "context": {"host": {"ip": "192.168.1.1"}, "discovered_ports": [22, 80]},
                    "system_prompt": "You are a security scanner planner."
                })

                assert response.status_code == 200
                data = response.json()
                assert "plan" in data
                assert "ai_call_log" in data
                assert data["plan"]["assessment"] == "needs_followup"
                assert len(data["plan"]["probes"]) == 1

    def test_plan_with_system_prompt(self, mock_env_deepseek):
        """Plan request passes system prompt to LLM."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            with patch("main.llm_service") as mock_service:
                mock_result = MagicMock()
                mock_result.text = '{"assessment": "complete", "probes": []}'
                mock_result.provider = "deepseek"
                mock_result.model = "deepseek-reasoner"
                mock_result.prompt_tokens = 100
                mock_result.completion_tokens = 50
                mock_result.cost_microcents = 1
                mock_result.latency_ms = 500
                mock_service.call.return_value = mock_result

                from main import app
                client = TestClient(app)

                response = client.post("/plan", json={
                    "context": {"host": {"ip": "192.168.1.1"}},
                    "system_prompt": "Custom system prompt here"
                })

                assert response.status_code == 200
                # Verify system prompt was passed
                call_args = mock_service.call.call_args
                assert call_args[1].get("system_prompt") == "Custom system prompt here" or \
                       call_args[0][1] == "Custom system prompt here"

    def test_plan_missing_context_returns_422(self, mock_env_deepseek):
        """Missing context field returns validation error."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            from main import app
            client = TestClient(app)

            response = client.post("/plan", json={
                "system_prompt": "test"
            })

            assert response.status_code == 422


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_returns_provider_status(self, mock_env_deepseek):
        """Health endpoint returns configured provider info."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            import main
            mock_service = MagicMock()
            mock_service.get_provider_info.return_value = [{"name": "deepseek", "model": "deepseek-reasoner"}]
            main.llm_service = mock_service

            client = TestClient(main.app)

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "providers" in data
            assert len(data["providers"]) > 0

    def test_health_shows_available_providers(self, mock_env_auto):
        """Health endpoint shows all configured providers in auto mode."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            import main
            mock_service = MagicMock()
            mock_service.get_provider_info.return_value = [
                {"name": "deepseek", "model": "deepseek-reasoner"},
                {"name": "gemini", "model": "gemini-2.0-flash"}
            ]
            main.llm_service = mock_service

            client = TestClient(main.app)

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            # In auto mode with both keys, should show multiple providers
            assert len(data["providers"]) >= 1


class TestErrorHandling:
    """Test error handling in endpoints."""

    def test_classify_llm_error_returns_500(self, mock_env_deepseek):
        """LLM service error returns 500 with error details."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            from services.llm_service import AllProvidersFailedError
            import main

            mock_service = MagicMock()
            mock_service.call.side_effect = AllProvidersFailedError("All providers failed")
            main.llm_service = mock_service

            client = TestClient(main.app)

            response = client.post("/classify", json={
                "prompt": "test"
            })

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data

    def test_plan_invalid_json_response_returns_500(self, mock_env_deepseek):
        """Invalid JSON from LLM returns 500."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            with patch("main.llm_service") as mock_service:
                mock_result = MagicMock()
                mock_result.text = "not valid json {"
                mock_result.provider = "deepseek"
                mock_service.call.return_value = mock_result

                from main import app
                client = TestClient(app)

                response = client.post("/plan", json={
                    "context": {"host": {"ip": "192.168.1.1"}},
                    "system_prompt": "test"
                })

                assert response.status_code == 500


class TestChatCompletionsEndpoint:
    """Test /v1/chat/completions endpoint (OpenAI-compatible)."""

    def test_chat_completions_success(self, mock_env_deepseek):
        """Successful chat completion returns OpenAI-compatible response."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            with patch("main.llm_service") as mock_service:
                mock_result = MagicMock()
                mock_result.text = "Hello, I can help with that."
                mock_result.provider = "deepseek"
                mock_result.model = "deepseek-reasoner"
                mock_result.prompt_tokens = 50
                mock_result.completion_tokens = 20
                mock_result.cost_microcents = 1
                mock_result.latency_ms = 300
                mock_service.call.return_value = mock_result

                from main import app
                client = TestClient(app)

                response = client.post("/v1/chat/completions", json={
                    "model": "default",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello"},
                    ],
                })

                assert response.status_code == 200
                data = response.json()
                assert data["choices"][0]["message"]["content"] == "Hello, I can help with that."
                assert data["choices"][0]["message"]["role"] == "assistant"
                assert data["usage"]["prompt_tokens"] == 50
                assert data["usage"]["completion_tokens"] == 20

    def test_chat_completions_passes_system_prompt(self, mock_env_deepseek):
        """System message is extracted and passed to LLM service."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            with patch("main.llm_service") as mock_service:
                mock_result = MagicMock()
                mock_result.text = "response"
                mock_result.provider = "deepseek"
                mock_result.model = "deepseek-reasoner"
                mock_result.prompt_tokens = 10
                mock_result.completion_tokens = 5
                mock_result.cost_microcents = 0
                mock_result.latency_ms = 100
                mock_service.call.return_value = mock_result

                from main import app
                client = TestClient(app)

                client.post("/v1/chat/completions", json={
                    "model": "default",
                    "messages": [
                        {"role": "system", "content": "Be concise."},
                        {"role": "user", "content": "What is 2+2?"},
                    ],
                })

                call_args = mock_service.call.call_args
                assert call_args[0][0] == "What is 2+2?"
                assert call_args[0][1] == "Be concise."

    def test_chat_completions_without_system_message(self, mock_env_deepseek):
        """Works without a system message."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            with patch("main.llm_service") as mock_service:
                mock_result = MagicMock()
                mock_result.text = "4"
                mock_result.provider = "deepseek"
                mock_result.model = "deepseek-reasoner"
                mock_result.prompt_tokens = 5
                mock_result.completion_tokens = 1
                mock_result.cost_microcents = 0
                mock_result.latency_ms = 50
                mock_service.call.return_value = mock_result

                from main import app
                client = TestClient(app)

                response = client.post("/v1/chat/completions", json={
                    "model": "default",
                    "messages": [
                        {"role": "user", "content": "What is 2+2?"},
                    ],
                })

                assert response.status_code == 200
                call_args = mock_service.call.call_args
                assert call_args[0][0] == "What is 2+2?"
                assert call_args[0][1] is None

    def test_chat_completions_empty_messages_returns_422(self, mock_env_deepseek):
        """Empty messages array returns validation error."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            from main import app
            client = TestClient(app)

            response = client.post("/v1/chat/completions", json={
                "model": "default",
                "messages": [],
            })

            assert response.status_code == 422

    def test_reasoning_content_forwarded_in_response(self, mock_env_deepseek):
        """When provider returns reasoning_content, it appears in response message."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            with patch("main.llm_service") as mock_service:
                mock_result = MagicMock()
                mock_result.text = None
                mock_result.provider = "deepseek"
                mock_result.model = "deepseek-reasoner"
                mock_result.prompt_tokens = 20
                mock_result.completion_tokens = 10
                mock_result.cost_microcents = 1
                mock_result.latency_ms = 300
                mock_result.finish_reason = "tool_calls"
                mock_result.tool_calls = []
                mock_result.reasoning_content = "I need to think about this carefully..."
                mock_service.call_with_tools.return_value = mock_result

                from main import app
                client = TestClient(app)

                response = client.post("/v1/chat/completions", json={
                    "model": "deepseek-reasoner",
                    "messages": [{"role": "user", "content": "scan 192.168.1.1"}],
                    "tools": [{"type": "function", "function": {"name": "run_nmap", "parameters": {}}}],
                })

                assert response.status_code == 200
                data = response.json()
                assert data["choices"][0]["message"]["reasoning_content"] == "I need to think about this carefully..."

    def test_reasoning_content_in_request_history_not_dropped(self, mock_env_deepseek):
        """When client sends assistant message with reasoning_content, it is not stripped."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            with patch("main.llm_service") as mock_service:
                mock_result = MagicMock()
                mock_result.text = "done"
                mock_result.provider = "deepseek"
                mock_result.model = "deepseek-reasoner"
                mock_result.prompt_tokens = 30
                mock_result.completion_tokens = 5
                mock_result.cost_microcents = 1
                mock_result.latency_ms = 200
                mock_result.finish_reason = "stop"
                mock_result.tool_calls = []
                mock_result.reasoning_content = None
                mock_service.call_with_tools.return_value = mock_result

                from main import app
                client = TestClient(app)

                response = client.post("/v1/chat/completions", json={
                    "model": "deepseek-reasoner",
                    "messages": [
                        {"role": "user", "content": "scan now"},
                        {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "thinking...",
                            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "run_nmap", "arguments": "{}"}}],
                        },
                        {"role": "tool", "content": "scan complete", "tool_call_id": "c1"},
                    ],
                    "tools": [{"type": "function", "function": {"name": "run_nmap", "parameters": {}}}],
                })

                assert response.status_code == 200
                call_args = mock_service.call_with_tools.call_args
                messages_sent = call_args[0][0]
                assistant_msg = next(m for m in messages_sent if m["role"] == "assistant")
                assert assistant_msg.get("reasoning_content") == "thinking..."

    def test_chat_completions_llm_error_returns_500(self, mock_env_deepseek):
        """LLM failure returns 500."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            from services.llm_service import AllProvidersFailedError
            import main

            mock_service = MagicMock()
            mock_service.call.side_effect = AllProvidersFailedError("All providers failed")
            main.llm_service = mock_service

            client = TestClient(main.app)

            response = client.post("/v1/chat/completions", json={
                "model": "default",
                "messages": [{"role": "user", "content": "test"}],
            })

            assert response.status_code == 500

    def test_model_override_passed_to_llm_service(self, mock_env_deepseek):
        """When request has model != 'default', call_with_tools receives model_override."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            with patch("main.llm_service") as mock_service:
                mock_result = MagicMock()
                mock_result.text = "response"
                mock_result.provider = "deepseek"
                mock_result.model = "deepseek-chat"
                mock_result.prompt_tokens = 10
                mock_result.completion_tokens = 5
                mock_result.cost_microcents = 0
                mock_result.latency_ms = 100
                mock_result.finish_reason = "stop"
                mock_result.tool_calls = []
                mock_result.reasoning_content = None
                mock_service.call_with_tools.return_value = mock_result

                from main import app
                client = TestClient(app)

                client.post("/v1/chat/completions", json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [{"type": "function", "function": {"name": "run_nmap", "parameters": {}}}],
                })

                call_kwargs = mock_service.call_with_tools.call_args.kwargs
                assert call_kwargs.get("model_override") == "deepseek-chat"

    def test_model_default_does_not_set_override(self, mock_env_deepseek):
        """When request has model='default', call_with_tools receives model_override=None."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            with patch("main.llm_service") as mock_service:
                mock_result = MagicMock()
                mock_result.text = "response"
                mock_result.provider = "deepseek"
                mock_result.model = "deepseek-reasoner"
                mock_result.prompt_tokens = 10
                mock_result.completion_tokens = 5
                mock_result.cost_microcents = 0
                mock_result.latency_ms = 100
                mock_result.finish_reason = "stop"
                mock_result.tool_calls = []
                mock_result.reasoning_content = None
                mock_service.call_with_tools.return_value = mock_result

                from main import app
                client = TestClient(app)

                client.post("/v1/chat/completions", json={
                    "model": "default",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [{"type": "function", "function": {"name": "run_nmap", "parameters": {}}}],
                })

                call_kwargs = mock_service.call_with_tools.call_args.kwargs
                assert call_kwargs.get("model_override") is None


class TestAuditLogging:
    """Test audit log records emitted per LLM call."""

    def test_chat_completions_logs_audit_record(self, mock_env_deepseek, caplog):
        """POST /v1/chat/completions emits an audit log record with request_id and token counts."""
        import logging

        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            with patch("main.llm_service") as mock_service:
                mock_result = MagicMock()
                mock_result.text = "response text"
                mock_result.provider = "deepseek"
                mock_result.model = "deepseek-reasoner"
                mock_result.prompt_tokens = 80
                mock_result.completion_tokens = 30
                mock_result.cost_microcents = 1
                mock_result.latency_ms = 200
                mock_result.finish_reason = "stop"
                mock_result.tool_calls = []
                mock_result.reasoning_content = None
                mock_service.call_with_tools.return_value = mock_result

                from main import app
                client = TestClient(app)

                with caplog.at_level(logging.INFO, logger="llm-gateway"):
                    client.post("/v1/chat/completions", json={
                        "model": "deepseek-reasoner",
                        "messages": [{"role": "user", "content": "scan 10.0.0.2"}],
                        "tools": [{"type": "function", "function": {"name": "run_nmap", "parameters": {}}}],
                    })

                log_messages = [r.message for r in caplog.records if "llm_gateway_call" in r.message]
                assert len(log_messages) >= 2, f"Expected >=2 audit records, got: {log_messages}"
                request_log = next(m for m in log_messages if "llm_gateway_call request_id=" in m)
                complete_log = next(m for m in log_messages if "llm_gateway_call_complete" in m)
                assert "model=" in request_log
                assert "messages=" in request_log
                assert "prompt_tokens=" in complete_log
                assert "completion_tokens=" in complete_log


class TestRequestValidation:
    """Test request body validation."""

    def test_classify_accepts_prompt_only(self, mock_env_deepseek):
        """Classify endpoint only requires prompt."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            with patch("main.llm_service") as mock_service:
                mock_result = MagicMock()
                mock_result.text = '{"owned": [], "third_party": [], "interesting": [], "ignore": []}'
                mock_result.provider = "deepseek"
                mock_result.model = "deepseek-reasoner"
                mock_result.prompt_tokens = 10
                mock_result.completion_tokens = 5
                mock_result.cost_microcents = 0
                mock_result.latency_ms = 100
                mock_service.call.return_value = mock_result

                from main import app
                client = TestClient(app)

                response = client.post("/classify", json={
                    "prompt": "classify these hosts"
                })

                assert response.status_code == 200

    def test_plan_requires_context_and_system_prompt(self, mock_env_deepseek):
        """Plan endpoint requires both context and system_prompt."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            from main import app
            client = TestClient(app)

            # Missing system_prompt
            response = client.post("/plan", json={
                "context": {"host": {"ip": "192.168.1.1"}}
            })
            assert response.status_code == 422

            # Missing context
            response = client.post("/plan", json={
                "system_prompt": "test"
            })
            assert response.status_code == 422

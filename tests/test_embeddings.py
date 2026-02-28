"""Tests for /embed endpoint - written first following TDD."""

import pytest
import json
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


class TestEmbedEndpoint:
    """Test /embed endpoint for generating text embeddings."""

    def test_embed_single_text_success(self, mock_env_openai):
        """Successful embedding of single text returns 1536-dim vector."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            with patch("services.embedding_service.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_openai.return_value = mock_client

                mock_response = MagicMock()
                mock_response.data = [MagicMock()]
                mock_response.data[0].embedding = [0.1] * 1536
                mock_response.usage = MagicMock()
                mock_response.usage.prompt_tokens = 5
                mock_response.usage.total_tokens = 5
                mock_client.embeddings.create.return_value = mock_response

                import main
                from services.embedding_service import EmbeddingService
                main.config = MagicMock()
                main.embedding_service = EmbeddingService("test-openai-key")

                client = TestClient(main.app)

                response = client.post("/embed", json={
                    "text": "test embedding"
                })

                assert response.status_code == 200
                data = response.json()
                assert "embeddings" in data
                assert len(data["embeddings"]) == 1
                assert len(data["embeddings"][0]) == 1536
                assert data["dimensions"] == 1536

    def test_embed_multiple_texts_success(self, mock_env_openai):
        """Embedding multiple texts returns list of vectors."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            with patch("services.embedding_service.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_openai.return_value = mock_client

                mock_response = MagicMock()
                mock_response.data = [MagicMock(), MagicMock()]
                mock_response.data[0].embedding = [0.1] * 1536
                mock_response.data[1].embedding = [0.2] * 1536
                mock_response.usage = MagicMock()
                mock_response.usage.prompt_tokens = 10
                mock_response.usage.total_tokens = 10
                mock_client.embeddings.create.return_value = mock_response

                import main
                from services.embedding_service import EmbeddingService
                main.config = MagicMock()
                main.embedding_service = EmbeddingService("test-openai-key")

                client = TestClient(main.app)

                response = client.post("/embed", json={
                    "text": ["first text", "second text"]
                })

                assert response.status_code == 200
                data = response.json()
                assert len(data["embeddings"]) == 2
                assert len(data["embeddings"][0]) == 1536
                assert len(data["embeddings"][1]) == 1536

    def test_embed_empty_text_returns_422(self, mock_env_openai):
        """Empty text returns validation error."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            import main
            main.config = MagicMock()
            main.config.openai_api_key = "test-openai-key"

            client = TestClient(main.app)

            response = client.post("/embed", json={
                "text": ""
            })

            assert response.status_code == 422

    def test_embed_empty_list_returns_422(self, mock_env_openai):
        """Empty list returns validation error."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            import main
            main.config = MagicMock()
            main.config.openai_api_key = "test-openai-key"

            client = TestClient(main.app)

            response = client.post("/embed", json={
                "text": []
            })

            assert response.status_code == 422

    def test_embed_missing_api_key_returns_500(self, mock_env_deepseek):
        """Missing OPENAI_API_KEY returns error."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            import main
            main.config = MagicMock()
            main.embedding_service = None

            client = TestClient(main.app)

            response = client.post("/embed", json={
                "text": "test"
            })

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "OPENAI_API_KEY" in data["detail"]

    def test_embed_includes_ai_call_log(self, mock_env_openai):
        """Embedding response includes AI call log for auditing."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            with patch("services.embedding_service.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_openai.return_value = mock_client

                mock_response = MagicMock()
                mock_response.data = [MagicMock()]
                mock_response.data[0].embedding = [0.1] * 1536
                mock_response.usage = MagicMock()
                mock_response.usage.prompt_tokens = 5
                mock_response.usage.total_tokens = 5
                mock_client.embeddings.create.return_value = mock_response

                import main
                from services.embedding_service import EmbeddingService
                main.config = MagicMock()
                main.embedding_service = EmbeddingService("test-openai-key")

                client = TestClient(main.app)

                response = client.post("/embed", json={
                    "text": "test embedding"
                })

                assert response.status_code == 200
                data = response.json()
                assert "ai_call_log" in data
                ai_log = data["ai_call_log"]
                assert ai_log["provider"] == "openai"
                assert "text-embedding" in ai_log["model"]
                assert ai_log["prompt_tokens"] == 5
                assert ai_log["success"] is True

    def test_embed_custom_model(self, mock_env_openai):
        """Can specify a custom embedding model."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            with patch("services.embedding_service.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_openai.return_value = mock_client

                mock_response = MagicMock()
                mock_response.data = [MagicMock()]
                mock_response.data[0].embedding = [0.1] * 3072
                mock_response.usage = MagicMock()
                mock_response.usage.prompt_tokens = 5
                mock_response.usage.total_tokens = 5
                mock_client.embeddings.create.return_value = mock_response

                import main
                from services.embedding_service import EmbeddingService
                main.config = MagicMock()
                main.embedding_service = EmbeddingService("test-openai-key")

                client = TestClient(main.app)

                response = client.post("/embed", json={
                    "text": "test embedding",
                    "model": "text-embedding-3-large"
                })

                assert response.status_code == 200
                mock_client.embeddings.create.assert_called_once()
                call_args = mock_client.embeddings.create.call_args
                assert call_args.kwargs.get("model") == "text-embedding-3-large"


class TestEmbeddingService:
    """Test EmbeddingService class directly."""

    def test_service_generates_embeddings(self, mock_env_openai):
        """EmbeddingService generates embeddings successfully."""
        with patch("services.embedding_service.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = [0.1] * 1536
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 5
            mock_response.usage.total_tokens = 5
            mock_client.embeddings.create.return_value = mock_response

            from services.embedding_service import EmbeddingService

            service = EmbeddingService("test-api-key")
            result = service.generate(["test text"])

            assert result.embeddings is not None
            assert len(result.embeddings) == 1
            assert len(result.embeddings[0]) == 1536

    def test_service_calculates_cost(self, mock_env_openai):
        """EmbeddingService calculates cost correctly."""
        with patch("services.embedding_service.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = [0.1] * 1536
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 1000
            mock_response.usage.total_tokens = 1000
            mock_client.embeddings.create.return_value = mock_response

            from services.embedding_service import EmbeddingService

            service = EmbeddingService("test-api-key")
            result = service.generate(["test text"])

            assert result.cost_microcents is not None
            assert result.cost_microcents > 0


class TestHealthEndpointEmbeddings:
    """Test /health endpoint includes embeddings status."""

    def test_health_shows_embeddings_available(self, mock_env_openai):
        """Health endpoint shows embeddings as available when OPENAI_API_KEY is set."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            import main
            mock_service = MagicMock()
            mock_service.get_provider_info.return_value = [{"name": "openai", "model": "gpt-4o-mini"}]
            main.llm_service = mock_service
            main.config = MagicMock()
            main.config.openai_api_key = "test-key"
            main.config.embeddings_available.return_value = True

            client = TestClient(main.app)

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert "embeddings_available" in data
            assert data["embeddings_available"] is True

    def test_health_shows_embeddings_unavailable(self, mock_env_deepseek):
        """Health endpoint shows embeddings as unavailable when OPENAI_API_KEY is not set."""
        with patch("main.validate_api_keys"), patch("main.verify_credentials"):
            import main
            mock_service = MagicMock()
            mock_service.get_provider_info.return_value = [{"name": "deepseek", "model": "deepseek-reasoner"}]
            main.llm_service = mock_service
            main.config = MagicMock()
            main.config.openai_api_key = None
            main.config.embeddings_available.return_value = False

            client = TestClient(main.app)

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert "embeddings_available" in data
            assert data["embeddings_available"] is False

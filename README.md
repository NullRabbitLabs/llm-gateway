# LLM Gateway

Multi-provider LLM gateway with automatic fallback and cost tracking. Provides a single HTTP API that routes requests across DeepSeek, Gemini, OpenAI, and Anthropic — trying cheaper providers first and falling back automatically on failure.

## Quick Start

```bash
# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set up at least one provider
export LLM_PROVIDER=deepseek
export DEEPSEEK_API_KEY=your-key
export DEEPSEEK_MODEL=deepseek-chat

# Start the server
python main.py
```

The server runs on `http://localhost:8090` by default.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/classify` | POST | Classify items using AI (returns JSON) |
| `/plan` | POST | Generate structured plans using AI (returns JSON) |
| `/embed` | POST | Generate text embeddings (requires OPENAI_API_KEY) |
| `/v1/chat/completions` | POST | OpenAI-compatible chat with optional tool call support |
| `/health` | GET | Health check with provider status |

### POST /classify

Send a prompt, get back a JSON classification response.

```bash
curl -X POST http://localhost:8090/classify \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Classify these items: ..."}'
```

### POST /plan

Generate a structured plan from context and a system prompt.

```bash
curl -X POST http://localhost:8090/plan \
  -H "Content-Type: application/json" \
  -d '{
    "context": {"task": "...", "constraints": []},
    "system_prompt": "You are a planner. Return JSON."
  }'
```

### POST /embed

Generate text embeddings using OpenAI's embedding models.

```bash
curl -X POST http://localhost:8090/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "text to embed"}'
```

Request body:
- `text`: String or list of strings to embed
- `model`: Embedding model (default: `text-embedding-ada-002`)

Response:
```json
{
  "embeddings": [[0.1, 0.2, ...]],
  "model": "text-embedding-ada-002",
  "dimensions": 1536,
  "ai_call_log": {
    "provider": "openai",
    "model": "text-embedding-ada-002",
    "prompt_tokens": 5,
    "completion_tokens": 0,
    "cost_microcents": 1,
    "latency_ms": 150,
    "success": true
  }
}
```

### POST /v1/chat/completions

OpenAI-compatible endpoint supporting optional tool calls. Provider-specific translation (e.g. Anthropic tool format) is handled transparently.

```bash
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### GET /health

Check service health and provider status.

```bash
curl http://localhost:8090/health
```

Response:
```json
{
  "status": "healthy",
  "providers": [{"name": "deepseek", "model": "deepseek-chat"}],
  "embeddings_available": true
}
```

## Configuration

All configuration is via environment variables. Copy `.env.example` to `.env` and fill in your keys.

### Provider Selection

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `auto` | Provider: `auto`, `deepseek`, `gemini`, `openai`, `anthropic` |

When `LLM_PROVIDER=auto`, providers are tried in cost-effectiveness order:
1. DeepSeek — $0.12/1M input, $0.20/1M output
2. Gemini — $0.10/1M input, $0.40/1M output
3. OpenAI — $0.15/1M input, $0.60/1M output
4. Anthropic — $3/1M input, $15/1M output

### Provider API Keys

| Variable | Description |
|----------|-------------|
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `DEEPSEEK_MODEL` | DeepSeek model (e.g., `deepseek-chat`) |
| `GEMINI_API_KEY` | Google Gemini API key |
| `GEMINI_MODEL` | Gemini model (e.g., `gemini-2.0-flash`) |
| `OPENAI_API_KEY` | OpenAI API key (also required for `/embed`) |
| `OPENAI_MODEL` | OpenAI model (e.g., `gpt-4o-mini`) |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `ANTHROPIC_MODEL` | Anthropic model (e.g., `claude-3-5-sonnet-20241022`) |

At least one provider must have both API key and model configured.

### Service Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8090` | HTTP port |
| `LOG_LEVEL` | `INFO` | Logging level |

## Development

### Running Tests

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=. --cov-report=term-missing

# Run specific test file
pytest tests/test_providers.py -v
```

### Docker

```bash
# Build
docker build -t llm-gateway .

# Run
docker run -p 8090:8090 \
  -e LLM_PROVIDER=auto \
  -e DEEPSEEK_API_KEY=key \
  -e DEEPSEEK_MODEL=deepseek-chat \
  llm-gateway
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Your Svc A  │     │ Your Svc B  │     │ Your Svc C  │
│             │     │             │     │             │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │ HTTP              │ HTTP              │ HTTP
       ▼                   ▼                   ▼
┌──────────────────────────────────────────────────────┐
│                  llm-gateway (Python)                 │
│  ┌────────────────────────────────────────────────┐  │
│  │ Providers: DeepSeek | Gemini | OpenAI | Anthropic│ │
│  │ Features: Auto-fallback, Cost tracking, Retries  │ │
│  │ Endpoints: /plan, /classify, /embed, /health    │ │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Context

LLM Gateway is the provider abstraction layer used by [NullRabbit's](https://nullrabbit.ai) AI agents for autonomous threat analysis across validator infrastructure and decentralised networks.

It is open-sourced as a standalone tool because multi-provider routing with cost tracking and automatic fallback is useful beyond security — if you're building AI agents or pipelines that need resilient LLM access, this does the job.

## License

MIT — see [LICENSE](LICENSE).

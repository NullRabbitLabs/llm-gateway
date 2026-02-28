# LLM Gateway

Multi-provider LLM gateway with automatic fallback and cost tracking. Provides a single HTTP API that routes requests across DeepSeek, Gemini, OpenAI, Anthropic, Ollama — and any OpenAI-compatible API — trying cheaper providers first and falling back automatically on failure.

## Quick Start

```bash
# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set up at least one provider
export LLM_PROVIDER=deepseek
export DEEPSEEK_API_KEY=your-key

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

All configuration is via environment variables. Copy `.env.example` to `.env` and fill in your keys. Provider definitions (pricing, timeouts, features) live in `providers.json`.

### Provider Selection

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `auto` | Provider: `auto`, `ollama`, `deepseek`, `gemini`, `openai`, `anthropic` |

When `LLM_PROVIDER=auto`, providers are tried in the priority order defined in `providers.json` (default: cheapest first). Only providers with configured env vars are used.

### Provider API Keys

| Variable | Description |
|----------|-------------|
| `OLLAMA_HOST` | Ollama server URL (e.g., `http://localhost:11434`) |
| `OLLAMA_MODEL` | Ollama model (e.g., `qwen2.5-coder:14b`) |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `DEEPSEEK_MODEL` | DeepSeek model (default: `deepseek-chat`) |
| `GEMINI_API_KEY` | Google Gemini API key |
| `GEMINI_MODEL` | Gemini model (default: `gemini-2.0-flash`) |
| `OPENAI_API_KEY` | OpenAI API key (also required for `/embed`) |
| `OPENAI_MODEL` | OpenAI model (default: `gpt-4o-mini`) |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `ANTHROPIC_MODEL` | Anthropic model (default: `claude-3-5-sonnet-20241022`) |

At least one provider must have its required env vars configured (API key, or host for Ollama). Model env vars are optional — defaults come from `providers.json`.

### Service Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8090` | HTTP port |
| `LOG_LEVEL` | `INFO` | Logging level |

## Adding a New Provider

### OpenAI-compatible providers (zero Python code)

Any provider with an OpenAI-compatible API (Groq, Together, Mistral, etc.) can be added with just a JSON entry. Add to `providers.json`:

```json
{
  "providers": {
    "groq": {
      "kind": "openai_compatible",
      "base_url": "https://api.groq.com/openai/v1",
      "env_key": "GROQ_API_KEY",
      "env_model": "GROQ_MODEL",
      "default_model": "llama-3.3-70b-versatile",
      "timeout": 60,
      "features": { "tool_calls": true, "json_mode": true },
      "pricing": { "input_per_1k_microcents": 0.59, "output_per_1k_microcents": 0.79 }
    }
  }
}
```

Then set `GROQ_API_KEY` in your environment. That's it — no Python changes needed.

### Custom providers

For providers with non-OpenAI APIs (like Anthropic or Gemini), create a provider class in `providers/` that extends `Provider`, then register its `kind` in `providers/registry.py`'s `_KIND_MAP`.

### Provider config fields

| Field | Required | Description |
|-------|----------|-------------|
| `kind` | Yes | Provider class: `openai_compatible`, `anthropic`, `gemini`, `ollama` |
| `env_key` | Yes* | Env var for API key (*or `env_host` for Ollama) |
| `env_model` | No | Env var to override default model |
| `default_model` | Yes | Fallback model if env var is unset |
| `base_url` | No | API base URL (omit for default OpenAI endpoint) |
| `timeout` | No | Request timeout in seconds (default: 300) |
| `api_params` | No | Extra API params: `max_tokens`, `temperature`, etc. |
| `features` | No | `tool_calls`, `json_mode`, `reasoning_content` |
| `pricing` | No | `input_per_1k_microcents`, `output_per_1k_microcents` |

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
│  │             providers.json (registry)           │  │
│  │  ┌──────────────────────────────────────────┐  │  │
│  │  │ OpenAI-compatible: DeepSeek, OpenAI, ... │  │  │
│  │  │ Custom: Anthropic, Gemini, Ollama        │  │  │
│  │  └──────────────────────────────────────────┘  │  │
│  │ Features: Auto-fallback, Cost tracking, Retries │  │
│  │ Endpoints: /classify, /plan, /embed, /health   │  │
│  │            /v1/chat/completions                 │  │
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

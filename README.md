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

All endpoints are served under `/api/v1.0/`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1.0/classify` | POST | Classify items using AI (returns JSON) |
| `/api/v1.0/plan` | POST | Generate structured plans using AI (returns JSON) |
| `/api/v1.0/embed` | POST | Generate text embeddings (requires OPENAI_API_KEY) |
| `/api/v1.0/chat/completions` | POST | OpenAI-compatible chat with optional tool call support |
| `/api/v1.0/health` | GET | Health check with provider status |

> **Deprecated endpoints:** The following unversioned routes still work but are deprecated and will be removed in a future release. Migrate to the `/api/v1.0/` equivalents above.
>
> | Legacy Endpoint | Replacement |
> |----------------|-------------|
> | `POST /classify` | `POST /api/v1.0/classify` |
> | `POST /plan` | `POST /api/v1.0/plan` |
> | `POST /embed` | `POST /api/v1.0/embed` |
> | `POST /v1/chat/completions` | `POST /api/v1.0/chat/completions` |
> | `GET /health` | `GET /api/v1.0/health` |

### Model Resolution

Every endpoint that accepts a `model` field resolves the model using a 3-tier priority:

1. **Request body** `model` field (highest priority — client override)
2. **Environment variable** per provider (e.g. `DEEPSEEK_MODEL`, `OPENAI_MODEL`)
3. **`providers.json` `default_model`** (fallback)

Omit `model` or pass `"default"` to use the server-configured default. Pass any model name to override per-request.

### POST /api/v1.0/classify

Send a prompt, get back a JSON classification response.

```bash
curl -X POST http://localhost:8090/api/v1.0/classify \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Classify these items: ...", "model": "deepseek-chat"}'
```

Request body:
- `prompt` *(string, required)* — The classification prompt
- `model` *(string, optional)* — Model override

Response:
```json
{
  "classification": {
    "owned": [{"subdomain": "app.example.com", "category": "owned"}],
    "third_party": [],
    "interesting": [],
    "ignore": []
  },
  "ai_call_log": {
    "provider": "deepseek",
    "model": "deepseek-chat",
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "cost_microcents": 12,
    "latency_ms": 500,
    "success": true
  }
}
```

### POST /api/v1.0/plan

Generate a structured plan from context and a system prompt.

```bash
curl -X POST http://localhost:8090/api/v1.0/plan \
  -H "Content-Type: application/json" \
  -d '{
    "context": {"task": "...", "constraints": []},
    "system_prompt": "You are a planner. Return JSON.",
    "model": "deepseek-chat"
  }'
```

Request body:
- `context` *(object, required)* — Arbitrary context dict for the planning task
- `system_prompt` *(string, required)* — System prompt for the LLM
- `model` *(string, optional)* — Model override

Response:
```json
{
  "plan": {
    "assessment": "needs_followup",
    "confidence": "medium",
    "rationale": "Discovered open ports need verification",
    "probes": [
      {
        "cmd": "curl -s -o /dev/null -w '%{http_code}' https://192.168.1.1:443",
        "timeout_s": 30,
        "note": "Verify HTTPS service"
      }
    ]
  },
  "ai_call_log": {
    "provider": "deepseek",
    "model": "deepseek-chat",
    "prompt_tokens": 200,
    "completion_tokens": 100,
    "cost_microcents": 24,
    "latency_ms": 800,
    "success": true
  }
}
```

### POST /api/v1.0/embed

Generate text embeddings using OpenAI's embedding models.

```bash
curl -X POST http://localhost:8090/api/v1.0/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "text to embed"}'
```

Request body:
- `text` *(string or array of strings, required)* — Text to embed
- `model` *(string, optional)* — Embedding model (default: `text-embedding-ada-002`)

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

### POST /api/v1.0/chat/completions

OpenAI-compatible endpoint supporting optional tool calls. Provider-specific translation (e.g. Anthropic tool format) is handled transparently.

```bash
curl -X POST http://localhost:8090/api/v1.0/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Request body:
- `model` *(string, optional)* — Model to use (`"default"` or omit for server default)
- `messages` *(array, required)* — Chat messages, each with `role` and `content`
- `tools` *(array, optional)* — Tool definitions for function calling

Request with tools:
```json
{
  "model": "deepseek-chat",
  "messages": [
    {"role": "user", "content": "scan 192.168.1.1"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "run_nmap",
        "description": "Run nmap scan",
        "parameters": {
          "type": "object",
          "properties": {
            "target": {"type": "string"}
          }
        }
      }
    }
  ]
}
```

Response:
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help?",
        "reasoning_content": null,
        "tool_calls": null
      },
      "finish_reason": "stop",
      "index": 0
    }
  ],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 20,
    "total_tokens": 70
  },
  "model": "deepseek-chat"
}
```

Response with tool calls:
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_xyz789",
            "type": "function",
            "function": {
              "name": "run_nmap",
              "arguments": "{\"target\": \"192.168.1.1\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls",
      "index": 0
    }
  ],
  "usage": {
    "prompt_tokens": 80,
    "completion_tokens": 30,
    "total_tokens": 110
  },
  "model": "deepseek-chat"
}
```

### GET /api/v1.0/health

Check service health and provider status.

```bash
curl http://localhost:8090/api/v1.0/health
```

Response:
```json
{
  "status": "healthy",
  "providers": [{"name": "deepseek", "model": "deepseek-chat"}],
  "embeddings_available": true
}
```

### Error Responses

All endpoints return a 500 with detail when all providers fail:

```json
{
  "detail": "All providers failed:\ndeepseek: timeout\ngemini: api error"
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

Pre-built images are published to GitHub Container Registry on every release.

```bash
# Pull and run
docker run -p 8090:8090 \
  -e LLM_PROVIDER=auto \
  -e DEEPSEEK_API_KEY=key \
  ghcr.io/nullrabbitlabs/llm-gateway:latest
```

Pin to a specific version in production:

```bash
docker pull ghcr.io/nullrabbitlabs/llm-gateway:1.0.0
```

To build locally instead:

```bash
docker build -t llm-gateway .
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
│  │ Endpoints: /api/v1.0/classify, /plan, /embed    │  │
│  │            /chat/completions, /health           │  │
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

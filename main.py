"""LLM Gateway - Multi-provider LLM gateway with automatic fallback and cost tracking."""

import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from services.llm_service import LLMService, AllProvidersFailedError
from services.embedding_service import EmbeddingService

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("llm-gateway")

# Global service instances
llm_service: Optional[LLMService] = None
config: Optional[Config] = None
embedding_service: Optional[EmbeddingService] = None


def validate_api_keys() -> None:
    """Validate that required API keys are configured."""
    global config
    config = Config.from_env()
    log.info(f"Configuration validated: provider={config.provider}")


def verify_credentials() -> None:
    """Verify API credentials by making minimal test calls."""
    global llm_service, config, embedding_service

    if config is None:
        raise RuntimeError("Configuration not loaded")

    llm_service = LLMService(config)
    log.info(f"LLM Service initialized with providers: {[p.name for p in llm_service.providers]}")

    if config.embeddings_available():
        embedding_service = EmbeddingService(os.getenv("OPENAI_API_KEY", ""))
        log.info("Embedding service initialized")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    log.info("Starting LLM Gateway...")
    try:
        validate_api_keys()
        verify_credentials()
        log.info("LLM Gateway started successfully")
    except Exception as e:
        log.error(f"Startup failed: {e}")
        raise

    yield

    # Shutdown
    log.info("LLM Gateway shutting down")


app = FastAPI(
    title="LLM Gateway",
    description="Multi-provider LLM gateway with automatic fallback and cost tracking.",
    version="1.0.0",
    lifespan=lifespan,
)


# Request/Response models


class ClassifyRequest(BaseModel):
    """Request body for /classify endpoint."""

    prompt: str = Field(..., min_length=1, description="Classification prompt")
    model: Optional[str] = Field(default=None, description="Model override (e.g. 'deepseek-chat')")

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt cannot be empty")
        return v


class AICallLog(BaseModel):
    """AI call metadata for auditing."""

    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_microcents: int
    latency_ms: int
    success: bool


class ClassifyResponse(BaseModel):
    """Response body for /classify endpoint."""

    classification: dict[str, Any]
    ai_call_log: AICallLog


class PlanRequest(BaseModel):
    """Request body for /plan endpoint."""

    context: dict[str, Any] = Field(..., description="Scan context for planning")
    system_prompt: str = Field(..., min_length=1, description="System prompt for the planner")
    model: Optional[str] = Field(default=None, description="Model override (e.g. 'deepseek-chat')")


class PlanResponse(BaseModel):
    """Response body for /plan endpoint."""

    plan: dict[str, Any]
    ai_call_log: AICallLog


class EmbedRequest(BaseModel):
    """Request body for /embed endpoint."""

    text: str | list[str] = Field(..., description="Text or list of texts to embed")
    model: str = Field(default="text-embedding-ada-002", description="Embedding model to use")

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str | list[str]) -> str | list[str]:
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("text cannot be empty")
        elif isinstance(v, list):
            if len(v) == 0:
                raise ValueError("text list cannot be empty")
            for item in v:
                if not item.strip():
                    raise ValueError("text items cannot be empty")
        return v


class EmbedResponse(BaseModel):
    """Response body for /embed endpoint."""

    embeddings: list[list[float]]
    model: str
    dimensions: int
    ai_call_log: AICallLog


class ChatMessage(BaseModel):
    """A single message in a chat conversation (OpenAI format)."""

    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    reasoning_content: str | None = None


class ToolFunction(BaseModel):
    """OpenAI function definition within a tool."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class Tool(BaseModel):
    """OpenAI-format tool schema."""

    type: str = "function"
    function: ToolFunction


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = "default"
    messages: list[ChatMessage] = Field(..., min_length=1)
    tools: list[Tool] | None = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    choices: list[dict[str, Any]]
    usage: dict[str, int]
    model: str


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str
    providers: list[dict[str, str]]
    embeddings_available: bool = False


class ErrorResponse(BaseModel):
    """Error response body."""

    error: str
    detail: Optional[str] = None


# Endpoints


@app.post("/classify", response_model=ClassifyResponse, responses={500: {"model": ErrorResponse}})
async def classify(request: ClassifyRequest) -> ClassifyResponse:
    """Classify items using AI."""
    if llm_service is None:
        raise HTTPException(status_code=500, detail="LLM service not initialized")

    try:
        result = llm_service.call(request.prompt, model_override=request.model)

        # Parse classification JSON
        if result.text is None:
            raise HTTPException(status_code=500, detail="LLM returned no text content")
        try:
            classification = json.loads(result.text)
        except json.JSONDecodeError as e:
            log.error(f"Failed to parse classification JSON: {e}\nResponse: {(result.text or '')[:500]}")
            raise HTTPException(
                status_code=500,
                detail=f"Invalid JSON response from LLM: {e}"
            )

        ai_call_log = AICallLog(
            provider=result.provider,
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_microcents=result.cost_microcents,
            latency_ms=result.latency_ms,
            success=True,
        )

        return ClassifyResponse(classification=classification, ai_call_log=ai_call_log)

    except AllProvidersFailedError as e:
        log.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/plan", response_model=PlanResponse, responses={500: {"model": ErrorResponse}})
async def plan(request: PlanRequest) -> PlanResponse:
    """Generate a structured plan using AI."""
    if llm_service is None:
        raise HTTPException(status_code=500, detail="LLM service not initialized")

    try:
        # Convert context to JSON string for the prompt
        context_json = json.dumps(request.context, ensure_ascii=False, separators=(",", ":"))

        result = llm_service.call(context_json, request.system_prompt, model_override=request.model)

        # Parse plan JSON
        if result.text is None:
            raise HTTPException(status_code=500, detail="LLM returned no text content")
        try:
            plan_data = json.loads(result.text)
        except json.JSONDecodeError as e:
            log.error(f"Failed to parse plan JSON: {e}\nResponse: {(result.text or '')[:500]}")
            raise HTTPException(
                status_code=500,
                detail=f"Invalid JSON response from LLM: {e}"
            )

        ai_call_log = AICallLog(
            provider=result.provider,
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_microcents=result.cost_microcents,
            latency_ms=result.latency_ms,
            success=True,
        )

        return PlanResponse(plan=plan_data, ai_call_log=ai_call_log)

    except AllProvidersFailedError as e:
        log.error(f"Plan generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed", response_model=EmbedResponse, responses={500: {"model": ErrorResponse}})
async def embed(request: EmbedRequest) -> EmbedResponse:
    """Generate text embeddings using OpenAI API.

    Requires OPENAI_API_KEY to be configured.
    """
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not loaded")

    if embedding_service is None:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is required for embeddings but not configured"
        )

    try:
        texts = [request.text] if isinstance(request.text, str) else request.text

        result = embedding_service.generate(texts, request.model)

        ai_call_log = AICallLog(
            provider="openai",
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=0,
            cost_microcents=result.cost_microcents,
            latency_ms=result.latency_ms,
            success=True,
        )

        return EmbedResponse(
            embeddings=result.embeddings,
            model=result.model,
            dimensions=result.dimensions,
            ai_call_log=ai_call_log,
        )

    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        log.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, responses={500: {"model": ErrorResponse}})
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """OpenAI-compatible chat completions endpoint with optional tool support.

    When tools are provided, provider-specific translation is handled
    transparently by the gateway.
    """
    if llm_service is None:
        raise HTTPException(status_code=500, detail="LLM service not initialized")

    request_id = str(uuid.uuid4())[:8]
    log.info(
        "llm_gateway_call request_id=%s model=%s messages=%d total_chars=%d",
        request_id,
        request.model,
        len(request.messages),
        sum(len(str(m.content or "")) for m in request.messages),
    )

    if log.isEnabledFor(logging.DEBUG):
        for i, msg in enumerate(request.messages):
            content_preview = str(msg.content or "")[:200]
            log.debug("llm_gateway_call msg[%d] role=%s preview=%r", i, msg.role, content_preview)

    try:
        if request.tools:
            # Tool-use path: pass full messages + tools through provider translation
            messages = [m.model_dump(exclude_none=True) for m in request.messages]
            tools = [t.model_dump() for t in request.tools]

            model_override = request.model if request.model != "default" else None
            result = llm_service.call_with_tools(messages, tools, model_override=model_override)

            log.info(
                "llm_gateway_call_complete request_id=%s model=%s prompt_tokens=%d completion_tokens=%d finish_reason=%s",
                request_id,
                result.model,
                result.prompt_tokens,
                result.completion_tokens,
                result.finish_reason,
            )

            message_dict: dict[str, Any] = {
                "role": "assistant",
                "content": result.text,
            }
            if result.tool_calls:
                message_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in result.tool_calls
                ]

            if result.reasoning_content is not None:
                message_dict["reasoning_content"] = result.reasoning_content

            return ChatCompletionResponse(
                choices=[{
                    "message": message_dict,
                    "finish_reason": result.finish_reason,
                    "index": 0,
                }],
                usage={
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.prompt_tokens + result.completion_tokens,
                },
                model=result.model,
            )

        # Text-only path: extract system/user messages
        system_prompt = None
        user_message = ""
        for msg in request.messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                user_message = msg.content or ""

        result = llm_service.call(user_message, system_prompt)

        log.info(
            "llm_gateway_call_complete request_id=%s model=%s prompt_tokens=%d completion_tokens=%d finish_reason=%s",
            request_id,
            result.model,
            result.prompt_tokens,
            result.completion_tokens,
            "stop",
        )

        return ChatCompletionResponse(
            choices=[{
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": "stop",
                "index": 0,
            }],
            usage={
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
            model=result.model,
        )

    except AllProvidersFailedError as e:
        log.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint with provider status."""
    providers = []
    if llm_service:
        providers = llm_service.get_provider_info()

    embeddings_available = config.embeddings_available() if config else False

    return HealthResponse(
        status="healthy",
        providers=providers,
        embeddings_available=embeddings_available,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8090"))
    uvicorn.run(app, host="0.0.0.0", port=port)

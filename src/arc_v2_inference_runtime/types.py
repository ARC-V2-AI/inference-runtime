from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict


class ToolFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunction


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ToolCallFunctionDelta(BaseModel):
    name: str | None = None
    arguments: str | None = None


class ToolCallDelta(BaseModel):
    index: int = 0
    id: str | None = None
    type: Literal["function"] | None = None
    function: ToolCallFunctionDelta | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None


ReasoningEffort = Literal["auto", "none", "on"]


class ToolChoiceFunction(BaseModel):
    name: str


class NamedToolChoice(BaseModel):
    type: Literal["function"] = "function"
    function: ToolChoiceFunction


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    stream: bool = True
    max_output_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: list[str] = Field(default_factory=list)
    response_format: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(GenerationRequest):
    messages: list[ChatMessage] = Field(min_length=1)
    tools: list[Tool] = Field(default_factory=list)
    tool_choice: Literal["auto", "none", "required"] | NamedToolChoice | None = None
    reasoning_effort: ReasoningEffort = "auto"


class CompletionRequest(GenerationRequest):
    prompt: str


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    # Qwen3 query embeddings benefit from an instruction prefix; document
    # embeddings are embedded without that query prefix. "raw" preserves the
    # original OpenAI-compatible behavior for existing clients.
    input_type: Literal["raw", "query", "document"] = "raw"
    instruction: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TokenizeRequest(BaseModel):
    input: str | list[str]
    include_tokens: bool = False
    add_bos: bool = False
    special: bool = True


class ConversationCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=256)


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None
    warning: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]


class ChatCompletionChunkDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCallDelta] | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None
    warning: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class EmbeddingData(BaseModel):
    index: int
    embedding: list[float]
    object: Literal["embedding"] = "embedding"


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    model: str
    data: list[EmbeddingData]


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "arc"
    tool_protocol: str = "unknown"
    supports_tools: bool = False
    supports_reasoning: bool = False
    reasoning_modes: list[str] = Field(default_factory=lambda: ["auto", "none", "on"])

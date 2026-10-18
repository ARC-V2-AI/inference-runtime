import json
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from arc_v2_inference_runtime.constants import ARC_RUNTIME_DEBUG
from arc_v2_inference_runtime.engine.engine import Engine, GenerationChunk
from arc_v2_inference_runtime.engine.context_manager import ContextLengthError
from arc_v2_inference_runtime.engine.conversations import ConversationNotFoundError
from arc_v2_inference_runtime.types import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionResponse,
    ChatMessage,
    ChatRequest,
    CompletionRequest,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelInfo,
    TokenizeRequest,
    ConversationCreateRequest,
    ToolCall,
    ToolCallDelta,
    ToolCallFunction,
    ToolCallFunctionDelta,
)

log = logging.getLogger(__name__)
app = FastAPI(debug=ARC_RUNTIME_DEBUG.value)
VERSION = "/v1"


def version_string(path: str) -> str:
    return f"{VERSION}{path}"


def _get_engine(request: Request) -> Engine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    return engine


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


class _Accumulator:
    def __init__(self) -> None:
        self.content_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self._tool_calls: dict[int, dict] = {}

    def add(self, chunk: GenerationChunk) -> None:
        if chunk.event:
            if chunk.event.kind == "content" and chunk.event.text:
                self.content_parts.append(chunk.event.text)
            elif chunk.event.kind == "reasoning" and chunk.event.text:
                self.reasoning_parts.append(chunk.event.text)
        if chunk.tool_call:
            tc = chunk.tool_call
            entry = self._tool_calls.setdefault(
                tc.index, {"id": None, "name": None, "arguments": ""}
            )
            if tc.id:
                entry["id"] = tc.id
            if tc.name:
                entry["name"] = tc.name
            entry["arguments"] += tc.arguments_fragment or ""

    def to_message(self) -> ChatMessage:
        tool_calls = None
        if self._tool_calls:
            tool_calls = [
                ToolCall(
                    id=entry["id"] or _new_id("call"),
                    function=ToolCallFunction(
                        name=entry["name"] or "", arguments=entry["arguments"] or "{}"
                    ),
                )
                for _, entry in sorted(self._tool_calls.items())
            ]
        return ChatMessage(
            role="assistant",
            content="".join(self.content_parts) or None,
            reasoning_content="".join(self.reasoning_parts) or None,
            tool_calls=tool_calls,
        )


async def _chat_completions(
    req: ChatRequest,
    request: Request,
    *,
    conversation_id: str | None = None,
):
    engine = _get_engine(request)
    completion_id = _new_id("chatcmpl")
    created = int(time.time())
    new_messages = list(req.messages)

    if conversation_id is not None:
        try:
            history = engine.conversations.get_messages(conversation_id)
        except ConversationNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Conversation not found") from exc
        # This endpoint deliberately accepts only the new turn. Send transcript
        # messages through GET /conversations/{id}; do not resend the old history.
        if any(message.role == "system" for message in new_messages) and history:
            raise HTTPException(
                status_code=400,
                detail="A persisted conversation already has history; send only new user/tool messages, not the system prompt or prior turns.",
            )
        req = req.model_copy(update={"messages": [*history, *new_messages]})

    try:
        req = engine.fit_chat_request(req)
    except ContextLengthError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if conversation_id is not None:
        try:
            engine.conversations.append_messages(conversation_id, new_messages)
        except ConversationNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Conversation not found") from exc

    if req.stream:

        async def event_stream():
            sent_role = False
            acc = _Accumulator()
            async for chunk in engine.generate_chat(req):
                acc.add(chunk)
                delta = ChatCompletionChunkDelta()
                if not sent_role:
                    delta.role = "assistant"
                    sent_role = True
                if chunk.event:
                    if chunk.event.kind == "content" and chunk.event.text:
                        delta.content = chunk.event.text
                    elif chunk.event.kind == "reasoning" and chunk.event.text:
                        delta.reasoning_content = chunk.event.text
                if chunk.tool_call:
                    tc = chunk.tool_call
                    delta.tool_calls = [
                        ToolCallDelta(
                            index=tc.index,
                            id=tc.id,
                            type="function" if tc.id or tc.name else None,
                            function=ToolCallFunctionDelta(
                                name=tc.name, arguments=tc.arguments_fragment or None
                            ),
                        )
                    ]
                out = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=engine.model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=delta,
                            finish_reason=chunk.finish_reason,
                            warning=chunk.warning,
                        )
                    ],
                )
                yield f"data: {out.model_dump_json()}\n\n"
            if conversation_id is not None:
                engine.conversations.append_messages(conversation_id, [acc.to_message()])
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    acc = _Accumulator()
    finish_reason = None
    warning = None
    async for chunk in engine.generate_chat(req):
        acc.add(chunk)
        if chunk.finish_reason:
            finish_reason = chunk.finish_reason
        if chunk.warning:
            warning = chunk.warning
    message = acc.to_message()
    if conversation_id is not None:
        engine.conversations.append_messages(conversation_id, [message])
    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=engine.model_name,
        choices=[ChatCompletionChoice(message=message, finish_reason=finish_reason, warning=warning)],
    )


@app.post(version_string("/chat/completions"))
async def chat_completions(req: ChatRequest, request: Request):
    return await _chat_completions(req, request)


@app.post(version_string("/conversations/{conversation_id}/chat/completions"))
async def conversation_chat_completions(
    conversation_id: str, req: ChatRequest, request: Request
):
    return await _chat_completions(req, request, conversation_id=conversation_id)


@app.post(version_string("/completions"))
async def completions(req: CompletionRequest, request: Request):
    engine = _get_engine(request)
    completion_id = _new_id("cmpl")
    created = int(time.time())
    try:
        req = engine.fit_completion_request(req)
    except ContextLengthError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    if req.stream:

        async def event_stream():
            async for chunk in engine.generate_completion(req):
                if not chunk.event or chunk.event.kind != "content":
                    continue
                payload = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": engine.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "text": chunk.event.text,
                            "finish_reason": chunk.finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    text_parts: list[str] = []
    finish_reason = None
    async for chunk in engine.generate_completion(req):
        if chunk.event and chunk.event.kind == "content":
            text_parts.append(chunk.event.text)
        if chunk.finish_reason:
            finish_reason = chunk.finish_reason
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": engine.model_name,
        "choices": [
            {"index": 0, "text": "".join(text_parts), "finish_reason": finish_reason}
        ],
    }


@app.post(version_string("/embeddings"))
async def embeddings(req: EmbeddingRequest, request: Request):
    engine = _get_engine(request)
    inputs = [req.input] if isinstance(req.input, str) else req.input
    try:
        vectors = await engine.embed(
            inputs, input_type=req.input_type, instruction=req.instruction
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return EmbeddingResponse(
        model=engine.embedding_model_name,
        data=[EmbeddingData(index=i, embedding=vec) for i, vec in enumerate(vectors)],
    )


@app.post(version_string("/tokenize"))
async def tokenize(req: TokenizeRequest, request: Request):
    engine = _get_engine(request)
    inputs = [req.input] if isinstance(req.input, str) else req.input
    data = []
    for index, value in enumerate(inputs):
        token_ids = engine.tokenize(value, add_bos=req.add_bos, special=req.special)
        item = {"index": index, "token_count": len(token_ids)}
        if req.include_tokens:
            item["tokens"] = token_ids
        data.append(item)
    return {"object": "list", "model": engine.model_name, "data": data}


@app.post(version_string("/chat/context"))
async def chat_context(req: ChatRequest, request: Request):
    engine = _get_engine(request)
    try:
        return {"model": engine.model_name, **engine.count_chat_context(req)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(version_string("/conversations"))
async def create_conversation(req: ConversationCreateRequest, request: Request):
    engine = _get_engine(request)
    return engine.conversations.create(req.title)


@app.get(version_string("/conversations"))
async def list_conversations(request: Request, limit: int = 100):
    engine = _get_engine(request)
    return {"object": "list", "data": engine.conversations.list(limit=limit)}


@app.get(version_string("/conversations/{conversation_id}"))
async def get_conversation(conversation_id: str, request: Request):
    engine = _get_engine(request)
    try:
        conversation = engine.conversations.get(conversation_id)
        messages = engine.conversations.get_messages(conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    return {**conversation, "messages": [message.model_dump(mode="json", exclude_none=True) for message in messages]}


@app.delete(version_string("/conversations/{conversation_id}"))
async def delete_conversation(conversation_id: str, request: Request):
    engine = _get_engine(request)
    if not engine.conversations.delete(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"id": conversation_id, "deleted": True}


@app.get(version_string("/models"))
async def models(request: Request):
    engine = _get_engine(request)
    protocol = engine.protocol
    info = ModelInfo(id=engine.model_name)
    if protocol:
        info.tool_protocol = str(protocol.tool_call_format)
        info.supports_tools = protocol.supports_tools
        info.supports_reasoning = protocol.supports_reasoning
    return {"object": "list", "data": [info.model_dump()]}


@app.get(version_string("/model"))
async def model(request: Request):
    engine = _get_engine(request)
    protocol = engine.protocol
    info = ModelInfo(id=engine.model_name)
    if protocol:
        info.tool_protocol = str(protocol.tool_call_format)
        info.supports_tools = protocol.supports_tools
        info.supports_reasoning = protocol.supports_reasoning
    return info

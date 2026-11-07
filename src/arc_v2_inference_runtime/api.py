from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from arc_v2_inference_runtime.constants import ARC_RUNTIME_DEBUG
from arc_v2_inference_runtime.engine.context_manager import ContextLengthError
from arc_v2_inference_runtime.engine.engine import Engine, GenerationChunk
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
        raise HTTPException(status_code=503, detail="Inference runtime is not initialized")
    return engine


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


class _Accumulator:
    def __init__(self) -> None:
        self.content_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self._tool_calls: dict[int, dict] = {}
        self.usage: dict[str, int] = {}
        self.finish_reason: str | None = None
        self.warning: str | None = None

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
        if chunk.usage:
            self.usage = chunk.usage
        if chunk.finish_reason:
            self.finish_reason = chunk.finish_reason
        if chunk.warning:
            self.warning = chunk.warning

    def to_message(self) -> ChatMessage:
        tool_calls = None
        # Streaming clients may have received partial tool-call deltas before
        # generation hit its token limit. Never expose those as executable
        # non-streaming calls unless the engine confirmed protocol completion.
        if self._tool_calls and self.finish_reason == "tool_calls":
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


async def _chat_completions(req: ChatRequest, request: Request):
    engine = _get_engine(request)
    completion_id = _new_id("chatcmpl")
    created = int(time.time())

    try:
        fitted = engine.fit_chat_request(req)
    except ContextLengthError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if fitted.stream:
        async def event_stream():
            sent_role = False
            acc = _Accumulator()
            async for chunk in engine.generate_chat(fitted):
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
                    usage=chunk.usage,
                )
                yield f"data: {out.model_dump_json(exclude_none=True)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    acc = _Accumulator()
    async for chunk in engine.generate_chat(fitted):
        acc.add(chunk)
    message = acc.to_message()
    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=engine.model_name,
        choices=[
            ChatCompletionChoice(
                message=message,
                finish_reason=acc.finish_reason,
                warning=acc.warning,
            )
        ],
        usage=acc.usage,
    )


@app.post(version_string("/chat/completions"))
async def chat_completions(req: ChatRequest, request: Request):
    return await _chat_completions(req, request)


@app.post(version_string("/completions"))
async def completions(req: CompletionRequest, request: Request):
    engine = _get_engine(request)
    completion_id = _new_id("cmpl")
    created = int(time.time())
    try:
        req = engine.fit_completion_request(req)
    except ContextLengthError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if req.stream:
        async def event_stream():
            async for chunk in engine.generate_completion(req):
                if (not chunk.event or chunk.event.kind != "content") and not chunk.usage:
                    continue
                payload = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": engine.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "text": chunk.event.text if chunk.event else "",
                            "finish_reason": chunk.finish_reason,
                        }
                    ],
                }
                if chunk.usage:
                    payload["usage"] = chunk.usage
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    text_parts: list[str] = []
    finish_reason = None
    usage: dict[str, int] = {}
    async for chunk in engine.generate_completion(req):
        if chunk.event and chunk.event.kind == "content":
            text_parts.append(chunk.event.text)
        if chunk.finish_reason:
            finish_reason = chunk.finish_reason
        if chunk.usage:
            usage = chunk.usage
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": engine.model_name,
        "choices": [
            {"index": 0, "text": "".join(text_parts), "finish_reason": finish_reason}
        ],
        "usage": usage,
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
    try:
        for index, value in enumerate(inputs):
            token_ids = engine.tokenize(value, add_bos=req.add_bos, special=req.special)
            item = {"index": index, "token_count": len(token_ids)}
            if req.include_tokens:
                item["tokens"] = token_ids
            data.append(item)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"object": "list", "model": engine.model_name, "data": data}


@app.post(version_string("/context/inspect"))
@app.post(version_string("/chat/context"), include_in_schema=False)
async def inspect_context(req: ChatRequest, request: Request):
    engine = _get_engine(request)
    try:
        return {"model": engine.model_name, **engine.count_chat_context(req)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get(version_string("/runtime/status"))
async def runtime_status(request: Request):
    return _get_engine(request).runtime_status()


@app.get(version_string("/runtime/memory"))
async def runtime_memory(request: Request):
    return _get_engine(request).memory_status()


@app.post(version_string("/runtime/cache/clear"))
async def clear_runtime_cache(request: Request):
    engine = _get_engine(request)
    try:
        result = await engine.clear_kv_cache()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        **result,
        "scope": "global_active_sequence",
        "allocated_vram_released": False,
    }


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

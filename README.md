# ARC-V2 Inference Runtime

A local-first HTTP inference service for ARC. The runtime owns the loaded model, native prompt rendering, tokenization, context-window enforcement, serialized inference and inference-resource cleanup. It does not own agent conversations or long-term memory.

## Design

- One main model instance is loaded for the lifetime of the service.
- Chat and completion requests are stateless: clients submit the complete prompt/message list for each request.
- The request queue serializes operations against the model context.
- KV sequence contents are cleared after every completed, cancelled or failed chat/completion generation. The allocated KV buffer remains loaded and is reused by later requests.
- Conversation history, summaries, agent state and persistent memory belong to ARC/client-side services.
- Optional multi-request KV sessions are not implemented in this version. They require sequence-ID ownership and a backend that can safely preserve multiple sequences.

## API

Base URL: `http://127.0.0.1:7842/v1`

| Method | Route | Purpose |
|---|---|---|
| POST | `/chat/completions` | Stateless chat inference; supports streaming and tool calls |
| POST | `/completions` | Stateless text completion |
| POST | `/embeddings` | Generate embeddings |
| POST | `/tokenize` | Count tokens and optionally return token IDs |
| POST | `/context/inspect` | Count the rendered prompt using the loaded model's chat template |
| GET | `/model` | Current model metadata and capabilities |
| GET | `/models` | Model listing |
| GET | `/runtime/status` | Runtime, queue and KV status |
| GET | `/runtime/memory` | Process RAM and NVIDIA GPU memory snapshot |
| POST | `/runtime/cache/clear` | Clear active KV sequence contents (global operation) |

`/chat/context` remains as a hidden-schema compatibility alias for `/context/inspect`.

### Context controls

Chat and completion requests accept:

```json
{
  "messages": [{"role": "user", "content": "Explain KV caching."}],
  "max_output_tokens": 512,
  "context_limit": 4096,
  "context_overflow": "truncate"
}
```

- `context_limit` may reduce the context budget for one request but cannot exceed the server's configured `ARC_RUNTIME_N_CTX`.
- `context_overflow="truncate"` removes oldest complete user turns while preserving system messages and the newest turn.
- `context_overflow="reject"` returns HTTP 413 rather than dropping history.
- Plain completion prompts are rejected if they do not fit; the runtime does not silently rewrite them.

Responses include `usage.prompt_tokens`, `usage.completion_tokens` and `usage.total_tokens`. For streamed responses, usage is attached to the final chunk.

### KV cleanup semantics

The runtime clears KV cache contents between requests to prevent a later request from reusing another request's prompt state. This does **not** deallocate model weights or the preallocated KV buffer and therefore does not necessarily reduce `nvidia-smi` memory usage. It makes the cache positions/data reusable while keeping the model hot.

`POST /runtime/cache/clear` clears the global active sequence after earlier queued inference completes. It is not a per-client operation. This runtime version has one shared sequence, not one cache allocation per agent.

Model resources are explicitly closed during service shutdown. Remote model load/unload endpoints are intentionally not exposed; safe live unloading needs request-admission control and should be added separately if required.

## Configuration

Important environment settings include:

- `ARC_RUNTIME_MODEL_PATH`
- `ARC_RUNTIME_N_CTX`
- `ARC_RUNTIME_MAX_OUTPUT_TOKENS`
- `ARC_RUNTIME_MAX_TOOL_CALL_TOKENS` (default `4096`; separate generation ceiling for requests with tools)
- `ARC_RUNTIME_CONTEXT_SAFETY_MARGIN`
- `ARC_RUNTIME_N_GPU_LAYERS`
- `ARC_RUNTIME_KV_CACHE_TYPE_K`
- `ARC_RUNTIME_KV_CACHE_TYPE_V`
- `ARC_RUNTIME_GEN_TIMEOUT_S`

The embedding model remains independently configurable and lazy-loaded. It is not loaded unless configured and requested (unless the explicit chat-model fallback is enabled).

## Run

Install with the repository's `uv` project and start it through the ARC service runner or the configured service entry point. The service binds to `127.0.0.1` by default.

The runtime API currently has no authentication or per-client authorization. Keep it bound to localhost unless an authenticated gateway is placed in front of it.


## 0.3.1 serialization fix

ARC `Constant` configuration objects are now converted to native Python values
when `EngineConfig` is created. Runtime status and memory responses also
normalize KV cache type labels to native strings before returning JSON. This
prevents Starlette JSON serialization errors for wrapped string configuration
values such as `ARC_RUNTIME_KV_CACHE_TYPE_K` and
`ARC_RUNTIME_KV_CACHE_TYPE_V`.


## 0.3.3 native tool-call streaming and generation budgets

Qwen-family XML function calls are parsed incrementally. Large string arguments
are JSON-escaped and emitted as argument fragments while the model is generating,
rather than buffering a complete `<parameter>` value until its closing tag.
Non-string arguments are decoded using the declared JSON Schema type and emitted
when the parameter closes. Function and parameter delimiters are carried by
`ModelProtocol`, allowing compatible template variants to configure their own
markers instead of relying on parser-wide fixed strings.

Tool-enabled chat requests use an independent output budget. The server defaults
`ARC_RUNTIME_MAX_TOOL_CALL_TOKENS` to `4096`. Clients may set
`max_tool_call_tokens` on a chat request to override it. Without that override,
the effective tool budget is the larger of the ordinary `max_output_tokens` and
the server tool-call budget. The effective value is used consistently by context
fitting, `/context/inspect`, and llama.cpp generation. A request is rejected or
history is pruned before generation if that budget cannot fit the context.
Ordinary chat and completion requests continue to use `max_output_tokens`.

A tool call is executable only after its outer native closing marker has been
observed. Truncated calls finish with `finish_reason="length"` and a
`tool_call_incomplete` warning. Non-streaming responses discard partial calls;
streaming clients must likewise require `finish_reason="tool_calls"` before
executing accumulated tool-call deltas.

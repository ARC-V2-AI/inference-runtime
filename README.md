# ARC-V2 Inference Runtime

A local-first inference service for ARC. The runtime provides a model-backed HTTP API and manages model loading, prompt rendering, tokenization, context limits, inference execution, and model resources.

It is designed to run as an ARC-managed service. Applications and other ARC services communicate with it through the HTTP API, using the inference client SDK or any compatible HTTP client.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Starting the Runtime](#starting-the-runtime)
- [Accessing the API](#accessing-the-api)
- [Using the Inference Client](#using-the-inference-client)
- [API Reference](#api-reference)
- [Context and Token Management](#context-and-token-management)
- [Streaming and Tool Calls](#streaming-and-tool-calls)
- [Configuration](#configuration)
  - [Environment File](#environment-file)
  - [Model Configuration](#model-configuration)
  - [Performance and Context](#performance-and-context)
  - [Generation and Sampling](#generation-and-sampling)
  - [Embedding Configuration](#embedding-configuration)
  - [Debugging](#debugging)
- [Runtime Resource Management](#runtime-resource-management)
- [Security](#security)

## Overview

The inference runtime provides a shared local inference backend for ARC and external applications.

Its responsibilities include:

- Loading and maintaining the active GGUF model.
- Rendering chat prompts using the model's native chat template.
- Tokenization and exact prompt-token accounting.
- Context-window validation and overflow handling.
- Serialized inference execution.
- Streaming text, reasoning, and native tool-call events.
- Model and runtime resource monitoring.
- Optional dedicated embedding-model support.

The runtime exposes an OpenAI-compatible HTTP interface with additional endpoints for model metadata, tokenization, context inspection, and runtime monitoring.

## Architecture

The runtime is designed around one active main model instance.

- A single model remains loaded for the lifetime of the service.
- Chat and completion requests are stateless. Clients submit the messages or prompt required for each request.
- A serialized request queue prevents concurrent operations from interfering with the shared model context.
- KV sequence contents are cleared after completed, cancelled, or failed inference requests. The allocated KV buffer remains loaded and is reused.
- Conversation history, agent state, orchestration, and persistent memory belong to ARC or the consuming application.
- Embedding models use a separate, independently configured model context and are loaded lazily.

The runtime does not implement agent orchestration or long-term memory.

## Installation

Install the repository through the ARC CLI. This installs the runtime into the ARC environment and registers it as an ARC service.

```bash
arc install https://github.com/ARC-V2-AI/inference-runtime
```

Alternatively, invoke the ARC CLI through `uv`:

```bash
uv run arc install https://github.com/ARC-V2-AI/inference-runtime
```

The runtime declares its service metadata in `pyproject.toml`. ARC uses this metadata to register and manage the service.

You do not need to install the runtime separately into every application that uses it. Applications communicate with the running service through its HTTP API.

## Starting the Runtime

Start ARC normally:

```bash
arc
```

Or, when running ARC through `uv`:

```bash
uv run arc
```

During startup, ARC builds the service startup tree and starts the registered inference runtime.

The runtime then:

1. Reads its configuration from the ARC environment.
2. Validates the configured model path and runtime settings.
3. Initializes the inference engine.
4. Loads the configured model.
5. Starts the HTTP server.
6. Reports readiness to the ARC service manager.

The default API address is:

```text
http://127.0.0.1:7842
```

The runtime binds to localhost by default.

## Accessing the API

Once the service is running, open the interactive API documentation:

**[http://127.0.0.1:7842/docs](http://127.0.0.1:7842/docs)**

The FastAPI documentation provides an interactive interface for inspecting endpoints, schemas, request parameters, and responses.

The OpenAPI schema is available at:

```text
http://127.0.0.1:7842/openapi.json
```

The versioned API root is:

```text
http://127.0.0.1:7842/v1
```

You can use the API directly through HTTP or connect through the ARC inference client.

## Using the Inference Client

The recommended way to consume the runtime from Python applications is the separate [ARC-V2 Inference Client](https://github.com/ARC-V2-AI/inference-client).

Install it in the environment of the application that will use inference:

```bash
uv add git+https://github.com/ARC-V2-AI/inference-client.git
```

Example:

```python
from arc_v2_inference_client import InferenceClient

with InferenceClient("http://127.0.0.1:7842") as client:
    response = client.chat.create(
        [
            {
                "role": "system",
                "content": "You are a concise assistant.",
            },
            {
                "role": "user",
                "content": "Explain how the ARC inference runtime works.",
            },
        ],
        max_output_tokens=512,
    )

    print(response.text)
```

The client can also access tokenization, context inspection, streaming, model metadata, embeddings, and tool-agent functionality.

For asynchronous applications, use `AsyncInferenceClient`.

The client and runtime are separate components: the runtime provides inference resources and the HTTP API, while the client provides application-facing Python interfaces and local tool execution.

## API Reference

All inference endpoints are available under `/v1`.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/chat/completions` | Chat inference, streaming, and tool calls |
| POST | `/completions` | Text completion |
| POST | `/embeddings` | Generate embeddings |
| POST | `/tokenize` | Token counts and optional token IDs |
| POST | `/context/inspect` | Inspect the rendered prompt and context budget |
| GET | `/model` | Active model metadata and capabilities |
| GET | `/models` | Model listing |
| GET | `/runtime/status` | Runtime, queue, and KV status |
| GET | `/runtime/memory` | Process RAM and NVIDIA GPU memory |
| POST | `/runtime/cache/clear` | Clear the active KV sequence contents |

`POST /chat/context` is retained as a hidden-schema compatibility alias for `/context/inspect`.

### Direct HTTP request

```bash
curl http://127.0.0.1:7842/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Explain local inference."
      }
    ],
    "max_output_tokens": 512,
    "stream": false
  }'
```

Responses include token usage:

```json
{
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 52,
    "total_tokens": 76
  }
}
```

For streamed responses, usage is attached to the final chunk.

## Context and Token Management

The runtime performs token accounting using the loaded model's tokenizer and actual chat template.

It does not estimate prompt size from character length.

Chat and completion requests support context controls:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Explain KV caching."
    }
  ],
  "max_output_tokens": 512,
  "context_limit": 4096,
  "context_overflow": "truncate"
}
```

- `context_limit` can reduce the context budget for a request but cannot exceed the configured server context window.
- `context_overflow="truncate"` removes the oldest complete user turns while preserving system messages and the newest turn.
- `context_overflow="reject"` returns HTTP 413 rather than silently dropping conversation history.
- Plain completion prompts are rejected when they exceed the available context.

The runtime reserves space for generation and a configurable safety margin before accepting a request.

Tool-enabled requests have an independent generation budget configured through `ARC_RUNTIME_MAX_TOOL_CALL_TOKENS`.

Use `/v1/context/inspect` to inspect the rendered prompt, token count, available context, and whether a request fits before generating a response.

## Streaming and Tool Calls

The runtime supports streaming text, reasoning fragments, and native tool-call events.

Qwen-family XML tool calls are parsed incrementally. Large string arguments are emitted as fragments rather than being held until the complete parameter value has arrived.

Function and parameter delimiters are defined through the model protocol, allowing compatible model-template variants to use their own markers.

Tool-call generation uses a separate token budget so large arguments, such as source files, are less likely to be truncated by the ordinary assistant output limit.

A streamed tool call is not executable until its complete outer tool-call closing marker has been received. Incomplete calls finish with `finish_reason="length"` and include a `tool_call_incomplete` warning.

Clients must only execute accumulated tool calls after the stream confirms `finish_reason="tool_calls"`. Partial tool-call deltas must not be executed.

## Configuration

The runtime reads its environment variables from ARC's environment configuration.

The current ARC environment file is:

```text
~/arc/.env
```

Settings can also be provided through exported shell environment variables before starting ARC.

### Environment File

Create or edit:

```bash
nano ~/arc/.env
```

Example configuration:

```dotenv
# Model
ARC_RUNTIME_MODEL_PATH=/absolute/path/to/model.gguf

# Model execution
ARC_RUNTIME_N_CTX=8192
ARC_RUNTIME_N_GPU_LAYERS=auto
ARC_RUNTIME_N_BATCH=256
ARC_RUNTIME_N_THREADS=0

# KV cache and performance
ARC_RUNTIME_FLASH_ATTN=true
ARC_RUNTIME_KV_CACHE_TYPE_K=f16
ARC_RUNTIME_KV_CACHE_TYPE_V=f16

# Generation budgets
ARC_RUNTIME_MAX_OUTPUT_TOKENS=3072
ARC_RUNTIME_MAX_TOOL_CALL_TOKENS=4096
ARC_RUNTIME_CONTEXT_SAFETY_MARGIN=16

# Runtime
ARC_RUNTIME_PORT=7842
ARC_RUNTIME_GEN_TIMEOUT_S=300
```

Alternatively, export settings in the shell:

```bash
export ARC_RUNTIME_MODEL_PATH="$HOME/models/model.gguf"
export ARC_RUNTIME_N_CTX=8192
export ARC_RUNTIME_N_GPU_LAYERS=auto

arc
```

Environment variables must be set before ARC starts the service. Restart ARC after changing its environment file or exported settings.

### Model Configuration

| Variable | Default | Description |
|---|---:|---|
| `ARC_RUNTIME_MODEL_PATH` | `~/arc/models/...` | Path to the main GGUF model |
| `ARC_RUNTIME_N_GPU_LAYERS` | `auto` | Number of model layers offloaded to the GPU; `auto` estimates a suitable value |
| `ARC_RUNTIME_N_CTX` | `8192` | Model context window |
| `ARC_RUNTIME_N_BATCH` | `256` | Prompt-processing batch size |
| `ARC_RUNTIME_N_THREADS` | `0` | CPU threads; `0` lets llama.cpp decide |

The configured model path must point to an existing GGUF file.

### Performance and Context

| Variable | Default | Description |
|---|---:|---|
| `ARC_RUNTIME_FLASH_ATTN` | `true` | Enable Flash Attention when supported by the backend |
| `ARC_RUNTIME_KV_CACHE_TYPE_K` | `f16` | KV cache key format: `f16`, `q8_0`, or `q4_0` |
| `ARC_RUNTIME_KV_CACHE_TYPE_V` | `f16` | KV cache value format: `f16`, `q8_0`, or `q4_0` |
| `ARC_RUNTIME_GEN_TIMEOUT_S` | `300` | Maximum generation operation timeout in seconds |
| `ARC_RUNTIME_CONTEXT_SAFETY_MARGIN` | `16` | Reserved safety margin in context tokens |
| `ARC_RUNTIME_MAX_OUTPUT_TOKENS` | `3072` | Default generation budget for ordinary output |
| `ARC_RUNTIME_MAX_TOOL_CALL_TOKENS` | `4096` | Default generation budget for tool-enabled requests |

The `auto` GPU-layer setting estimates offloading based on available VRAM and model information. Set an integer to use a fixed number of layers.

Quantized KV cache formats can reduce memory consumption, but they may affect inference performance or numerical fidelity. Use formats supported by the installed llama.cpp backend.

### Generation and Sampling

| Variable | Default | Description |
|---|---:|---|
| `ARC_RUNTIME_TOOL_PROTOCOL` | `auto` | Tool protocol detection or selection |
| `ARC_RUNTIME_GRAMMAR_CONSTRAINED_TOOLS` | `false` | Constrain supported native tool-call output with a grammar |
| `ARC_RUNTIME_STOP_SEQUENCES` | empty | Comma-separated stop sequences |
| `ARC_RUNTIME_SAMPLING_TEMPERATURE` | `0.7` | Default sampling temperature |
| `ARC_RUNTIME_SAMPLING_TOP_P` | `0.8` | Default nucleus sampling value |
| `ARC_RUNTIME_SAMPLING_TOP_K` | `20` | Default top-k sampling value |
| `ARC_RUNTIME_SAMPLING_MIN_P` | `0.0` | Default minimum probability sampling value |
| `ARC_RUNTIME_SAMPLING_REPEAT_PENALTY` | `1.0` | Repetition penalty |
| `ARC_RUNTIME_SAMPLING_TEMPERATURE_THINKING` | `0.6` | Temperature used in thinking mode |
| `ARC_RUNTIME_SAMPLING_TOP_P_THINKING` | `0.95` | Top-p used in thinking mode |
| `ARC_RUNTIME_SAMPLING_TOP_K_THINKING` | `20` | Top-k used in thinking mode |
| `ARC_RUNTIME_SAMPLING_MIN_P_THINKING` | `0.0` | Min-p used in thinking mode |

Request-level sampling parameters can override the runtime defaults where supported by the API.

### Embedding Configuration

Embeddings use a dedicated model configuration. The embedding model is loaded lazily when the first embedding request is received.

| Variable | Default | Description |
|---|---:|---|
| `ARC_RUNTIME_EMBEDDING_MODEL_PATH` | empty | Dedicated embedding GGUF path; disabled when empty |
| `ARC_RUNTIME_EMBEDDING_POOLING` | `last` | Pooling strategy: `unspecified`, `mean`, `cls`, or `last` |
| `ARC_RUNTIME_EMBEDDING_N_CTX` | `8192` | Embedding model context window |
| `ARC_RUNTIME_EMBEDDING_N_BATCH` | `2048` | Embedding batch size |
| `ARC_RUNTIME_EMBEDDING_N_GPU_LAYERS` | `auto` | Embedding model GPU layers |
| `ARC_RUNTIME_EMBEDDING_FORMAT` | `qwen3` | Input formatting: `qwen3` or `plain` |
| `ARC_RUNTIME_EMBEDDING_NORMALIZE` | `true` | Normalize returned embedding vectors |
| `ARC_RUNTIME_EMBEDDING_FALLBACK_TO_CHAT` | `false` | Explicitly allow a second copy of the chat model to be used for embeddings |

The recommended configuration is to use a dedicated embedding model instead of loading another copy of the chat model.

For Qwen3 embedding models, use `ARC_RUNTIME_EMBEDDING_POOLING=last`.

When no embedding model is configured, embedding requests return an error rather than silently using the chat model.

### Debugging

| Variable | Default | Description |
|---|---:|---|
| `ARC_RUNTIME_DEBUG` | `true` | Enable runtime and parser debug tracing |
| `ARC_RUNTIME_VERBOSE` | `false` | Enable verbose llama.cpp logging |

Debug tracing can expose detailed prompt and parser information. Disable it during normal operation unless troubleshooting requires it.

## Runtime Resource Management

The runtime maintains one shared main model context and serializes operations against it.

### KV cache behavior

After each chat or completion request, the active KV sequence contents are cleared.

This prevents a later request from reusing another request's prompt state. It does not unload model weights or deallocate the preallocated KV buffer. Consequently, GPU memory usage reported by `nvidia-smi` may remain high even after a request completes.

`POST /v1/runtime/cache/clear` clears the active global sequence after earlier queued inference operations have completed. It is a global runtime operation, not a per-client or per-agent cache.

### Model lifecycle

Model loading and unloading are managed by the ARC service lifecycle.

The runtime does not expose remote model load or unload endpoints. Model resources are released during service shutdown.

## Security

The runtime currently exposes no authentication or per-client authorization.

It binds to `127.0.0.1` by default and is intended for local use. Do not expose the API directly to an external network. Use an authenticated gateway and suitable access controls if remote access is required.

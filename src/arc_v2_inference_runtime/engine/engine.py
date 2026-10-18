"""ARC inference engine.

The engine is intentionally thin:

    request -> model renderer -> llama.cpp -> protocol parser -> GenerationChunk

Model metadata, chat-template detection and native protocol selection belong to
``Model``.  The engine owns request orchestration, streaming, sampling,
cancellation and queueing.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import threading
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from llama_cpp import (
    LLAMA_POOLING_TYPE_CLS,
    LLAMA_POOLING_TYPE_LAST,
    LLAMA_POOLING_TYPE_MEAN,
    LLAMA_POOLING_TYPE_UNSPECIFIED,
    Llama,
)

try:
    from llama_cpp import LlamaGrammar
except ImportError:  # pragma: no cover
    LlamaGrammar = None  # type: ignore[assignment,misc]

from arc_v2_inference_runtime.engine.agentic_parsing import (
    AgenticStreamParser,
    ToolCallFragment,
)
from arc_v2_inference_runtime.engine.parsing import ParsedEvent, StreamParser
from arc_v2_inference_runtime.engine.config import EngineConfig, _auto_gpu_layers
from arc_v2_inference_runtime.engine.context_manager import (
    ContextFit,
    ContextLengthError,
    fit_chat_context,
    validate_prompt_fits,
)
from arc_v2_inference_runtime.engine.conversations import ConversationStore
from arc_v2_inference_runtime.engine.model import Model
from arc_v2_inference_runtime.engine.model_types import (
    ModelProtocol,
    ReasoningFormat,
    ToolCallFormat,
)
from arc_v2_inference_runtime.engine.grammar import build_tool_call_grammar
from arc_v2_inference_runtime.engine.queue import RequestQueue
from arc_v2_inference_runtime.types import (
    ChatRequest,
    CompletionRequest,
    NamedToolChoice,
)

logger = logging.getLogger(__name__)


class _EmbeddingLlama(Llama):
    """Llama wrapper that can parse special markers in embedding text inputs.

    llama-cpp-python's embedding helper calls `tokenize()` without a `special`
    argument. Qwen3 embedding inputs need the literal end-of-text marker parsed
    as a real token, so only the Qwen3-configured embedding instance enables it.
    """

    def __init__(self, *args: Any, parse_special_tokens: bool = False, **kwargs: Any) -> None:
        self._arc_parse_special_tokens = parse_special_tokens
        super().__init__(*args, **kwargs)

    def tokenize(
        self, text: bytes, add_bos: bool = True, special: bool = False
    ) -> list[int]:
        return super().tokenize(
            text,
            add_bos=add_bos,
            special=special or self._arc_parse_special_tokens,
        )


_POOLING_TYPES = {
    "unspecified": LLAMA_POOLING_TYPE_UNSPECIFIED,
    "mean": LLAMA_POOLING_TYPE_MEAN,
    "cls": LLAMA_POOLING_TYPE_CLS,
    "last": LLAMA_POOLING_TYPE_LAST,
}


@dataclass(slots=True)
class GenerationChunk:
    """Normalized streaming unit exposed to the API layer."""

    event: ParsedEvent | None = None
    tool_call: ToolCallFragment | None = None
    finish_reason: str | None = None
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class _ChatGeneration:
    """All derived state needed for one chat generation."""

    prompt: str
    include_tools: bool
    force_tool_call: bool
    named_tool: str | None
    thinking_enabled: bool
    initial_mode: Literal["auto", "reasoning", "content"]
    stop: tuple[str, ...]
    sampling: dict[str, Any]


class Engine:
    """Owns the loaded model and coordinates inference."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._model: Model | None = None
        self._queue = RequestQueue()
        self._embedding_model: Llama | None = None
        self._embedding_model_name: str | None = None
        self._embedding_lock = asyncio.Lock()
        self._conversations = ConversationStore(config.conversation_db_path)

        if config.debug_trace:
            logger.setLevel(logging.DEBUG)
            logging.getLogger("arc.services.runtime.engine.agentic_parsing").setLevel(
                logging.DEBUG
            )

        if config.grammar_constrained_tools and LlamaGrammar is None:
            logger.warning(
                "ARC_RUNTIME_GRAMMAR_CONSTRAINED_TOOLS is enabled, but this "
                "llama-cpp-python build does not export LlamaGrammar; "
                "continuing without grammar constraints."
            )

    # ------------------------------------------------------------------
    # Public model state
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        if self._model is None:
            return Path(self.config.model_path).name
        try:
            return self._model.info.name
        except Exception:
            return Path(self.config.model_path).name

    @property
    def protocol(self) -> ModelProtocol | None:
        if self._model is None or not self._model.loaded:
            return None
        return self._model.protocol

    @property
    def model(self) -> Model:
        if self._model is None or not self._model.loaded:
            raise RuntimeError("Model is not loaded")
        return self._model

    @property
    def conversations(self) -> ConversationStore:
        return self._conversations

    @property
    def embedding_model_name(self) -> str:
        if self._embedding_model_name:
            return self._embedding_model_name
        if self.config.embedding_model_path is not None:
            return self.config.embedding_model_path.name
        if self.config.embedding_fallback_to_chat:
            return Path(self.config.model_path).name
        return "not-configured"

    def tokenize(
        self, text: str, *, add_bos: bool = False, special: bool = True
    ) -> list[int]:
        """Tokenize with the loaded generation model's own GGUF vocabulary."""
        return self.model.instance.tokenize(
            text.encode("utf-8"), add_bos=add_bos, special=special
        )

    def count_tokens(self, text: str, *, add_bos: bool = False, special: bool = True) -> int:
        return len(self.tokenize(text, add_bos=add_bos, special=special))

    def count_chat_context(self, req: ChatRequest) -> dict[str, int | bool]:
        """Count the full rendered native-template prompt without pruning it.

        Explicitly convert numeric settings to Python built-in ``int`` values.
        ARC configuration constants may implement integer-like behavior without
        being JSON-serializable by Starlette's standard JSON encoder.
        """
        context = self._prepare_chat(req)
        prompt_tokens = int(
            self.count_tokens(context.prompt, add_bos=True, special=True)
        )
        output_tokens = int(req.max_output_tokens or self.config.max_output_tokens)
        safety_margin = int(self.config.context_safety_margin_tokens)
        context_window = int(self.config.n_ctx)
        fits = bool(
            prompt_tokens + output_tokens + safety_margin <= context_window
        )
        return {
            "prompt_tokens": prompt_tokens,
            "max_output_tokens": output_tokens,
            "safety_margin_tokens": safety_margin,
            "context_window": context_window,
            "remaining_tokens": max(
                0,
                context_window - prompt_tokens - output_tokens - safety_margin,
            ),
            "available_output_tokens": max(
                0,
                context_window - prompt_tokens - safety_margin,
            ),
            "fits": fits,
        }

    def fit_chat_request(self, req: ChatRequest) -> ChatRequest:
        """Return a request that fits, pruning only older complete turns."""
        output_tokens = req.max_output_tokens or self.config.max_output_tokens
        result: ContextFit = fit_chat_context(
            req.messages,
            render=lambda messages: self._prepare_chat(
                req.model_copy(update={"messages": messages, "max_output_tokens": output_tokens})
            ).prompt,
            count_tokens=lambda prompt: self.count_tokens(
                prompt, add_bos=True, special=True
            ),
            context_window=self.config.n_ctx,
            output_tokens=output_tokens,
            safety_margin_tokens=self.config.context_safety_margin_tokens,
        )
        if result.dropped_turns:
            logger.info(
                "Context budget dropped %d oldest turn(s): prompt_tokens=%d output_tokens=%d context=%d",
                result.dropped_turns,
                result.prompt_tokens,
                result.output_tokens,
                result.context_window,
            )
        return req.model_copy(
            update={"messages": result.messages, "max_output_tokens": output_tokens}
        )

    def fit_completion_request(self, req: CompletionRequest) -> CompletionRequest:
        output_tokens = req.max_output_tokens or self.config.max_output_tokens
        prompt_tokens = self.count_tokens(req.prompt, add_bos=True, special=True)
        validate_prompt_fits(
            prompt_tokens,
            context_window=self.config.n_ctx,
            output_tokens=output_tokens,
            safety_margin_tokens=self.config.context_safety_margin_tokens,
        )
        return req.model_copy(update={"max_output_tokens": output_tokens})

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._model is not None:
            raise RuntimeError("Engine is already started")

        model = Model(self.config)
        await model.load()

        self._model = model
        self._queue.start()

        logger.info(
            "Inference engine ready: model=%s protocol=%s",
            self.model_name,
            model.protocol.tool_call_format,
        )

    async def stop(self) -> None:
        await self._queue.stop()

        if self._model is not None:
            self._model.unload()
            self._model = None

        if self._embedding_model is not None:
            close = getattr(self._embedding_model, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("Embedding model close failed", exc_info=True)
        self._embedding_model = None
        self._embedding_model_name = None

    async def unload(self) -> None:
        """Compatibility alias for callers using the old engine lifecycle."""

        await self.stop()

    # ------------------------------------------------------------------
    # Chat preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _tool_choice_kind(req: ChatRequest) -> str | None:
        choice = req.tool_choice

        if choice is None:
            return "auto" if req.tools else None

        if isinstance(choice, NamedToolChoice):
            return "named"

        return choice

    @staticmethod
    def _named_tool(req: ChatRequest) -> str | None:
        if isinstance(req.tool_choice, NamedToolChoice):
            return req.tool_choice.function.name
        return None

    def _initial_parser_mode(
        self,
        prompt: str,
        protocol: ModelProtocol,
        reasoning_effort: str,
    ) -> Literal["auto", "reasoning", "content"]:
        """Determine only the parser's starting state.

        ``auto`` means the model/template's native behavior.  Explicit
        ``none`` always starts in content mode.  When a template has already
        opened a reasoning section in the rendered assistant prefix, the
        parser must enter that section because its opening token will not be
        present in the streamed completion.
        """

        if reasoning_effort == "none":
            return "content"

        if protocol.reasoning_format == ReasoningFormat.THINK_TAG:
            open_tag = protocol.think_open
            close_tag = protocol.think_close
            if open_tag and open_tag in prompt:
                if prompt.rfind(open_tag) > prompt.rfind(close_tag or ""):
                    return "reasoning"

        if protocol.reasoning_format == ReasoningFormat.HARMONY_CHANNEL:
            analysis_marker = "<|channel|>analysis<|message|>"
            if analysis_marker in prompt:
                if prompt.rfind(analysis_marker) > prompt.rfind("<|end|>"):
                    return "reasoning"

        return "auto"

    def _thinking_enabled(
        self,
        req: ChatRequest,
        initial_mode: str,
    ) -> bool:
        """Resolve sampler mode without making template parsing the policy.

        Explicit request values are authoritative.  ``auto`` follows the
        model-native rendered state when the template opened reasoning.
        """

        if req.reasoning_effort == "none":
            return False
        if req.reasoning_effort == "on":
            return True
        return initial_mode == "reasoning"

    def _render_chat(
        self,
        req: ChatRequest,
        include_tools: bool,
    ) -> str:
        """Render through the model-owned native chat template.

        The renderer API is deliberately kept as the source of truth.  New
        renderers may accept template kwargs; older renderers need no special
        handling here.
        """

        renderer = self.model.renderer
        kwargs: dict[str, Any] = {
            "messages": req.messages,
            "tools": req.tools if include_tools else None,
            "add_generation_prompt": True,
        }

        # Some renderer revisions expose template_kwargs.  Use it when
        # available without making the Engine depend on a particular renderer
        # implementation.
        try:
            parameters = inspect.signature(renderer.render).parameters
        except (TypeError, ValueError):
            parameters = {}

        if "template_kwargs" in parameters:
            template_kwargs: dict[str, Any] = {}
            if req.reasoning_effort == "none":
                template_kwargs["enable_thinking"] = False
            elif req.reasoning_effort == "on":
                template_kwargs["enable_thinking"] = True
            kwargs["template_kwargs"] = template_kwargs

        return renderer.render(**kwargs)

    def _prepare_chat(self, req: ChatRequest) -> _ChatGeneration:
        protocol = self.model.protocol
        choice_kind = self._tool_choice_kind(req)
        named_tool = self._named_tool(req)

        include_tools = bool(req.tools) and choice_kind != "none"
        force_tool_call = include_tools and choice_kind in {"required", "named"}

        prompt = self._render_chat(req, include_tools)

        # Do not blindly append <tool_call>. Native templates may place the
        # assistant turn inside a reasoning channel or use a different prefix.
        # A natural-language forcing instruction is only a fallback for the
        # selection policy; the parser still receives the real force flag.
        if force_tool_call:
            if named_tool:
                prompt += f"\nUse the function {named_tool} now."
            else:
                prompt += "\nCall one of the available tools now."

        initial_mode = self._initial_parser_mode(
            prompt,
            protocol,
            req.reasoning_effort,
        )
        thinking_enabled = self._thinking_enabled(req, initial_mode)

        stops = list(
            dict.fromkeys(
                [
                    *self.config.stop_sequences,
                    *req.stop,
                ]
            )
        )

        sampling = self._sampling_kwargs(req, thinking_enabled)

        return _ChatGeneration(
            prompt=prompt,
            include_tools=include_tools,
            force_tool_call=force_tool_call,
            named_tool=named_tool,
            thinking_enabled=thinking_enabled,
            initial_mode=initial_mode,
            stop=tuple(stops),
            sampling=sampling,
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sampling_kwargs(
        self,
        req: ChatRequest | CompletionRequest,
        thinking_enabled: bool,
    ) -> dict[str, Any]:
        if thinking_enabled:
            temperature = self.config.sampling_temperature_thinking
            top_p = self.config.sampling_top_p_thinking
            top_k = self.config.sampling_top_k_thinking
            min_p = self.config.sampling_min_p_thinking
        else:
            temperature = self.config.sampling_temperature
            top_p = self.config.sampling_top_p
            top_k = self.config.sampling_top_k
            min_p = self.config.sampling_min_p

        result: dict[str, Any] = {
            "stream": True,
            "temperature": (
                req.temperature if req.temperature is not None else temperature
            ),
            "top_p": req.top_p if req.top_p is not None else top_p,
            "top_k": req.top_k if req.top_k is not None else top_k,
            "min_p": req.min_p if req.min_p is not None else min_p,
            "repeat_penalty": self.config.sampling_repeat_penalty,
        }

        if req.max_output_tokens is not None:
            result["max_tokens"] = req.max_output_tokens

        return result

    # ------------------------------------------------------------------
    # Grammar
    # ------------------------------------------------------------------

    def _build_grammar(
        self,
        req: ChatRequest,
        context: _ChatGeneration,
    ) -> Any | None:
        if (
            not context.include_tools
            or not self.config.grammar_constrained_tools
            or LlamaGrammar is None
        ):
            return None

        protocol = self.model.protocol

        if protocol.tool_call_format not in {
            ToolCallFormat.QWEN_XML,
            ToolCallFormat.TAGGED_JSON,
        }:
            return None

        grammar_text = build_tool_call_grammar(
            protocol,
            req.tools,
            needs_reasoning_close=context.initial_mode == "reasoning",
            think_close=protocol.think_close or "</think>",
        )

        if not grammar_text:
            return None

        try:
            return LlamaGrammar.from_string(grammar_text, verbose=False)
        except Exception:
            logger.exception("Failed to compile tool-call grammar")
            return None

    # ------------------------------------------------------------------
    # Chat generation
    # ------------------------------------------------------------------

    async def generate_chat(
        self,
        req: ChatRequest,
    ) -> AsyncIterator[GenerationChunk]:
        req = self.fit_chat_request(req)
        context = self._prepare_chat(req)
        # Pass the exact token IDs used for budgeting. This avoids a second
        # tokenizer pass and ensures Qwen's native template markers are treated
        # as special tokens consistently in counting and generation.
        prompt_token_ids = self.tokenize(context.prompt, add_bos=True, special=True)
        prompt_token_count = len(prompt_token_ids)

        protocol = self.model.protocol
        tool_schemas = {
            tool.function.name: (tool.function.parameters or {}).get("properties", {})
            for tool in req.tools
        }

        parser = AgenticStreamParser(
            protocol=protocol,
            tool_schemas=tool_schemas,
            force_tool_call=context.force_tool_call,
            initial_mode=context.initial_mode,
        )

        sampling = dict(context.sampling)
        grammar = self._build_grammar(req, context)
        if grammar is not None:
            sampling["grammar"] = grammar

        logger.debug(
            "Chat generation: tools=%s choice=%s named=%s "
            "reasoning=%s initial_mode=%s grammar=%s prompt_tokens=%d output_tokens=%d",
            context.include_tools,
            self._tool_choice_kind(req),
            context.named_tool,
            context.thinking_enabled,
            context.initial_mode,
            grammar is not None,
            prompt_token_count,
            req.max_output_tokens,
        )

        def _run(cancel_event: threading.Event) -> Iterator[GenerationChunk]:
            last_finish: str | None = None
            raw_parts: list[str] = []

            try:
                for output in self.model.instance(
                    prompt_token_ids,  # type: ignore[arg-type] -- llama.cpp accepts pre-tokenized prompts
                    stop=list(context.stop),
                    **sampling,
                ):
                    if cancel_event.is_set():
                        return

                    choice = output["choices"][0]  # pyright: ignore[reportArgumentType]
                    text = choice.get("text", "")  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
                    finish = choice.get("finish_reason")  # pyright: ignore[reportAttributeAccessIssue]

                    if text:
                        raw_parts.append(text)

                    if finish:
                        last_finish = finish

                    for event in parser.feed(text):
                        if isinstance(event, ToolCallFragment):
                            yield GenerationChunk(tool_call=event)
                        else:
                            yield GenerationChunk(event=event)

                    # A single inference turn yields at most one completed
                    # tool call. `parser.saw_tool_call` only flips once the
                    # parser has consumed the native closing marker for that
                    # call (see AgenticStreamParser._step_tool_tail) -- not
                    # on the first header fragment (name-only, no arguments
                    # yet). Breaking on the first fragment cut generation off
                    # mid tool-call, right after `<function=NAME>`, before
                    # any parameters or the closing tags were ever produced.
                    if parser.saw_tool_call:
                        break

                for event in parser.finalize():
                    if isinstance(event, ToolCallFragment):
                        yield GenerationChunk(tool_call=event)
                    else:
                        yield GenerationChunk(event=event)

                warning = None
                if parser.incomplete_tool_call_mode:
                    warning = (
                        "tool_call_incomplete: parser stopped in "
                        f"{parser.incomplete_tool_call_mode!r}"
                    )
                    logger.warning(
                        "%s model=%s stop=%s",
                        warning,
                        self.model_name,
                        context.stop,
                    )

                yield GenerationChunk(
                    event=ParsedEvent(kind="content", text=""),
                    finish_reason=(
                        "tool_calls"
                        if parser.saw_tool_call
                        else (last_finish or "stop")
                    ),
                    warning=warning,
                )

                logger.debug("Raw generation: %s", "".join(raw_parts))

            except Exception:
                logger.exception("Chat generation failed")
                raise

        async for chunk in self._queue.submit(
            _run,
            timeout_s=self.config.generation_timeout_s,
        ):
            yield chunk

    # ------------------------------------------------------------------
    # Text completion
    # ------------------------------------------------------------------

    async def generate_completion(
        self,
        req: CompletionRequest,
    ) -> AsyncIterator[GenerationChunk]:
        req = self.fit_completion_request(req)
        prompt_token_ids = self.tokenize(req.prompt, add_bos=True, special=True)
        sampling = self._sampling_kwargs(req, thinking_enabled=False)

        parser = StreamParser(
            think_open="<think>",
            think_close="</think>",
        )

        def _run(cancel_event: threading.Event) -> Iterator[GenerationChunk]:
            last_finish: str | None = None

            for output in self.model.instance(
                prompt_token_ids,  # type: ignore[arg-type] -- llama.cpp accepts pre-tokenized prompts
                stop=list(dict.fromkeys([*self.config.stop_sequences, *req.stop])),
                **sampling,
            ):
                if cancel_event.is_set():
                    return

                choice = output["choices"][0]  # pyright: ignore[reportArgumentType]
                text = choice.get("text", "")
                finish = choice.get("finish_reason")

                if finish:
                    last_finish = finish

                for event in parser.feed(text):
                    yield GenerationChunk(event=event)

            for event in parser.finalize():
                yield GenerationChunk(event=event)

            yield GenerationChunk(finish_reason=last_finish or "stop")

        async for chunk in self._queue.submit(
            _run,
            timeout_s=self.config.generation_timeout_s,
        ):
            yield chunk

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def embed(
        self,
        inputs: list[str],
        *,
        input_type: str = "raw",
        instruction: str | None = None,
    ) -> list[list[float]]:
        if not inputs:
            return []
        model = await self._ensure_embedding_model()
        formatted_inputs = [
            self._format_embedding_input(text, input_type=input_type, instruction=instruction)
            for text in inputs
        ]

        # Fail explicitly instead of letting llama.cpp silently truncate an
        # embedding input. Its context size is independently configured.
        for index, text in enumerate(formatted_inputs):
            token_count = len(
                model.tokenize(
                    text.encode("utf-8"),
                    add_bos=True,
                    special=self.config.embedding_format == "qwen3",
                )
            )
            max_input_tokens = min(self.config.embedding_n_ctx, model.n_batch)
            if token_count > max_input_tokens:
                raise ValueError(
                    f"Embedding input {index} has {token_count} tokens, exceeding "
                    f"the safe embedding input limit of {max_input_tokens} tokens "
                    f"(n_ctx={self.config.embedding_n_ctx}, n_batch={model.n_batch}). "
                    "Split the input into smaller chunks before embedding."
                )

        def _run(cancel_event: threading.Event) -> Iterator[list[list[float]]]:
            if cancel_event.is_set():
                return
            result = model.create_embedding(formatted_inputs)
            vectors: list[list[float]] = []
            for item in result["data"]:
                vector = [float(value) for value in item["embedding"]]
                if self.config.embedding_normalize:
                    norm = math.sqrt(sum(value * value for value in vector))
                    if norm > 0.0:
                        vector = [value / norm for value in vector]
                vectors.append(vector)
            yield vectors

        async for vectors in self._queue.submit(
            _run,
            timeout_s=self.config.generation_timeout_s,
        ):
            return vectors
        return []

    def _format_embedding_input(
        self, text: str, *, input_type: str, instruction: str | None
    ) -> str:
        if input_type not in {"raw", "query", "document"}:
            raise ValueError("input_type must be 'raw', 'query', or 'document'")

        if self.config.embedding_format == "qwen3":
            if input_type == "query":
                task = (instruction or "Represent this sentence for searching relevant passages").strip()
                text = f"Instruct: {task}\nQuery: {text}"
            # Qwen3-Embedding GGUFs use the end-of-text token as the last token.
            if not text.endswith("<|endoftext|>"):
                text = f"{text}<|endoftext|>"
        elif input_type == "query" and instruction:
            # Generic instruction-aware embedding models can opt into a simple
            # text prefix by setting EMBEDDING_FORMAT=plain; raw/document inputs
            # remain untouched.
            text = f"{instruction.strip()}\n{text}"
        return text

    async def _ensure_embedding_model(self) -> Llama:
        async with self._embedding_lock:
            if self._embedding_model is not None:
                return self._embedding_model

            model_path = self.config.embedding_model_path
            if model_path is None:
                if not self.config.embedding_fallback_to_chat:
                    raise RuntimeError(
                        "No dedicated embedding model is configured. Set "
                        "ARC_RUNTIME_EMBEDDING_MODEL_PATH to an embedding GGUF "
                        "(recommended: Qwen3-Embedding-0.6B-Q8_0.gguf). To explicitly "
                        "allow loading a second copy of the chat model, set "
                        "ARC_RUNTIME_EMBEDDING_FALLBACK_TO_CHAT=true."
                    )
                model_path = self.config.model_path
                logger.warning(
                    "Explicit embedding fallback enabled: loading a second copy "
                    "of the chat model for embeddings (%s).",
                    model_path,
                )

            pooling = _POOLING_TYPES.get(
                self.config.embedding_pooling,
                LLAMA_POOLING_TYPE_UNSPECIFIED,
            )

            def _load(cancel_event: threading.Event) -> Iterator[Llama]:
                if cancel_event.is_set():
                    return
                yield self._construct_embedding_model(model_path, pooling)

            # Load via the same serialized worker as generation. This avoids
            # constructing a second GPU model while a chat decode is active.
            async for loaded_model in self._queue.submit(
                _load, timeout_s=self.config.generation_timeout_s
            ):
                self._embedding_model = loaded_model
                self._embedding_model_name = model_path.name
                logger.info(
                    "Embedding model loaded: model=%s pooling=%s n_ctx=%d n_batch=%d",
                    model_path.name,
                    self.config.embedding_pooling,
                    self.config.embedding_n_ctx,
                    self.config.embedding_n_batch,
                )
                return loaded_model
            raise RuntimeError("Embedding model load was cancelled before completion")

    def _construct_embedding_model(self, model_path: Path, pooling_type: int) -> Llama:
        if (
            self.config.embedding_format == "qwen3"
            and pooling_type != LLAMA_POOLING_TYPE_LAST
        ):
            raise RuntimeError(
                "Qwen3-Embedding requires ARC_RUNTIME_EMBEDDING_POOLING=last"
            )

        gpu_setting = self.config.embedding_n_gpu_layers.strip().lower()
        if gpu_setting == "auto":
            # Resolve after the main model is loaded, using the *currently free*
            # VRAM rather than the startup-time free memory estimate.
            n_gpu_layers = _auto_gpu_layers(model_path)
        else:
            try:
                n_gpu_layers = int(gpu_setting)
            except ValueError as exc:
                raise RuntimeError(
                    "ARC_RUNTIME_EMBEDDING_N_GPU_LAYERS must be 'auto' or an integer"
                ) from exc

        def construct(pooling: int) -> Llama:
            llama_class = _EmbeddingLlama if self.config.embedding_format == "qwen3" else Llama
            extra: dict[str, Any] = (
                {"parse_special_tokens": True}
                if self.config.embedding_format == "qwen3"
                else {}
            )
            return llama_class(
                model_path=str(model_path),
                n_ctx=self.config.embedding_n_ctx,
                n_threads=self.config.n_threads,
                n_batch=min(self.config.embedding_n_batch, self.config.embedding_n_ctx),
                n_gpu_layers=n_gpu_layers,
                embedding=True,
                pooling_type=pooling,
                verbose=self.config.verbose,
                **extra,
            )

        try:
            return construct(pooling_type)
        except (ValueError, TypeError) as exc:
            if self.config.embedding_format == "qwen3" and pooling_type == LLAMA_POOLING_TYPE_LAST:
                raise RuntimeError(
                    "The installed llama-cpp-python build could not initialize LAST pooling, "
                    "which Qwen3-Embedding requires. Update/rebuild llama-cpp-python with "
                    "a recent llama.cpp backend rather than silently using incorrect pooling."
                ) from exc
            if pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED:
                raise
            logger.warning(
                "Embedding model rejected pooling_type=%s; retrying with unspecified pooling.",
                pooling_type,
            )
            return construct(LLAMA_POOLING_TYPE_UNSPECIFIED)

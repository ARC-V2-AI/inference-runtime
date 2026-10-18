import asyncio
import logging

from llama_cpp import Llama

from arc_v2_inference_runtime.engine.config import EngineConfig
from arc_v2_inference_runtime.engine.model_types import (
    ModelCapabilities,
    ModelInfo,
    ModelProtocol,
    ReasoningFormat,
    ToolCallFormat,
    detect_model_protocols,
)
from arc_v2_inference_runtime.engine.templating import ChatTemplateRenderer

# GGML KV-cache quantization type constants. Not all llama-cpp-python builds
# expose these; we degrade gracefully (skip the kwarg) if they're missing.
try:
    from llama_cpp import GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0
except ImportError:  # pragma: no cover
    GGML_TYPE_F16 = GGML_TYPE_Q4_0 = GGML_TYPE_Q8_0 = None  # type: ignore[assignment]  # pyright: ignore[reportConstantRedefinition]


_KV_CACHE_TYPES = {"f16": GGML_TYPE_F16, "q8_0": GGML_TYPE_Q8_0, "q4_0": GGML_TYPE_Q4_0}

logger = logging.getLogger(__name__)


class Model:
    def __init__(
        self,
        config: EngineConfig,
    ) -> None:
        self.config: EngineConfig = config
        self._info: ModelInfo | None = None
        self._capabilities: ModelCapabilities | None = None
        self._model: Llama | None = None
        self._renderer: ChatTemplateRenderer | None = None
        self._protocol: ModelProtocol | None = None

    @property
    def loaded(self) -> bool:
        """Return whether the model is currently loaded."""
        return (
            self._model is not None
            and self._renderer is not None
            and self._protocol is not None
        )

    @property
    def instance(self) -> Llama:
        """Return the underlying llama.cpp model instance.

        Raises:
            RuntimeError: If the model has not been loaded yet.

        Returns:
            Llama: The loaded llama.cpp model instance.
        """
        if self._model is None:
            raise RuntimeError("Model is not loaded")
        return self._model

    @property
    def renderer(self) -> ChatTemplateRenderer:
        """Return the chat template renderer for the loaded model.

        The renderer is created when the model is loaded and is specific to that
        model instance. It must not be accessed before the model has been loaded.

        Raises:
            RuntimeError: If the model instance has not been loaded and the
                renderer has not been initialized.

        Returns:
            ChatTemplateRenderer: The renderer configured for the loaded model.
        """
        if self._renderer is None:
            raise RuntimeError("Renderer is not set")

        return self._renderer

    @property
    def protocol(self) -> ModelProtocol:
        """Return the output protocol detected for the loaded model.

        The protocol is detected from the loaded model's chat template and is
        specific to that model instance. It must not be accessed before the model
        has been loaded.

        Raises:
            RuntimeError: If the model instance has not been loaded and the
                protocol has not been initialized.

        Returns:
            ModelProtocol: The protocol detected for the loaded model.
        """
        if self._protocol is None:
            raise RuntimeError("Protocols are not set")

        return self._protocol

    @property
    def info(self) -> ModelInfo:
        """Return metadata reported by the loaded model.

        The metadata is read from the underlying llama.cpp model and converted
        into the normalized :class:`ModelInfo` model. The result is cached for
        subsequent accesses.

        The returned values describe the model itself and are not derived from
        runtime configuration:

        - ``name``: Model name reported by the GGUF metadata.
        - ``architecture``: Model architecture reported by the GGUF metadata.
        - ``context_length``: Maximum context length reported by the model.
        - ``raw_metadata``: Complete GGUF metadata exposed by llama.cpp.
        - ``chat_template``: Chat template reported by the model, if available.
        - ``bos_token_id``: Token ID used as the model's beginning-of-sequence
        token, if valid metadata is available.
        - ``bos_token``: String representation of the beginning-of-sequence
        token, if its ID can be resolved from the model vocabulary.
        - ``eos_token_id``: Token ID used as the model's end-of-sequence token,
        if valid metadata is available.
        - ``eos_token``: String representation of the end-of-sequence token,
        if its ID can be resolved from the model vocabulary.

        Values that cannot be obtained or validated from the model metadata are
        represented as ``None`` where the corresponding field is optional.

        Raises:
            RuntimeError: If the model has not been loaded yet.

        Returns:
            ModelInfo: Normalized metadata describing the loaded model.
        """
        if self._info:
            return self._info

        if not self._model:
            raise RuntimeError("Get get model info: model is not loaded!")

        _metadata = self._model.metadata

        architecture = _metadata.get("general.architecture", "unknown")

        tokens = _metadata.get("tokenizer.ggml.tokens")
        bos_id = _metadata.get("tokenizer.ggml.bos_token_id")
        eos_id = _metadata.get("tokenizer.ggml.eos_token_id")

        if (
            isinstance(tokens, list)
            and isinstance(bos_id, int)
            and 0 <= bos_id < len(tokens)
        ):
            bos_token = tokens[bos_id]
        else:
            bos_id = None
            bos_token = None

        if (
            isinstance(tokens, list)
            and isinstance(eos_id, int)
            and 0 <= eos_id < len(tokens)
        ):
            eos_token = tokens[eos_id]
        else:
            eos_id = None
            eos_token = None

        return ModelInfo(
            name=_metadata.get("general.name", "unknown"),
            architecture=architecture,
            context_length=int(
                _metadata.get(
                    f"{architecture}.context_length",
                    "unknown",
                ),
            ),
            raw_metadata=_metadata,
            chat_template=_metadata.get("tokenizer.chat_template", "unknown"),
            bos_token=bos_token,
            bos_token_id=bos_id,
            eos_token=eos_token,
            eos_token_id=eos_id,
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return the capabilities detected for the loaded model.

        Capabilities are derived from the model's chat template using
        :func:`detect_model_protocols`. The detected :class:`ModelProtocol`
        describes the model's reasoning and tool-calling formats, while this
        object provides normalized capability flags for whether those features
        are supported.

        The context length is taken directly from :attr:`info` and represents
        the context length reported by the model's metadata, not the runtime
        configuration.

        Raises:
            RuntimeError: If the model is not loaded.

        Returns:
            ModelCapabilities: The detected model capabilities, including
            reasoning support, tool-calling support, the detected model protocol,
            and the model's reported context length.
        """
        if not self._model:
            raise RuntimeError("Model is not loaded")

        _info = self.info
        _template = _info.chat_template

        _protocols = detect_model_protocols(_template)

        return ModelCapabilities(
            reasoning=_protocols.reasoning_format != ReasoningFormat.NONE,
            tool_calling=_protocols.tool_call_format != ToolCallFormat.NONE,
            model_protocols=_protocols,
            context_length=_info.context_length,
        )

    @staticmethod
    def protocol_for_override(value: str, detected: ModelProtocol) -> ModelProtocol:

        try:
            tool_protocol = ToolCallFormat(value)
        except ValueError as exc:
            raise RuntimeError(f"Unknown ARC_RUNTIME_TOOL_PROTOCOL={value!r}") from exc
        defaults = {
            ToolCallFormat.QWEN_XML: ("<tool_call>", "</tool_call>"),
            ToolCallFormat.TAGGED_JSON: ("<tool_call>", "</tool_call>"),
        }
        tool_open, tool_close = defaults.get(tool_protocol, (None, None))
        return ModelProtocol(
            tool_call_format=tool_protocol,
            reasoning_format=detected.reasoning_format,
            think_open=detected.think_open,
            think_close=detected.think_close,
            tool_open=tool_open,
            tool_close=tool_close,
        )

    def _perf_kwargs(self):  # pyright: ignore[reportUnknownParameterType]
        """Optional constructor kwargs for flash attention / KV cache quant.

        Built separately so _construct_llama can retry without them if the
        installed llama-cpp-python build doesn't support one or more of
        these -- older builds raise TypeError on an unknown kwarg.
        """
        kwargs = {}

        if self.config.flash_attn:
            kwargs["flash_attn"] = True

        type_k = _KV_CACHE_TYPES.get(self.config.kv_cache_type_k)
        type_v = _KV_CACHE_TYPES.get(self.config.kv_cache_type_v)

        if type_k is not None:
            kwargs["type_k"] = type_k
        elif self.config.kv_cache_type_k != "f16":
            logger.warning(
                f"Unknown or unsupported kv_cache_type_k={self.config.kv_cache_type_k}; leaving llama.cpp default",
            )

        if type_v is not None:
            kwargs["type_v"] = type_v
        elif self.config.kv_cache_type_v != "f16":
            logger.warning(
                f"Unknown or unsupported kv_cache_type_v={self.config.kv_cache_type_v}; leaving llama.cpp default",
            )

        return kwargs  # pyright: ignore[reportUnknownVariableType]

    def _load(self):
        n_gpu_layers = self.config.n_gpu_layers
        attempts = [n_gpu_layers]

        # If GPU layers are requested
        if n_gpu_layers > 0:
            # Reduce GPU layers by 8 for each fallback attempt.
            # This creates a list of fallback GPU-layer counts, decreasing by 8 each attempt.
            current = n_gpu_layers
            while current > 0:
                current = max(0, current - 8)
                if current not in attempts:
                    attempts.append(current)

        perf_kwargs = self._perf_kwargs()  # pyright: ignore[reportUnknownVariableType]
        last_error: Exception | None = None

        # Try each GPU-layer configuration with and without optional performance settings.
        for layers in attempts:
            for kwargs in (perf_kwargs, {}) if perf_kwargs else ({},):  # pyright: ignore[reportUnknownVariableType]
                try:
                    if kwargs:
                        logger.info(
                            "Trying Llama() with n_gpu_layers=%d perf_kwargs=%s",
                            layers,
                            kwargs,  # pyright: ignore[reportUnknownArgumentType]
                        )
                    else:
                        logger.info(
                            "Trying Llama() with n_gpu_layers=%d (no perf kwargs)",
                            layers,
                        )

                    self._model = Llama(
                        model_path=str(self.config.model_path),
                        n_ctx=self.config.n_ctx,
                        n_threads=self.config.n_threads,
                        n_batch=self.config.n_batch,
                        n_gpu_layers=layers,
                        embedding=False,
                        verbose=self.config.verbose,
                        **kwargs,  # pyright: ignore[reportUnknownArgumentType]
                    )
                    return

                except TypeError as exc:
                    # Unknown kwarg for this llama-cpp-python build -- retry
                    # this same n_gpu_layers without the extra perf kwargs
                    # rather than treating it as an OOM-style failure.
                    last_error = exc
                    if kwargs:
                        logger.warning(
                            "Llama() rejected perf kwargs %s (%s); retrying without them",
                            kwargs,  # pyright: ignore[reportUnknownArgumentType]
                            exc,
                        )
                    continue
                except Exception as exc:
                    last_error = exc
                    logger.exception(
                        "Llama() construction failed with n_gpu_layers=%d kwargs=%s",
                        layers,
                        kwargs,  # pyright: ignore[reportUnknownArgumentType]
                    )
        assert last_error is not None
        raise last_error

    async def load(self) -> None:
        if self._model is not None:
            return

        await asyncio.to_thread(self._load)

        _infos = self.info

        self._renderer = ChatTemplateRenderer(
            _infos.chat_template,
            bos_token=_infos.bos_token or "",
            eos_token=_infos.eos_token or "",
        )

        self._protocol = self.capabilities.model_protocols
        if self.config.tool_protocol != "auto":
            self._protocol = self.protocol_for_override(
                self.config.tool_protocol, self._protocol
            )

        logger.info(
            "Model protocol: tools=%s reasoning=%s tool_protocol=%s",
            self._protocol.supports_tools,
            self._protocol.supports_reasoning,
            self._protocol.tool_call_format,
        )
        if self._renderer.using_fallback:
            logger.warning(
                "GGUF has no tokenizer.chat_template; using minimal fallback template"
            )
        logger.debug(
            "Protocol markers: tool_open=%r tool_close=%r think_open=%r think_close=%r",
            self._protocol.tool_open,
            self._protocol.tool_close,
            self._protocol.think_open,
            self._protocol.think_close,
        )

    def unload(self) -> None:
        self._model = None

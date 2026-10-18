import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import gguf

from arc_v2_inference_runtime.constants import (
    ARC_RUNTIME_DEBUG,
    ARC_RUNTIME_CONVERSATION_DB_PATH,
    ARC_RUNTIME_CONTEXT_SAFETY_MARGIN,
    ARC_RUNTIME_EMBEDDING_FALLBACK_TO_CHAT,
    ARC_RUNTIME_EMBEDDING_FORMAT,
    ARC_RUNTIME_EMBEDDING_N_CTX,
    ARC_RUNTIME_EMBEDDING_N_BATCH,
    ARC_RUNTIME_EMBEDDING_N_GPU_LAYERS,
    ARC_RUNTIME_EMBEDDING_NORMALIZE,
    ARC_RUNTIME_EMBEDDING_MODEL_PATH,
    ARC_RUNTIME_EMBEDDING_POOLING,
    ARC_RUNTIME_FLASH_ATTN,
    ARC_RUNTIME_GEN_TIMEOUT_S,
    ARC_RUNTIME_GRAMMAR_CONSTRAINED_TOOLS,
    ARC_RUNTIME_KV_CACHE_TYPE_K,
    ARC_RUNTIME_KV_CACHE_TYPE_V,
    ARC_RUNTIME_MODEL_PATH,
    ARC_RUNTIME_MAX_OUTPUT_TOKENS,
    ARC_RUNTIME_N_BATCH,
    ARC_RUNTIME_N_CTX,
    ARC_RUNTIME_N_GPU_LAYERS,
    ARC_RUNTIME_N_THREADS,
    ARC_RUNTIME_SAMPLING_MIN_P,
    ARC_RUNTIME_SAMPLING_MIN_P_THINKING,
    ARC_RUNTIME_SAMPLING_REPEAT_PENALTY,
    ARC_RUNTIME_SAMPLING_TEMPERATURE,
    ARC_RUNTIME_SAMPLING_TEMPERATURE_THINKING,
    ARC_RUNTIME_SAMPLING_TOP_K,
    ARC_RUNTIME_SAMPLING_TOP_K_THINKING,
    ARC_RUNTIME_SAMPLING_TOP_P,
    ARC_RUNTIME_SAMPLING_TOP_P_THINKING,
    ARC_RUNTIME_STOP_SEQUENCES,
    ARC_RUNTIME_TOOL_PROTOCOL,
    ARC_RUNTIME_VERBOSE,
)

logger = logging.getLogger(__name__)
_POOLING_VALUES = {"unspecified", "mean", "cls", "last"}


def _free_vram_mb() -> int | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            timeout=5,
        )
        values = [int(x.strip()) for x in out.decode().splitlines() if x.strip()]
        return max(values) if values else None
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def _gguf_layer_estimate(model_path: Path) -> tuple[int | None, float | None]:
    try:
        reader = gguf.GGUFReader(model_path)
        n_layers = None
        for field_name, field_ in reader.fields.items():
            if field_name.endswith("block_count"):
                n_layers = int(field_.parts[field_.data[0]][0])
                break
        if not n_layers:
            return None, None
        file_size = os.path.getsize(model_path)
        return n_layers, (file_size * 0.90) / n_layers
    except Exception:
        return None, None


def _auto_gpu_layers(model_path: Path) -> int:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return -1
    free_mb = _free_vram_mb()
    if free_mb is None:
        return 0
    n_layers, bytes_per_layer = _gguf_layer_estimate(model_path)
    if not n_layers or not bytes_per_layer:
        return 0
    usable_bytes = free_mb * 1024 * 1024 * 0.60
    return max(0, min(int(usable_bytes // bytes_per_layer), n_layers))


@dataclass
class EngineConfig:
    model_path: Path
    n_ctx: int = 4096
    n_threads: int | None = None
    # NOTE: prefer a power-of-2 batch size (256/512/1024) for prompt-processing
    # throughput. A non-power-of-2 value (e.g. 225) works but is suboptimal.
    n_batch: int = 256
    n_gpu_layers: int = 0
    embedding_pooling: str = "last"
    generation_timeout_s: float = 120.0
    verbose: bool = False
    tool_protocol: str = "auto"
    stop_sequences: list[str] = field(default_factory=list)

    # --- Performance ---
    # Both require a llama-cpp-python build with the corresponding llama.cpp
    # features compiled in. The engine falls back gracefully (with a logged
    # warning) if the installed version doesn't support these constructor
    # kwargs, so it's safe to leave these on by default.
    flash_attn: bool = True
    kv_cache_type_k: str = "f16"  # "f16" | "q8_0" | "q4_0"
    kv_cache_type_v: str = "f16"

    # --- Dedicated embedding model ---
    # Dedicated embedding models are lazy-loaded. Do not duplicate the chat model
    # unless ARC_RUNTIME_EMBEDDING_FALLBACK_TO_CHAT=true is explicitly requested.
    embedding_model_path: Path | None = None
    embedding_n_ctx: int = 8192
    embedding_n_batch: int = 2048
    embedding_n_gpu_layers: str = "auto"
    embedding_format: str = "qwen3"  # "qwen3" | "plain"
    embedding_normalize: bool = True
    embedding_fallback_to_chat: bool = False

    # --- Context and transcript persistence ---
    max_output_tokens: int = 1024
    context_safety_margin_tokens: int = 16
    conversation_db_path: Path = field(
        default_factory=lambda: Path("~/.local/share/arc-v2/inference/conversations.sqlite3").expanduser()
    )

    # --- Sampling defaults ---
    # Used whenever a request doesn't set its own value. Thinking vs
    # non-thinking defaults differ, matching Qwen3's own recommended sampler
    # settings per mode (thinking mode wants a tighter, more deterministic
    # sampler than freeform chat).
    sampling_temperature_thinking: float = 0.6
    sampling_top_p_thinking: float = 0.95
    sampling_top_k_thinking: int = 20
    sampling_min_p_thinking: float = 0.0
    sampling_temperature: float = 0.7
    sampling_top_p: float = 0.8
    sampling_top_k: int = 20
    sampling_min_p: float = 0.0
    # Applied unconditionally: llama.cpp's own default (1.0) is "off", which
    # is the single biggest reason quantized models loop/repeat in long
    # agentic tool sessions.
    sampling_repeat_penalty: float = 1.1

    # --- Reliability ---
    # Constrains native tool-call output (qwen_xml / tagged_json) to
    # well-formed structure via a GBNF grammar built from the request's own
    # tool list. This does not fight the model's native format -- it enforces
    # the same structure the prompt already asks for -- so it doesn't add
    # meaningful latency, it just prevents malformed/incomplete calls.
    grammar_constrained_tools: bool = False

    # --- Debugging ---
    # Flips the engine + parser loggers to DEBUG so you can trace protocol
    # detection, rendered prompts, and parser state transitions without
    # touching your global logging config.
    debug_trace: bool = False

    @classmethod
    def from_env(cls) -> "EngineConfig":
        model_path = ARC_RUNTIME_MODEL_PATH
        if not model_path:
            raise RuntimeError(
                "ARC_RUNTIME_MODEL_PATH is not set. Point it at a local .gguf file."
            )

        if not model_path.is_file():
            raise RuntimeError(f"ARC_RUNTIME_MODEL_PATH does not exist: {model_path}")

        gpu_env = ARC_RUNTIME_N_GPU_LAYERS
        gpu_layers = (
            _auto_gpu_layers(model_path)
            if gpu_env.strip().lower() == "auto"
            else int(gpu_env)
        )

        # NOTE: N_THREADS = 0 means let llama.cpp decide
        threads = int(ARC_RUNTIME_N_THREADS) if ARC_RUNTIME_N_THREADS != 0 else None

        # NOTE: STOP_SEQUENCE is a list of comma separated values like <|im_end|>,<|endoftext|>
        stops = [s for s in ARC_RUNTIME_STOP_SEQUENCES.split(",") if s.strip()]

        embedding_model_path = (
            None
            if ARC_RUNTIME_EMBEDDING_MODEL_PATH == ""
            else Path(os.path.expandvars(os.path.expanduser(ARC_RUNTIME_EMBEDDING_MODEL_PATH)))
        )

        if embedding_model_path is not None and not embedding_model_path.is_file():
            raise RuntimeError(
                f"ARC_RUNTIME_EMBEDDING_MODEL_PATH does not exist: {embedding_model_path}"
            )

        if ARC_RUNTIME_N_CTX < 256:
            raise RuntimeError("ARC_RUNTIME_N_CTX must be at least 256")
        if ARC_RUNTIME_MAX_OUTPUT_TOKENS < 1:
            raise RuntimeError("ARC_RUNTIME_MAX_OUTPUT_TOKENS must be positive")
        if ARC_RUNTIME_CONTEXT_SAFETY_MARGIN < 0:
            raise RuntimeError("ARC_RUNTIME_CONTEXT_SAFETY_MARGIN cannot be negative")
        if ARC_RUNTIME_EMBEDDING_N_CTX < 1:
            raise RuntimeError("ARC_RUNTIME_EMBEDDING_N_CTX must be positive")
        if ARC_RUNTIME_EMBEDDING_N_BATCH < 1:
            raise RuntimeError("ARC_RUNTIME_EMBEDDING_N_BATCH must be positive")
        if ARC_RUNTIME_EMBEDDING_FORMAT.strip().lower() not in {"qwen3", "plain"}:
            raise RuntimeError("ARC_RUNTIME_EMBEDDING_FORMAT must be 'qwen3' or 'plain'")
        if ARC_RUNTIME_EMBEDDING_POOLING.strip().lower() not in _POOLING_VALUES:
            raise RuntimeError(
                "ARC_RUNTIME_EMBEDDING_POOLING must be one of: unspecified, mean, cls, last"
            )
        if (
            ARC_RUNTIME_EMBEDDING_FORMAT.strip().lower() == "qwen3"
            and ARC_RUNTIME_EMBEDDING_POOLING.strip().lower() != "last"
        ):
            raise RuntimeError(
                "Qwen3-Embedding requires ARC_RUNTIME_EMBEDDING_POOLING=last"
            )

        return cls(
            model_path=model_path,
            n_ctx=ARC_RUNTIME_N_CTX,
            n_threads=threads,
            n_batch=ARC_RUNTIME_N_BATCH,
            n_gpu_layers=gpu_layers,
            embedding_pooling=ARC_RUNTIME_EMBEDDING_POOLING,
            generation_timeout_s=ARC_RUNTIME_GEN_TIMEOUT_S,
            verbose=ARC_RUNTIME_VERBOSE,
            tool_protocol=ARC_RUNTIME_TOOL_PROTOCOL,
            stop_sequences=stops,
            flash_attn=ARC_RUNTIME_FLASH_ATTN,
            kv_cache_type_k=ARC_RUNTIME_KV_CACHE_TYPE_K,
            kv_cache_type_v=ARC_RUNTIME_KV_CACHE_TYPE_V,
            embedding_model_path=embedding_model_path,
            embedding_n_ctx=ARC_RUNTIME_EMBEDDING_N_CTX,
            embedding_n_batch=ARC_RUNTIME_EMBEDDING_N_BATCH,
            embedding_n_gpu_layers=ARC_RUNTIME_EMBEDDING_N_GPU_LAYERS,
            embedding_format=ARC_RUNTIME_EMBEDDING_FORMAT.strip().lower(),
            embedding_normalize=ARC_RUNTIME_EMBEDDING_NORMALIZE,
            embedding_fallback_to_chat=ARC_RUNTIME_EMBEDDING_FALLBACK_TO_CHAT,
            max_output_tokens=ARC_RUNTIME_MAX_OUTPUT_TOKENS,
            context_safety_margin_tokens=ARC_RUNTIME_CONTEXT_SAFETY_MARGIN,
            conversation_db_path=ARC_RUNTIME_CONVERSATION_DB_PATH,
            sampling_temperature_thinking=ARC_RUNTIME_SAMPLING_TEMPERATURE_THINKING,
            sampling_top_p_thinking=ARC_RUNTIME_SAMPLING_TOP_P_THINKING,
            sampling_top_k_thinking=ARC_RUNTIME_SAMPLING_TOP_K_THINKING,
            sampling_min_p_thinking=ARC_RUNTIME_SAMPLING_MIN_P_THINKING,
            sampling_temperature=ARC_RUNTIME_SAMPLING_TEMPERATURE,
            sampling_top_p=ARC_RUNTIME_SAMPLING_TOP_P,
            sampling_top_k=ARC_RUNTIME_SAMPLING_TOP_K,
            sampling_min_p=ARC_RUNTIME_SAMPLING_MIN_P,
            sampling_repeat_penalty=ARC_RUNTIME_SAMPLING_REPEAT_PENALTY,
            grammar_constrained_tools=ARC_RUNTIME_GRAMMAR_CONSTRAINED_TOOLS,
            debug_trace=ARC_RUNTIME_DEBUG,
        )

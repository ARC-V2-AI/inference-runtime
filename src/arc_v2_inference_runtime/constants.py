from arc_v2_util.constants import Constant

ARC_RUNTIME_PORT = Constant(default=7842, type=int)
ARC_RUNTIME_DEBUG = Constant(default=True, type=bool)
ARC_RUNTIME_EMBEDDING_MODEL_PATH = Constant(default="", type=str)
ARC_RUNTIME_EMBEDDING_POOLING = Constant(default="last", type=str)
ARC_RUNTIME_FLASH_ATTN = Constant(default=True, type=bool)
ARC_RUNTIME_GEN_TIMEOUT_S = Constant(default=120.0, type=float)
ARC_RUNTIME_GRAMMAR_CONSTRAINED_TOOLS = Constant(default=True, type=bool)
ARC_RUNTIME_KV_CACHE_TYPE_K = Constant(default="f16", type=str)
ARC_RUNTIME_KV_CACHE_TYPE_V = Constant(default="f16", type=str)
ARC_RUNTIME_MODEL_PATH = Constant(
    default="~/arc/models/unsloth__Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf", type=str
).to_path()
ARC_RUNTIME_N_BATCH = Constant(default=256, type=int)
ARC_RUNTIME_N_CTX = Constant(default=4096, type=int)
ARC_RUNTIME_N_GPU_LAYERS = Constant(default="auto", type=str)
ARC_RUNTIME_N_THREADS = Constant(default=0, type=int)
ARC_RUNTIME_SAMPLING_MIN_P = Constant(default=0.0, type=float)
ARC_RUNTIME_SAMPLING_MIN_P_THINKING = Constant(default=0.0, type=float)
ARC_RUNTIME_SAMPLING_REPEAT_PENALTY = Constant(default=1.1, type=float)
ARC_RUNTIME_SAMPLING_TEMPERATURE = Constant(default=0.7, type=float)
ARC_RUNTIME_SAMPLING_TEMPERATURE_THINKING = Constant(default=0.6, type=float)
ARC_RUNTIME_SAMPLING_TOP_K = Constant(default=20, type=int)
ARC_RUNTIME_SAMPLING_TOP_K_THINKING = Constant(default=20, type=int)
ARC_RUNTIME_SAMPLING_TOP_P = Constant(default=0.8, type=float)
ARC_RUNTIME_SAMPLING_TOP_P_THINKING = Constant(default=0.95, type=float)
ARC_RUNTIME_STOP_SEQUENCES = Constant(default="", type=str)
ARC_RUNTIME_TOOL_PROTOCOL = Constant(default="auto", type=str)
ARC_RUNTIME_VERBOSE = Constant(default=False, type=bool)

# Token-aware context management.
ARC_RUNTIME_MAX_OUTPUT_TOKENS = Constant(default=1024, type=int)
ARC_RUNTIME_CONTEXT_SAFETY_MARGIN = Constant(default=16, type=int)

# Persistent chat transcripts (separate from model/KV-cache state).
ARC_RUNTIME_CONVERSATION_DB_PATH = Constant(
    default="~/.local/share/arc-v2/inference/conversations.sqlite3", type=str
).to_path()

# Dedicated embedding model settings. Qwen3 Embedding requires last-token pooling.
ARC_RUNTIME_EMBEDDING_N_CTX = Constant(default=8192, type=int)
ARC_RUNTIME_EMBEDDING_N_BATCH = Constant(default=2048, type=int)
ARC_RUNTIME_EMBEDDING_N_GPU_LAYERS = Constant(default="auto", type=str)
ARC_RUNTIME_EMBEDDING_FORMAT = Constant(default="qwen3", type=str)
ARC_RUNTIME_EMBEDDING_FALLBACK_TO_CHAT = Constant(default=False, type=bool)
ARC_RUNTIME_EMBEDDING_NORMALIZE = Constant(default=True, type=bool)

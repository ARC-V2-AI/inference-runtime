from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ReasoningFormat(str, Enum):
    UNKNOWN = "unknown"
    NONE = "none"
    THINK_TAG = "think_tag"
    SEPARATE_FIELD = "separate_field"
    HARMONY_CHANNEL = "harmony_channel"


class ToolCallFormat(Enum):
    NONE = "none"
    QWEN_XML = "qwen_xml"
    TAGGED_JSON = "tagged_json"
    HARMONY = "harmony"
    PYTHON_TAG = "python_tag"
    BRACKET_JSON = "bracket_json"
    GENERIC_JSON = "generic_json"


class ModelInfo(BaseModel):
    name: str
    architecture: str
    context_length: int
    raw_metadata: dict[str, str]
    chat_template: str

    bos_token_id: int | None = None
    bos_token: str | None = None
    eos_token_id: int | None = None
    eos_token: str | None = None


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    context_length: int = Field(default=10240, gt=0)
    gpu_layers: int = -1
    batch_size: int = Field(default=512, gt=0)


class ModelProtocol(BaseModel):
    tool_call_format: ToolCallFormat
    reasoning_format: ReasoningFormat
    think_open: str | None = None
    think_close: str | None = None
    tool_open: str | None = None
    tool_close: str | None = None

    @property
    def supports_tools(self) -> bool:
        return self.tool_call_format != ToolCallFormat.NONE

    @property
    def supports_reasoning(self) -> bool:
        return self.reasoning_format != ReasoningFormat.NONE


def detect_model_protocols(template: str | None) -> ModelProtocol:
    """Detect the output protocol from the model's own Jinja template.

    The template is treated as the source of truth. We intentionally detect
    only syntax that is explicit in the template and otherwise fall back to a
    generic JSON extractor rather than inventing a model-specific format.
    """
    if not template:
        return ModelProtocol(
            tool_call_format=ToolCallFormat.GENERIC_JSON,
            reasoning_format=ReasoningFormat.THINK_TAG,
            think_open="<think>",
            think_close="</think>",
            tool_open="<tool_call>",
            tool_close="</tool_call>",
        )

    is_qwen_xml = all(
        x in template for x in ("<tool_call>", "<function=", "<parameter=")
    )
    if is_qwen_xml:
        reasoning = (
            ReasoningFormat.THINK_TAG
            if "<think>" in template or "</think>" in template
            else ReasoningFormat.NONE
        )
        return ModelProtocol(
            tool_call_format=ToolCallFormat.QWEN_XML,
            reasoning_format=reasoning,
            think_open="<think>" if "<think>" in template else None,
            think_close="</think>" if "</think>" in template else None,
            tool_open="<tool_call>",
            tool_close="</tool_call>",
        )

    if "<|channel|>analysis<|message|>" in template and "to=functions." in template:
        return ModelProtocol(
            tool_call_format=ToolCallFormat.HARMONY,
            reasoning_format=ReasoningFormat.HARMONY_CHANNEL,
        )

    if "<|python_tag|>" in template:
        return ModelProtocol(
            tool_call_format=ToolCallFormat.PYTHON_TAG,
            reasoning_format=(
                ReasoningFormat.THINK_TAG
                if "<think>" in template
                else ReasoningFormat.NONE
            ),
            think_open="<think>" if "<think>" in template else None,
            think_close="</think>" if "</think>" in template else None,
        )

    if "[TOOL_CALLS]" in template:
        return ModelProtocol(
            tool_call_format=ToolCallFormat.BRACKET_JSON,
            reasoning_format=(
                ReasoningFormat.THINK_TAG
                if "<think>" in template
                else ReasoningFormat.NONE
            ),
            think_open="<think>" if "<think>" in template else None,
            think_close="</think>" if "</think>" in template else None,
        )

    if "<tool_call>" in template:
        return ModelProtocol(
            tool_call_format=ToolCallFormat.TAGGED_JSON,
            reasoning_format=(
                ReasoningFormat.THINK_TAG
                if "<think>" in template
                else ReasoningFormat.NONE
            ),
            think_open="<think>" if "<think>" in template else None,
            think_close="</think>" if "</think>" in template else None,
            tool_open="<tool_call>",
            tool_close="</tool_call>",
        )

    return ModelProtocol(
        tool_call_format=ToolCallFormat.GENERIC_JSON,
        reasoning_format=(
            ReasoningFormat.THINK_TAG if "<think>" in template else ReasoningFormat.NONE
        ),
        think_open="<think>" if "<think>" in template else None,
        think_close="</think>" if "</think>" in template else None,
    )


class ModelCapabilities(BaseModel):
    reasoning: bool = False
    tool_calling: bool = False
    model_protocols: ModelProtocol
    context_length: int = Field(gt=0)

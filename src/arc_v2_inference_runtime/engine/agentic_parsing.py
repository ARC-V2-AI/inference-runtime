"""Streaming parser for model-native reasoning and tool-call output.

The parser no longer selects an XML/JSON "fallback dialect" itself.  The loaded
Model supplies a ModelProtocol detected from the GGUF's native chat template,
and this parser only implements the output syntax described by that protocol.

The normalized output remains:
    ParsedEvent(kind="reasoning" | "content", ...)
    ToolCallFragment(index, id, name, arguments_fragment)

This keeps the parser independent from llama.cpp's structured function-calling
handler while allowing ARC to use the model's native chat template/protocol.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from arc_v2_inference_runtime.engine.model_types import ModelProtocol, ToolCallFormat
from arc_v2_inference_runtime.engine.parsing import ParsedEvent

_Mode = Literal[
    "auto",
    "content",
    "reasoning",
    "tagged_json_header",
    "tagged_json_args_seek",
    "tagged_json_args_stream",
    "tool_tail",
    "xml_function_header",
    "xml_param_seek",
    "xml_param_value",
]


_FUNCTION_OPEN_RE = re.compile(r"<function=([^>]+)>")
_PARAM_OPEN_RE = re.compile(r"<parameter=([^>]+)>")
_FUNCTION_CLOSE = "</function>"
_PARAM_CLOSE = "</parameter>"
_NAME_RE = re.compile(r'"name"\s*:\s*"((?:[^"\\]|\\.)*)"')
_ARGS_KEY_RE = re.compile(r'"(?:arguments|parameters)"\s*:\s*')


@dataclass
class ToolCallFragment:
    """A partial normalized tool call emitted while generation streams."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments_fragment: str = ""


def _new_call_id() -> str:
    return f"call-{uuid.uuid4().hex}"


def _trim_param_value(raw: str) -> str:
    # Native Qwen function-parameter templates add one framing newline.
    raw = raw.removeprefix("\n")
    raw = raw.removesuffix("\n")
    return raw


def _coerce_param_value(raw: str, schema_type: str | None) -> Any:
    """Convert native raw parameter text using the request tool schema."""
    if schema_type == "boolean":
        value = raw.strip().lower()
        if value in {"true", "false"}:
            return value == "true"
        return raw

    if schema_type == "integer":
        try:
            return int(raw.strip())
        except ValueError:
            return raw

    if schema_type == "number":
        try:
            return float(raw.strip())
        except ValueError:
            return raw

    if schema_type in {"array", "object"}:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    return raw


class _BalancedJsonScanner:
    """
    Incrementally find the end of a JSON value.


    """

    __slots__: tuple[str, ...] = ("depth", "escape", "in_string", "root_kind")

    def __init__(self) -> None:
        self.depth: int = 0
        self.root_kind: str | None = None
        self.in_string: bool = False
        self.escape: bool = False

    def consume(self, text: str) -> int:
        for i, ch in enumerate(text):
            if self.escape:
                self.escape = False
                continue

            if self.in_string:
                if ch == "\\":
                    self.escape = True
                elif ch == '"':
                    self.in_string = False
                continue

            if ch == '"':
                self.in_string = True
                if self.root_kind is None:
                    self.root_kind = '"'
                    return i + 1
            elif ch in "[{":
                if self.root_kind is None:
                    self.root_kind = ch
                self.depth += 1
            elif ch in "]}":
                if self.depth:
                    self.depth -= 1
                    if self.depth == 0:
                        return i + 1
            elif self.root_kind is None and ch in "-0123456789tfn":
                self.root_kind = "scalar"

        return -1


ParserEvent = ParsedEvent | ToolCallFragment


class AgenticStreamParser:
    """Parse output according to the model's detected native protocol."""

    def __init__(
        self,
        *,
        protocol: ModelProtocol,
        tool_schemas: dict[str, dict[str, Any]] | None = None,
        force_tool_call: bool = False,
        initial_mode: Literal["auto", "content", "reasoning"] = "auto",
    ) -> None:
        self.protocol: ModelProtocol = protocol
        self._tool_schemas: dict[str, dict[str, Any]] = tool_schemas or {}

        self._think_open: str = protocol.think_open or "<think>"
        self._think_close: str = protocol.think_close or "</think>"
        self._tool_open: str | None = protocol.tool_open
        self._tool_close: str | None = protocol.tool_close

        self._buf: str = ""
        self._tool_index: int = -1
        self._completed_indices: set[int] = set()
        self._scanner: _BalancedJsonScanner | None = None

        self._current_function_name: str | None = None
        self._current_param_key: str | None = None
        self._pending_args: dict[str, Any] = {}

        self.incomplete_tool_call_mode: str | None = None

        self._max_tag_len: int = max(
            len(x)
            for x in (
                self._think_open,
                self._think_close,
                self._tool_open or "",
                self._tool_close or "",
                _FUNCTION_CLOSE,
                _PARAM_CLOSE,
            )
        )

        # Forced calls are entered after the engine has already placed the
        # model-native tool prefix in the prompt.  Therefore the first streamed
        # bytes are the protocol-specific body, not <tool_call> itself.
        if force_tool_call and self._supports_tool_protocol:
            self._tool_index = 0
            self._mode: _Mode = self._native_tool_header_mode
        else:
            self._mode = initial_mode

    @property
    def _supports_tool_protocol(self) -> bool:
        return self.protocol.tool_call_format in {
            ToolCallFormat.QWEN_XML,
            ToolCallFormat.TAGGED_JSON,
        }

    @property
    def _native_tool_header_mode(self) -> _Mode:
        if self.protocol.tool_call_format == ToolCallFormat.QWEN_XML:
            return "xml_function_header"
        if self.protocol.tool_call_format == ToolCallFormat.TAGGED_JSON:
            return "tagged_json_header"
        raise RuntimeError(
            f"No streaming parser for native tool protocol {self.protocol.tool_call_format!r}"
        )

    @property
    def saw_tool_call(self) -> bool:
        return bool(self._completed_indices)

    def feed(self, chunk: str) -> list[ParserEvent]:
        self._buf += chunk
        out: list[ParserEvent] = []
        while self._step(out):
            pass
        return out

    def finalize(self) -> list[ParserEvent]:
        out: list[ParserEvent] = []

        if self._mode in {"auto", "content", "reasoning"} and self._buf:
            out.append(
                ParsedEvent(
                    kind="reasoning" if self._mode == "reasoning" else "content",
                    text=self._buf,
                )
            )

        elif self._mode == "tagged_json_args_stream" and self._buf:
            out.append(
                ToolCallFragment(
                    index=self._tool_index,
                    arguments_fragment=self._buf,
                )
            )

        elif self._mode in {
            "tagged_json_header",
            "tagged_json_args_seek",
            "tool_tail",
            "xml_function_header",
            "xml_param_seek",
            "xml_param_value",
        }:
            self.incomplete_tool_call_mode = self._mode
            if self._buf:
                out.append(ParsedEvent(kind="content", text=self._buf))

        self._buf = ""
        return out

    def _step(self, out: list[ParserEvent]) -> bool:
        handlers = {
            "auto": self._step_content,
            "content": self._step_content,
            "reasoning": self._step_reasoning,
            "tagged_json_header": self._step_tagged_json_header,
            "tagged_json_args_seek": self._step_tagged_json_args_seek,
            "tagged_json_args_stream": self._step_tagged_json_args_stream,
            "tool_tail": self._step_tool_tail,
            "xml_function_header": self._step_xml_function_header,
            "xml_param_seek": self._step_xml_param_seek,
            "xml_param_value": self._step_xml_param_value,
        }
        return handlers[self._mode](out)

    def _step_content(self, out: list[ParserEvent]) -> bool:
        think_idx = self._buf.find(self._think_open)

        tool_idx = -1
        if self._tool_open:
            tool_idx = self._buf.find(self._tool_open)

        candidates = [i for i in (think_idx, tool_idx) if i >= 0]
        if not candidates:
            safe_len = max(0, len(self._buf) - self._max_tag_len)
            if safe_len:
                out.append(ParsedEvent(kind="content", text=self._buf[:safe_len]))
                self._buf = self._buf[safe_len:]
            return False

        idx = min(candidates)

        if idx:
            out.append(ParsedEvent(kind="content", text=self._buf[:idx]))

        if idx == think_idx:
            self._buf = self._buf[idx + len(self._think_open) :]
            self._mode = "reasoning"
            return True

        self._buf = self._buf[idx + len(self._tool_open or "") :]
        self._tool_index += 1
        self._mode = self._native_tool_header_mode
        return True

    def _step_reasoning(self, out: list[ParserEvent]) -> bool:
        idx = self._buf.find(self._think_close)
        if idx < 0:
            safe_len = max(0, len(self._buf) - self._max_tag_len)
            if safe_len:
                out.append(ParsedEvent(kind="reasoning", text=self._buf[:safe_len]))
                self._buf = self._buf[safe_len:]
            return False

        if idx:
            out.append(ParsedEvent(kind="reasoning", text=self._buf[:idx]))

        self._buf = self._buf[idx + len(self._think_close) :]
        self._mode = "content"
        return True

    # ------------------------------------------------------------------
    # Tagged JSON native protocol
    # ------------------------------------------------------------------

    def _step_tagged_json_header(self, out: list[ParserEvent]) -> bool:
        match = _NAME_RE.search(self._buf)
        if not match:
            return False

        name = json.loads(f'"{match.group(1)}"')
        out.append(
            ToolCallFragment(
                index=self._tool_index,
                id=_new_call_id(),
                name=name,
            )
        )

        self._buf = self._buf[match.end() :]
        self._mode = "tagged_json_args_seek"
        return True

    def _step_tagged_json_args_seek(self, out: list[ParserEvent]) -> bool:  # pyright: ignore[reportUnusedParameter]
        match = _ARGS_KEY_RE.search(self._buf)
        if not match:
            return False

        rest = self._buf[match.end() :]
        brace_idx = rest.find("{")
        if brace_idx < 0:
            return False

        self._buf = rest[brace_idx:]
        self._scanner = _BalancedJsonScanner()
        self._mode = "tagged_json_args_stream"
        return True

    def _step_tagged_json_args_stream(self, out: list[ParserEvent]) -> bool:
        assert self._scanner is not None

        end = self._scanner.consume(self._buf)

        if end < 0:
            if self._buf:
                out.append(
                    ToolCallFragment(
                        index=self._tool_index,
                        arguments_fragment=self._buf,
                    )
                )
                self._buf = ""
            return False

        out.append(
            ToolCallFragment(
                index=self._tool_index,
                arguments_fragment=self._buf[:end],
            )
        )
        self._buf = self._buf[end:]
        self._scanner = None
        self._mode = "tool_tail"
        return True

    # ------------------------------------------------------------------
    # Qwen native function/parameter protocol
    # ------------------------------------------------------------------

    def _step_xml_function_header(self, out: list[ParserEvent]) -> bool:
        match = _FUNCTION_OPEN_RE.search(self._buf)
        if not match:
            return False

        name = match.group(1).strip()
        self._current_function_name = name
        self._pending_args = {}

        out.append(
            ToolCallFragment(
                index=self._tool_index,
                id=_new_call_id(),
                name=name,
            )
        )

        self._buf = self._buf[match.end() :]
        self._mode = "xml_param_seek"
        return True

    def _step_xml_param_seek(self, out: list[ParserEvent]) -> bool:
        function_close_idx = self._buf.find(_FUNCTION_CLOSE)
        match = _PARAM_OPEN_RE.search(self._buf)
        param_idx = match.start() if match else -1

        if param_idx < 0 and function_close_idx < 0:
            return False

        if param_idx >= 0 and (
            function_close_idx < 0 or param_idx < function_close_idx
        ):
            assert match is not None
            self._current_param_key = match.group(1).strip()
            self._buf = self._buf[match.end() :]
            self._mode = "xml_param_value"
            return True

        schema = self._tool_schemas.get(self._current_function_name or "", {})
        typed_args = {
            key: _coerce_param_value(
                raw,
                (schema.get(key) or {}).get("type"),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            )
            for key, raw in self._pending_args.items()
        }

        out.append(
            ToolCallFragment(
                index=self._tool_index,
                arguments_fragment=json.dumps(
                    typed_args,
                    separators=(",", ":"),
                ),
            )
        )

        self._buf = self._buf[function_close_idx + len(_FUNCTION_CLOSE) :]
        self._mode = "tool_tail"
        return True

    def _step_xml_param_value(self, out: list[ParserEvent]) -> bool:  # pyright: ignore[reportUnusedParameter]
        idx = self._buf.find(_PARAM_CLOSE)
        if idx < 0:
            return False

        assert self._current_param_key is not None

        self._pending_args[self._current_param_key] = _trim_param_value(self._buf[:idx])
        self._buf = self._buf[idx + len(_PARAM_CLOSE) :]
        self._current_param_key = None
        self._mode = "xml_param_seek"
        return True

    def _step_tool_tail(self, out: list[ParserEvent]) -> bool:  # pyright: ignore[reportUnusedParameter]
        if not self._tool_close:
            # A native protocol without an explicit outer close cannot be
            # finalized here. Current supported native protocols both have it.
            self._completed_indices.add(self._tool_index)
            self._mode = "content"
            return True

        idx = self._buf.find(self._tool_close)
        if idx < 0:
            safe_len = max(0, len(self._buf) - self._max_tag_len)
            self._buf = self._buf[safe_len:]
            return False

        self._buf = self._buf[idx + len(self._tool_close) :]
        self._completed_indices.add(self._tool_index)
        self._mode = "content"
        return True

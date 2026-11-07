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
import math
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


def _coerce_param_value(raw: str, schema_type: str | list[str] | None) -> Any:
    """Convert native raw parameter text using the request tool schema."""
    if isinstance(schema_type, list):
        schema_type = next(
            (item for item in schema_type if item != "null"),
            None,
        )

    if schema_type == "null":
        return None if raw.strip() in {"", "null"} else raw

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
            value = float(raw.strip())
            return value if math.isfinite(value) else raw
        except ValueError:
            return raw

    if schema_type in {"array", "object"}:
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw
        expected_type = list if schema_type == "array" else dict
        return value if isinstance(value, expected_type) else raw

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
        self._current_param_type: str | None = None
        self._current_param_is_string: bool = True
        self._current_param_first_chunk: bool = True
        self._current_param_buffer: list[str] = []
        self._xml_args_count: int = 0

        self.incomplete_tool_call_mode: str | None = None

        self._max_tag_len: int = max(
            len(x)
            for x in (
                self._think_open,
                self._think_close,
                self._tool_open or "",
                self._tool_close or "",
                protocol.function_open or "",
                protocol.function_close or "",
                protocol.parameter_open or "",
                protocol.parameter_close or "",
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
            partial_tool_length = 0
            if self._tool_open:
                partial_tool_length = max(
                    (
                        length
                        for length in range(1, len(self._tool_open))
                        if self._buf.endswith(self._tool_open[:length])
                    ),
                    default=0,
                )
            if partial_tool_length:
                prefix = self._buf[:-partial_tool_length]
                if prefix:
                    out.append(
                        ParsedEvent(
                            kind="reasoning" if self._mode == "reasoning" else "content",
                            text=prefix,
                        )
                    )
                self.incomplete_tool_call_mode = "tool_open"
            else:
                out.append(
                    ParsedEvent(
                        kind="reasoning" if self._mode == "reasoning" else "content",
                        text=self._buf,
                    )
                )

        elif self._mode == "tagged_json_args_stream":
            self.incomplete_tool_call_mode = self._mode

        elif self._mode in {
            "tagged_json_header",
            "tagged_json_args_seek",
            "tool_tail",
            "xml_function_header",
            "xml_param_seek",
            "xml_param_value",
        }:
            self.incomplete_tool_call_mode = self._mode
            # Do not leak unfinished protocol markup into assistant prose.
            # Tool-call deltas already streamed to clients are marked invalid by
            # the final finish_reason/warning and must not be executed.

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
        # Qwen sometimes issues a tool call from *inside* its thinking block and
        # never writes the closing think tag. A real call is the tool-open tag
        # followed by the protocol's call syntax; treat it as an implicit end of
        # reasoning so the call is not swallowed as reasoning text.
        if self._tool_open and self._supports_tool_protocol:
            tool_idx = self._buf.find(self._tool_open)
            close_idx = self._buf.find(self._think_close)
            if tool_idx >= 0 and (close_idx < 0 or tool_idx < close_idx):
                probe = self._buf[tool_idx + len(self._tool_open) :].lstrip()
                expected = (
                    self.protocol.function_open or "<function="
                    if self.protocol.tool_call_format == ToolCallFormat.QWEN_XML
                    else "{"
                )
                if probe.startswith(expected):
                    if tool_idx:
                        out.append(
                            ParsedEvent(kind="reasoning", text=self._buf[:tool_idx])
                        )
                    self._buf = self._buf[tool_idx + len(self._tool_open) :]
                    self._tool_index += 1
                    self._mode = self._native_tool_header_mode
                    return True
                if len(probe) < len(expected) and expected.startswith(probe):
                    return False  # not enough text yet to tell; wait for more
                # The tag was only mentioned in prose: keep it as reasoning.
                end = tool_idx + len(self._tool_open)
                out.append(ParsedEvent(kind="reasoning", text=self._buf[:end]))
                self._buf = self._buf[end:]
                return True

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
    # Native XML function/parameter protocol (Qwen template family)
    # ------------------------------------------------------------------

    def _find_protocol_tag(self, prefix: str) -> tuple[int, int, str] | None:
        """Find a complete ``<name=...>``-style protocol tag in the buffer."""
        start = self._buf.find(prefix)
        if start < 0:
            return None
        end = self._buf.find(">", start + len(prefix))
        if end < 0:
            return None
        value = self._buf[start + len(prefix) : end].strip()
        if not value:
            return None
        return start, end + 1, value

    def _schema_type_for(self, function_name: str, parameter_name: str) -> str | None:
        """Resolve the JSON-schema type for one native XML parameter."""
        function_schema = self._tool_schemas.get(function_name, {})
        properties = function_schema.get("properties", function_schema)
        if not isinstance(properties, dict):
            return None
        parameter_schema = properties.get(parameter_name, {})
        if not isinstance(parameter_schema, dict):
            return None
        schema_type = parameter_schema.get("type")
        if schema_type is None:
            for composition in ("anyOf", "oneOf", "allOf"):
                choices = parameter_schema.get(composition)
                if isinstance(choices, list):
                    for choice in choices:
                        if isinstance(choice, dict) and choice.get("type"):
                            schema_type = choice["type"]
                            break
                if schema_type is not None:
                    break
        if isinstance(schema_type, list):
            schema_type = next((item for item in schema_type if item != "null"), None)
        return str(schema_type) if schema_type is not None else None

    def _xml_argument_prefix(self, key: str) -> str:
        separator = "," if self._xml_args_count else "{"
        return separator + json.dumps(key, ensure_ascii=False) + ":"

    @staticmethod
    def _json_string_fragment(value: str) -> str:
        """Encode a fragment for insertion inside an already-open JSON string."""
        return json.dumps(value, ensure_ascii=False)[1:-1]

    def _consume_xml_string_value(
        self,
        raw: str,
        *,
        final: bool,
        out: list[ParserEvent],
    ) -> None:
        # Qwen's native template frames parameter values with one newline after
        # the opening tag and one before the closing tag. Strip only those
        # framing newlines; preserve any additional user data newlines.
        if self._current_param_first_chunk:
            raw = raw.removeprefix("\n")
            self._current_param_first_chunk = False
        if final:
            raw = raw.removesuffix("\n")
        if raw:
            out.append(
                ToolCallFragment(
                    index=self._tool_index,
                    arguments_fragment=self._json_string_fragment(raw),
                )
            )

    def _step_xml_function_header(self, out: list[ParserEvent]) -> bool:
        prefix = self.protocol.function_open or "<function="
        tag = self._find_protocol_tag(prefix)
        if tag is None:
            return False

        _, end, name = tag
        self._current_function_name = name
        self._current_param_key = None
        self._current_param_type = None
        self._current_param_buffer = []
        self._xml_args_count = 0

        out.append(
            ToolCallFragment(
                index=self._tool_index,
                id=_new_call_id(),
                name=name,
            )
        )
        self._buf = self._buf[end:]
        self._mode = "xml_param_seek"
        return True

    def _step_xml_param_seek(self, out: list[ParserEvent]) -> bool:
        function_close = self.protocol.function_close or "</function>"
        parameter_open = self.protocol.parameter_open or "<parameter="
        function_close_idx = self._buf.find(function_close)
        parameter_tag = self._find_protocol_tag(parameter_open)
        param_idx = parameter_tag[0] if parameter_tag else -1
        partial_parameter_idx = self._buf.find(parameter_open)

        if (
            parameter_tag is None
            and partial_parameter_idx >= 0
            and (function_close_idx < 0 or partial_parameter_idx < function_close_idx)
        ):
            # A parameter opener can be split across model-token chunks. Do not
            # mistake a later function close for a valid end-of-arguments.
            return False

        if param_idx < 0 and function_close_idx < 0:
            return False

        if param_idx >= 0 and (
            function_close_idx < 0 or param_idx < function_close_idx
        ):
            assert parameter_tag is not None
            _, end, key = parameter_tag
            self._current_param_key = key
            self._current_param_type = self._schema_type_for(
                self._current_function_name or "", key
            )
            self._current_param_is_string = self._current_param_type in {None, "string"}
            self._current_param_first_chunk = True
            self._current_param_buffer = []

            prefix = self._xml_argument_prefix(key)
            if self._current_param_is_string:
                prefix += '"'
            if prefix:
                out.append(
                    ToolCallFragment(
                        index=self._tool_index,
                        arguments_fragment=prefix,
                    )
                )
            self._buf = self._buf[end:]
            self._mode = "xml_param_value"
            return True

        if self._xml_args_count == 0:
            arguments_suffix = "{}"
        else:
            arguments_suffix = "}"
        out.append(
            ToolCallFragment(
                index=self._tool_index,
                arguments_fragment=arguments_suffix,
            )
        )
        self._buf = self._buf[function_close_idx + len(function_close) :]
        self._mode = "tool_tail"
        return True

    def _step_xml_param_value(self, out: list[ParserEvent]) -> bool:
        parameter_close = self.protocol.parameter_close or "</parameter>"
        idx = self._buf.find(parameter_close)
        if idx >= 0:
            raw = self._buf[:idx]
            self._buf = self._buf[idx + len(parameter_close) :]
            if self._current_param_is_string:
                self._consume_xml_string_value(raw, final=True, out=out)
                out.append(
                    ToolCallFragment(index=self._tool_index, arguments_fragment='"')
                )
            else:
                if self._current_param_first_chunk:
                    raw = raw.removeprefix("\n")
                    self._current_param_first_chunk = False
                raw = raw.removesuffix("\n")
                self._current_param_buffer.append(raw)
                value = _coerce_param_value(
                    "".join(self._current_param_buffer), self._current_param_type
                )
                encoded_value = json.dumps(
                    value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                out.append(
                    ToolCallFragment(
                        index=self._tool_index,
                        arguments_fragment=encoded_value,
                    )
                )
            self._xml_args_count += 1
            self._current_param_key = None
            self._current_param_type = None
            self._current_param_buffer = []
            self._mode = "xml_param_seek"
            return True

        # Keep only enough tail to recognize a closing tag split across model
        # token chunks. String values are JSON-escaped and emitted immediately;
        # non-string values are buffered until their type can be decoded.
        # Retain the closing-tag width plus one character for Qwen's framing
        # newline immediately before the closing tag.
        safe_length = max(0, len(self._buf) - len(parameter_close))
        if safe_length <= 0:
            return False
        raw = self._buf[:safe_length]
        self._buf = self._buf[safe_length:]
        if self._current_param_is_string:
            self._consume_xml_string_value(raw, final=False, out=out)
        else:
            if self._current_param_first_chunk:
                raw = raw.removeprefix("\n")
                self._current_param_first_chunk = False
            self._current_param_buffer.append(raw)
        return bool(raw)

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

from __future__ import annotations

import json
import unittest

from arc_v2_inference_runtime.engine.agentic_parsing import (
    AgenticStreamParser,
    ToolCallFragment,
)
from arc_v2_inference_runtime.engine.model_types import (
    ModelProtocol,
    ReasoningFormat,
    ToolCallFormat,
    detect_model_protocols,
)
from arc_v2_inference_runtime.engine.parsing import ParsedEvent
from arc_v2_inference_runtime.engine.grammar import build_tool_call_grammar
from arc_v2_inference_runtime.types import ChatRequest, Tool, ToolFunction


def qwen_protocol(**overrides: str) -> ModelProtocol:
    values = {
        "tool_call_format": ToolCallFormat.QWEN_XML,
        "reasoning_format": ReasoningFormat.NONE,
        "tool_open": "<tool_call>",
        "tool_close": "</tool_call>",
        "function_open": "<function=",
        "function_close": "</function>",
        "parameter_open": "<parameter=",
        "parameter_close": "</parameter>",
    }
    values.update(overrides)
    return ModelProtocol(**values)


def parse_in_chunks(text: str, parser: AgenticStreamParser, chunk_size: int = 5):
    events = []
    for offset in range(0, len(text), chunk_size):
        events.extend(parser.feed(text[offset : offset + chunk_size]))
    events.extend(parser.finalize())
    return events


def collect_call(events: list[ParsedEvent | ToolCallFragment]):
    calls: dict[int, dict[str, str | None]] = {}
    for event in events:
        if not isinstance(event, ToolCallFragment):
            continue
        call = calls.setdefault(
            event.index,
            {"id": None, "name": None, "arguments": ""},
        )
        if event.id:
            call["id"] = event.id
        if event.name:
            call["name"] = event.name
        call["arguments"] = str(call["arguments"]) + event.arguments_fragment
    return calls


class QwenXMLParserTests(unittest.TestCase):
    def test_string_arguments_are_valid_json_across_arbitrary_chunk_boundaries(self):
        payload = 'line one\nline "two" \\ path 雪 <node>\nlast line'
        native = (
            "<tool_call>\n"
            "<function=write_file>\n"
            "<parameter=path>\nsource.py\n</parameter>\n"
            "<parameter=content>\n"
            f"{payload}\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        parser = AgenticStreamParser(
            protocol=qwen_protocol(),
            tool_schemas={
                "write_file": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                }
            },
        )
        events = parse_in_chunks(native, parser, chunk_size=3)
        calls = collect_call(events)

        self.assertTrue(parser.saw_tool_call)
        self.assertIsNone(parser.incomplete_tool_call_mode)
        call = calls[0]
        self.assertEqual(call["name"], "write_file")
        args = json.loads(str(call["arguments"]))
        self.assertEqual(args, {"path": "source.py", "content": payload})

    def test_large_string_argument_is_emitted_before_parameter_closes(self):
        payload = "0123456789abcdef" * 4096
        parser = AgenticStreamParser(protocol=qwen_protocol())
        prefix = "<tool_call><function=write_file><parameter=content>\n"
        events = parser.feed(prefix)
        for offset in range(0, len(payload), 64):
            events.extend(parser.feed(payload[offset : offset + 64]))

        emitted = sum(
            len(event.arguments_fragment)
            for event in events
            if isinstance(event, ToolCallFragment)
        )
        self.assertGreater(emitted, 60_000)
        self.assertEqual(parser._mode, "xml_param_value")
        self.assertLessEqual(len(parser._buf), len("</parameter>"))

        events.extend(parser.feed("\n</parameter></function></tool_call>"))
        events.extend(parser.finalize())
        call = collect_call(events)[0]
        decoded = json.loads(str(call["arguments"]))
        self.assertEqual(decoded["content"], payload)
        self.assertTrue(parser.saw_tool_call)

    def test_schema_types_are_decoded(self):
        native = (
            "<tool_call><function=set_config>"
            "<parameter=count>\n42\n</parameter>"
            "<parameter=enabled>\ntrue\n</parameter>"
            "<parameter=threshold>\n0.125\n</parameter>"
            "<parameter=options>\n{\"mode\":\"fast\"}\n</parameter>"
            "</function></tool_call>"
        )
        parser = AgenticStreamParser(
            protocol=qwen_protocol(),
            tool_schemas={
                "set_config": {
                    "count": {"type": "integer"},
                    "enabled": {"type": "boolean"},
                    "threshold": {"type": "number"},
                    "options": {"type": "object"},
                }
            },
        )
        events = parse_in_chunks(native, parser, chunk_size=7)
        call = collect_call(events)[0]
        self.assertEqual(
            json.loads(str(call["arguments"])),
            {
                "count": 42,
                "enabled": True,
                "threshold": 0.125,
                "options": {"mode": "fast"},
            },
        )
        self.assertTrue(parser.saw_tool_call)

    def test_empty_arguments_produce_empty_object(self):
        parser = AgenticStreamParser(protocol=qwen_protocol())
        events = parse_in_chunks(
            "<tool_call><function=refresh></function></tool_call>",
            parser,
            chunk_size=2,
        )
        call = collect_call(events)[0]
        self.assertEqual(json.loads(str(call["arguments"])), {})
        self.assertTrue(parser.saw_tool_call)

    def test_truncated_parameter_is_marked_incomplete_and_not_content(self):
        parser = AgenticStreamParser(protocol=qwen_protocol())
        events = parser.feed(
            "<tool_call><function=write_file><parameter=content>\n"
            "partial source that has not closed"
        )
        events.extend(parser.finalize())

        self.assertFalse(parser.saw_tool_call)
        self.assertEqual(parser.incomplete_tool_call_mode, "xml_param_value")
        self.assertFalse(
            any(
                isinstance(event, ParsedEvent)
                and "partial source" in event.text
                for event in events
            )
        )



    def test_tagged_json_qwen_template_remains_supported(self):
        protocol = detect_model_protocols(
            '<tool_call>{"name":"lookup","arguments":{"query":"x"}}</tool_call>'
        )
        self.assertEqual(protocol.tool_call_format, ToolCallFormat.TAGGED_JSON)
        parser = AgenticStreamParser(protocol=protocol)
        events = parse_in_chunks(
            '<tool_call>{"name":"lookup","arguments":{"query":"x"}}</tool_call>',
            parser,
            chunk_size=2,
        )
        call = collect_call(events)[0]
        self.assertEqual(call["name"], "lookup")
        self.assertEqual(json.loads(str(call["arguments"])), {"query": "x"})
        self.assertTrue(parser.saw_tool_call)

    def test_partial_tool_opener_is_not_returned_as_assistant_content(self):
        parser = AgenticStreamParser(protocol=qwen_protocol())
        events = parser.feed("ordinary text <tool_ca")
        events.extend(parser.finalize())
        rendered_content = "".join(
            event.text for event in events if isinstance(event, ParsedEvent)
        )
        self.assertEqual(rendered_content, "ordinary text ")
        self.assertEqual(parser.incomplete_tool_call_mode, "tool_open")

    def test_grammar_uses_protocol_delimiters(self):
        protocol = qwen_protocol(
            function_open="<function:",
            function_close="</function>",
            parameter_open="<arg:",
            parameter_close="</arg>",
        )
        tools = [
            Tool(function=ToolFunction(name="write_file", parameters={"type": "object"}))
        ]
        grammar = build_tool_call_grammar(protocol, tools)
        self.assertIsNotNone(grammar)
        assert grammar is not None
        self.assertIn('"<function:"', grammar)
        self.assertIn('"<arg:"', grammar)
        self.assertIn('"</arg>"', grammar)

    def test_chat_request_accepts_per_request_tool_budget(self):
        request = ChatRequest(
            messages=[{"role": "user", "content": "write a file"}],
            tools=[{"function": {"name": "write_file", "parameters": {}}}],
            max_output_tokens=1024,
            max_tool_call_tokens=6144,
        )
        self.assertEqual(request.max_output_tokens, 1024)
        self.assertEqual(request.max_tool_call_tokens, 6144)

    def test_protocol_markers_come_from_model_template(self):
        protocol = detect_model_protocols(
            "<tool_call><function=tool><parameter=arg></parameter></function></tool_call>"
        )
        self.assertEqual(protocol.tool_call_format, ToolCallFormat.QWEN_XML)
        self.assertEqual(protocol.function_open, "<function=")
        self.assertEqual(protocol.parameter_close, "</parameter>")

    def test_custom_xml_markers_are_supported(self):
        protocol = qwen_protocol(
            function_open="<function:",
            function_close="</function>",
            parameter_open="<arg:",
            parameter_close="</arg>",
        )
        parser = AgenticStreamParser(protocol=protocol)
        events = parse_in_chunks(
            "<tool_call><function:write_file><arg:content>\nabc\n"
            "</arg></function></tool_call>",
            parser,
            chunk_size=4,
        )
        call = collect_call(events)[0]
        self.assertEqual(json.loads(str(call["arguments"])), {"content": "abc"})
        self.assertTrue(parser.saw_tool_call)


if __name__ == "__main__":
    unittest.main()

"""GBNF grammar construction for constraining native tool-call output.

This layers a structural constraint on top of the model's own prompt-
instructed tool-call format -- it does not invent a new format, it just
forces the model to emit *valid* instances of the format it's already being
asked to use (qwen_xml's <tool_call><function=...> or tagged_json's
<tool_call>{"name": ...}). Because it mirrors the native format rather than
fighting it, it doesn't add meaningful latency; what it buys you is zero
malformed/truncated tool calls, which is a common source of an agent
silently losing track of what it already did and re-issuing calls.

CRITICAL: when generation starts mid-reasoning (the chat template already
opened <think> before the model's turn begins -- the normal case for Qwen3
auto-thinking), the grammar MUST include an explicit path for the model to
emit the literal think-close tag before falling into the plain-text/tool-call
structure. An earlier version of this grammar omitted that phase entirely --
since plain-text forbids any '<' character except as the exact opener of
<tool_call>, a model starting mid-reasoning was structurally unable to ever
emit "</think>" and would ramble inside an unclosable reasoning block until
it hit the token/timeout limit. `needs_reasoning_close` exists specifically
to avoid repeating that bug.

Known limitation (kept deliberately simple): to keep the grammar tractable,
plain assistant prose, reasoning text, and any text preceding a tool call are
restricted to not contain a literal '<' character (other than the specific
literals this grammar defines: the tool-call opener and the think-close tag).
This is invisible for normal responses; it only bites if the model wants to
write literal HTML-like syntax or a "<" comparison in prose. Disable via
ARC_RUNTIME_GRAMMAR_CONSTRAINED_TOOLS=false if that's a problem for you.

Only qwen_xml and tagged_json are covered -- these are your two target
protocols. Other protocols return None (no grammar; falls back to the
existing prompt-only reliance).
"""

from __future__ import annotations

import json

from arc_v2_inference_runtime.engine.model_types import ModelProtocol, ToolCallFormat
from arc_v2_inference_runtime.types import Tool

# Minimal standard JSON grammar, used as the `arguments` payload for
# tagged_json tool calls (whose arguments aren't schema-typed the way the
# qwen_xml <parameter=...> tags are, so we just accept arbitrary valid JSON).
_JSON_GRAMMAR = r"""
json-value   ::= json-object | json-array | json-string | json-number | json-bool | json-null
json-object  ::= "{" ws (json-member ("," ws json-member)*)? ws "}"
json-member  ::= json-string ws ":" ws json-value
json-array   ::= "[" ws (json-value ("," ws json-value)*)? ws "]"
json-string  ::= "\"" json-char* "\""
json-char    ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
json-number  ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
json-bool    ::= "true" | "false"
json-null    ::= "null"
ws           ::= [ \t\n]*
"""


def _function_name_rule(tools: list[Tool]) -> str:
    """Build a GBNF alternation of quoted tool-name literals.

    json.dumps() on a plain identifier string produces a GBNF string literal
    (e.g. "read_file") that matches the bare, unquoted text `read_file` in
    generated output -- exactly what we want whether it's used inside
    <function=NAME> or inside a JSON "name" field.
    """
    names = [json.dumps(t.function.name) for t in tools]
    return " | ".join(names) if names else '"__no_tools_available__"'


def _body_grammar(protocol: ModelProtocol, tools: list[Tool]) -> str | None:
    """Grammar for 'the rest of the turn': optional prose, optionally
    followed by one or more well-formed tool calls. Defines `body` plus
    everything it depends on, including the shared `plain-char` rule that
    the reasoning-close phase (see build_tool_call_grammar) also reuses."""
    function_name_rule = _function_name_rule(tools)

    if protocol.tool_call_format == ToolCallFormat.QWEN_XML:
        return (
            "body          ::= plain-text | (plain-text tool-call)\n"
            "plain-text    ::= plain-char*\n"
            "plain-char    ::= [^<]\n"
            + r'tool-call     ::= "<tool_call>\n<function=" function-name ">\n" parameter* "</function>\n</tool_call>" "\n"?'
            + "\n"
            + f"function-name ::= {function_name_rule}\n"
            + r'parameter     ::= "<parameter=" param-name ">\n" param-value "\n</parameter>\n"'
            + "\n"
            "param-name    ::= plain-char+\n"
            "param-value   ::= [^<]*\n"
        )

    if protocol.tool_call_format == ToolCallFormat.TAGGED_JSON:
        return (
            "body          ::= plain-text | (plain-text tool-call)\n"
            "plain-text    ::= plain-char*\n"
            "plain-char    ::= [^<]\n"
            + r'tool-call     ::= "<tool_call>\n{\"name\": \"" function-name "\", \"arguments\": " json-value "}\n</tool_call>" "\n"?'
            + "\n"
            + f"function-name ::= {function_name_rule}\n"
            + _JSON_GRAMMAR
        )

    return None


def build_tool_call_grammar(
    protocol: ModelProtocol,
    tools: list[Tool],
    *,
    needs_reasoning_close: bool = False,
    think_close: str = "</think>",
) -> str | None:
    """Build a GBNF grammar for this request's tool list, or None if the
    protocol isn't covered (caller should skip grammar constraint and rely
    on the prompt alone).

    needs_reasoning_close must be True whenever generation starts already
    inside an open reasoning block (i.e. the engine's initial parser mode
    for this request is "reasoning" -- pass initial_mode == "reasoning" from
    the caller). Getting this wrong in the True direction when it should be
    False is harmless (adds an always-satisfiable prefix); getting it wrong
    in the False direction when it should be True reproduces the "model can
    never close </think>" bug this module exists to avoid.
    """
    if not tools:
        return None

    body = _body_grammar(protocol, tools)
    if body is None:
        return None

    if needs_reasoning_close:
        prefix = (
            "root           ::= reasoning-text think-close body\n"
            "reasoning-text ::= plain-char*\n"
            f'think-close    ::= {json.dumps(think_close)} "\\n"*\n'
        )
    else:
        prefix = "root           ::= body\n"

    return prefix + body

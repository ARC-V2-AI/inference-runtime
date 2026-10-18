"""Model-native Jinja chat template rendering."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import jinja2

from arc_v2_inference_runtime.types import ChatMessage, Tool

_FALLBACK_TEMPLATE = """\
{%- for message in messages %}
<|im_start|>{{ message.role }}
{{ message.content or '' }}<|im_end|>
{%- endfor %}
{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}
"""


def _safe_json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _raise_exception(message: str) -> None:
    raise jinja2.exceptions.TemplateError(message)


def _strftime_now(fmt: str) -> str:
    return datetime.now().strftime(fmt)


class ChatTemplateRenderer:
    def __init__(
        self, chat_template: str | None, *, bos_token: str = "", eos_token: str = ""
    ) -> None:
        self.template_source = chat_template or _FALLBACK_TEMPLATE
        self.bos_token = bos_token
        self.eos_token = eos_token
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        env.globals.update(
            raise_exception=_raise_exception,
            strftime_now=_strftime_now,
        )
        if "tojson" not in env.filters:
            env.filters["tojson"] = lambda obj, **_kwargs: json.dumps(
                obj, ensure_ascii=False
            )
        self._template = env.from_string(self.template_source)
        self._using_fallback = chat_template is None

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback

    def render(
        self,
        messages: list[ChatMessage],
        tools: list[Tool] | None = None,
        *,
        add_generation_prompt: bool = True,
        template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        rendered_messages: list[dict[str, Any]] = []
        for m in messages:
            msg: dict[str, Any] = {
                "role": m.role,
                "content": m.content or "",
                "name": m.name,
                "tool_call_id": m.tool_call_id,
                "reasoning_content": m.reasoning_content,
            }
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": _safe_json_loads(tc.function.arguments),
                        },
                    }
                    for tc in m.tool_calls
                ]
            rendered_messages.append(msg)

        rendered_tools = None
        if tools:
            rendered_tools = [
                {
                    "type": t.type,
                    "function": {
                        "name": t.function.name,
                        "description": t.function.description,
                        "parameters": t.function.parameters,
                    },
                }
                for t in tools
            ]

        context: dict[str, Any] = {
            "messages": rendered_messages,
            "tools": rendered_tools,
            "add_generation_prompt": add_generation_prompt,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "chat_template_kwargs": {},
        }
        if template_kwargs:
            # Keep both flattened variables (Qwen-style) and the aggregate
            # chat_template_kwargs object (llama.cpp-style) available.
            context.update(template_kwargs)
            if "chat_template_kwargs" not in template_kwargs:
                context["chat_template_kwargs"] = {
                    k: v for k, v in template_kwargs.items() if not isinstance(v, dict)
                }

        return self._template.render(**context)

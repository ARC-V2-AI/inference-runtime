"""Token-aware context budgeting for ARC's rendered chat prompts.

The engine owns rendering and tokenization. This module stays independent from
llama.cpp so the pruning policy can be tested with a deterministic fake renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from arc_v2_inference_runtime.types import ChatMessage


class ContextLengthError(ValueError):
    """Raised when a prompt cannot fit without discarding the newest turn."""


@dataclass(frozen=True, slots=True)
class ContextFit:
    messages: list[ChatMessage]
    prompt: str
    prompt_tokens: int
    output_tokens: int
    context_window: int
    safety_margin_tokens: int
    dropped_turns: int

    @property
    def total_reserved_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens + self.safety_margin_tokens

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.context_window - self.total_reserved_tokens)


Render = Callable[[list[ChatMessage]], str]
Count = Callable[[str], int]


def _split_system_and_turns(messages: Sequence[ChatMessage]) -> tuple[list[ChatMessage], list[list[ChatMessage]]]:
    """Keep system messages and split the rest at user-message boundaries.

    A turn includes the user message and all assistant/tool messages that follow
    it, including a tool-call cycle. A prefix before the first user message is
    attached to the first turn rather than being silently discarded.
    """
    system_messages = [message for message in messages if message.role == "system"]
    conversation = [message for message in messages if message.role != "system"]
    if not conversation:
        return system_messages, []

    turns: list[list[ChatMessage]] = []
    current: list[ChatMessage] = []
    saw_user = False
    for message in conversation:
        if message.role == "user" and saw_user and current:
            turns.append(current)
            current = []
        if message.role == "user":
            saw_user = True
        current.append(message)
    if current:
        turns.append(current)

    return system_messages, turns


def fit_chat_context(
    messages: Sequence[ChatMessage],
    *,
    render: Render,
    count_tokens: Count,
    context_window: int,
    output_tokens: int,
    safety_margin_tokens: int = 16,
) -> ContextFit:
    """Drop oldest complete user turns until prompt + output fit.

    System messages and the newest user turn are never discarded. If that
    minimum viable prompt is still too large, raise ContextLengthError and let
    the caller return a useful client error rather than silently truncating it.
    """
    if context_window < 1:
        raise ValueError("context_window must be positive")
    if output_tokens < 1:
        raise ValueError("output_tokens must be positive")
    if safety_margin_tokens < 0:
        raise ValueError("safety_margin_tokens cannot be negative")
    if output_tokens + safety_margin_tokens >= context_window:
        raise ContextLengthError(
            f"Reserved output ({output_tokens}) plus safety margin "
            f"({safety_margin_tokens}) leaves no usable prompt context in a "
            f"{context_window}-token window."
        )

    system_messages, turns = _split_system_and_turns(messages)
    dropped_turns = 0

    while True:
        kept = [*system_messages, *(message for turn in turns for message in turn)]
        prompt = render(kept)
        prompt_tokens = count_tokens(prompt)
        if prompt_tokens + output_tokens + safety_margin_tokens <= context_window:
            return ContextFit(
                messages=kept,
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                context_window=context_window,
                safety_margin_tokens=safety_margin_tokens,
                dropped_turns=dropped_turns,
            )

        if len(turns) <= 1:
            raise ContextLengthError(
                "The system instructions and newest conversation turn do not fit "
                f"the configured context window: prompt={prompt_tokens}, "
                f"output_reserved={output_tokens}, safety_margin={safety_margin_tokens}, "
                f"context_window={context_window}. Shorten the current input or "
                "increase ARC_RUNTIME_N_CTX."
            )

        turns.pop(0)
        dropped_turns += 1


def validate_prompt_fits(
    prompt_tokens: int,
    *,
    context_window: int,
    output_tokens: int,
    safety_margin_tokens: int = 16,
) -> None:
    """Validate a plain completion prompt without rewriting user input."""
    reserved = prompt_tokens + output_tokens + safety_margin_tokens
    if reserved > context_window:
        raise ContextLengthError(
            f"Prompt exceeds the context budget: prompt={prompt_tokens}, "
            f"output_reserved={output_tokens}, safety_margin={safety_margin_tokens}, "
            f"context_window={context_window}. Shorten the prompt, reduce "
            "max_output_tokens, or increase ARC_RUNTIME_N_CTX."
        )

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EventKind = Literal["reasoning", "content"]


@dataclass
class ParsedEvent:
    kind: EventKind
    text: str = ""


class StreamParser:
    def __init__(
        self,
        think_open: str = "<think>",
        think_close: str = "</think>",
    ) -> None:
        self._think_open = think_open
        self._think_close = think_close
        self._max_tag_len = max(len(think_open), len(think_close))

        self._buf = ""
        self._mode: Literal["content", "reasoning"] = "content"

    def feed(self, chunk: str) -> list[ParsedEvent]:
        self._buf += chunk
        events: list[ParsedEvent] = []

        while True:
            if self._mode == "content":
                idx = self._buf.find(self._think_open)
                if idx == -1:
                    # Emit everything except a tail that might be a partial tag opener.
                    safe_len = max(0, len(self._buf) - self._max_tag_len)
                    if safe_len > 0:
                        events.append(
                            ParsedEvent(kind="content", text=self._buf[:safe_len])
                        )
                        self._buf = self._buf[safe_len:]
                    break
                if idx > 0:
                    events.append(ParsedEvent(kind="content", text=self._buf[:idx]))
                self._buf = self._buf[idx + len(self._think_open) :]
                self._mode = "reasoning"
                continue

            if self._mode == "reasoning":
                idx = self._buf.find(self._think_close)
                if idx == -1:
                    safe_len = max(0, len(self._buf) - self._max_tag_len)
                    if safe_len > 0:
                        events.append(
                            ParsedEvent(kind="reasoning", text=self._buf[:safe_len])
                        )
                        self._buf = self._buf[safe_len:]
                    break
                if idx > 0:
                    events.append(ParsedEvent(kind="reasoning", text=self._buf[:idx]))
                self._buf = self._buf[idx + len(self._think_close) :]
                self._mode = "content"
                continue

        return events

    def finalize(self) -> list[ParsedEvent]:
        """Flush whatever remains in the buffer at end-of-stream."""
        events: list[ParsedEvent] = []
        if self._buf:
            events.append(ParsedEvent(kind=self._mode, text=self._buf))
        self._buf = ""
        return events

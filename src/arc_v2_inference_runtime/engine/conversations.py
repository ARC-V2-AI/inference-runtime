"""SQLite-backed conversation storage, separate from the model KV cache."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arc_v2_inference_runtime.types import ChatMessage


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ConversationNotFoundError(KeyError):
    """Requested conversation does not exist."""


class ConversationStore:
    """Small local SQLite store for persistent chat transcripts.

    Each operation owns its own connection, which works with FastAPI worker
    threads and avoids sharing sqlite3 connection objects across threads.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id          TEXT PRIMARY KEY,
                    title       TEXT,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    ordinal         INTEGER NOT NULL,
                    role            TEXT NOT NULL,
                    message_json    TEXT NOT NULL,
                    created_at      TEXT NOT NULL,
                    UNIQUE(conversation_id, ordinal)
                );

                CREATE INDEX IF NOT EXISTS idx_conversation_messages_order
                    ON conversation_messages(conversation_id, ordinal);
                """
            )

    def create(self, title: str | None = None) -> dict[str, str | None]:
        conversation_id = uuid.uuid4().hex
        now = _now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conversation_id, title, now, now),
            )
        return {
            "id": conversation_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
        }

    def exists(self, conversation_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        return row is not None

    def get(self, conversation_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            raise ConversationNotFoundError(conversation_id)
        return dict(row)

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT c.id, c.title, c.created_at, c.updated_at, "
                "COUNT(m.id) AS message_count "
                "FROM conversations c LEFT JOIN conversation_messages m "
                "ON m.conversation_id = c.id "
                "GROUP BY c.id ORDER BY c.updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_messages(self, conversation_id: str) -> list[ChatMessage]:
        if not self.exists(conversation_id):
            raise ConversationNotFoundError(conversation_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT message_json FROM conversation_messages "
                "WHERE conversation_id = ? ORDER BY ordinal",
                (conversation_id,),
            ).fetchall()
        return [ChatMessage.model_validate(json.loads(row["message_json"])) for row in rows]

    def append_messages(self, conversation_id: str, messages: list[ChatMessage]) -> None:
        if not messages:
            return
        now = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if exists is None:
                connection.rollback()
                raise ConversationNotFoundError(conversation_id)
            ordinal_row = connection.execute(
                "SELECT COALESCE(MAX(ordinal), -1) + 1 AS next_ordinal "
                "FROM conversation_messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            ordinal = int(ordinal_row["next_ordinal"])
            for message in messages:
                payload = message.model_dump(mode="json", exclude_none=True)
                connection.execute(
                    "INSERT INTO conversation_messages "
                    "(conversation_id, ordinal, role, message_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        conversation_id,
                        ordinal,
                        message.role,
                        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                        now,
                    ),
                )
                ordinal += 1
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            connection.commit()

    def delete(self, conversation_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
            return cursor.rowcount > 0

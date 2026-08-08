"""
Lightweight SQLite persistence for conversation logging and feedback.

Kept intentionally simple (stdlib sqlite3, no ORM) so the project has zero
extra infrastructure to run locally, similar in spirit to the reference
fitness-assistant project's Postgres logging, just swapped for a
zero-setup embedded database.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from app.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    document_id TEXT,
    grounded INTEGER NOT NULL,
    citations_json TEXT NOT NULL,
    retrieval_debug_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    feedback INTEGER NOT NULL,
    comment TEXT,
    created_at TEXT NOT NULL
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def save_conversation(
    conversation_id: str,
    question: str,
    answer: str,
    document_id: Optional[str],
    grounded: bool,
    citations: list,
    retrieval_debug: Optional[dict],
):
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO conversations
               (conversation_id, question, answer, document_id, grounded,
                citations_json, retrieval_debug_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                conversation_id,
                question,
                answer,
                document_id,
                int(grounded),
                json.dumps(citations),
                json.dumps(retrieval_debug) if retrieval_debug else None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def save_feedback(conversation_id: str, feedback: int, comment: Optional[str]):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO feedback (conversation_id, feedback, comment, created_at)
               VALUES (?, ?, ?, ?)""",
            (conversation_id, feedback, comment, datetime.now(timezone.utc).isoformat()),
        )

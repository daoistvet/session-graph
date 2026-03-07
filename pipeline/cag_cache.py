"""
SQLite cache for session-scoped CAG triples.

Stores the most recently extracted CAG triples per session so that
incremental extraction can see prior state and produce a complete,
refined set each time.
"""

import json
import sqlite3
from pathlib import Path


_CACHE_DIR = Path(__file__).parent / "cache"
_CACHE_DIR.mkdir(exist_ok=True)
_CACHE_PATH = _CACHE_DIR / "cag_cache.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_CACHE_PATH))
    conn.execute("""CREATE TABLE IF NOT EXISTS cag_cache (
        session_id TEXT PRIMARY KEY,
        triples_json TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    return conn


def get_cached_cag(session_id: str) -> list[dict] | None:
    """Return cached CAG triples for a session, or None on cache miss."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT triples_json FROM cag_cache WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


def cache_cag(session_id: str, triples: list[dict]) -> None:
    """Store extracted CAG triples, keyed by session_id (INSERT OR REPLACE)."""
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO cag_cache (session_id, triples_json, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
        (session_id, json.dumps(triples)),
    )
    conn.commit()
    conn.close()

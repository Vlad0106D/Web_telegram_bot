from __future__ import annotations

import os
import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Sequence

import psycopg
from psycopg_pool import ConnectionPool

log = logging.getLogger("mm_v2.db")


def _db_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty. Set it in environment (Neon connection string).")
    return url


# Single shared pool for MM v2.
# Important: keep it local to mm_v2 and do not reuse old project DB code.
_POOL: Optional[ConnectionPool] = None


def get_pool() -> ConnectionPool:
    global _POOL
    if _POOL is not None:
        return _POOL

    # Neon обычно требует sslmode=require (как правило уже в DATABASE_URL).
    # open=False -> соединения создаются лениво при первом запросе.
    _POOL = ConnectionPool(
        conninfo=_db_url(),
        min_size=int(os.getenv("MM_DB_POOL_MIN", "1")),
        max_size=int(os.getenv("MM_DB_POOL_MAX", "5")),
        open=False,
        kwargs={
            # autocommit False по умолчанию; управляем транзакциями через context
        },
    )
    return _POOL


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    """
    Context manager returning a pooled psycopg connection.

    Usage:
      with get_conn() as conn:
          conn.execute(...)
    """
    pool = get_pool()
    with pool.connection() as conn:
        yield conn


def close_pool() -> None:
    """Close the global pool (optional on shutdown)."""
    global _POOL
    if _POOL is not None:
        try:
            _POOL.close()
        finally:
            _POOL = None


def ping() -> bool:
    """
    Simple health-check to validate DATABASE_URL and pool connectivity.
    Returns True if SELECT 1 succeeded.
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        return True
    except Exception:
        log.exception("DB ping failed")
        return False


def execute(sql: str, params: Sequence[Any] | None = None) -> None:
    """Execute a statement (no results)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()


def fetch_one(sql: str, params: Sequence[Any] | None = None) -> Optional[tuple]:
    """Fetch a single row."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        return row


def fetch_all(sql: str, params: Sequence[Any] | None = None) -> list[tuple]:
    """Fetch all rows."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return rows
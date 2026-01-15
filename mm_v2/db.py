from __future__ import annotations

import os
import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Sequence
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import psycopg
from psycopg_pool import ConnectionPool

log = logging.getLogger("mm_v2.db")


def _mask_db_url(url: str) -> str:
    # скрываем пароль в логах
    try:
        u = urlparse(url)
        if u.password:
            netloc = u.netloc.replace(u.password, "***")
            return urlunparse((u.scheme, netloc, u.path, u.params, u.query, u.fragment))
        return url
    except Exception:
        return "<unparsable>"


def _ensure_sslmode_require(url: str) -> str:
    """
    Neon почти всегда требует SSL.
    Если sslmode не указан — добавим sslmode=require.
    """
    u = urlparse(url)
    qs = dict(parse_qsl(u.query, keep_blank_values=True))
    if "sslmode" not in qs:
        qs["sslmode"] = "require"
        new_query = urlencode(qs)
        u = u._replace(query=new_query)
        return urlunparse(u)
    return url


def _db_url() -> str:
    raw = os.getenv("DATABASE_URL", "").strip()
    if not raw:
        raise RuntimeError("DATABASE_URL is empty. Set it in Render Environment (Neon connection string).")
    fixed = _ensure_sslmode_require(raw)
    return fixed


# Single shared pool for MM v2
_POOL: Optional[ConnectionPool] = None


def get_pool() -> ConnectionPool:
    global _POOL
    if _POOL is not None:
        return _POOL

    url = _db_url()
    log.info("DB url (masked): %s", _mask_db_url(url))

    _POOL = ConnectionPool(
        conninfo=url,
        min_size=int(os.getenv("MM_DB_POOL_MIN", "1")),
        max_size=int(os.getenv("MM_DB_POOL_MAX", "5")),
        open=False,  # lazy open
        kwargs={
            # полезно для Neon, чтобы быстрее падало и было видно проблему
            "connect_timeout": int(os.getenv("MM_DB_CONNECT_TIMEOUT", "8")),
        },
    )
    return _POOL


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    pool = get_pool()
    with pool.connection() as conn:
        yield conn


def close_pool() -> None:
    global _POOL
    if _POOL is not None:
        try:
            _POOL.close()
        finally:
            _POOL = None


def ping() -> bool:
    """
    Health-check. Логируем точную ошибку подключения/авторизации/SSL.
    """
    try:
        url = os.getenv("DATABASE_URL", "").strip()
        if not url:
            log.error("DB ping failed: DATABASE_URL is missing/empty in environment")
            return False

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()

        return True

    except Exception as e:
        log.exception("DB ping failed: %r", e)
        return False


def execute(sql: str, params: Sequence[Any] | None = None) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()


def fetch_one(sql: str, params: Sequence[Any] | None = None) -> Optional[tuple]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        return row


def fetch_all(sql: str, params: Sequence[Any] | None = None) -> list[tuple]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return rows
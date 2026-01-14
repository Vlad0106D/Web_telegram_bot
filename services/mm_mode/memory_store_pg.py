from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional, Tuple, List

from psycopg_pool import AsyncConnectionPool

log = logging.getLogger(__name__)

_POOL: Optional[AsyncConnectionPool] = None
_POOL_LOCK = asyncio.Lock()

# сколько раз пробуем повторить запись при сетевых/SSL обрывах
_MAX_RETRIES = 2


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _get_dsn() -> str:
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set")

    # DATABASE_URL должен быть URL (postgresql://...), без "psql '...'"
    dsn = dsn.strip().strip("'").strip('"')
    if dsn.lower().startswith("psql "):
        raise RuntimeError("DATABASE_URL looks like a psql command. Put only the postgresql://... URL")

    return dsn


def _to_jsonable(obj: Any) -> Any:
    """Приводит объект к JSON-совместимому виду (dict/list/str/int/float/bool/None)."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, datetime):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x) for x in obj]

    # pydantic v2
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return _to_jsonable(obj.model_dump())
        except Exception:
            pass

    # pydantic v1 / обычные .dict()
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return _to_jsonable(obj.dict())
        except Exception:
            pass

    # dataclass
    if dataclasses.is_dataclass(obj):
        try:
            return _to_jsonable(dataclasses.asdict(obj))
        except Exception:
            pass

    # fallback: __dict__
    if hasattr(obj, "__dict__"):
        try:
            d = {k: v for k, v in obj.__dict__.items() if not str(k).startswith("_")}
            return _to_jsonable(d)
        except Exception:
            pass

    return str(obj)


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _as_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _as_text(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        return str(v)
    except Exception:
        return None


def _tf_from_source_mode(source_mode: str) -> Optional[str]:
    """
    Маппинг source_mode -> tf (единый формат для БД: 1h/4h/1d/1w).
    """
    sm = (source_mode or "").lower()
    if sm.startswith("h1_") or sm == "1h":
        return "1h"
    if sm.startswith("h4_") or sm == "4h":
        return "4h"
    if sm.startswith("daily_") or sm in ("1d", "d", "day"):
        return "1d"
    if sm.startswith("weekly_") or sm in ("1w", "w", "week"):
        return "1w"
    return None


def _is_transient_db_error(e: Exception) -> bool:
    """Ловим сетевые/SSL/обрыв соединения."""
    msg = (str(e) or "").lower()
    return any(
        s in msg
        for s in [
            "ssl connection has been closed unexpectedly",
            "server closed the connection unexpectedly",
            "terminating connection",
            "connection reset by peer",
            "connection refused",
            "network is unreachable",
            "connection timed out",
            "broken pipe",
            "eof detected",
            "consuming input failed",
        ]
    )


async def _close_pool() -> None:
    global _POOL
    if _POOL is None:
        return
    try:
        await _POOL.close()
    except Exception:
        pass
    _POOL = None


async def _get_pool() -> AsyncConnectionPool:
    """Ленивая инициализация пула. Если пул умер — пересоздаём."""
    global _POOL

    async with _POOL_LOCK:
        if _POOL is not None:
            return _POOL

        dsn = _get_dsn()

        _POOL = AsyncConnectionPool(
            conninfo=dsn,
            min_size=1,
            max_size=3,
            timeout=10,
            open=False,
        )
        await _POOL.open()
        return _POOL


def _split_symbols(symbols: str) -> List[str]:
    out: List[str] = []
    for s in (symbols or "").split(","):
        s = s.strip()
        if s:
            out.append(s)
    return out or ["BTCUSDT"]


def _pick_exchange_okx_only(snap: Any) -> str:
    """
    Жёсткое правило проекта: все price/OI берём только с OKX.
    exchange в БД фиксируем как 'okx'.
    """
    ex = _as_text(_safe_get(snap, "exchange"))
    if ex:
        ex = ex.strip().lower()
        # мягкий режим: если кто-то передал не okx — всё равно пишем okx
        # (если хочешь жёстко падать — скажи, сделаю raise)
    return "okx"


def _pick_oi_source_okx_only(snap: Any) -> str:
    src = _as_text(_safe_get(snap, "oi_source"))
    if src:
        src = src.strip().lower()
    return "okx"


def _extract_ohlcv(obj: Any) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Пытаемся извлечь OHLCV из разных возможных форматов.
    Допускаем варианты ключей: open/high/low/close/volume или o/h/l/c/v.
    Если есть только price -> кладём её в open/high/low/close (но лучше, чтобы реальный OHLC приходил из market_data).
    """
    if obj is None:
        return None, None, None, None, None

    if isinstance(obj, dict):
        o = _as_float(obj.get("open", obj.get("o")))
        h = _as_float(obj.get("high", obj.get("h")))
        l = _as_float(obj.get("low", obj.get("l")))
        c = _as_float(obj.get("close", obj.get("c")))
        v = _as_float(obj.get("volume", obj.get("v")))

        if o is None and c is None:
            p = _as_float(obj.get("price"))
            if p is not None:
                o = h = l = c = p

        return o, h, l, c, v

    o = _as_float(_safe_get(obj, "open", _safe_get(obj, "o")))
    h = _as_float(_safe_get(obj, "high", _safe_get(obj, "h")))
    l = _as_float(_safe_get(obj, "low", _safe_get(obj, "l")))
    c = _as_float(_safe_get(obj, "close", _safe_get(obj, "c")))
    v = _as_float(_safe_get(obj, "volume", _safe_get(obj, "v")))

    if o is None and c is None:
        p = _as_float(_safe_get(obj, "price"))
        if p is not None:
            o = h = l = c = p

    return o, h, l, c, v


def _extract_oi(obj: Any) -> Tuple[Optional[float], Optional[float]]:
    """Извлекаем open_interest / open_interest_usd, если есть."""
    if obj is None:
        return None, None

    if isinstance(obj, dict):
        oi = _as_float(obj.get("open_interest", obj.get("oi")))
        oi_usd = _as_float(obj.get("open_interest_usd", obj.get("oi_usd")))
        return oi, oi_usd

    oi = _as_float(_safe_get(obj, "open_interest", _safe_get(obj, "oi")))
    oi_usd = _as_float(_safe_get(obj, "open_interest_usd", _safe_get(obj, "oi_usd")))
    return oi, oi_usd


def _extract_regime(snap: Any) -> Tuple[Optional[str], Optional[int], Optional[float], Optional[str]]:
    """
    regime: market_regime ('trend_up'|'range'|'trend_down'),
    trend_dir (-1|0|1),
    trend_strength (float),
    regime_source (text)
    """
    market_regime = _as_text(_safe_get(snap, "market_regime"))
    trend_dir = _as_int(_safe_get(snap, "trend_dir"))
    trend_strength = _as_float(_safe_get(snap, "trend_strength"))
    regime_source = _as_text(_safe_get(snap, "regime_source"))

    if market_regime is None:
        market_regime = _as_text(_safe_get(_safe_get(snap, "regime"), "market_regime"))

    return market_regime, trend_dir, trend_strength, regime_source


def _subsnap_for_symbol(snap: Any, symbol: str) -> Any:
    """
    Пытаемся взять "подснап" по конкретному активу:
    BTCUSDT -> snap['btc'] / snap.btc
    ETHUSDT -> snap['eth'] / snap.eth
    иначе возвращаем исходный snap.
    """
    s = (symbol or "").upper()
    if s.startswith("BTC"):
        sub = _safe_get(snap, "btc")
        return sub if sub is not None else snap
    if s.startswith("ETH"):
        sub = _safe_get(snap, "eth")
        return sub if sub is not None else snap
    return snap


async def append_snapshot(
    *,
    snap: Any,
    source_mode: str,
    symbols: str = "BTCUSDT,ETHUSDT",
    ts_utc: Optional[datetime] = None,
) -> None:
    """
    Outcomes 2.0:
    Пишем снапшоты в public.mm_snapshots (НОВАЯ СХЕМА):
      - 1 строка = 1 symbol/exchange/timeframe/ts
      - OHLC обязателен
      - OI/Regime/Features — как факты для будущих расчётов
    ВАЖНО: outcomes/расчёты НЕ делают запросы к OKX — только по БД.
    """
    snap_now = _safe_get(snap, "now_dt")
    if ts_utc is None and isinstance(snap_now, datetime):
        ts_utc = snap_now
    ts_utc = ts_utc or _now_utc()

    timeframe = _as_text(_safe_get(snap, "tf")) or _tf_from_source_mode(source_mode) or "1h"

    exchange = _pick_exchange_okx_only(snap)
    oi_source = _pick_oi_source_okx_only(snap)

    market_regime, trend_dir, trend_strength, regime_source = _extract_regime(snap)

    # базовые features: кладём весь снап как jsonb (можно потом сузить)
    base_features = _to_jsonable(snap)
    if isinstance(base_features, dict):
        base_features.setdefault("tf", timeframe)
        base_features.setdefault("exchange", exchange)

    last_err: Optional[Exception] = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            pool = await _get_pool()

            async with pool.connection() as conn:
                async with conn.transaction():
                    for symbol in _split_symbols(symbols):
                        sub = _subsnap_for_symbol(snap, symbol)

                        o, h, l, c, v = _extract_ohlcv(sub)
                        if o is None or h is None or l is None or c is None:
                            log.warning(
                                "MM memory: skip snapshot (missing OHLC) symbol=%s tf=%s source_mode=%s",
                                symbol, timeframe, source_mode,
                            )
                            continue

                        oi, oi_usd = _extract_oi(sub)

                        feat = base_features
                        if isinstance(feat, dict):
                            feat = dict(feat)
                            feat["symbol"] = symbol
                            feat["source_mode"] = source_mode
                            if isinstance(sub, dict):
                                feat["symbol_block"] = _to_jsonable(sub)

                        features_json = json.dumps(_to_jsonable(feat), ensure_ascii=False)

                        await conn.execute(
                            """
                            INSERT INTO public.mm_snapshots (
                              ts, symbol, exchange, timeframe,
                              open, high, low, close, volume,
                              open_interest, open_interest_usd, oi_source,
                              market_regime, trend_dir, trend_strength, regime_source,
                              features
                            )
                            VALUES (
                              %s, %s, %s, %s,
                              %s, %s, %s, %s, %s,
                              %s, %s, %s,
                              %s, %s, %s, %s,
                              %s::jsonb
                            )
                            ON CONFLICT (symbol, exchange, timeframe, ts)
                            DO UPDATE SET
                              open = EXCLUDED.open,
                              high = EXCLUDED.high,
                              low  = EXCLUDED.low,
                              close = EXCLUDED.close,
                              volume = EXCLUDED.volume,
                              open_interest = EXCLUDED.open_interest,
                              open_interest_usd = EXCLUDED.open_interest_usd,
                              oi_source = EXCLUDED.oi_source,
                              market_regime = EXCLUDED.market_regime,
                              trend_dir = EXCLUDED.trend_dir,
                              trend_strength = EXCLUDED.trend_strength,
                              regime_source = EXCLUDED.regime_source,
                              features = EXCLUDED.features
                            """,
                            (
                                ts_utc, symbol, exchange, timeframe,
                                o, h, l, c, v,
                                oi, oi_usd, oi_source,
                                market_regime, trend_dir, trend_strength, regime_source,
                                features_json,
                            ),
                        )

            return

        except Exception as e:
            last_err = e

            if _is_transient_db_error(e) and attempt < _MAX_RETRIES:
                log.warning("MM memory: transient DB error, retry %s/%s: %s", attempt + 1, _MAX_RETRIES, e)
                async with _POOL_LOCK:
                    await _close_pool()
                await asyncio.sleep(0.6 * (attempt + 1))
                continue

            break

    log.exception("MM memory: append_snapshot failed (%s)", source_mode, exc_info=last_err)
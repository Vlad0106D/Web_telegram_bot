from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional

import httpx
import pandas as pd

log = logging.getLogger(__name__)

# ---- OKX helpers ----
_TF_OKX = {"1h": "1H", "4h": "4H", "1d": "1D"}
_TF_SEC = {"1h": 3600, "4h": 14400, "1d": 86400}

def _sym_okx(symbol: str) -> str:
    s = symbol.upper().replace("_", "").replace("-", "")
    if s.endswith("USDT"):
        return s[:-4] + "-USDT"
    return s

def _df_from_okx(arr) -> pd.DataFrame:
    # OKX: [ts, o, h, l, c, vol, volCcy, ...]
    rows = []
    for it in arr:
        ts, o, h, l, c, vol, *_ = it
        rows.append([int(ts), float(o), float(h), float(l), float(c), float(vol)])
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

async def _okx_history_candles(
    inst_id: str,
    tf: str,
    *,
    before_ms: int,
    limit: int = 100,
) -> pd.DataFrame:
    """
    Пытаемся получить свечи "до" before_ms через history endpoint.
    OKX иногда по-разному трактует before/after у разных ручек/версий,
    поэтому делаем fallback: если before не дал данных — пробуем after.
    """
    bar = _TF_OKX.get(tf)
    if not bar:
        raise ValueError(f"tf not supported for OKX: {tf}")

    url = "https://www.okx.com/api/v5/market/history-candles"
    params_before = {"instId": inst_id, "bar": bar, "limit": str(limit), "before": str(before_ms)}
    params_after  = {"instId": inst_id, "bar": bar, "limit": str(limit), "after": str(before_ms)}

    async with httpx.AsyncClient(timeout=12) as client:
        # 1) try before
        r = await client.get(url, params=params_before)
        r.raise_for_status()
        j = r.json()
        arr = j.get("data") or []
        if arr:
            return _df_from_okx(arr)

        # 2) fallback after
        r2 = await client.get(url, params=params_after)
        r2.raise_for_status()
        j2 = r2.json()
        arr2 = j2.get("data") or []
        if arr2:
            return _df_from_okx(arr2)

    return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

async def _get_window_okx(
    symbol: str,
    tf: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """
    Гарантированно стараемся покрыть [start_ms; end_ms] историей.
    Тянем пачками по 100 свечей назад, пока не закроем start_ms.
    """
    inst = _sym_okx(symbol)
    sec = _TF_SEC[tf]
    step_ms = sec * 1000

    # хотим минимум столько свечей, сколько нужно по окну
    need = int((end_ms - start_ms) // step_ms) + 5
    # OKX history обычно max 100 за раз, поэтому пагинация
    pages = max(1, min(30, (need // 90) + 1))

    dfs = []
    cursor = end_ms
    oldest = None

    for _ in range(pages):
        df = await _okx_history_candles(inst, tf, before_ms=cursor, limit=100)
        if df is None or df.empty:
            break

        dfs.append(df)
        oldest = int(df["time"].iloc[0])

        # если уже дошли до старта — хватит
        if oldest <= start_ms:
            break

        # двигаем курсор ещё назад
        cursor = oldest - 1

    if not dfs:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    out = pd.concat(dfs, ignore_index=True)
    out.drop_duplicates(subset=["time"], inplace=True)
    out.sort_values("time", inplace=True)
    out.reset_index(drop=True, inplace=True)

    # отфильтруем с запасом (чтобы price0 можно было взять "предыдущую")
    pad = 2 * step_ms
    out = out[(out["time"] >= start_ms - pad) & (out["time"] <= end_ms)]
    out.reset_index(drop=True, inplace=True)
    return out

def _floor_to_tf(ms: int, tf: str) -> int:
    sec = _TF_SEC[tf]
    step = sec * 1000
    return ms - (ms % step)

def _pick_price0(df: pd.DataFrame, event_ms: int, tf: str) -> Optional[float]:
    """
    price0 берём как close свечи, которая "содержит" event_ts (floor по TF).
    Если exact нет — берём последнюю свечу ДО этой точки.
    """
    if df is None or df.empty:
        return None
    t0 = _floor_to_tf(event_ms, tf)
    exact = df[df["time"] == t0]
    if not exact.empty:
        return float(exact["close"].iloc[0])

    before = df[df["time"] < t0]
    if before.empty:
        return None
    return float(before["close"].iloc[-1])

def _max_high_min_low(df: pd.DataFrame, start_ms: int, end_ms: int) -> Tuple[Optional[float], Optional[float], int]:
    if df is None or df.empty:
        return None, None, 0
    w = df[(df["time"] > start_ms) & (df["time"] <= end_ms)]
    if w.empty:
        return None, None, 0
    return float(w["high"].max()), float(w["low"].min()), int(len(w))

def _close_at_or_before(df: pd.DataFrame, t_ms: int) -> Optional[float]:
    if df is None or df.empty:
        return None
    w = df[df["time"] <= t_ms]
    if w.empty:
        return None
    return float(w["close"].iloc[-1])

def _safe_pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or a == 0:
        return None
    return (b - a) / a

async def calc_event_outcomes(
    *,
    symbol: str,
    event_ts_utc: datetime,
    tf_for_calc: str = "1h",
) -> Dict[str, Tuple[Optional[float], Optional[float], Optional[float], str]]:
    """
    Возвращает dict:
      horizon -> (max_up_pct, max_down_pct, close_pct, outcome_type)

    outcome_type:
      ok | error
    """
    if event_ts_utc.tzinfo is None:
        event_ts_utc = event_ts_utc.replace(tzinfo=timezone.utc)

    tf = tf_for_calc
    if tf not in _TF_SEC:
        raise ValueError(f"Unsupported tf_for_calc: {tf}")

    event_ms = int(event_ts_utc.timestamp() * 1000)

    horizons = {"1h": 3600, "4h": 14400, "1d": 86400}
    out: Dict[str, Tuple[Optional[float], Optional[float], Optional[float], str]] = {}

    # Чтобы одним запросом покрыть все горизонты:
    end_ms = event_ms + max(horizons.values()) * 1000
    start_ms = event_ms - 2 * _TF_SEC[tf] * 1000  # запас, чтобы price0 нашёлся

    try:
        df = await _get_window_okx(symbol, tf, start_ms, end_ms)
        price0 = _pick_price0(df, event_ms, tf)

        if price0 is None:
            # НЕ падаем — отдадим error, watcher запишет outcome и забудет этот event
            for h in horizons.keys():
                out[h] = (None, None, None, "error")
            return out

        for h, sec in horizons.items():
            target_ms = event_ms + sec * 1000
            hi, lo, _n = _max_high_min_low(df, event_ms, target_ms)
            close_t = _close_at_or_before(df, target_ms)

            # метрики
            close_pct = _safe_pct(price0, close_t)
            mfe = _safe_pct(price0, hi)          # max favorable
            mae = _safe_pct(price0, lo)          # max adverse (обычно <= 0)

            # max_up/max_down в терминах outcomes-таблицы:
            max_up = mfe
            max_down = mae

            out[h] = (max_up, max_down, close_pct, "ok")

        return out

    except Exception as e:
        log.exception("calc_event_outcomes failed: %s %s", symbol, event_ts_utc.isoformat())
        for h in horizons.keys():
            out[h] = (None, None, None, "error")
        return out
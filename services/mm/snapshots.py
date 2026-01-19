# services/mm/snapshots.py
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import httpx
import psycopg
from psycopg.types.json import Jsonb


SYMBOLS = ["BTC-USDT", "ETH-USDT"]
TFS = ["H1", "H4", "D1", "W1"]

OKX_CANDLES = "https://www.okx.com/api/v5/market/candles"
OKX_FUNDING = "https://www.okx.com/api/v5/public/funding-rate"
OKX_OI = "https://www.okx.com/api/v5/public/open-interest"


def _db_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    return url


def _tf_to_okx(tf: str) -> str:
    m = {"H1": "1H", "H4": "4H", "D1": "1D", "W1": "1W"}
    if tf not in m:
        raise ValueError(f"Unsupported tf: {tf}")
    return m[tf]


def _swap_inst_id(spot_symbol: str) -> str:
    # BTC-USDT -> BTC-USDT-SWAP
    return f"{spot_symbol}-SWAP"


def _floor_ts(tf: str, ts: datetime) -> datetime:
    ts = ts.astimezone(timezone.utc).replace(microsecond=0)

    if tf == "H1":
        return ts.replace(minute=0, second=0)

    if tf == "H4":
        h = (ts.hour // 4) * 4
        return ts.replace(hour=h, minute=0, second=0)

    if tf == "D1":
        return ts.replace(hour=0, minute=0, second=0)

    if tf == "W1":
        base = ts.replace(hour=0, minute=0, second=0)
        return base - timedelta(days=base.weekday())

    raise ValueError(tf)


def _parse_okx_candle(row: List[str]) -> Dict[str, float]:
    return {
        "ts_ms": int(row[0]),
        "open": float(row[1]),
        "high": float(row[2]),
        "low": float(row[3]),
        "close": float(row[4]),
        "volume": float(row[5]),
    }


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _safe_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


async def fetch_last_closed_candle(client: httpx.AsyncClient, symbol: str, tf: str) -> Tuple[datetime, Dict]:
    params = {"instId": symbol, "bar": _tf_to_okx(tf), "limit": "2"}
    r = await client.get(OKX_CANDLES, params=params)
    r.raise_for_status()
    data = (r.json().get("data") or [])
    if len(data) < 2:
        raise RuntimeError(f"Not enough candles for {symbol} {tf}")

    row = data[1]  # закрытая свеча
    cndl = _parse_okx_candle(row)

    ts = datetime.fromtimestamp(cndl["ts_ms"] / 1000, tz=timezone.utc)
    ts = _floor_ts(tf, ts)
    return ts, cndl


async def fetch_funding_rate(client: httpx.AsyncClient, spot_symbol: str) -> Dict:
    inst_id = _swap_inst_id(spot_symbol)
    params = {"instId": inst_id}
    r = await client.get(OKX_FUNDING, params=params)
    r.raise_for_status()
    data = (r.json().get("data") or [])
    if not data:
        return {}

    d0 = data[0]
    return {
        "instId": inst_id,
        "funding_rate": _safe_float(d0.get("fundingRate")),
        "funding_time_ms": _safe_int(d0.get("fundingTime")),
        "next_funding_rate": _safe_float(d0.get("nextFundingRate")),
        "next_funding_time_ms": _safe_int(d0.get("nextFundingTime")),
    }


async def fetch_open_interest(client: httpx.AsyncClient, spot_symbol: str) -> Dict:
    inst_id = _swap_inst_id(spot_symbol)
    params = {"instType": "SWAP", "instId": inst_id}
    r = await client.get(OKX_OI, params=params)
    r.raise_for_status()
    data = (r.json().get("data") or [])
    if not data:
        return {}

    d0 = data[0]
    return {
        "instId": inst_id,
        "open_interest": _safe_float(d0.get("oi")),
        "open_interest_ccy": _safe_float(d0.get("oiCcy")),
        "oi_ts_ms": _safe_int(d0.get("ts")),
    }


def _load_last_snapshot_ts(
    conn: psycopg.Connection,
    *,
    symbols: List[str],
    tfs: List[str],
) -> Dict[Tuple[str, str], Optional[datetime]]:
    """
    Возвращает map[(symbol, tf)] = max(ts) из mm_snapshots.
    Нужно, чтобы НЕ дергать цепочку дальше, если новая закрытая свеча еще не появилась.
    """
    out: Dict[Tuple[str, str], Optional[datetime]] = {(s, tf): None for s in symbols for tf in tfs}

    sql = """
    SELECT symbol, tf, MAX(ts) AS last_ts
    FROM mm_snapshots
    WHERE symbol = ANY(%s) AND tf = ANY(%s)
    GROUP BY symbol, tf;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbols, tfs))
        for symbol, tf, last_ts in cur.fetchall():
            out[(symbol, tf)] = last_ts
    return out


def upsert_snapshot(
    conn: psycopg.Connection,
    *,
    ts: datetime,
    tf: str,
    symbol: str,
    o: float,
    h: float,
    l: float,
    c: float,
    v: float,
    meta: Optional[dict] = None,
) -> int:
    sql = """
    INSERT INTO mm_snapshots (ts, tf, symbol, open, high, low, close, volume, meta_json)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (symbol, tf, ts) DO UPDATE SET
      open=EXCLUDED.open,
      high=EXCLUDED.high,
      low=EXCLUDED.low,
      close=EXCLUDED.close,
      volume=EXCLUDED.volume,
      meta_json=EXCLUDED.meta_json
    RETURNING id;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ts, tf, symbol, o, h, l, c, v, Jsonb(meta or {})))
        row = cur.fetchone()
    return int(row[0])


async def run_snapshots_once(
    symbols: List[str] = SYMBOLS,
    tfs: List[str] = TFS,
) -> List[str]:
    results: List[str] = []

    async with httpx.AsyncClient(timeout=20) as client:
        funding_map: Dict[str, Dict] = {}
        oi_map: Dict[str, Dict] = {}

        for symbol in symbols:
            try:
                funding_map[symbol] = await fetch_funding_rate(client, symbol)
            except Exception:
                funding_map[symbol] = {}

            try:
                oi_map[symbol] = await fetch_open_interest(client, symbol)
            except Exception:
                oi_map[symbol] = {}

        with psycopg.connect(_db_url()) as conn:
            conn.execute("SET TIME ZONE 'UTC';")

            # 1) Смотрим, что уже записано (чтобы НЕ писать каждый тик)
            last_ts_map = _load_last_snapshot_ts(conn, symbols=symbols, tfs=tfs)

            for symbol in symbols:
                for tf in tfs:
                    ts, cndl = await fetch_last_closed_candle(client, symbol, tf)

                    last_ts = last_ts_map.get((symbol, tf))
                    # Если закрытая свеча уже есть в БД — ничего не делаем (и не триггерим downstream)
                    if last_ts is not None and last_ts >= ts:
                        continue

                    meta = {
                        "src": "okx",
                        "bar": _tf_to_okx(tf),
                        "ts_ms": cndl["ts_ms"],
                        "funding": funding_map.get(symbol, {}),
                        "open_interest": oi_map.get(symbol, {}),
                        "metrics_fetched_at_ms": int(datetime.now(timezone.utc).timestamp() * 1000),
                    }

                    snap_id = upsert_snapshot(
                        conn,
                        ts=ts,
                        tf=tf,
                        symbol=symbol,
                        o=cndl["open"],
                        h=cndl["high"],
                        l=cndl["low"],
                        c=cndl["close"],
                        v=cndl["volume"],
                        meta=meta,
                    )
                    results.append(f"{symbol} {tf} {ts.isoformat()} id={snap_id}")

            conn.commit()

    return results
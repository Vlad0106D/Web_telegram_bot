# services/market_data.py
# Асинхронная загрузка рыночных данных с OKX и KuCoin.
# Возвращаем единый формат свечей: pandas.DataFrame c колонками:
# ["open","high","low","close","volume"] и индексом-UTC datetime.
# Функции:
#   - await get_candles(symbol, tf, limit=300) -> (df, exchange)
#   - await get_price(symbol) -> (price, exchange)
#   - await get_price_safe(symbol) -> float | None

from __future__ import annotations

import asyncio
from typing import Optional, Tuple, Dict

import pandas as pd
import numpy as np
import httpx

# -----------------------
# Константы и маппинги
# -----------------------

OKX_BASE = "https://www.okx.com"
KCS_BASE = "https://api.kucoin.com"

# Маппинг таймфреймов в обозначения бирж
_OKX_BAR = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "10m": "10m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "6h": "6H",
    "12h": "12H",
    "1d": "1D",
}
_KCS_TYPE = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1hour",
    "4h": "4hour",
    "1d": "1day",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MrTradeBot/1.0)"}

# -----------------------
# Утилиты
# -----------------------

def _to_okx_inst(symbol: str) -> str:
    # "BTCUSDT" -> "BTC-USDT"
    s = symbol.upper().strip()
    if "-" in s:
        return s
    if s.endswith("USDT"):
        return s[:-4] + "-USDT"
    return s

def _to_kcs_symbol(symbol: str) -> str:
    # "BTCUSDT" -> "BTC-USDT"
    return _to_okx_inst(symbol)

def _build_df_ohlc(raw: list, order: str, scheme: str) -> pd.DataFrame:
    """
    Унифицируем разные форматы массивов бирж в общий DataFrame.
    order: "desc" или "asc" — порядок входящих строк (последнее -> первое).
    scheme:
      - "okx": [ts, o, h, l, c, vol, volccy, ...]
      - "kcs": [ts, o, c, h, l, vol, turnover]
    """
    if not raw:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    rows = raw[:]
    if order == "desc":
        rows = list(reversed(rows))  # делаем возрастание времени

    opens, highs, lows, closes, vols, tss = [], [], [], [], [], []
    for r in rows:
        if scheme == "okx":
            # OKX: ts(ms), o,h,l,c, vol(base), volccy, volCcyQuote, confirm
            ts_ms = int(r[0])
            o, h, l, c = map(float, (r[1], r[2], r[3], r[4]))
            vol = float(r[5]) if len(r) > 5 else np.nan
        else:
            # KuCoin: ts(sec), open, close, high, low, volume, turnover
            ts_ms = int(r[0]) * 1000
            o = float(r[1]); c = float(r[2]); h = float(r[3]); l = float(r[4])
            vol = float(r[5]) if len(r) > 5 else np.nan

        tss.append(pd.to_datetime(ts_ms, unit="ms", utc=True))
        opens.append(o); highs.append(h); lows.append(l); closes.append(c); vols.append(vol)

    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=pd.DatetimeIndex(tss, name="time")
    )
    return df

# -----------------------
# Загрузка свечей
# -----------------------

async def _okx_candles(symbol: str, tf: str, limit: int = 300) -> Optional[pd.DataFrame]:
    bar = _OKX_BAR.get(tf.lower())
    if not bar:
        return None
    inst = _to_okx_inst(symbol)
    url = f"{OKX_BASE}/api/v5/market/candles"
    params = {"instId": inst, "bar": bar, "limit": str(min(limit, 1000))}
    async with httpx.AsyncClient(timeout=15.0, headers=HEADERS) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    if not isinstance(data, dict) or data.get("code") != "0":
        return None
    arr = data.get("data", [])
    df = _build_df_ohlc(arr, order="desc", scheme="okx")
    return df.tail(limit)

async def _kcs_candles(symbol: str, tf: str, limit: int = 300) -> Optional[pd.DataFrame]:
    ctype = _KCS_TYPE.get(tf.lower())
    if not ctype:
        return None
    sym = _to_kcs_symbol(symbol)
    url = f"{KCS_BASE}/api/v1/market/candles"
    params = {"type": ctype, "symbol": sym}
    async with httpx.AsyncClient(timeout=15.0, headers=HEADERS) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    if not isinstance(data, dict) or data.get("code") != "200000":
        return None
    arr = data.get("data", [])
    df = _build_df_ohlc(arr, order="desc", scheme="kcs")
    return df.tail(limit)

async def get_candles(symbol: str, tf: str, limit: int = 300) -> Tuple[pd.DataFrame, str]:
    """
    Пробуем OKX, затем KuCoin. Возвращаем (df, "OKX"/"KuCoin").
    Бросаем ValueError, если данных нет.
    """
    # 1) OKX
    try:
        df = await _okx_candles(symbol, tf, limit)
        if df is not None and not df.empty:
            return df, "OKX"
    except Exception:
        pass

    # 2) KuCoin
    try:
        df = await _kcs_candles(symbol, tf, limit)
        if df is not None and not df.empty:
            return df, "KuCoin"
    except Exception:
        pass

    raise ValueError(f"No candles for {symbol} {tf}")

# -----------------------
# Текущая цена
# -----------------------

async def _okx_price(symbol: str) -> Optional[Tuple[float, str]]:
    inst = _to_okx_inst(symbol)
    url = f"{OKX_BASE}/api/v5/market/ticker"
    params = {"instId": inst}
    async with httpx.AsyncClient(timeout=10.0, headers=HEADERS) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    if not isinstance(data, dict) or data.get("code") != "0":
        return None
    arr = data.get("data", [])
    if not arr:
        return None
    last = float(arr[0]["last"])
    return last, "OKX"

async def _kcs_price(symbol: str) -> Optional[Tuple[float, str]]:
    sym = symbol.upper()
    if "-" not in sym:
        if sym.endswith("USDT"):
            sym = sym[:-4] + "-USDT"
    url = f"{KCS_BASE}/api/v1/market/orderbook/level1"
    params = {"symbol": sym}
    async with httpx.AsyncClient(timeout=10.0, headers=HEADERS) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    if not isinstance(data, dict) or data.get("code") != "200000":
        return None
    d = data.get("data") or {}
    price = d.get("price")
    if price is None:
        return None
    return float(price), "KuCoin"

async def get_price(symbol: str) -> Tuple[float, str]:
    """
    Пробуем OKX, затем KuCoin. Возвращаем (price, exchange) или ValueError.
    """
    try:
        p = await _okx_price(symbol)
        if p:
            return p
    except Exception:
        pass
    try:
        p = await _kcs_price(symbol)
        if p:
            return p
    except Exception:
        pass
    raise ValueError(f"No price for {symbol}")

async def get_price_safe(symbol: str) -> Optional[float]:
    try:
        price, _ = await get_price(symbol)
        return price
    except Exception:
        return None
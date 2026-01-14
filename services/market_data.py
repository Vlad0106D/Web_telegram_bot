# services/market_data.py
import logging
import re
from typing import Tuple, List

import httpx
import pandas as pd

log = logging.getLogger(__name__)

OKX_BASE = "https://www.okx.com"

# OKX candle bars mapping
_TF_OKX = {
    "5m": "5m",
    "10m": "10m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
    "1w": "1W",
}


def _sym_okx_spot(symbol: str) -> str:
    """
    BTCUSDT -> BTC-USDT (OKX spot instId)
    """
    s = symbol.upper().replace("_", "").replace("-", "")
    if s.endswith("USDT"):
        return s[:-4] + "-USDT"
    return s


def _df_ohlc_from_okx(arr) -> pd.DataFrame:
    """
    OKX candles schema:
      [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    Возвращаем DataFrame с колонками:
      [time, open, high, low, close, volume]
    time в UTC ms, сортировка asc
    """
    rows = []
    for it in arr:
        ts, o, h, l, c, vol, *_ = it
        rows.append([int(ts), float(o), float(h), float(l), float(c), float(vol)])

    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


async def get_candles(symbol: str, tf: str = "1h", limit: int = 300) -> Tuple[pd.DataFrame, str]:
    """
    OKX-only.
    Возвращает (df, exchange), где df: columns [time, open, high, low, close, volume] UTC ms asc
    """
    if tf not in _TF_OKX:
        raise ValueError(f"Unsupported tf for OKX: {tf}")

    okx_url = f"{OKX_BASE}/api/v5/market/candles"
    params = {"instId": _sym_okx_spot(symbol), "bar": _TF_OKX[tf], "limit": str(limit)}

    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(okx_url, params=params)
        r.raise_for_status()
        data = r.json()
        arr = data.get("data") or []
        if not arr:
            raise ValueError(f"No candles for {symbol} {tf} on OKX")

        df = _df_ohlc_from_okx(arr)
        return df, "OKX"


async def get_price(symbol: str) -> Tuple[float, str]:
    """
    OKX-only ticker price.
    Возвращает (price, exchange)
    """
    okx_ticker = f"{OKX_BASE}/api/v5/market/ticker"
    params = {"instId": _sym_okx_spot(symbol)}

    async with httpx.AsyncClient(timeout=8) as client:
        r = await client.get(okx_ticker, params=params)
        r.raise_for_status()
        j = r.json()
        data = (j.get("data") or [{}])[0]
        last = data.get("last")
        if last is None:
            raise ValueError(f"No last price for {symbol} on OKX")
        return float(last), "OKX"


def _norm(sym: str) -> str:
    """
    "BTC-USDT" -> "BTCUSDT"
    """
    s = sym.replace("-", "").replace("_", "").upper()
    return re.sub(r"[^A-Z0-9]", "", s)


async def search_symbols(query: str, limit: int = 50) -> List[str]:
    """
    OKX-only.
    Ищем пары по подстроке среди SPOT-инструментов OKX.
    Возвращаем тикеры вида BTCUSDT (только USDT), без дублей.
    """
    q = (query or "").strip().lower()
    if not q:
        return []

    out: List[str] = []
    seen = set()

    async with httpx.AsyncClient(timeout=10) as client:
        r_okx = await client.get(
            f"{OKX_BASE}/api/v5/public/instruments",
            params={"instType": "SPOT"},
        )
        r_okx.raise_for_status()
        data = r_okx.json()

        for inst in data.get("data", []):
            inst_id = inst.get("instId", "")  # BTC-USDT
            n = _norm(inst_id)
            if not n.endswith("USDT"):
                continue
            if q in n.lower() and n not in seen:
                seen.add(n)
                out.append(n)
                if len(out) >= limit:
                    break

    return out
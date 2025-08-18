# services/market_data.py
import asyncio
from typing import List, Tuple, Optional
import httpx
import pandas as pd
from datetime import datetime

KUCOIN_BASE = "https://api.kucoin.com"  # без ключей, публичные эндпоинты

def _kc_symbol(symbol: str) -> str:
    """BTCUSDT -> BTC-USDT (KuCoin формат)"""
    s = symbol.upper().replace("_", "")
    if s.endswith("USDT"):
        return s[:-4] + "-USDT"
    return s

async def get_price(symbol: str) -> Optional[float]:
    """
    Последняя цена с KuCoin (orderbook level1).
    Возвращает float или None.
    """
    sym = _kc_symbol(symbol)
    url = f"{KUCOIN_BASE}/api/v1/market/orderbook/level1"
    params = {"symbol": sym}
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, params=params)
        data = r.json()
        if data.get("code") == "200000":
            return float(data["data"]["price"])
    return None

async def get_ohlcv(symbol: str, interval: str = "1hour", limit: int = 300) -> pd.DataFrame:
    """
    OHLCV с KuCoin:
      interval: 1min, 5min, 15min, 30min, 1hour, 4hour, 1day
    Возвращает DataFrame с колонками: time, open, high, low, close, volume
    """
    sym = _kc_symbol(symbol)
    url = f"{KUCOIN_BASE}/api/v1/market/candles"
    params = {"symbol": sym, "type": interval}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url, params=params)
        data = r.json()

    if data.get("code") != "200000":
        raise RuntimeError(f"KuCoin klines error: {data}")

    # KuCoin отдаёт массивы: [time, open, close, high, low, volume, turnover]
    raw: List[List[str]] = data["data"][:limit]
    # данные приходят от новых к старым — развернём
    raw = list(reversed(raw))

    rows = []
    for it in raw:
        ts = int(it[0])  # milliseconds
        rows.append({
            "time": datetime.utcfromtimestamp(ts/1000),
            "open": float(it[1]),
            "close": float(it[2]),
            "high": float(it[3]),
            "low": float(it[4]),
            "volume": float(it[5]),
        })
    df = pd.DataFrame(rows)
    return df
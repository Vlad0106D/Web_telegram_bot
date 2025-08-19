# services/market_data.py
import os
import asyncio
import logging
from typing import Tuple
import httpx
import pandas as pd

log = logging.getLogger(__name__)

# --- helpers -------------------------------------------------------------
_TF_OKX = {
    "5m": "5m", "10m": "10m", "15m": "15m", "30m": "30m",
    "1h": "1H", "4h": "4H", "1d": "1D"
}
_TF_KUCOIN = {
    "5m": "5min", "10m": "10min", "15m": "15min", "30m": "30min",
    "1h": "1hour", "4h": "4hour", "1d": "1day"
}

def _sym_okx(symbol: str) -> str:
    # BTCUSDT -> BTC-USDT
    symbol = symbol.upper().replace("_", "")
    if symbol.endswith("USDT"):
        return symbol[:-4] + "-USDT"
    return symbol

def _sym_kucoin(symbol: str) -> str:
    # KuCoin тоже любит дефис
    return _sym_okx(symbol)

def _df_ohlc_from_array(arr, schema="okx") -> pd.DataFrame:
    """
    Преобразует массив свечей в DataFrame с колонками [time, open, high, low, close, volume]
    OKX: [ts, o, h, l, c, vol, volCcy]
    KuCoin: [ts, o, c, h, l, vol, turnover]
    """
    rows = []
    if schema == "okx":
        for it in arr:
            ts, o, h, l, c, vol, *_ = it
            rows.append([int(ts), float(o), float(h), float(l), float(c), float(vol)])
    else:  # kucoin
        for it in arr:
            ts, o, c, h, l, vol, *_ = it
            rows.append([int(float(ts) * 1000), float(o), float(h), float(l), float(c), float(vol)])

    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# --- public API ----------------------------------------------------------
async def get_candles(symbol: str, tf: str = "1h", limit: int = 300) -> Tuple[pd.DataFrame, str]:
    """
    Возвращает (df, exchange), где df: columns [time, open, high, low, close, volume] UTC ms asc
    Порядок попыток: OKX -> KuCoin. Берём первую успешную.
    """
    symbol_okx = _sym_okx(symbol)
    symbol_ku = _sym_kucoin(symbol)

    # --- OKX
    if tf in _TF_OKX:
        okx_url = "https://www.okx.com/api/v5/market/candles"
        params = {"instId": symbol_okx, "bar": _TF_OKX[tf], "limit": str(limit)}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(okx_url, params=params)
                r.raise_for_status()
                data = r.json()
                arr = data.get("data") or []
                if len(arr) > 0:
                    df = _df_ohlc_from_array(arr, schema="okx")
                    return df, "OKX"
        except Exception as e:
            log.warning("OKX candles error for %s %s: %s", symbol, tf, e)

    # --- KuCoin
    if tf in _TF_KUCOIN:
        ku_url = "https://api.kucoin.com/api/v1/market/candles"
        params = {"symbol": symbol_ku, "type": _TF_KUCOIN[tf]}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(ku_url, params=params)
                r.raise_for_status()
                data = r.json()
                arr = data.get("data") or []
                if len(arr) > 0:
                    df = _df_ohlc_from_array(arr, schema="kucoin")
                    return df, "KuCoin"
        except Exception as e:
            log.warning("KuCoin candles error for %s %s: %s", symbol, tf, e)

    raise ValueError(f"No candles for {symbol} {tf}")


async def get_price(symbol: str) -> Tuple[float, str]:
    """
    Текущая цена: сначала OKX, затем KuCoin.
    Возвращает (price, exchange)
    """
    # OKX
    try:
        okx_ticker = "https://www.okx.com/api/v5/market/ticker"
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(okx_ticker, params={"instId": _sym_okx(symbol)})
            r.raise_for_status()
            j = r.json()
            data = (j.get("data") or [{}])[0]
            last = float(data.get("last"))
            return last, "OKX"
    except Exception as e:
        log.warning("OKX price fail %s: %s", symbol, e)

    # KuCoin
    try:
        ku_ticker = "https://api.kucoin.com/api/v1/market/orderbook/level1"
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(ku_ticker, params={"symbol": _sym_kucoin(symbol)})
            r.raise_for_status()
            j = r.json()
            data = j.get("data") or {}
            last = float(data.get("price"))
            return last, "KuCoin"
    except Exception as e:
        log.warning("KuCoin price fail %s: %s", symbol, e)

    raise ValueError(f"No price for {symbol}")
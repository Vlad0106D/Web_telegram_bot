# services/market_data.py
import asyncio
import aiohttp
import logging
import pandas as pd
from typing import Tuple, Optional

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "MrTradeBot/1.0 (+okx+kucoin)"}
TIMEOUT = aiohttp.ClientTimeout(total=15)

OKX_TICKER = "https://www.okx.com/api/v5/market/ticker"
OKX_CANDLES = "https://www.okx.com/api/v5/market/candles"
KCS_L1     = "https://api.kucoin.com/api/v1/market/orderbook/level1"
KCS_CANDLES= "https://api.kucoin.com/api/v1/market/candles"

def _to_okx(symbol: str) -> str:
    s = symbol.upper().replace("_","").replace("-","")
    if not s.endswith("USDT"):
        s += "USDT"
    base = s[:-4]
    return f"{base}-USDT"

def _to_kucoin(symbol: str) -> str:
    s = symbol.upper().replace("_","").replace("-","")
    if not s.endswith("USDT"):
        s += "USDT"
    base = s[:-4]
    return f"{base}-USDT"

async def _okx_price(session: aiohttp.ClientSession, symbol: str) -> Optional[Tuple[float,str]]:
    try:
        inst = _to_okx(symbol)
        async with session.get(OKX_TICKER, params={"instId": inst}) as r:
            data = await r.json()
        if data.get("code") == "0" and data.get("data"):
            last = float(data["data"][0]["last"])
            return last, "OKX"
    except Exception as e:
        log.warning("OKX price error for %s: %s", symbol, e)
    return None

async def _kcs_price(session: aiohttp.ClientSession, symbol: str) -> Optional[Tuple[float,str]]:
    try:
        sym = _to_kucoin(symbol)
        async with session.get(KCS_L1, params={"symbol": sym}) as r:
            data = await r.json()
        if data.get("code") == "200000" and data.get("data") and data["data"].get("price"):
            last = float(data["data"]["price"])
            return last, "KuCoin"
    except Exception as e:
        log.warning("KuCoin price error for %s: %s", symbol, e)
    return None

async def get_price(symbol: str) -> Tuple[Optional[float], Optional[str]]:
    """Возвращает (цена, биржа)."""
    symbol = symbol.upper()
    async with aiohttp.ClientSession(headers=HEADERS, timeout=TIMEOUT) as session:
        res = await _okx_price(session, symbol)
        if res:
            return res
        res = await _kcs_price(session, symbol)
        if res:
            return res
    return None, None

# ====== КЛИНЫ ======
_BAR_MAP_OKX = {
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D"
}
_BAR_MAP_KCS = {
    "15m": "15min",
    "30m": "30min",
    "1h": "1hour",
    "4h": "4hour",
    "1d": "1day"
}

def _df_from_okx(raw) -> pd.DataFrame:
    # OKX candles: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    rows = []
    for it in raw:
        ts = int(it[0])
        o,h,l,c = map(float, it[1:5])
        rows.append((ts, o, h, l, c))
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close"])
    df = df.sort_values("ts").reset_index(drop=True)
    return df

def _df_from_kcs(raw) -> pd.DataFrame:
    # KCS candles: [time, open, close, high, low, volume, turnover]
    rows = []
    for it in raw:
        ts = int(float(it[0])*1000)  # сек -> мс
        o = float(it[1]); c = float(it[2]); h = float(it[3]); l = float(it[4])
        rows.append((ts, o, h, l, c))
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close"])
    df = df.sort_values("ts").reset_index(drop=True)
    return df

async def _okx_candles(session, symbol: str, tf: str, limit: int) -> Optional[pd.DataFrame]:
    try:
        bar = _BAR_MAP_OKX.get(tf)
        if not bar:
            return None
        inst = _to_okx(symbol)
        async with session.get(OKX_CANDLES, params={"instId": inst, "bar": bar, "limit": limit}) as r:
            data = await r.json()
        if data.get("code") == "0" and data.get("data"):
            return _df_from_okx(data["data"])
    except Exception as e:
        log.warning("OKX candles error %s %s: %s", symbol, tf, e)
    return None

async def _kcs_candles(session, symbol: str, tf: str, limit: int) -> Optional[pd.DataFrame]:
    try:
        tp = _BAR_MAP_KCS.get(tf)
        if not tp:
            return None
        sym = _to_kucoin(symbol)
        async with session.get(KCS_CANDLES, params={"symbol": sym, "type": tp}) as r:
            data = await r.json()
        if data.get("code") == "200000" and isinstance(data.get("data"), list):
            # KuCoin вернёт от нового к старому — переведём в возрастающий
            raw = list(reversed(data["data"]))[-limit:]
            return _df_from_kcs(raw)
    except Exception as e:
        log.warning("KuCoin candles error %s %s: %s", symbol, tf, e)
    return None

async def get_candles(symbol: str, tf: str = "1h", limit: int = 300) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Возвращает (DataFrame OHLC, биржа-источник) или (None, None)."""
    async with aiohttp.ClientSession(headers=HEADERS, timeout=TIMEOUT) as session:
        df = await _okx_candles(session, symbol, tf, limit)
        if df is not None and len(df):
            return df, "OKX"
        df = await _kcs_candles(session, symbol, tf, limit)
        if df is not None and len(df):
            return df, "KuCoin"
    return None, None
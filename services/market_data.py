# services/market_data.py
import asyncio
import time
from typing import Tuple, Optional, List

import aiohttp
import pandas as pd

try:
    from config import EXCHANGE as DEFAULT_EXCHANGE
except Exception:
    DEFAULT_EXCHANGE = "okx"


# ---- Таймфреймы-синонимы (вход -> канонический) ----
_CANON = {
    "1h": "1h", "1hour": "1h",
    "4h": "4h", "4hour": "4h",
    "30m": "30m", "30min": "30m",
    "15m": "15m", "15min": "15m",
}

# ---- Маппинги таймфреймов под API бирж ----
TF_OKX = {"1h": "1H", "4h": "4H", "30m": "30m", "15m": "15m"}
TF_BINANCE = {"1h": "1h", "4h": "4h", "30m": "30m", "15m": "15m"}
TF_KUCOIN = {"1h": "1hour", "4h": "4hour", "30m": "30min", "15m": "15min"}


def _canon_tf(tf: str) -> str:
    tf = (tf or "").strip().lower()
    if tf not in _CANON:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return _CANON[tf]


def _okx_symbol(symbol: str) -> str:
    # OKX использует дефис: BTC-USDT
    s = symbol.upper().replace("_", "-")
    if "-" not in s:
        s = s.replace("USDT", "-USDT")
    return s


def _kucoin_symbol(symbol: str) -> str:
    # KuCoin тоже с дефисом
    s = symbol.upper().replace("_", "-")
    if "-" not in s:
        s = s.replace("USDT", "-USDT")
    return s


def _binance_symbol(symbol: str) -> str:
    # Binance — без дефиса
    return symbol.upper().replace("-", "")


async def _fetch_json(session: aiohttp.ClientSession, url: str, params: dict | None = None) -> dict | list:
    for attempt in range(3):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep(0.4)
    return {}  # на всякий


async def _candles_okx(session: aiohttp.ClientSession, symbol: str, tf: str, limit: int) -> pd.DataFrame:
    inst = _okx_symbol(symbol)
    bar = TF_OKX[tf]
    # https://www.okx.com/api/v5/market/candles?instId=BTC-USDT&bar=1H&limit=100
    url = "https://www.okx.com/api/v5/market/candles"
    data = await _fetch_json(session, url, params={"instId": inst, "bar": bar, "limit": str(limit)})
    # Ответ: {"code":"0","data":[ [ts, o,h,l,c,vol,volCcy,volCcyQuote,confirm] , ... ]}
    arr = (data or {}).get("data") or []
    if not arr:
        return pd.DataFrame()
    rows = []
    for item in arr:
        # item: [ts, o, h, l, c, vol, ...]
        ts = int(item[0])
        o = float(item[1]); h = float(item[2]); l = float(item[3]); c = float(item[4])
        vol = float(item[5])
        rows.append((ts, o, h, l, c, vol))
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


async def _candles_binance(session: aiohttp.ClientSession, symbol: str, tf: str, limit: int) -> pd.DataFrame:
    sym = _binance_symbol(symbol)
    interval = TF_BINANCE[tf]
    # https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=100
    url = "https://api.binance.com/api/v3/klines"
    data = await _fetch_json(session, url, params={"symbol": sym, "interval": interval, "limit": str(limit)})
    # Элементы: [ openTime, open, high, low, close, volume, closeTime, ... ]
    if not data:
        return pd.DataFrame()
    rows = []
    for k in data:
        ts = int(k[0])
        o = float(k[1]); h = float(k[2]); l = float(k[3]); c = float(k[4])
        vol = float(k[5])
        rows.append((ts, o, h, l, c, vol))
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


async def _candles_kucoin(session: aiohttp.ClientSession, symbol: str, tf: str, limit: int) -> pd.DataFrame:
    sym = _kucoin_symbol(symbol)
    ktype = TF_KUCOIN[tf]
    # https://api.kucoin.com/api/v1/market/candles?type=1hour&symbol=BTC-USDT
    url = "https://api.kucoin.com/api/v1/market/candles"
    data = await _fetch_json(session, url, params={"type": ktype, "symbol": sym})
    # Ответ: {"code":"200000","data":[ [time, open, close, high, low, volume, turnover], ... ]}
    arr = (data or {}).get("data") or []
    if not arr:
        return pd.DataFrame()
    # KuCoin возвращает в обратном порядке (от новой к старой), приведём к хронологии
    rows = []
    for item in reversed(arr[-limit:]):
        # item: [time, open, close, high, low, volume, turnover]
        # time — строка timestamp в секундах
        ts = int(float(item[0])) * 1000
        o = float(item[1]); c = float(item[2]); h = float(item[3]); l = float(item[4])
        vol = float(item[5])
        rows.append((ts, o, h, l, c, vol))
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


async def get_candles(
    symbol: str,
    timeframe: str,
    limit: int = 300,
    exchange: Optional[str] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Универсальный загрузчик свечей.
    Возвращает (df, exchange_used). Если данных нет — ValueError.
    """
    tf = _canon_tf(timeframe)

    # Порядок попыток: указанная биржа → остальные
    priority: List[str] = []
    if exchange:
        priority.append(exchange.lower())
    if DEFAULT_EXCHANGE and (not priority or DEFAULT_EXCHANGE.lower() not in priority):
        priority.append(DEFAULT_EXCHANGE.lower())
    for ex in ("okx", "binance", "kucoin"):
        if ex not in priority:
            priority.append(ex)

    async with aiohttp.ClientSession(headers={"User-Agent": "mr.trade-bot/1.0"}) as session:
        for ex in priority:
            try:
                if ex == "okx":
                    df = await _candles_okx(session, symbol, tf, limit)
                elif ex == "binance":
                    df = await _candles_binance(session, symbol, tf, limit)
                elif ex == "kucoin":
                    df = await _candles_kucoin(session, symbol, tf, limit)
                else:
                    continue

                if not df.empty:
                    return df, ex
            except Exception:
                # тихо пробуем следующую биржу
                await asyncio.sleep(0.2)
                continue

    raise ValueError(f"No candles for {symbol} {timeframe}")


# Утилита для синхронного теста (локально):
if __name__ == "__main__":
    async def _test():
        for ex in (None, "okx", "binance", "kucoin"):
            try:
                df, used = await get_candles("BTCUSDT", "1h", 100, exchange=ex)
                print("OK:", used, len(df), "rows", "last close:", df["close"].iloc[-1])
                break
            except Exception as e:
                print("fail", ex, e)

    asyncio.run(_test())
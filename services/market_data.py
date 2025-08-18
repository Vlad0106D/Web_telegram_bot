# services/market_data.py
import asyncio
from typing import Tuple, Optional, List, Union

import aiohttp
import pandas as pd

try:
    from config import EXCHANGE as DEFAULT_EXCHANGE
except Exception:
    DEFAULT_EXCHANGE = "binance"

# ===== Канонизация таймфреймов =====
_CANON = {
    "1h": "1h", "1hour": "1h", "60m": "1h",
    "4h": "4h", "4hour": "4h", "240m": "4h",
    "30m": "30m", "30min": "30m",
    "15m": "15m", "15min": "15m",
}

TF_OKX     = {"1h": "1H", "4h": "4H", "30m": "30m", "15m": "15m"}
TF_BINANCE = {"1h": "1h", "4h": "4h", "30m": "30m", "15m": "15m"}
TF_KUCOIN  = {"1h": "1hour", "4h": "4hour", "30m": "30min", "15m": "15min"}

def _canon_tf(tf: str) -> str:
    tf = (tf or "").strip().lower()
    if tf not in _CANON:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return _CANON[tf]

def _okx_symbol(symbol: str) -> str:
    s = symbol.upper().replace("_", "-")
    if "-" not in s:
        s = s.replace("USDT", "-USDT")
    return s

def _kucoin_symbol(symbol: str) -> str:
    s = symbol.upper().replace("_", "-")
    if "-" not in s:
        s = s.replace("USDT", "-USDT")
    return s

def _binance_symbol(symbol: str) -> str:
    return symbol.upper().replace("-", "")

async def _fetch_json(session: aiohttp.ClientSession, url: str, params: dict | None = None) -> Union[dict, list]:
    # 3 попытки с небольшой задержкой
    last_err: Optional[Exception] = None
    for _ in range(3):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=12)) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.5)
    if last_err:
        raise last_err
    return {}

# ---------- Загрузчики по биржам ----------
async def _candles_binance(session: aiohttp.ClientSession, symbol: str, tf: str, limit: int) -> pd.DataFrame:
    sym = _binance_symbol(symbol)
    interval = TF_BINANCE[tf]
    url = "https://api.binance.com/api/v3/klines"
    data = await _fetch_json(session, url, params={"symbol": sym, "interval": interval, "limit": str(limit)})
    if not data or isinstance(data, dict) and data.get("code"):
        return pd.DataFrame()
    rows = []
    for k in data:
        ts = int(k[0])
        o = float(k[1]); h = float(k[2]); l = float(k[3]); c = float(k[4]); vol = float(k[5])
        rows.append((ts, o, h, l, c, vol))
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

async def _candles_okx(session: aiohttp.ClientSession, symbol: str, tf: str, limit: int) -> pd.DataFrame:
    inst = _okx_symbol(symbol)
    bar = TF_OKX[tf]
    url = "https://www.okx.com/api/v5/market/candles"
    data = await _fetch_json(session, url, params={"instId": inst, "bar": bar, "limit": str(limit)})
    arr = (data or {}).get("data") or []
    if not arr:
        return pd.DataFrame()
    rows = []
    for item in arr:
        ts = int(item[0])
        o = float(item[1]); h = float(item[2]); l = float(item[3]); c = float(item[4]); vol = float(item[5])
        rows.append((ts, o, h, l, c, vol))
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

async def _candles_kucoin(session: aiohttp.ClientSession, symbol: str, tf: str, limit: int) -> pd.DataFrame:
    sym = _kucoin_symbol(symbol)
    ktype = TF_KUCOIN[tf]
    url = "https://api.kucoin.com/api/v1/market/candles"
    data = await _fetch_json(session, url, params={"type": ktype, "symbol": sym})
    arr = (data or {}).get("data") or []
    if not arr:
        return pd.DataFrame()
    rows = []
    # у KuCoin порядок обратный — перевернём и ограничим
    for item in reversed(arr[-limit:]):
        ts = int(float(item[0])) * 1000
        o = float(item[1]); c = float(item[2]); h = float(item[3]); l = float(item[4]); vol = float(item[5])
        rows.append((ts, o, h, l, c, vol))
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ---------- Публичная функция ----------
async def get_candles(
    symbol: str,
    timeframe: str,
    limit: int = 300,
    exchange: Optional[str] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Возвращает (DataFrame, имя_биржи). Бросает ValueError, если не удалось
    взять свечи ни на одной из бирж.
    """
    tf = _canon_tf(timeframe)

    # Порядок приоритетов: запрошенная → DEFAULT_EXCHANGE → binance → okx → kucoin
    priority: List[str] = []
    if exchange:
        priority.append(exchange.lower())
    if DEFAULT_EXCHANGE and DEFAULT_EXCHANGE.lower() not in priority:
        priority.append(DEFAULT_EXCHANGE.lower())
    for ex in ("binance", "okx", "kucoin"):
        if ex not in priority:
            priority.append(ex)

    errors: List[str] = []

    async with aiohttp.ClientSession(headers={"User-Agent": "mr.trade-bot/1.0"}) as session:
        for ex in priority:
            try:
                if ex == "binance":
                    df = await _candles_binance(session, symbol, tf, limit)
                elif ex == "okx":
                    df = await _candles_okx(session, symbol, tf, limit)
                elif ex == "kucoin":
                    df = await _candles_kucoin(session, symbol, tf, limit)
                else:
                    continue

                if not df.empty:
                    return df, ex
                errors.append(f"{ex}: empty")
            except Exception as e:
                errors.append(f"{ex}: {type(e).__name__} {e}")

    # Детальная причина уйдёт в исключении — увидим её в логах/сообщении бота
    raise ValueError(f"No candles for {symbol} {timeframe} | tried: " + " ; ".join(errors))


# Локальный тест
if __name__ == "__main__":
    async def _test():
        try:
            df, used = await get_candles("BTCUSDT", "1h", 120)
            print("OK", used, len(df), "last:", df.close.iloc[-1])
        except Exception as e:
            print("ERR", e)
    asyncio.run(_test())
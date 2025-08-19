# services/kucoin_client.py
import aiohttp
from typing import List, Dict
from config import HTTP_TIMEOUT_SEC

KUCOIN_BASE = "https://api.kucoin.com"

# Map нашего интервала -> KuCoin type
TYPE_MAP = {
    "15": "15min",
    "60": "1hour",
    "15m": "15min",
    "1h": "1hour",
    "1H": "1hour"
}

def to_kucoin_symbol(symbol: str) -> str:
    # KuCoin тоже использует дефис: BTC-USDT
    s = symbol.replace(" ", "").upper()
    return s if "-" in s else f"{s[:-4]}-{s[-4:]}"

async def fetch_kucoin_klines(symbol: str, interval: str = "15", limit: int = 200) -> List[Dict]:
    sym = to_kucoin_symbol(symbol)
    typ = TYPE_MAP.get(interval, interval)
    # KuCoin: GET /api/v1/market/candles?type=15min&symbol=BTC-USDT
    # Возвращает [[time, open, close, high, low, volume, turnover], ...] от старой к новой или наоборот — нормализуем
    params = {"type": typ, "symbol": sym}
    url = f"{KUCOIN_BASE}/api/v1/market/candles"
    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SEC)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        async with sess.get(url, params=params) as r:
            r.raise_for_status()
            data = await r.json()
            if data.get("code") != "200000":
                raise RuntimeError(f"KuCoin error: {data}")
            rows = data["data"]
            # Преобразуем и сортируем по времени возрастанию (rows[i][0] — Unix ts секунд)
            parsed = []
            for row in rows:
                ts = int(row[0]) * 1000  # приводим к ms (как у OKX), не критично
                o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
                v = float(row[5])
                parsed.append({"t": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
            parsed.sort(key=lambda x: x["t"])
            # Ограничим до limit последних
            return parsed[-limit:]
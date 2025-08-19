# services/okx_client.py
import aiohttp
from typing import List, Dict
from config import HTTP_TIMEOUT_SEC

OKX_BASE = "https://www.okx.com"

# Map нашего интервала -> OKX bar
BAR_MAP = {
    "15": "15m",
    "60": "1H",
    "15m": "15m",
    "1h": "1H",
    "1H": "1H"
}

def to_okx_symbol(symbol: str) -> str:
    # Приводим BTCUSDT/BTC-USDT к OKX "BTC-USDT"
    s = symbol.replace(" ", "").upper()
    return s if "-" in s else f"{s[:-4]}-{s[-4:]}"  # ...USDT

async def fetch_okx_klines(symbol: str, interval: str = "15", limit: int = 200) -> List[Dict]:
    inst_id = to_okx_symbol(symbol)
    bar = BAR_MAP.get(interval, interval)
    params = {"instId": inst_id, "bar": bar, "limit": str(limit)}
    url = f"{OKX_BASE}/api/v5/market/candles"
    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SEC)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        async with sess.get(url, params=params) as r:
            r.raise_for_status()
            data = await r.json()
            if data.get("code") not in (None, "0") and data.get("code") != 0:
                raise RuntimeError(f"OKX error: {data}")
            rows = data["data"]  # список списков: [ts, o, h, l, c, vol, volCcy, ...]
            # OKX отдаёт от новой к старой; разворачиваем по возрастанию времени
            rows_sorted = sorted(rows, key=lambda x: int(x[0]))
            out = []
            for row in rows_sorted:
                ts = int(row[0])
                o, h, l, c = map(float, row[1:5])
                vol = float(row[5])
                out.append({"t": ts, "open": o, "high": h, "low": l, "close": c, "volume": vol})
            return out
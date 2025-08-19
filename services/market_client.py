# services/market_client.py
import asyncio
from typing import List, Dict
from config import HTTP_RETRIES
from services.okx_client import fetch_okx_klines
from services.kucoin_client import fetch_kucoin_klines

async def fetch_klines(symbol: str, interval: str = "15", limit: int = 200) -> List[Dict]:
    last_err = None
    for attempt in range(HTTP_RETRIES):
        # 1) OKX
        try:
            return await fetch_okx_klines(symbol, interval=interval, limit=limit)
        except Exception as e:
            last_err = e
        # 2) KuCoin
        try:
            return await fetch_kucoin_klines(symbol, interval=interval, limit=limit)
        except Exception as e:
            last_err = e
        await asyncio.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"All sources failed after {HTTP_RETRIES} retries: {last_err}")
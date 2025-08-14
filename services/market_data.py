# services/market_data.py — безопасное получение цены c OKX/KuCoin
import aiohttp
import logging

log = logging.getLogger("services.market_data")

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MrTradeBot/1.0)"}

OKX_TICKER = "https://www.okx.com/api/v5/market/ticker"
KCS_L1     = "https://api.kucoin.com/api/v1/market/orderbook/level1"

async def _okx_price(session: aiohttp.ClientSession, symbol: str):
    # OKX формат тикера: BTC-USDT
    inst_id = symbol.upper().replace("-", "")
    inst_id = f"{inst_id[:-4]}-{inst_id[-4:]}" if not "-" in inst_id else inst_id
    try:
        async with session.get(OKX_TICKER, params={"instId": inst_id}, timeout=10) as r:
            data = await r.json()
            if isinstance(data, dict) and data.get("code") == "0":
                arr = data.get("data") or []
                if arr:
                    last = float(arr[0]["last"])
                    return last, "OKX"
            log.warning(f"OKX price miss for {symbol}: {data}")
    except Exception as e:
        log.warning(f"OKX price error for {symbol}: {e}")
    return None, None

async def _kucoin_price(session: aiohttp.ClientSession, symbol: str):
    # KuCoin формат тикера: BTC-USDT (но в level1 — без дефиса тоже работает)
    sym = symbol.upper().replace("-", "")
    try:
        async with session.get(KCS_L1, params={"symbol": sym}, timeout=10) as r:
            data = await r.json()
            # успешный ответ: {"code":"200000","data":{"price":"12345.6", ...}}
            if isinstance(data, dict) and data.get("code") == "200000" and data.get("data"):
                price = float(data["data"]["price"])
                return price, "KuCoin"
            log.warning(f"KuCoin price miss for {symbol}: {data}")
    except Exception as e:
        log.warning(f"KuCoin price error for {symbol}: {e}")
    return None, None

async def get_price(symbol: str):
    symbol = symbol.upper().replace("-", "")
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        p, ex = await _okx_price(session, symbol)
        if p is not None:
            return p, ex
        p, ex = await _kucoin_price(session, symbol)
        if p is not None:
            return p, ex
    return None, None

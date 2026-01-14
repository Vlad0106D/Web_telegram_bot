from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import httpx

log = logging.getLogger(__name__)

OKX_BASE = "https://www.okx.com"

# Для MM режима используем PERP/SWAP
# BTCUSDT -> BTC-USDT-SWAP
def to_okx_swap_inst_id(symbol: str) -> str:
    s = symbol.upper().replace("-", "").replace("_", "")
    if s.endswith("USDT"):
        base = s[:-4]
        return f"{base}-USDT-SWAP"
    return s


@dataclass
class DerivativesSnap:
    inst_id: str
    exchange: str

    open_interest: Optional[float]        # raw OI (units/contracts)
    open_interest_usd: Optional[float]    # OI × price

    funding_rate: Optional[float]         # 0.0001 == 0.01%
    next_funding_time_ms: Optional[int]


async def get_open_interest_okx(inst_id: str) -> Optional[float]:
    """
    OKX V5 Public: Open Interest
    GET /api/v5/public/open-interest?instType=SWAP&instId=...
    """
    url = f"{OKX_BASE}/api/v5/public/open-interest"
    params = {"instType": "SWAP", "instId": inst_id}

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            j = r.json()
            data = j.get("data") or []
            if not data:
                return None

            oi = data[0].get("oi")
            return float(oi) if oi is not None else None

    except Exception as e:
        log.warning("OKX open-interest failed (%s): %s", inst_id, e)
        return None


async def get_funding_rate_okx(inst_id: str) -> Tuple[Optional[float], Optional[int]]:
    """
    OKX V5 Public: Funding Rate
    GET /api/v5/public/funding-rate?instId=...
    """
    url = f"{OKX_BASE}/api/v5/public/funding-rate"
    params = {"instId": inst_id}

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            j = r.json()
            data = j.get("data") or []
            if not data:
                return None, None

            fr = data[0].get("fundingRate")
            nft = data[0].get("nextFundingTime")

            funding_rate = float(fr) if fr is not None else None
            next_funding_time_ms = int(nft) if nft is not None else None
            return funding_rate, next_funding_time_ms

    except Exception as e:
        log.warning("OKX funding-rate failed (%s): %s", inst_id, e)
        return None, None


async def get_derivatives_snapshot(
    symbol: str,
    *,
    price: Optional[float] = None,
) -> DerivativesSnap:
    """
    ЕДИНСТВЕННЫЙ источник деривативных данных — OKX (public API).

    price:
      close последней ЗАКРЫТОЙ свечи соответствующего TF
      (нужен для расчёта open_interest_usd)
    """
    inst_id = to_okx_swap_inst_id(symbol)

    oi = await get_open_interest_okx(inst_id)
    fr, nft = await get_funding_rate_okx(inst_id)

    oi_usd: Optional[float] = None
    if oi is not None and price is not None:
        try:
            oi_usd = float(oi) * float(price)
        except Exception:
            oi_usd = None

    return DerivativesSnap(
        inst_id=inst_id,
        exchange="okx",
        open_interest=oi,
        open_interest_usd=oi_usd,
        funding_rate=fr,
        next_funding_time_ms=nft,
    )
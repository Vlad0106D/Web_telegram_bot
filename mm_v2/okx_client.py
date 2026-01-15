from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

log = logging.getLogger("mm_v2.okx")


OKX_BASE_URL = "https://www.okx.com"


# -------------------------
# Symbol mapping (MM v2 scope)
# -------------------------
# Мы работаем только с BTCUSDT / ETHUSDT, но OKX ожидает instId в своём формате.
SPOT_INST = {
    "BTCUSDT": "BTC-USDT",
    "ETHUSDT": "ETH-USDT",
}

# Для OI/Funding используем perpetual swap (это ок для нашей логики pressure/unwind/distribution)
SWAP_INST = {
    "BTCUSDT": "BTC-USDT-SWAP",
    "ETHUSDT": "ETH-USDT-SWAP",
}

# TF mapping OKX "bar"
OKX_BAR = {
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
    "1w": "1W",
}


@dataclass(frozen=True)
class Candle:
    ts: datetime  # UTC close/open time boundary (OKX timestamp)
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class DerivMetrics:
    open_interest: Optional[float]
    funding_rate: Optional[float]


def _ms_to_dt_utc(ms: str) -> datetime:
    return datetime.fromtimestamp(int(ms) / 1000.0, tz=timezone.utc)


def _get_json(path: str, params: dict, timeout_sec: float = 10.0, retries: int = 2) -> dict:
    url = f"{OKX_BASE_URL}{path}"
    last_err: Optional[Exception] = None

    for i in range(retries + 1):
        try:
            with httpx.Client(timeout=timeout_sec) as client:
                r = client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
                if not isinstance(data, dict):
                    raise RuntimeError(f"OKX: unexpected response type: {type(data)}")
                # OKX success code обычно "0"
                if data.get("code") not in ("0", 0, None):
                    raise RuntimeError(f"OKX error: code={data.get('code')} msg={data.get('msg')}")
                return data
        except Exception as e:
            last_err = e
            if i < retries:
                time.sleep(0.5 * (i + 1))
                continue
            break

    raise RuntimeError(f"OKX request failed: {url} params={params} err={last_err!r}")


# -------------------------
# Candles (OHLCV)
# -------------------------
def fetch_candles(
    *,
    symbol: str,
    tf: str,
    limit: int = 100,
) -> list[Candle]:
    """
    Returns candles in ASC order by ts.
    Uses OKX market candles endpoint:
      GET /api/v5/market/candles?instId=BTC-USDT&bar=1H&limit=100
    """
    if symbol not in SPOT_INST:
        raise ValueError(f"Unsupported symbol for MM v2: {symbol}")
    if tf not in OKX_BAR:
        raise ValueError(f"Unsupported tf for MM v2: {tf}")

    inst_id = SPOT_INST[symbol]
    bar = OKX_BAR[tf]

    data = _get_json(
        "/api/v5/market/candles",
        params={"instId": inst_id, "bar": bar, "limit": str(int(limit))},
    )

    rows = data.get("data") or []
    out: list[Candle] = []

    # OKX отдаёт candles в порядке DESC (свежие первые)
    # Формат: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    for r in reversed(rows):
        try:
            ts = _ms_to_dt_utc(r[0])
            out.append(
                Candle(
                    ts=ts,
                    open=float(r[1]),
                    high=float(r[2]),
                    low=float(r[3]),
                    close=float(r[4]),
                    volume=float(r[5]),
                )
            )
        except Exception:
            log.exception("Failed to parse candle row: %r", r)

    return out


# -------------------------
# Derivatives: OI + Funding
# -------------------------
def fetch_deriv_metrics(symbol: str) -> DerivMetrics:
    """
    Fetch OI and funding from perpetual swap for the given symbol.
    If OKX doesn't return something, value becomes None.

    OI:
      GET /api/v5/public/open-interest?instType=SWAP&instId=BTC-USDT-SWAP
    Funding:
      GET /api/v5/public/funding-rate?instId=BTC-USDT-SWAP
    """
    if symbol not in SWAP_INST:
        raise ValueError(f"Unsupported symbol for MM v2: {symbol}")

    inst_id = SWAP_INST[symbol]

    # Open Interest
    oi_val: Optional[float] = None
    try:
        data_oi = _get_json(
            "/api/v5/public/open-interest",
            params={"instType": "SWAP", "instId": inst_id},
        )
        rows = data_oi.get("data") or []
        if rows:
            # поле: oi (обычно в контрактах). Для нашей логики важно относительное изменение.
            oi_raw = rows[0].get("oi")
            if oi_raw is not None and str(oi_raw).strip() != "":
                oi_val = float(oi_raw)
    except Exception:
        log.exception("OI fetch failed for %s (%s)", symbol, inst_id)

    # Funding rate
    fr_val: Optional[float] = None
    try:
        data_fr = _get_json(
            "/api/v5/public/funding-rate",
            params={"instId": inst_id},
        )
        rows = data_fr.get("data") or []
        if rows:
            fr_raw = rows[0].get("fundingRate")
            if fr_raw is not None and str(fr_raw).strip() != "":
                fr_val = float(fr_raw)
    except Exception:
        log.exception("Funding fetch failed for %s (%s)", symbol, inst_id)

    return DerivMetrics(open_interest=oi_val, funding_rate=fr_val)
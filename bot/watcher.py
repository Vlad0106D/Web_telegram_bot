import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple, List

import pandas as pd

from services.state import get_favorites
from services.market_data import get_candles
from strategy.breakout_watcher import classify_breakout

log = logging.getLogger(__name__)

# Детектор (общие базовые параметры)
BB_PERIOD = 20
BB_K = 2.0
LOOKBACK_RANGE_BY_TF = {
    "5m": 60,     # ~5 часов истории
    "15m": 48,    # ~12 часов
    "1h": 50,     # ~2 суток
}
BB_SQUEEZE_PCT_BY_TF = {
    "5m": 3.0,
    "15m": 3.5,
    "1h": 4.0,
}
PROXIMITY_PCT_BY_TF = {
    "5m": 0.5,
    "15m": 0.6,
    "1h": 0.7,
}
BREAK_EPS_PCT_BY_TF = {
    "5m": 0.10,
    "15m": 0.12,
    "1h": 0.15,
}

# Анти-спам: кэш последних алертов (ключ: (tf, symbol))
_LAST_ALERT: Dict[Tuple[str, str], Tuple[str, float, datetime]] = {}
COOLDOWN_BY_TF = {
    "5m": timedelta(minutes=8),
    "15m": timedelta(minutes=12),
    "1h": timedelta(minutes=20),
}
PRICE_JITTER = 0.15  # % — если уровень почти тот же, подавляем дубль

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _fmt(v: float) -> str:
    if v >= 1000:
        return f"{v:,.2f}".replace(",", " ")
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.6f}".rstrip("0").rstrip(".")

def _should_alert(tf: str, symbol: str, state: str, ref_level: float) -> bool:
    now = _now_utc()
    key = (tf, symbol)
    last = _LAST_ALERT.get(key)
    if not last:
        _LAST_ALERT[key] = (state, ref_level, now)
        return True

    last_state, last_level, last_time = last
    cooldown = COOLDOWN_BY_TF.get(tf, timedelta(minutes=15))
    if (now - last_time) > cooldown:
        _LAST_ALERT[key] = (state, ref_level, now)
        return True

    level_pct = abs((ref_level - last_level) / (last_level or 1) * 100.0)
    if state == last_state and level_pct <= PRICE_JITTER:
        return False

    _LAST_ALERT[key] = (state, ref_level, now)
    return True

def _format_alert(symbol: str, tf: str, ex: str, d: Dict) -> str:
    state = d["state"]
    price = _fmt(d["price"])
    bw = d.get("bb_width_pct")
    r_hi = d.get("range_high")
    r_lo = d.get("range_low")
    bw_txt = f"{bw:.2f}%" if isinstance(bw, (int, float)) else "—"
    hi_txt = _fmt(r_hi) if isinstance(r_hi, (int, float)) else "—"
    lo_txt = _fmt(r_lo) if isinstance(r_lo, (int, float)) else "—"
    label = {
        "break_up": "🚀 Пробой вверх",
        "break_down": "📉 Пробой вниз",
        "possible_up": "⚠️ Возможен пробой вверх (squeeze)",
        "possible_down": "⚠️ Возможен пробой вниз (squeeze)",
    }.get(state, "ℹ️ Сигнал")

    return (
        "🔔 BREAKOUT ALERT\n"
        "━━━━━━━━━━━━\n"
        f"Пара: {symbol}\n"
        f"Состояние: {label}\n"
        f"Цена: {price}\n"
        f"ТФ: {tf}\n"
        f"Биржа: {ex}\n"
        "━━━━━━━━━━━━\n"
        f"H/L({LOOKBACK_RANGE_BY_TF.get(tf, 50)}): {hi_txt} / {lo_txt}\n"
        f"BB width: {bw_txt}\n"
        f"Обновлено: {_now_utc().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        "━━━━━━━━━━━━"
    )

def _limits_by_tf(tf: str) -> dict:
    return {
        "lookback": LOOKBACK_RANGE_BY_TF.get(tf, 50),
        "squeeze": BB_SQUEEZE_PCT_BY_TF.get(tf, 4.0),
        "prox": PROXIMITY_PCT_BY_TF.get(tf, 0.7),
        "eps": BREAK_EPS_PCT_BY_TF.get(tf, 0.15),
    }

async def _check_symbol(symbol: str, tf: str):
    limit = 400 if tf in ("1h", "4h", "1d") else 600  # побольше свечей на младших ТФ
    df, ex = await get_candles(symbol, tf, limit=limit)
    if df.empty:
        return None

    closes: List[float] = df["close"].astype(float).tolist()
    highs:  List[float] = df["high"].astype(float).tolist()
    lows:   List[float] = df["low"].astype(float).tolist()

    p = _limits_by_tf(tf)
    d = classify_breakout(
        closes=closes,
        highs=highs,
        lows=lows,
        period_bb=BB_PERIOD,
        bb_k=BB_K,
        lookback_range=p["lookback"],
        bb_squeeze_pct=p["squeeze"],
        proximity_pct=p["prox"],
        break_eps_pct=p["eps"],
    )
    state = d.get("state", "none")
    if state == "none":
        return None

    ref_level = d.get("range_high") if "up" in state else d.get("range_low")
    if ref_level is None:
        ref_level = d.get("price", 0.0)

    if not _should_alert(tf, symbol, state, float(ref_level)):
        return None

    msg = _format_alert(symbol, tf, ex, d)
    return msg

async def breakout_job(context):
    """
    JobQueue callback. TF берём из job.data["tf"] (по умолчанию 1h)
    """
    tf = (context.job.data or {}).get("tf", "1h")
    try:
        symbols = get_favorites()
        tasks = [asyncio.create_task(_check_symbol(s, tf)) for s in symbols]
        for t in tasks:
            try:
                msg = await t
                if msg:
                    await context.bot.send_message(
                        chat_id=context.job.chat_id or context._chat_id,
                        text=msg
                    )
            except Exception:
                log.exception("breakout task error")
    except Exception:
        log.exception("breakout job error")
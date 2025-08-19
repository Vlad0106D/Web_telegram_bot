# bot/watcher.py
import time
from typing import Dict
from telegram.ext import ContextTypes
from services.market_client import fetch_klines  # OKX→KuCoin фейловер
from strategy.breakout_watcher import classify_breakout
from bot.watchlist_provider import get_watchlist_from_context
from config import (
    WATCHER_INTERVAL_SEC, LOOKBACK_RANGE, BB_PERIOD, BB_STD,
    BB_SQUEEZE_PCT, PROXIMITY_PCT, BREAK_EPS_PCT, COOLDOWN_MIN, ALERT_CHAT_ID
)

LAST_ALERTS: Dict[str, float] = {}

def _cooldown_ok(symbol: str, state: str) -> bool:
    key = f"{symbol}:{state}"
    last = LAST_ALERTS.get(key, 0.0)
    if time.time() - last >= COOLDOWN_MIN * 60:
        LAST_ALERTS[key] = time.time()
        return True
    return False

def _fmt(symbol: str, state: str, price: float, bbw: float, r_high: float, r_low: float) -> str:
    arrow = "⬆️" if "up" in state else "⬇️"
    title = "ВОЗМОЖЕН ПРОБОЙ" if "possible" in state else "ПРОБОЙ"
    lines = [
        f"💎 {title} {arrow}",
        "━━━━━━━━━━━━",
        f"🔹 Пара: {symbol}",
        f"💵 Цена: {price:,.4f}".replace(",", " "),
        f"📊 BB width: {bbw:.2f}%",
        f"📏 Диапазон: H={r_high:.4f} • L={r_low:.4f}",
        "🕒 ТФ: 15m (контекст 1h)",
        "🗓 Источник: OKX / KuCoin",
        "━━━━━━━━━━━━",
    ]
    return "\n".join(lines)

async def breakout_job(context: ContextTypes.DEFAULT_TYPE):
    symbols = get_watchlist_from_context(context)
    if not symbols:
        # Ничего не делаем, ждём, пока пользователь добавит пары через /list
        return

    for sym in symbols:
        try:
            kl15 = await fetch_klines(sym, interval="15", limit=max(200, LOOKBACK_RANGE+BB_PERIOD+5))
            if len(kl15) < LOOKBACK_RANGE + BB_PERIOD:
                continue

            closes = [k["close"] for k in kl15]
            highs  = [k["high"]  for k in kl15]
            lows   = [k["low"]   for k in kl15]

            res = classify_breakout(
                closes, highs, lows,
                period_bb=BB_PERIOD, bb_k=BB_STD, lookback_range=LOOKBACK_RANGE,
                bb_squeeze_pct=BB_SQUEEZE_PCT, proximity_pct=PROXIMITY_PCT, break_eps_pct=BREAK_EPS_PCT
            )
            state = res.get("state", "none")
            if state == "none" or not _cooldown_ok(sym, state):
                continue

            text = _fmt(sym, state, res["price"], res["bb_width_pct"], res["range_high"], res["range_low"])
            await context.bot.send_message(chat_id=ALERT_CHAT_ID, text=text, disable_web_page_preview=True)

        except Exception:
            # Не спамим ошибками; market_client сам фейловерит OKX→KuCoin
            continue
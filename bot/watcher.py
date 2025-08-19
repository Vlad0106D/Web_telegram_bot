# bot/watcher.py
import logging
from typing import List, Dict

from telegram.ext import ContextTypes

from config import ALERT_CHAT_ID
from services.state import get_favorites
from services.market_data import get_candles
from strategy.breakout_watcher import classify_breakout

log = logging.getLogger(__name__)

# Параметры классификатора пробоя — можно будет вынести в .env при желании
BB_PERIOD = 20
BB_STD_K = 2.0
LOOKBACK_RANGE = 50          # сколько последних свечей берём для локального high/low
BB_SQUEEZE_PCT = 8.0         # «узкие» Боллинджеры (в процентах)
PROXIMITY_PCT = 0.6          # близость к границе/уровню (в процентах)
BREAK_EPS_PCT = 0.15         # на сколько % выше/ниже уровня считаем пробой

STATE_EMOJI = {
    "break_up": "🚀",
    "break_down": "📉",
    "possible_up": "🟢",
    "possible_down": "🔴",
    "none": "·",
}

def _fmt_price(x: float) -> str:
    if x >= 1000:
        return f"{x:,.2f}".replace(",", " ")
    if x >= 1:
        return f"{x:.3f}"
    return f"{x:.6f}".rstrip("0").rstrip(".")

def _fmt_msg(symbol: str, tf: str, res: Dict) -> str:
    st = res.get("state", "none")
    em = STATE_EMOJI.get(st, "·")
    px = _fmt_price(res.get("price", 0.0))
    rb = res.get("range_high")
    rl = res.get("range_low")
    bbw = res.get("bb_width_pct")
    parts = [
        f"{em} {symbol} • {tf}",
        f"Цена: {px}",
        f"Диапазон {LOOKBACK_RANGE}: H={_fmt_price(rb)} / L={_fmt_price(rl)}" if rb and rl else "",
        f"BB width≈{bbw:.2f}%" if isinstance(bbw, (int, float)) else "",
        f"Сигнал: {st}",
    ]
    text = "\n".join([p for p in parts if p])
    return "🔔 Breakout Watch\n" + text

async def _scan_one_tf(context: ContextTypes.DEFAULT_TYPE, chat_id: int, symbols: List[str], tf: str) -> None:
    for sym in symbols:
        try:
            df, _ex = await get_candles(sym, tf, limit=BB_PERIOD + max(LOOKBACK_RANGE, 60))
            if df is None or df.empty:
                continue

            closes = df["close"].astype(float).tolist()
            highs = df["high"].astype(float).tolist()
            lows  = df["low"].astype(float).tolist()

            res = classify_breakout(
                closes=closes,
                highs=highs,
                lows=lows,
                period_bb=BB_PERIOD,
                bb_k=BB_STD_K,
                lookback_range=LOOKBACK_RANGE,
                bb_squeeze_pct=BB_SQUEEZE_PCT,
                proximity_pct=PROXIMITY_PCT,
                break_eps_pct=BREAK_EPS_PCT,
            )

            state = res.get("state", "none")
            if state in ("break_up", "break_down", "possible_up", "possible_down"):
                await context.bot.send_message(chat_id=chat_id, text=_fmt_msg(sym, tf, res))

        except Exception:
            log.exception("watcher: error on %s %s", sym, tf)

async def breakout_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Периодически сканирует избранные пары на нескольких ТФ и шлёт алерты при пробоях/квазипробоях.
    chat_id берём из job.chat_id (если был задан при run_repeating) или из конфигурации.
    """
    job = getattr(context, "job", None)
    chat_id = getattr(job, "chat_id", None) or ALERT_CHAT_ID
    if not chat_id:
        log.warning("breakout_job: no chat_id configured, skip send")
        return

    data = getattr(job, "data", {}) if job else {}
    tfs: List[str] = data.get("tfs") or ["1h"]

    # Список тикеров — текущие «избранные»
    symbols = get_favorites()
    if not symbols:
        return

    for tf in tfs:
        await _scan_one_tf(context, chat_id, symbols, tf)
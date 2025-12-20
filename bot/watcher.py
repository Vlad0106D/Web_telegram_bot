from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import Iterable, Sequence, List, Dict, Any, Tuple

from telegram import Update
from telegram.ext import (
    Application,
    ContextTypes,
    CommandHandler,
)

# базовые настройки и брейкер
from config import (
    WATCHER_TFS,
    WATCHER_INTERVAL_SEC,
    ALERT_CHAT_ID,
    BREAKER_LOOKBACK,
    BREAKER_EPS,
    BREAKER_COOLDOWN_SEC,
)

# опциональные настройки для «старого» вотчера (если нет в config — используем дефолты)
try:
    from config import SIGNAL_MIN_CONF  # int
except Exception:
    SIGNAL_MIN_CONF = 70

try:
    from config import SIGNAL_COOLDOWN_SEC  # int (сек)
except Exception:
    SIGNAL_COOLDOWN_SEC = 900

# опциональный кулдаун для разворотов
try:
    from config import REVERSAL_COOLDOWN_SEC
except Exception:
    REVERSAL_COOLDOWN_SEC = 900

# Fusion настройки
try:
    from config import FUSION_ENABLED, FUSION_COOLDOWN_SEC
except Exception:
    FUSION_ENABLED, FUSION_COOLDOWN_SEC = True, 900

# --- FIBO настройки ---
try:
    from config import FIBO_ENABLED, FIBO_TFS, FIBO_COOLDOWN_SEC
except Exception:
    FIBO_ENABLED, FIBO_TFS, FIBO_COOLDOWN_SEC = False, [], 1200

from services.state import get_favorites
from services.breaker import detect_breakout, format_breakout_message
from services.reversal import detect_reversals, format_reversal_message

from services.analyze import analyze_symbol
from services.signal_text import build_signal_message

from services.fusion import analyze_fusion, format_fusion_message

# новый модуль Фибо
from strategy.fibo_watcher import analyze_fibo, format_fibo_message

# агрегатор ATTENTION (TrueTrading без торговли)
from services.true_trading import get_tt

log = logging.getLogger(__name__)

__all__ = [
    "schedule_watcher_jobs",
    "register_watch_handlers",
    "cmd_watch_on",
    "cmd_watch_off",
    "cmd_watch_status",
]

# ----------------------- утилиты -----------------------

def _normalize_tfs(tfs: Iterable[str] | None) -> List[str]:
    if not tfs:
        return []
    uniq: List[str] = []
    for tf in tfs:
        s = str(tf).strip()
        if s and s not in uniq:
            uniq.append(s)
    return uniq


def _job_name(tf: str) -> str:
    return f"watch_{tf}"


def _jobs_summary(app: Application) -> str:
    jq = app.job_queue
    names = []
    for tf in _normalize_tfs(WATCHER_TFS):
        if jq.get_jobs_by_name(_job_name(tf)):
            names.append(_job_name(tf))
    return ", ".join(names) if names else "—"


# ----------------------- тик вотчера -----------------------

async def _watch_tick(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Один тик вотчера для конкретного TF.
    Блоки:
      1) Breaker (пробой диапазона)
      2) Strategy (analyze_symbol, порог уверенности)
      3) Reversal (RSI-дивергенции 1h/4h + импульсные развороты 5m/10m)
      4) Fusion (сводный сигнал по конвергенции модулей)
      5) Fibo (уровни Фибоначчи)
      6) ATTENTION (объединённый сигнал Fibo + Fusion)
    """
    data: Dict[str, Any] = context.job.data or {}
    tf: str = data.get("tf", "?")
    chat_id: int | None = data.get("chat_id") or ALERT_CHAT_ID

    app = context.application
    now = time.time()

    # Кулдауны в app.bot_data
    breaker_last: Dict[Tuple[str, str, str], float] = app.bot_data.setdefault("breaker_last", {})
    signal_last: Dict[Tuple[str, str, str], float] = app.bot_data.setdefault("signal_last", {})
    reversal_last: Dict[Tuple[str, str, str], float] = app.bot_data.setdefault("reversal_last", {})
    fusion_last: Dict[Tuple[str, str, str], float] = app.bot_data.setdefault("fusion_last", {})
    fibo_last: Dict[Tuple[str, str, str, float, str], float] = app.bot_data.setdefault("fibo_last", {})

    try:
        favs = get_favorites()
        if not favs:
            log.info("Watcher tick: tf=%s (favorites empty)", tf)
            return

        sent_breaker = 0
        sent_signal = 0
        sent_reversal = 0
        sent_fusion = 0
        sent_fibo = 0
        sent_attention = 0

        for sym in favs:
            # ---------- 1) BREAKER ----------
            try:
                ev = await detect_breakout(
                    symbol=sym,
                    tf=tf,
                    lookback=BREAKER_LOOKBACK,
                    eps=BREAKER_EPS,
                )
                if ev:
                    key_b = (ev.symbol, ev.tf, ev.direction)
                    last_ts = breaker_last.get(key_b, 0.0)
                    if now - last_ts >= float(BREAKER_COOLDOWN_SEC):
                        if chat_id:
                            await context.bot.send_message(
                                chat_id=chat_id,
                                text=format_breakout_message(ev),
                                parse_mode="HTML",
                            )
                        breaker_last[key_b] = now
                        sent_breaker += 1
            except Exception:
                log.exception("Breaker failed for %s %s", sym, tf)

            # ---------- 2) STRATEGY ----------
            try:
                # ВАЖНО: analyze_symbol может быть без аргумента tf.
                # Делаем совместимую попытку.
                try:
                    res = await analyze_symbol(sym, tf=tf)
                except TypeError:
                    res = await analyze_symbol(sym)

                signal = (res.get("signal") or res.get("direction") or "none").lower()
                conf = int(res.get("confidence") or 0)

                if signal in ("long", "short") and conf >= int(SIGNAL_MIN_CONF):
                    key_s = (res.get("symbol", sym).upper(), tf, signal)
                    last_ts = signal_last.get(key_s, 0.0)
                    if now - last_ts >= float(SIGNAL_COOLDOWN_SEC):
                        if chat_id:
                            await context.bot.send_message(
                                chat_id=chat_id,
                                text=build_signal_message(res),
                                parse_mode="HTML",
                            )
                        signal_last[key_s] = now
                        sent_signal += 1
            except Exception:
                log.exception("Strategy analyze failed for %s %s", sym, tf)

            # ---------- 3) REVERSAL ----------
            try:
                rev_events = await detect_reversals(sym)
                for ev in rev_events:
                    key_r = (ev.symbol, ev.tf, ev.kind)
                    last_ts = reversal_last.get(key_r, 0.0)
                    if now - last_ts >= float(REVERSAL_COOLDOWN_SEC):
                        if chat_id:
                            await context.bot.send_message(
                                chat_id=chat_id,
                                text=format_reversal_message(ev),
                                parse_mode="HTML",
                            )
                        reversal_last[key_r] = now
                        sent_reversal += 1
            except Exception:
                log.exception("Reversal detection failed for %s", sym)

            # ---------- 4) FUSION ----------
            try:
                if FUSION_ENABLED:
                    fev = await analyze_fusion(sym, tf)
                    if fev:
                        key_f = (fev.symbol, fev.tf, fev.side)
                        last_ts = fusion_last.get(key_f, 0.0)
                        if now - last_ts >= float(FUSION_COOLDOWN_SEC):
                            if chat_id:
                                await context.bot.send_message(
                                    chat_id=chat_id,
                                    text=format_fusion_message(fev),
                                    parse_mode="HTML",
                                )
                            fusion_last[key_f] = now
                            sent_fusion += 1

                        # кэш для ATTENTION
                        try:
                            get_tt(app).update_fusion(
                                sym, tf,
                                {
                                    "symbol": fev.symbol,
                                    "tf": fev.tf,
                                    "side": getattr(fev, "side", None),
                                    "score": int(getattr(fev, "score", getattr(fev, "confidence", 0) or 0)),
                                    "trend1d": getattr(fev, "trend_1d", None),
                                },
                            )
                        except Exception:
                            log.exception("Failed to cache fusion for ATTENTION %s %s", sym, tf)
            except Exception:
                log.exception("Fusion analysis failed for %s %s", sym, tf)

            # ---------- 5) FIBO ----------
            try:
                if FIBO_ENABLED and tf in FIBO_TFS:
                    fibo_events = await analyze_fibo(sym, tf)
                    for ev in fibo_events:
                        key_fi = (ev.symbol, ev.tf, ev.side, round(ev.level_pct, 1), ev.scenario)
                        last_ts = fibo_last.get(key_fi, 0.0)
                        if now - last_ts >= float(FIBO_COOLDOWN_SEC):
                            if chat_id:
                                await context.bot.send_message(
                                    chat_id=chat_id,
                                    text=format_fibo_message(ev),
                                    parse_mode="HTML",
                                )
                            fibo_last[key_fi] = now
                            sent_fibo += 1

                        # кэш для ATTENTION (берём план сделки из Фибо)
                        try:
                            get_tt(app).update_fibo(
                                sym, tf,
                                {
                                    "symbol": ev.symbol,
                                    "tf": ev.tf,
                                    "side": getattr(ev, "side", "").lower(),
                                    "level_kind": getattr(ev, "level_kind", None),
                                    "level_pct": float(getattr(ev, "level_pct", 0.0)),
                                    "trend_1d": getattr(ev, "trend_1d", None),

                                    # торговый план
                                    "entry": float(getattr(ev, "entry", getattr(ev, "touch_price", 0.0))),
                                    "sl": float(getattr(ev, "sl", 0.0)),
                                    "tp1": float(getattr(ev, "tp1", 0.0)),
                                    "tp2": float(getattr(ev, "tp2", 0.0)),
                                    "tp3": float(getattr(ev, "tp3", 0.0)),

                                    "rr_tp1": float(getattr(ev, "rr_tp1", 0.0)),
                                    "rr_tp2": float(getattr(ev, "rr_tp2", 0.0)),
                                    "rr_tp3": float(getattr(ev, "rr_tp3", 0.0)),
                                },
                            )
                        except Exception:
                            log.exception("Failed to cache fibo for ATTENTION %s %s", sym, tf)
            except Exception:
                log.exception("Fibo watcher failed for %s %s", sym, tf)

            # ---------- 6) ATTENTION (Fibo + Fusion агрегатор) ----------
            try:
                attn_ev = await get_tt(app).maybe_send_attention(context, chat_id, sym, tf)
                if attn_ev:
                    sent_attention += 1
            except Exception:
                log.exception("ATTENTION aggregator failed for %s %s", sym, tf)

        log.info(
            "Watcher tick: tf=%s, favorites=%d, alerts: breaker=%d, strategy=%d, reversal=%d, fusion=%d, fibo=%d, attention=%d",
            tf, len(favs), sent_breaker, sent_signal, sent_reversal, sent_fusion, sent_fibo, sent_attention
        )

    except Exception:
        log.exception("Watcher tick failed for tf=%s", tf)


# ----------------------- планировщик -----------------------

def schedule_watcher_jobs(
    app: Application,
    tfs: Iterable[str],
    interval_sec: int,
    chat_id: int | None = None,
) -> Sequence[str]:
    jq = app.job_queue
    created: List[str] = []
    norm_tfs = _normalize_tfs(tfs)

    if chat_id is not None:
        app.bot_data["watch_chat_id"] = int(chat_id)
    default_chat = app.bot_data.get("watch_chat_id") or ALERT_CHAT_ID

    for tf in norm_tfs:
        name = _job_name(tf)

        # удалим все одноимённые задачи (если были)
        for old in jq.get_jobs_by_name(name):
            try:
                old.schedule_removal()
            except Exception:
                log.exception("Failed to remove old job '%s'", name)

        jq.run_repeating(
            _watch_tick,
            interval=timedelta(seconds=int(interval_sec)),
            first=5,
            name=name,
            data={"tf": tf, "chat_id": default_chat},
        )
        created.append(name)
        log.info(
            "Watcher job scheduled: name=%s, interval=%ss, chat_id=%s",
            name, interval_sec, default_chat,
        )

    log.info(
        "Watcher scheduled for TFs=%s, interval=%ss",
        ", ".join(norm_tfs) if norm_tfs else "[]",
        interval_sec,
    )
    return created


def _stop_all_watcher_jobs(app: Application) -> int:
    jq = app.job_queue
    count = 0
    for tf in _normalize_tfs(WATCHER_TFS):
        name = _job_name(tf)
        jobs = jq.get_jobs_by_name(name)
        for j in jobs:
            try:
                j.schedule_removal()
                count += 1
            except Exception:
                log.exception("Failed to remove job '%s'", name)
    return count


# ----------------------- команды -----------------------

async def cmd_watch_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    chat_id = update.effective_chat.id if update.effective_chat else ALERT_CHAT_ID
    app.bot_data["watch_chat_id"] = int(chat_id)

    created = schedule_watcher_jobs(
        app=app,
        tfs=WATCHER_TFS,
        interval_sec=int(WATCHER_INTERVAL_SEC),
        chat_id=chat_id,
    )

    text = (
        "✅ Вотчер запущен.\n"
        f"Чат: <code>{chat_id}</code>\n"
        f"TF: {', '.join(_normalize_tfs(WATCHER_TFS)) or '—'}\n"
        f"Интервал: {WATCHER_INTERVAL_SEC} сек.\n"
        f"Порог сигнала: {SIGNAL_MIN_CONF}\n"
        f"Jobs: {', '.join(created) if created else '—'}"
    )
    if update.effective_message:
        await update.effective_message.reply_text(text, parse_mode="HTML")


async def cmd_watch_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    removed = _stop_all_watcher_jobs(app)
    text = f"⛔ Вотчер остановлен. Удалено задач: {removed}"
    if update.effective_message:
        await update.effective_message.reply_text(text)


async def cmd_watch_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    app = context.application
    chat_id = app.bot_data.get("watch_chat_id") or ALERT_CHAT_ID
    jobs = _jobs_summary(app)
    text = (
        "📟 Статус вотчера\n"
        f"Чат: <code>{chat_id or '—'}</code>\n"
        f"TF: {', '.join(_normalize_tfs(WATCHER_TFS)) or '—'}\n"
        f"Интервал: {WATCHER_INTERVAL_SEC} сек.\n"
        f"Порог сигнала: {SIGNAL_MIN_CONF}\n"
        f"Активные jobs: {jobs}"
    )
    if update.effective_message:
        await update.effective_message.reply_text(text, parse_mode="HTML")


def register_watch_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("watch_on", cmd_watch_on))
    app.add_handler(CommandHandler("watch_off", cmd_watch_off))
    app.add_handler(CommandHandler("watch_status", cmd_watch_status))
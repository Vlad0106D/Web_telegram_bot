# services/true_trading.py
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

# Базовые флаги TrueTrading (вкл/выкл и параметры риска пока не используются для торговли)
from config import (
    TT_ENABLED, TT_RISK_PCT, TT_MAX_OPEN_POS, TT_DAILY_LOSS_LIMIT_PCT,
    TT_SYMBOL_COOLDOWN_MIN, TT_ORDER_SLIPPAGE_BPS, TT_REQUIRE_1D_TREND,
    TT_MIN_RR_TP1,
    BYBIT_API_KEY, BYBIT_API_SECRET,  # только для статуса "ключи есть/нет", торговля не ведётся
)

# Настройки агрегатора ATTENTION (если нет в config.py — возьмём дефолты здесь)
try:
    from config import (
        ATTN_ENABLED, ATTN_FUSION_MIN, ATTN_REQUIRE_1D_TREND,
        ATTN_MIN_RR_TP1, ATTN_COOLDOWN_SEC,
    )
except Exception:
    ATTN_ENABLED = True
    ATTN_FUSION_MIN = 72
    ATTN_REQUIRE_1D_TREND = True
    ATTN_MIN_RR_TP1 = 1.6
    ATTN_COOLDOWN_SEC = 900


@dataclass
class TTStatus:
    enabled: bool
    since_ts: Optional[int]
    # ниже — для информации в /tt_status (хотя мы не торгуем)
    risk_pct: float
    max_open_pos: int
    daily_loss_limit_pct: float
    symbol_cooldown_min: int
    slippage_bps: int
    require_trend_1d: bool
    min_rr_tp1: float
    exchange_connected: bool  # просто "ключи заданы или нет"


@dataclass
class AttentionEvent:
    symbol: str
    tf: str
    side: str                # "long" | "short"
    fusion_score: int
    fibo_level: str          # "retr 38.2%" / "ext 161.8%"
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    rr_tp1: Optional[float]
    rr_tp2: Optional[float]
    rr_tp3: Optional[float]
    bad_sl: bool = False     # SL по «не той» стороне


# ---------- локальные утилиты ----------

def _fmt_price(x: float) -> str:
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}".rstrip("0").rstrip(".")

def _fmt_rr(rr: Optional[float], bad_side: bool = False) -> str:
    if rr is None:
        return "(RR=–)"
    # защита от «космоса»
    if rr > 9999:
        return "(RR>9999)"
    if rr < -9999:
        return "(RR<-9999)"
    s = f"{rr:.2f}"
    if (rr < 0) or bad_side:
        return f"(RR={s} ⚠)"
    return f"(RR={s})"

def _norm_side(v: Any) -> str:
    s = str(v or "").strip().lower()
    if s in ("long", "buy", "bull", "up"):
        return "long"
    if s in ("short", "sell", "bear", "down"):
        return "short"
    return "none"

def _norm_trend(v: Any) -> str:
    s = str(v or "").strip().lower()
    if s in ("up", "bull", "bullish", "long", "buy", "1", "true", "yes"):
        return "up"
    if s in ("down", "bear", "bearish", "short", "sell", "-1", "false", "no"):
        return "down"
    return "none"

def _as_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _safe_rr(side: str, entry: float, sl: float, tp: float) -> Tuple[Optional[float], bool]:
    """
    Считает RR безопасно. Возвращает (rr, bad_sl),
    где bad_sl=True если SL/TP на «не тех» сторонах.
    RR положительный — TP на «правильной» стороне; отрицательный — TP на «неправильной».
    """
    side = _norm_side(side)
    e = float(entry); s = float(sl); t = float(tp)

    # проверка «плохого SL»
    bad_sl = (side == "long" and s >= e) or (side == "short" and s <= e)

    # риск считаем по модулю: это убирает деление на ~0, когда SL ошибочно по другой стороне
    risk = abs(e - s)
    if risk <= 0:
        return None, True

    reward = abs(t - e)
    rr = reward / risk

    # знак RR по стороне TP
    if (side == "long" and t <= e) or (side == "short" and t >= e):
        rr = -rr  # TP не на той стороне

    return rr, bad_sl

def _confluence_hit(
    fibo_price: Optional[float],
    fusion_center: Optional[float],
    fusion_halfwidth: Optional[float],
    *,
    allow_abs: Optional[float] = None,
    allow_pct: float = 0.0015,  # ~0.15% допуск
) -> bool:
    """
    Истинный конфлюэнс по цене: Фибо-уровень попадает в зону Fusion (center ± halfwidth),
    расширенную на допуск (abs или %). Если данных Fusion нет — возвращаем True (не блокируем).
    """
    if fibo_price is None or fusion_center is None:
        return True  # нет данных для строгой проверки — не режем
    half = abs(fusion_halfwidth or 0.0)
    allow = allow_abs if allow_abs is not None else abs(fusion_center) * allow_pct
    return abs(fibo_price - fusion_center) <= (half + allow)

# ---------- формат ATTENTION ----------

def format_attention_message(ev: AttentionEvent) -> str:
    arrow = "↑" if ev.side == "long" else "↓"
    return (
        "⚠️ <b>[ATTENTION]</b>\n"
        f"{ev.symbol} {ev.tf}\n"
        f"{ev.side.upper()} {arrow} | {ev.fibo_level} | Fusion={ev.fusion_score}\n"
        "━━━━━━━━━━━━\n"
        f"Вход: <code>{_fmt_price(ev.entry)}</code>\n"
        f"SL:   <code>{_fmt_price(ev.sl)}</code>{'  ⚠ неверная сторона SL' if ev.bad_sl else ''}\n"
        f"TP1:  <code>{_fmt_price(ev.tp1)}</code> {_fmt_rr(ev.rr_tp1, ev.bad_sl)}\n"
        f"TP2:  <code>{_fmt_price(ev.tp2)}</code> {_fmt_rr(ev.rr_tp2, ev.bad_sl)}\n"
        f"TP3:  <code>{_fmt_price(ev.tp3)}</code> {_fmt_rr(ev.rr_tp3, ev.bad_sl)}\n"
        "━━━━━━━━━━━━"
    )


class TrueTrading:
    """
    Агрегатор Fibo + Fusion:
      - НЕ торгует, только шлёт объединённые сигналы [ATTENTION]
      - хранит флаги, кэши последних fibo/fusion по (symbol, tf)
      - кулдаун для антиспама
    """
    def __init__(self) -> None:
        # Раньше self._enabled зависел от TT_ENABLED, из-за чего ATTENTION молчал при TT_ENABLED=False.
        # Держим агрегатор включённым независимо от реальной торговли.
        self._enabled: bool = True
        self._since_ts: Optional[int] = int(time.time())

        # Кэш последних событий по ключу (symbol, tf)
        self._last_fibo: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._last_fusion: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # Кулдаун по отправке ATTENTION: ключ (symbol, tf, side) -> ts
        self._attention_last: Dict[Tuple[str, str, str], float] = {}

        # Для статуса — просто признак, что ключи заданы (торговли всё равно нет)
        self._exchange_connected = bool(BYBIT_API_KEY and BYBIT_API_SECRET)

    # ---------- ВКЛ/ВЫКЛ/СТАТУС ----------
    def enable(self) -> None:
        if not self._enabled:
            self._enabled = True
            self._since_ts = int(time.time())

    def disable(self) -> None:
        self._enabled = False

    def is_enabled(self) -> bool:
        return self._enabled

    def status(self) -> TTStatus:
        return TTStatus(
            enabled=self._enabled,
            since_ts=self._since_ts,
            risk_pct=TT_RISK_PCT,
            max_open_pos=TT_MAX_OPEN_POS,
            daily_loss_limit_pct=TT_DAILY_LOSS_LIMIT_PCT,
            symbol_cooldown_min=TT_SYMBOL_COOLDOWN_MIN,
            slippage_bps=TT_ORDER_SLIPPAGE_BPS,
            require_trend_1d=TT_REQUIRE_1D_TREND,
            min_rr_tp1=TT_MIN_RR_TP1,
            exchange_connected=self._exchange_connected,
        )

    # ---------- Приём данных от блоков ----------
    def update_fibo(self, symbol: str, tf: str, fibo_dict: Dict[str, Any]) -> None:
        """
        Ожидаем словарь из fibo_watcher с полями как у FiboEvent:
          side, level_kind, level_pct, entry, sl, tp1..tp3, rr_tp1..rr_tp3, trend_1d ...
        """
        key = (symbol.upper(), tf)
        self._last_fibo[key] = dict(fibo_dict or {})

    def update_fusion(self, symbol: str, tf: str, fusion_dict: Dict[str, Any]) -> None:
        """
        Ожидаем словарь из analyze_fusion:
          side, score (или confidence), trend1d (опц.), zone_center/zone_halfwidth (опц.), tf, symbol ...
        """
        key = (symbol.upper(), tf)
        fd = dict(fusion_dict or {})
        # Поддержка старого поля confidence, если score не передали
        if "score" not in fd and "confidence" in fd:
            try:
                fd["score"] = int(fd.get("confidence") or 0)
            except Exception:
                fd["score"] = 0
        self._last_fusion[key] = fd

    # ---------- Решение и отправка ATTENTION ----------
    async def maybe_send_attention(
        self,
        context,
        chat_id: Optional[int],
        symbol: str,
        tf: str,
        *,
        debug: bool = False
    ) -> Optional[AttentionEvent]:
        """
        Если TrueTrading включён, есть свежие fibo+fusion по (symbol, tf) и они проходят пороги —
        отправляет единый [ATTENTION]-сигнал в Telegram и возвращает AttentionEvent.
        """
        # ВАЖНО: не зависим от TT_ENABLED — ATTENTION работает автономно
        if not ATTN_ENABLED or not chat_id:
            return None

        key = (symbol.upper(), tf)
        fibo = self._last_fibo.get(key) or {}
        fusion = self._last_fusion.get(key) or {}
        if not fibo or not fusion:
            if debug:
                await context.bot.send_message(chat_id=chat_id, text=f"[ATTN:SKIP] {key} — нет fibo/fusion данных")
            return None

        # 1) Совпадение направления
        side_fibo = _norm_side(fibo.get("side"))
        side_fusion = _norm_side(fusion.get("side"))
        if side_fibo not in ("long", "short") or side_fusion != side_fibo:
            if debug:
                await context.bot.send_message(chat_id=chat_id, text=f"[ATTN:SKIP] {key} — side mismatch: fibo={side_fibo}, fusion={side_fusion}")
            return None

        # 2) Порог Fusion (score или confidence)
        try:
            score = int(fusion.get("score") or fusion.get("confidence") or 0)
        except Exception:
            score = 0
        if score < int(ATTN_FUSION_MIN):
            if debug:
                await context.bot.send_message(chat_id=chat_id, text=f"[ATTN:SKIP] {key} — fusion.score {score} < {ATTN_FUSION_MIN}")
            return None

        # 3) Тренд 1D (по флагу)
        if ATTN_REQUIRE_1D_TREND:
            t1d_raw = fibo.get("trend_1d") or fusion.get("trend1d")
            t1d = _norm_trend(t1d_raw)
            need = "up" if side_fibo == "long" else "down"
            if t1d != need:
                if debug:
                    await context.bot.send_message(chat_id=chat_id, text=f"[ATTN:SKIP] {key} — trend1d={t1d} need={need}")
                return None

        # 4) Уровни из Fibo (не доверяем rr_*, пересчитаем сами)
        try:
            entry = float(fibo["entry"])
            sl    = float(fibo["sl"])
            tp1   = float(fibo["tp1"])
            tp2   = float(fibo["tp2"])
            tp3   = float(fibo["tp3"])
        except Exception:
            if debug:
                await context.bot.send_message(chat_id=chat_id, text=f"[ATTN:SKIP] {key} — неполные Fibo уровни")
            return None

        # 4.1 Истинный конфлюэнс по цене (если Fusion прислал зону)
        fusion_center = _as_float(fusion.get("zone_center") or fusion.get("level_price"))
        fusion_half   = _as_float(fusion.get("zone_halfwidth") or fusion.get("band"))
        fibo_ref_price = entry  # можно заменить на конкретный уровень зоны Фибо, если требуется
        if not _confluence_hit(fibo_ref_price, fusion_center, fusion_half):
            if debug:
                fc = _fmt_price(fusion_center) if fusion_center is not None else "—"
                fh = _fmt_price(fusion_half or 0.0)
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"[ATTN:SKIP] {key} — no price confluence: fibo≈{_fmt_price(fibo_ref_price)} vs fusion≈{fc}±{fh}"
                )
            return None

        # 5) RR-проверка (TP1 должен быть на «правильной» стороне и RR достаточный)
        rr1, bad_sl_1 = _safe_rr(side_fibo, entry, sl, tp1)
        rr2, bad_sl_2 = _safe_rr(side_fibo, entry, sl, tp2)
        rr3, bad_sl_3 = _safe_rr(side_fibo, entry, sl, tp3)
        bad_sl = bool(bad_sl_1 or bad_sl_2 or bad_sl_3)

        if rr1 is None or rr1 <= 0 or rr1 < float(ATTN_MIN_RR_TP1):
            if debug:
                await context.bot.send_message(chat_id=chat_id, text=f"[ATTN:SKIP] {key} — RR1={rr1} < {ATTN_MIN_RR_TP1}")
            return None

        # 6) Кулдаун
        now = time.time()
        cd_key = (key[0], key[1], side_fibo)
        last_ts = float(self._attention_last.get(cd_key, 0.0))
        if now - last_ts < float(ATTN_COOLDOWN_SEC):
            if debug:
                await context.bot.send_message(chat_id=chat_id, text=f"[ATTN:SKIP] {key} — cooldown {int(now-last_ts)}s/{ATTN_COOLDOWN_SEC}s")
            return None

        # 7) Формирование события
        lvl_kind = str(fibo.get("level_kind") or "ext").lower()
        lvl_pct_raw = fibo.get("level_pct")
        try:
            lvl_pct = float(lvl_pct_raw)
            lvl_str = f"{'retr' if lvl_kind=='retr' else 'ext'} {lvl_pct:.1f}%"
        except Exception:
            lvl_str = f"{'retr' if lvl_kind=='retr' else 'ext'}"

        ev = AttentionEvent(
            symbol=key[0], tf=key[1], side=side_fibo,
            fusion_score=score, fibo_level=lvl_str,
            entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
            rr_tp1=rr1, rr_tp2=rr2, rr_tp3=rr3, bad_sl=bad_sl,
        )

        # 8) Отправка сообщения
        text = format_attention_message(ev)
        try:
            await context.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
            self._attention_last[cd_key] = now
        except Exception:
            # не роняем тикер, просто не записываем кулдаун если отправка не удалась
            pass

        return ev


# --- Singleton доступ --- #

def get_tt(app) -> TrueTrading:
    """Singleton доступ к агрегатору через app.bot_data."""
    tt = app.bot_data.get("true_trading")
    if not isinstance(tt, TrueTrading):
        tt = TrueTrading()
        app.bot_data["true_trading"] = tt
    return tt
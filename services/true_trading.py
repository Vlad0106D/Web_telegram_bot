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
    rr_tp1: float
    rr_tp2: float
    rr_tp3: float


class TrueTrading:
    """
    Агрегатор Fibo + Fusion:
      - НЕ торгует, только шлёт объединённые сигналы [ATTENTION]
      - хранит флаги, кэши последних fibo/fusion по (symbol, tf)
      - кулдаун для антиспама
    """
    def __init__(self) -> None:
        self._enabled: bool = bool(TT_ENABLED)
        self._since_ts: Optional[int] = int(time.time()) if self._enabled else None

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
          side, score, trend1d, tf, symbol ...
        """
        key = (symbol.upper(), tf)
        self._last_fusion[key] = dict(fusion_dict or {})

    # ---------- Решение и отправка ATTENTION ----------
    async def maybe_send_attention(self, context, chat_id: Optional[int], symbol: str, tf: str) -> Optional[AttentionEvent]:
        """
        Если TrueTrading включён, есть свежие fibo+fusion по (symbol, tf) и они проходят пороги —
        отправляет единый [ATTENTION]-сигнал в Telegram и возвращает AttentionEvent.
        """
        if not ATTN_ENABLED or not self._enabled or not chat_id:
            return None

        key = (symbol.upper(), tf)
        fibo = self._last_fibo.get(key) or {}
        fusion = self._last_fusion.get(key) or {}
        if not fibo or not fusion:
            return None

        # 1) Совпадение направления
        side_fibo = (fibo.get("side") or "").lower()
        side_fusion = (fusion.get("side") or "").lower()
        if side_fibo not in ("long", "short") or side_fusion != side_fibo:
            return None

        # 2) Порог Fusion
        score = int(fusion.get("score") or 0)
        if score < int(ATTN_FUSION_MIN):
            return None

        # 3) Тренд 1D (по флагу)
        if ATTN_REQUIRE_1D_TREND:
            t1d = (fibo.get("trend_1d") or fusion.get("trend1d") or "").lower()
            if side_fibo == "long" and t1d != "up":
                return None
            if side_fibo == "short" and t1d != "down":
                return None

        # 4) RR минимальный
        rr1 = float(fibo.get("rr_tp1") or 0.0)
        rr2 = float(fibo.get("rr_tp2") or 0.0)
        rr3 = float(fibo.get("rr_tp3") or 0.0)
        if rr1 < float(ATTN_MIN_RR_TP1):
            return None

        # 5) Кулдаун
        now = time.time()
        cd_key = (key[0], key[1], side_fibo)
        last_ts = float(self._attention_last.get(cd_key, 0.0))
        if now - last_ts < float(ATTN_COOLDOWN_SEC):
            return None

        # 6) Формирование события
        lvl_kind = fibo.get("level_kind")
        lvl_pct = fibo.get("level_pct")
        lvl_str = f"{'retr' if lvl_kind=='retr' else 'ext'} {float(lvl_pct):.1f}%"

        ev = AttentionEvent(
            symbol=key[0], tf=key[1], side=side_fibo,
            fusion_score=score, fibo_level=lvl_str,
            entry=float(fibo["entry"]), sl=float(fibo["sl"]),
            tp1=float(fibo["tp1"]), tp2=float(fibo["tp2"]), tp3=float(fibo["tp3"]),
            rr_tp1=rr1, rr_tp2=rr2, rr_tp3=rr3,
        )

        # 7) Отправка сообщения
        text = format_attention_message(ev)
        try:
            await context.bot.send_message(chat_id=chat_id, text=text)
            self._attention_last[cd_key] = now
        except Exception:
            # не роняем тикер, просто не записываем кулдаун если отправка не удалась
            pass

        return ev


# --- Утилиты форматирования и singleton-доступ --- #

def _fmt_price(x: float) -> str:
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}".rstrip("0").rstrip(".")

def format_attention_message(ev: AttentionEvent) -> str:
    arrow = "↑" if ev.side == "long" else "↓"
    return (
        "⚠️ <b>[ATTENTION]</b>\n"
        f"{ev.symbol} {ev.tf}\n"
        f"{ev.side.upper()} {arrow} | {ev.fibo_level} | Fusion={ev.fusion_score}\n"
        "━━━━━━━━━━━━\n"
        f"Вход: <code>{_fmt_price(ev.entry)}</code>\n"
        f"SL:   <code>{_fmt_price(ev.sl)}</code>\n"
        f"TP1:  <code>{_fmt_price(ev.tp1)}</code> (RR={ev.rr_tp1:.2f})\n"
        f"TP2:  <code>{_fmt_price(ev.tp2)}</code> (RR={ev.rr_tp2:.2f})\n"
        f"TP3:  <code>{_fmt_price(ev.tp3)}</code> (RR={ev.rr_tp3:.2f})\n"
        "━━━━━━━━━━━━"
    )

def get_tt(app) -> TrueTrading:
    """Singleton доступ к агрегатору через app.bot_data."""
    tt = app.bot_data.get("true_trading")
    if not isinstance(tt, TrueTrading):
        tt = TrueTrading()
        app.bot_data["true_trading"] = tt
    return tt
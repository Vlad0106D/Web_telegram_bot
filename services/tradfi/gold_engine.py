from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import httpx

BASE = "https://www.okx.com"
SWAP = "XAU-USDT-SWAP"
INDEX = "XAU-USDT"
BARS = {"M1": "1m", "M5": "5m", "M15": "15m", "H1": "1H"}


class GoldDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class Candle:
    ts: datetime
    o: float
    h: float
    l: float
    c: float
    volume: float


async def _get(client: httpx.AsyncClient, path: str, **params) -> List:
    response = await client.get(BASE + path, params=params)
    response.raise_for_status()
    payload = response.json()
    if str(payload.get("code")) != "0":
        raise GoldDataError(payload.get("msg") or path)
    return payload.get("data") or []


async def _candles(client: httpx.AsyncClient, tf: str) -> List[Candle]:
    rows = await _get(client, "/api/v5/market/candles", instId=SWAP, bar=BARS[tf], limit="200")
    result = [
        Candle(datetime.fromtimestamp(int(x[0]) / 1000, timezone.utc), *map(float, x[1:6]))
        for x in rows if len(x) >= 9 and str(x[8]) == "1"
    ]
    result.sort(key=lambda x: x.ts)
    if len(result) < 60:
        raise GoldDataError(f"Недостаточно закрытых свечей {tf}: {len(result)}")
    return result


def _ema(values: Sequence[float], period: int) -> float:
    value, alpha = float(values[0]), 2.0 / (period + 1)
    for item in values[1:]:
        value = alpha * item + (1 - alpha) * value
    return value


def _atr(items: Sequence[Candle], period: int = 14) -> float:
    values = [
        max(b.h - b.l, abs(b.h - a.c), abs(b.l - a.c))
        for a, b in zip(items[:-1], items[1:])
    ][-period:]
    return sum(values) / len(values)


def _context(items: Sequence[Candle]) -> Tuple[str, int]:
    closes = [x.c for x in items]
    e20, e50 = _ema(closes[-80:], 20), _ema(closes[-120:], 50)
    slope = _ema(closes[-20:], 8) - _ema(closes[-28:-8], 8)
    if closes[-1] > e20 > e50 and slope > 0:
        return "рост / HH-HL", 1
    if closes[-1] < e20 < e50 and slope < 0:
        return "снижение / LH-LL", -1
    if closes[-1] > e20 and slope > 0:
        return "локальное давление вверх", 1
    if closes[-1] < e20 and slope < 0:
        return "локальное давление вниз", -1
    return "диапазон / конфликт", 0


def _levels(price: float, sets: Sequence[Sequence[Candle]]) -> Tuple[Optional[float], Optional[float]]:
    levels = []
    for items in sets:
        for i in range(2, len(items) - 2):
            part = items[i - 2:i] + items[i + 1:i + 3]
            if all(items[i].h >= x.h for x in part):
                levels.append(items[i].h)
            if all(items[i].l <= x.l for x in part):
                levels.append(items[i].l)
    above = sorted({x for x in levels[-80:] if x > price})
    below = sorted({x for x in levels[-80:] if x < price}, reverse=True)
    return (above[0] if above else None, below[0] if below else None)


def _event_chain(items: Sequence[Candle], side: str, level: Optional[float], atr: float) -> Tuple[List[str], int]:
    if level is None or len(items) < 8:
        return [], 0
    recent = items[-8:]
    tolerance = max(0.05, atr * 0.12)
    chain: List[str] = []
    acceptance_at = None
    for i in range(1, len(recent)):
        a, b = recent[i - 1], recent[i]
        accepted = (
            a.c < level - tolerance and b.c < level - tolerance
            if side == "SHORT"
            else a.c > level + tolerance and b.c > level + tolerance
        )
        if accepted and acceptance_at is None:
            acceptance_at = i
    if acceptance_at is not None:
        chain.append("accept below" if side == "SHORT" else "accept above")
        after = recent[acceptance_at + 1:]
        for candle in after:
            retest = (
                candle.h >= level - tolerance and candle.c < level
                if side == "SHORT"
                else candle.l <= level + tolerance and candle.c > level
            )
            if retest:
                chain.append("retest снизу" if side == "SHORT" else "retest сверху")
                rejection = candle.c < candle.o if side == "SHORT" else candle.c > candle.o
                if rejection:
                    chain.append("bearish rejection" if side == "SHORT" else "bullish rejection")
                break
    current = recent[-1]
    previous = recent[-2]
    displacement = (
        current.c < current.o and current.c <= current.l + .25 * max(current.h-current.l, 1e-9)
        if side == "SHORT"
        else current.c > current.o and current.c >= current.h - .25 * max(current.h-current.l, 1e-9)
    )
    sweep = (
        current.h > max(x.h for x in recent[-5:-1]) and current.c < previous.h
        if side == "SHORT"
        else current.l < min(x.l for x in recent[-5:-1]) and current.c > previous.l
    )
    if sweep:
        chain.append("sweep high" if side == "SHORT" else "sweep low")
    if displacement:
        chain.append("bearish displacement" if side == "SHORT" else "bullish displacement")
    chain = list(dict.fromkeys(chain))
    score = min(25, (10 if any("accept" in x for x in chain) else 0)
                + (8 if any("retest" in x for x in chain) else 0)
                + (7 if any("rejection" in x for x in chain) else 0)
                + (9 if any("displacement" in x for x in chain) else 0)
                + (6 if any("sweep" in x for x in chain) else 0))
    return chain, score


def _side_plan(side: str, price: float, above: Optional[float], below: Optional[float],
               atr1: float, atr5: float, votes: Dict[str, int],
               candles: Dict[str, List[Candle]]) -> Dict:
    sign = 1 if side == "LONG" else -1
    higher_raw = 2 * votes["H1"] + 2 * votes["M15"]
    aligned = higher_raw * sign
    setup_type = "TREND" if aligned > 0 else "COUNTERTREND" if aligned < 0 else "NEUTRAL"
    level = below if side == "LONG" else above
    target = above if side == "LONG" else below
    chain, event_score = _event_chain(candles["M1"], side, level, atr1)
    local = sign * (2 * votes["M5"] + votes["M1"])
    context_score = 15 if aligned >= 3 else 10 if aligned > 0 else 5 if aligned < 0 else 7
    structure_score = 20 if local >= 3 else 14 if local >= 1 else 5
    proximity = 0 if level is None else max(0, min(15, round(15 * (1-abs(price-level)/max(1.5*atr5,.01)))))
    buffer = max(.12, .35*atr1)
    stop = (level-buffer if side=="LONG" else level+buffer) if level is not None else None
    risk = abs(price-stop) if stop is not None else 0
    reward = abs(target-price) if target is not None else 0
    rr = reward/risk if risk else 0
    parts = {"context":context_score, "structure":structure_score, "liquidity":proximity,
             "event":event_score, "rr":min(15,round(7.5*rr)), "market":10}
    return {"side":side, "type":setup_type, "score":min(100,sum(parts.values())),
            "parts":parts, "chain":chain, "level":level, "stop":stop,
            "target":target, "rr":rr}

def _n(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def assess_gold_now() -> Dict:
    async with httpx.AsyncClient(timeout=12) as client:
        tasks = [_candles(client, tf) for tf in BARS]
        tasks += [
            _get(client, "/api/v5/market/ticker", instId=SWAP),
            _get(client, "/api/v5/public/mark-price", instType="SWAP", instId=SWAP),
            _get(client, "/api/v5/market/index-tickers", instId=INDEX),
            _get(client, "/api/v5/public/funding-rate", instId=SWAP),
            _get(client, "/api/v5/public/open-interest", instType="SWAP", instId=SWAP),
        ]
        data = await asyncio.gather(*tasks)
    candles = dict(zip(BARS, data[:4]))
    ticker, mark, index, funding, oi = (x[0] if x else {} for x in data[4:])
    price = float(ticker["last"])
    contexts, votes = {}, {}
    for tf in BARS:
        contexts[tf], votes[tf] = _context(candles[tf])
    higher_raw = 2 * votes["H1"] + 2 * votes["M15"]
    higher_bias = "LONG" if higher_raw >= 2 else "SHORT" if higher_raw <= -2 else "NEUTRAL"
    above, below = _levels(price, [candles["M1"], candles["M5"], candles["M15"], candles["H1"]])
    atr1, atr5 = _atr(candles["M1"]), _atr(candles["M5"])
    impulse = (candles["M1"][-1].h-candles["M1"][-1].l)/atr1
    long_plan = _side_plan("LONG", price, above, below, atr1, atr5, votes, candles)
    short_plan = _side_plan("SHORT", price, above, below, atr1, atr5, votes, candles)
    selected = long_plan if long_plan["score"] >= short_plan["score"] else short_plan
    if abs(long_plan["score"]-short_plan["score"]) < 6:
        tactical_side = "NEUTRAL"
    else:
        tactical_side = selected["side"]
    idx = _n(index.get("idxPx"))
    basis = (price/idx-1)*100 if idx else None
    stale = (datetime.now(timezone.utc)-(candles["M1"][-1].ts+timedelta(minutes=1))).total_seconds()
    market_block = stale > 600 or (basis is not None and abs(basis) > .20)
    confirmation_threshold = 80 if selected["type"] == "COUNTERTREND" else 75
    has_confirmation = selected["parts"]["event"] >= (18 if selected["type"]=="COUNTERTREND" else 9)
    if tactical_side == "NEUTRAL" or market_block:
        decision = "WAIT"
    elif selected["score"] >= confirmation_threshold and has_confirmation and impulse <= 2:
        decision = selected["side"]
    elif selected["score"] >= 55:
        decision = "SETUP WATCH"
    else:
        decision = "WAIT"
    trigger_text = " → ".join(selected["chain"]) if selected["chain"] else "цепочка подтверждения не сформирована"
    return dict(now=datetime.now(timezone.utc), price=price, bid=_n(ticker.get("bidPx")),
                ask=_n(ticker.get("askPx")), mark=_n(mark.get("markPx")), index=idx,
                basis=basis, funding=_n(funding.get("fundingRate")), oi=_n(oi.get("oi")),
                contexts=contexts, higher_bias=higher_bias, direction=tactical_side,
                setup_type=selected["type"], decision=decision, score=selected["score"],
                long_score=long_plan["score"], short_score=short_plan["score"],
                parts=selected["parts"], event_chain=selected["chain"], above=above,
                below=below, stop=selected["stop"], target=selected["target"], atr5=atr5,
                impulse=impulse, stale=stale, trigger_text=trigger_text)


def _p(x) -> str:
    return "—" if x is None else f"{x:,.2f}".replace(",", " ")


def render_gold(a: Dict) -> str:
    labels = {"LONG":"🟢 LONG ПОДТВЕРЖДЁН", "SHORT":"🔴 SHORT ПОДТВЕРЖДЁН",
              "SETUP WATCH":"👀 СЕТАП ФОРМИРУЕТСЯ", "WAIT":"🟡 ЖДАТЬ"}
    p, c = a["parts"], a["contexts"]
    lines = ["🥇 XAUUSD+ — ОЦЕНКА СЕЙЧАС", f"🕒 {a['now']:%d.%m.%Y %H:%M UTC}",
             "Источник: OKX XAU │ исполнение: Bybit CFD", "", "💵 ЦЕНА-ОРИЕНТИР",
             f"Trade: {_p(a['price'])} │ Bid/Ask: {_p(a['bid'])}/{_p(a['ask'])}",
             f"Mark: {_p(a['mark'])} │ Index: {_p(a['index'])}",
             f"Basis: {a['basis']:+.3f}%" if a["basis"] is not None else "Basis: —", "",
             f"{labels[a['decision']]}", f"Старший bias: {a['higher_bias']}",
             f"Локальный сетап: {a['direction']} │ {a['setup_type']} │ Entry: {a['score']}/100",
             f"Long Entry: {a['long_score']}/100 │ Short Entry: {a['short_score']}/100", "",
             "🧭 КОНТЕКСТ"] + [f"• {tf}: {c[tf]}" for tf in ("H1","M15","M5","M1")]
    lines += ["", "⚙️ PRECISION ENTRY",
              f"Контекст {p['context']}/15 │ структура {p['structure']}/20 │ ликвидность {p['liquidity']}/15",
              f"События {p['event']}/25 │ RR {p['rr']}/15 │ рынок {p['market']}/10",
              f"Триггер: {a['trigger_text']}", "", "🧲 ЛИКВИДНОСТЬ",
              f"Сверху: {_p(a['above'])}", f"Снизу: {_p(a['below'])}"]
    if a["stop"] is not None:
        risk = abs(a["price"]-a["stop"]) + .12
        lines += ["", "🎯 ПЛАН ПРИ ПОДТВЕРЖДЕНИИ" if a["decision"] in ("WAIT", "SETUP WATCH") else "🎯 АКТИВНЫЙ ПЛАН", f"Stop: {_p(a['stop'])}", f"Цель: {_p(a['target'])}",
                  f"Риск 0.01 lot: около ${risk:.2f} + проскальзывание"]
    if a["impulse"] > 2:
        lines += ["", f"⚡ Вход заблокирован: M1 импульс {a['impulse']:.1f}× ATR"]
    if a["stale"] > 600:
        lines += ["", "⛔ M1 устарела: рынок закрыт или поток остановлен"]
    lines += ["", f"M5 ATR: ${a['atr5']:.2f}",
              "⚠️ Уровни рассчитаны по OKX. Перед сделкой сверить Bid/Ask в Bybit."]
    return "\n".join(lines)

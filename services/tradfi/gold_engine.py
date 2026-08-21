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
    highs, lows = [], []
    for items in sets:
        for i in range(2, len(items) - 2):
            part = items[i - 2:i] + items[i + 1:i + 3]
            if all(items[i].h >= x.h for x in part):
                highs.append(items[i].h)
            if all(items[i].l <= x.l for x in part):
                lows.append(items[i].l)
    above = sorted({x for x in highs[-40:] if x > price})
    below = sorted({x for x in lows[-40:] if x < price}, reverse=True)
    return (above[0] if above else None, below[0] if below else None)


def _trigger(items: Sequence[Candle], direction: str) -> Tuple[str, int]:
    current, previous = items[-1], items[-2]
    recent = items[-5:-1]
    if direction == "LONG":
        sweep = current.l < min(x.l for x in recent) and current.c > previous.l
        impulse = current.c > current.o and current.c >= current.h - .25 * (current.h - current.l)
    elif direction == "SHORT":
        sweep = current.h > max(x.h for x in recent) and current.c < previous.h
        impulse = current.c < current.o and current.c <= current.l + .25 * (current.h - current.l)
    else:
        return "нет направления", 0
    if sweep and impulse:
        return "свип + reclaim M1", 15
    if impulse:
        return "displacement M1", 9
    return "подтверждения M1 нет", 0


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
    raw = 2 * votes["H1"] + 2 * votes["M15"] + votes["M5"]
    direction = "LONG" if raw >= 2 else "SHORT" if raw <= -2 else "NEUTRAL"
    above, below = _levels(price, [candles["M5"], candles["M15"], candles["H1"]])
    atr1, atr5 = _atr(candles["M1"]), _atr(candles["M5"])
    impulse = (candles["M1"][-1].h - candles["M1"][-1].l) / atr1
    trigger_text, trigger = _trigger(candles["M1"], direction)
    working = below if direction == "LONG" else above
    proximity = 0 if working is None else max(0, min(20, round(20 * (1 - abs(price-working) / (1.5*atr5)))))
    idx = _n(index.get("idxPx"))
    basis = (price / idx - 1) * 100 if idx else None
    stop = (below - max(.12, .35*atr1)) if direction == "LONG" and below else (
        above + max(.12, .35*atr1) if direction == "SHORT" and above else None
    )
    target = above if direction == "LONG" else below
    risk = abs(price-stop) if stop else 0
    reward = abs(target-price) if target else 0
    rr = reward/risk if risk else 0
    parts = {
        "context": min(15, abs(raw)*3), "structure": 20 if abs(raw)>=4 else 12 if abs(raw)>=2 else 2,
        "liquidity": proximity, "trigger": trigger, "rr": min(15, round(7.5*rr)), "market": 10,
    }
    stale = (datetime.now(timezone.utc) - (candles["M1"][-1].ts + timedelta(minutes=1))).total_seconds()
    blocked = direction == "NEUTRAL" or impulse > 2 or stale > 600 or (basis is not None and abs(basis) > .20) or not stop or not target
    score = min(100, sum(parts.values()))
    decision = "WAIT" if blocked else direction if score >= 75 and trigger >= 9 else "SETUP WATCH" if score >= 55 else "WAIT"
    return dict(now=datetime.now(timezone.utc), price=price, bid=_n(ticker.get("bidPx")), ask=_n(ticker.get("askPx")),
                mark=_n(mark.get("markPx")), index=idx, basis=basis, funding=_n(funding.get("fundingRate")),
                oi=_n(oi.get("oi")), contexts=contexts, direction=direction, decision=decision, score=score,
                parts=parts, above=above, below=below, stop=stop, target=target, atr5=atr5,
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
             f"{labels[a['decision']]}", f"Направление: {a['direction']} │ Entry: {a['score']}/100", "",
             "🧭 КОНТЕКСТ"] + [f"• {tf}: {c[tf]}" for tf in ("H1","M15","M5","M1")]
    lines += ["", "⚙️ PRECISION ENTRY",
              f"Контекст {p['context']}/15 │ структура {p['structure']}/20 │ ликвидность {p['liquidity']}/20",
              f"M1 trigger {p['trigger']}/15 │ RR {p['rr']}/15 │ рынок {p['market']}/10",
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

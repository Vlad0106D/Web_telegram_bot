from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import httpx

BASE = "https://www.okx.com"
SWAP = "XAU-USDT-SWAP"
INDEX = "XAU-USDT"
BARS = {"M1": "1m", "M5": "5m", "M15": "15m", "H1": "1H"}
EXECUTION_TZ = ZoneInfo("America/Chicago")


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


def _execution_market_open(now: datetime) -> bool:
    """Conservative XAU CFD session gate in Chicago exchange time.

    The OKX reference can keep printing during the weekend while the Bybit CFD
    used for execution is closed.  Gold is considered tradable Sunday 17:00
    through Friday 16:00 CT, excluding the regular 16:00-17:00 CT break.
    ZoneInfo keeps the UTC boundary correct across daylight-saving changes.
    """
    local = now.astimezone(EXECUTION_TZ)
    weekday = local.weekday()
    minute = local.hour * 60 + local.minute
    if weekday == 5:  # Saturday
        return False
    if weekday == 6:  # Sunday
        return minute >= 17 * 60
    if weekday == 4:  # Friday
        return minute < 16 * 60
    return not (16 * 60 <= minute < 17 * 60)


def _market_activity(items: Sequence[Candle], now: datetime) -> Dict:
    """Reject fresh-looking but frozen/micro-tick reference candles."""
    recent = list(items[-6:])
    stale = (now - (recent[-1].ts + timedelta(minutes=1))).total_seconds()
    price_range = max(x.h for x in recent) - min(x.l for x in recent)
    distinct_closes = len({round(x.c, 2) for x in recent})
    active = stale <= 180 and price_range >= .12 and distinct_closes >= 3
    reason = "active"
    if stale > 180:
        reason = "stale_m1"
    elif price_range < .12 or distinct_closes < 3:
        reason = "frozen_quotes"
    return {
        "active": active,
        "reason": reason,
        "range": price_range,
        "distinct_closes": distinct_closes,
        "stale": stale,
    }


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


def _swing_centers(items: Sequence[Candle]) -> List[Tuple[float, int]]:
    result: List[Tuple[float, int]] = []
    for i in range(2, len(items) - 2):
        neighbours = items[i - 2:i] + items[i + 1:i + 3]
        if all(items[i].h >= x.h for x in neighbours):
            result.append((items[i].h, i))
        if all(items[i].l <= x.l for x in neighbours):
            result.append((items[i].l, i))
    return result


def _build_zones(candles: Dict[str, List[Candle]]) -> List[Dict]:
    atr_h1, atr_m15 = _atr(candles["H1"]), _atr(candles["M15"])
    raw: List[Dict] = []
    for tf, base, fraction, atr in (
        ("H1", 70, .08, atr_h1),
        ("M15", 45, .12, atr_m15),
    ):
        swings = _swing_centers(candles[tf])[-24:]
        half_width = max(.08, fraction * atr)
        total = len(candles[tf])
        for center, index in swings:
            age = total - 1 - index
            recency = max(-15, -age // 12)
            raw.append({
                "tf": tf, "sources": [tf], "center": center,
                "low": center - half_width, "high": center + half_width,
                "strength": max(20, base + recency),
            })
    raw.sort(key=lambda z: z["center"])
    merge_distance = max(.10, .15 * atr_m15)
    clusters: List[Dict] = []
    for zone in raw:
        if clusters and zone["low"] <= clusters[-1]["high"] + merge_distance:
            current = clusters[-1]
            old_weight, new_weight = current["strength"], zone["strength"]
            current["center"] = (
                current["center"] * old_weight + zone["center"] * new_weight
            ) / (old_weight + new_weight)
            current["low"] = min(current["low"], zone["low"])
            current["high"] = max(current["high"], zone["high"])
            current["sources"] = sorted(set(current["sources"] + zone["sources"]))
            current["tf"] = "+".join(current["sources"])
            confluence = 20 if len(current["sources"]) > 1 else 0
            current["strength"] = min(100, max(old_weight, new_weight) + confluence)
        else:
            clusters.append(dict(zone))
    return clusters


def _working_zones(price: float, zones: Sequence[Dict], atr_m15: float) -> Tuple[Optional[Dict], Optional[Dict]]:
    above = [z for z in zones if z["low"] > price]
    below = [z for z in zones if z["high"] < price]
    containing = [z for z in zones if z["low"] <= price <= z["high"]]
    def priority(zone: Dict) -> float:
        distance = 0.0 if zone["low"] <= price <= zone["high"] else min(
            abs(price-zone["low"]), abs(price-zone["high"])
        )
        return zone["strength"] / (1 + distance / max(atr_m15, .01))
    upper = max(above, key=priority) if above else None
    lower = max(below, key=priority) if below else None
    if containing:
        active = max(containing, key=priority)
        # The current zone may act as support for LONG and resistance for SHORT.
        lower = lower or active
        upper = upper or active
    return upper, lower


def _event_chain(items: Sequence[Candle], side: str, zone: Optional[Dict], atr: float) -> Tuple[List[str], int]:
    if zone is None or len(items) < 8:
        return [], 0
    recent = items[-8:]
    boundary = zone["low"] if side == "SHORT" else zone["high"]
    tolerance = max(.05, atr * .12)
    chain: List[str] = []
    acceptance_at = None
    for i in range(1, len(recent)):
        first, second = recent[i-1], recent[i]
        accepted = (
            first.c < boundary-tolerance and second.c < boundary-tolerance
            if side == "SHORT"
            else first.c > boundary+tolerance and second.c > boundary+tolerance
        )
        if accepted and acceptance_at is None:
            acceptance_at = i
    if acceptance_at is not None:
        chain.append("accept below" if side=="SHORT" else "accept above")
        for candle in recent[acceptance_at+1:]:
            retest = (
                candle.h >= boundary-tolerance and candle.c < boundary
                if side=="SHORT"
                else candle.l <= boundary+tolerance and candle.c > boundary
            )
            if retest:
                chain.append("retest снизу" if side=="SHORT" else "retest сверху")
                if (side=="SHORT" and candle.c<candle.o) or (side=="LONG" and candle.c>candle.o):
                    chain.append("bearish rejection" if side=="SHORT" else "bullish rejection")
                break
    current, previous = recent[-1], recent[-2]
    displacement = (
        current.c<current.o and current.c<=current.l+.25*max(current.h-current.l,1e-9)
        if side=="SHORT" else
        current.c>current.o and current.c>=current.h-.25*max(current.h-current.l,1e-9)
    )
    sweep = (
        current.h>max(x.h for x in recent[-5:-1]) and current.c<previous.h
        if side=="SHORT" else
        current.l<min(x.l for x in recent[-5:-1]) and current.c>previous.l
    )
    if sweep:
        chain.append("sweep high" if side=="SHORT" else "sweep low")
    if displacement:
        chain.append("bearish displacement" if side=="SHORT" else "bullish displacement")
    chain = list(dict.fromkeys(chain))
    score = min(25, (10 if any("accept" in x for x in chain) else 0)
                +(8 if any("retest" in x for x in chain) else 0)
                +(7 if any("rejection" in x for x in chain) else 0)
                +(9 if any("displacement" in x for x in chain) else 0)
                +(6 if any("sweep" in x for x in chain) else 0))
    return chain, score


def _side_plan(side: str, price: float, upper_zone: Optional[Dict], lower_zone: Optional[Dict],
               atr1: float, atr5: float, votes: Dict[str, int],
               candles: Dict[str, List[Candle]]) -> Dict:
    sign = 1 if side=="LONG" else -1
    higher_raw = 2*votes["H1"] + 2*votes["M15"]
    aligned = higher_raw*sign
    setup_type = "TREND" if aligned>0 else "COUNTERTREND" if aligned<0 else "NEUTRAL"
    working_zone = lower_zone if side=="LONG" else upper_zone
    target_zone = upper_zone if side=="LONG" else lower_zone
    chain, event_score = _event_chain(candles["M1"], side, working_zone, atr1)
    local = sign*(2*votes["M5"]+votes["M1"])
    context_score = 15 if aligned>=3 else 10 if aligned>0 else 5 if aligned<0 else 7
    structure_score = 20 if local>=3 else 14 if local>=1 else 5
    if working_zone is None:
        proximity = 0
    else:
        distance = 0 if working_zone["low"]<=price<=working_zone["high"] else min(
            abs(price-working_zone["low"]), abs(price-working_zone["high"])
        )
        proximity = max(0,min(15,round(15*(1-distance/max(1.5*atr5,.01)))))
        proximity = round(proximity * working_zone["strength"]/100)
    buffer = max(.12,.35*atr1)
    stop = None
    if working_zone is not None:
        stop = working_zone["low"]-buffer if side=="LONG" else working_zone["high"]+buffer
    target = target_zone["center"] if target_zone is not None else None
    risk = abs(price-stop) if stop is not None else 0
    reward = abs(target-price) if target is not None else 0
    rr = reward/risk if risk else 0
    parts={"context":context_score,"structure":structure_score,"liquidity":proximity,
           "event":event_score,"rr":min(15,round(7.5*rr)),"market":10}
    return {"side":side,"type":setup_type,"score":min(100,sum(parts.values())),
            "parts":parts,"chain":chain,"zone":working_zone,"stop":stop,
            "target":target,"rr":rr}

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
    atr1, atr5 = _atr(candles["M1"]), _atr(candles["M5"])
    atr15 = _atr(candles["M15"])
    zones = _build_zones(candles)
    upper_zone, lower_zone = _working_zones(price, zones, atr15)
    above = upper_zone["center"] if upper_zone else None
    below = lower_zone["center"] if lower_zone else None
    impulse = (candles["M1"][-1].h-candles["M1"][-1].l)/atr1
    long_plan = _side_plan("LONG", price, upper_zone, lower_zone, atr1, atr5, votes, candles)
    short_plan = _side_plan("SHORT", price, upper_zone, lower_zone, atr1, atr5, votes, candles)
    selected = long_plan if long_plan["score"] >= short_plan["score"] else short_plan
    if abs(long_plan["score"]-short_plan["score"]) < 6:
        tactical_side = "NEUTRAL"
    else:
        tactical_side = selected["side"]
    idx = _n(index.get("idxPx"))
    basis = (price/idx-1)*100 if idx else None
    now = datetime.now(timezone.utc)
    session_open = _execution_market_open(now)
    activity = _market_activity(candles["M1"], now)
    stale = activity["stale"]
    market_ready = session_open and activity["active"]
    market_reason = "active" if market_ready else (
        "execution_session_closed" if not session_open else activity["reason"]
    )
    market_block = not market_ready or (basis is not None and abs(basis) > .20)
    for plan in (long_plan, short_plan):
        plan["parts"]["market"] = 10 if market_ready else 0
        plan["score"] = min(100, sum(plan["parts"].values()))
    selected = long_plan if selected["side"] == "LONG" else short_plan
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
    return dict(now=now, price=price, bid=_n(ticker.get("bidPx")),
                ask=_n(ticker.get("askPx")), mark=_n(mark.get("markPx")), index=idx,
                basis=basis, funding=_n(funding.get("fundingRate")), oi=_n(oi.get("oi")),
                contexts=contexts, higher_bias=higher_bias, direction=tactical_side,
                setup_type=selected["type"], decision=decision, score=selected["score"],
                long_score=long_plan["score"], short_score=short_plan["score"],
                parts=selected["parts"], event_chain=selected["chain"], above=above,
                below=below, upper_zone=upper_zone, lower_zone=lower_zone,
                active_zone=selected["zone"], stop=selected["stop"],
                target=selected["target"], atr5=atr5,
                impulse=impulse, stale=stale, trigger_text=trigger_text,
                market_open=session_open, market_active=activity["active"],
                market_ready=market_ready, market_reason=market_reason,
                activity_range=activity["range"],
                activity_distinct_closes=activity["distinct_closes"])


def _p(x) -> str:
    return "—" if x is None else f"{x:,.2f}".replace(",", " ")


def _zone_text(zone: Optional[Dict]) -> str:
    if not zone:
        return "—"
    return (
        f"{_p(zone['low'])}–{_p(zone['high'])} | "
        f"{zone['tf']} | сила {zone['strength']}/100"
    )


def render_gold(a: Dict) -> str:
    labels = {"LONG":"🟢 LONG ПОДТВЕРЖДЁН", "SHORT":"🔴 SHORT ПОДТВЕРЖДЁН",
              "SETUP WATCH":"👀 СЕТАП ФОРМИРУЕТСЯ", "WAIT":"🟡 ЖДАТЬ"}
    p, c = a["parts"], a["contexts"]
    if not a.get("market_open", True):
        status_label = "🔒 РЫНОК BYBIT CFD ЗАКРЫТ"
    elif not a.get("market_active", True):
        status_label = "⏸ КОТИРОВКИ НЕАКТИВНЫ"
    else:
        status_label = labels[a["decision"]]
    lines = ["🥇 XAUUSD+ — ОЦЕНКА СЕЙЧАС", f"🕒 {a['now']:%d.%m.%Y %H:%M UTC}",
             "Источник: OKX XAU │ исполнение: Bybit CFD", "", "💵 ЦЕНА-ОРИЕНТИР",
             f"Trade: {_p(a['price'])} │ Bid/Ask: {_p(a['bid'])}/{_p(a['ask'])}",
             f"Mark: {_p(a['mark'])} │ Index: {_p(a['index'])}",
             f"Basis: {a['basis']:+.3f}%" if a["basis"] is not None else "Basis: —", "",
             status_label, f"Старший bias: {a['higher_bias']}",
             f"Локальный сетап: {a['direction']} │ {a['setup_type']} │ Entry: {a['score']}/100",
             f"Long Entry: {a['long_score']}/100 │ Short Entry: {a['short_score']}/100", "",
             "🧭 КОНТЕКСТ"] + [f"• {tf}: {c[tf]}" for tf in ("H1","M15","M5","M1")]
    lines += ["", "⚙️ PRECISION ENTRY",
              f"Контекст {p['context']}/15 │ структура {p['structure']}/20 │ ликвидность {p['liquidity']}/15",
              f"События {p['event']}/25 │ RR {p['rr']}/15 │ рынок {p['market']}/10",
              f"Триггер: {a['trigger_text']}", "", "🧲 ЛИКВИДНОСТЬ",
              f"Сверху: {_zone_text(a.get('upper_zone'))}",
              f"Снизу: {_zone_text(a.get('lower_zone'))}"]
    if a["stop"] is not None:
        risk = abs(a["price"]-a["stop"]) + .12
        lines += ["", "🎯 ПЛАН ПРИ ПОДТВЕРЖДЕНИИ" if a["decision"] in ("WAIT", "SETUP WATCH") else "🎯 АКТИВНЫЙ ПЛАН", f"Stop: {_p(a['stop'])}", f"Цель: {_p(a['target'])}",
                  f"Риск 0.01 lot: около ${risk:.2f} + проскальзывание"]
    if a["impulse"] > 2:
        lines += ["", f"⚡ Вход заблокирован: M1 импульс {a['impulse']:.1f}× ATR"]
    if a["stale"] > 600:
        lines += ["", "⛔ M1 устарела: рынок закрыт или поток остановлен"]
    if not a.get("market_ready", True):
        lines += ["", "Автоматические сетапы и входы приостановлены.",
                  "После открытия нужен прогрев свежих M1/M5 данных."]
    lines += ["", f"M5 ATR: ${a['atr5']:.2f}",
              "⚠️ Уровни рассчитаны по OKX. Перед сделкой сверить Bid/Ask в Bybit."]
    return "\n".join(lines)

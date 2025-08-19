# bot/watchlist_provider.py
from typing import List

def _normalize(sym: str) -> str:
    # Принимаем 'BTCUSDT' или 'BTC-USDT' → приводим к 'BTCUSDT'
    return sym.replace("-", "").replace(" ", "").upper()

def get_watchlist_from_context(context) -> List[str]:
    """
    Берём избранные пары из application.bot_data['favorites'] (или context.bot_data['favorites']).
    Ожидаемый формат: список строк, например ['BTCUSDT', 'SOLUSDT'] или с дефисом.
    НИКАКИХ фолбэков на ENV: если пусто — вернуть [].
    """
    # где /list их хранит — используем оба варианта на всякий случай
    app_bd = getattr(getattr(context, "application", None), "bot_data", {}) or {}
    ctx_bd = getattr(context, "bot_data", {}) or {}

    favs = app_bd.get("favorites")
    if not isinstance(favs, list) or len(favs) == 0:
        favs = ctx_bd.get("favorites")

    if not isinstance(favs, list):
        return []

    # нормализуем и убираем дубли
    return list({ _normalize(s) for s in favs if isinstance(s, str) and s.strip() })
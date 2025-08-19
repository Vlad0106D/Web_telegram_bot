# services/state.py
import os
import json
import threading
from typing import List

_LOCK = threading.Lock()

FAVORITES_PATH = os.getenv("FAVORITES_PATH", "data/favorites.json")

# Стартовый список — из переменной окружения WATCHLIST или дефолт
_DEFAULT_FAVS = [
    s.strip().upper()
    for s in os.getenv("WATCHLIST", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
    if s.strip()
]


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _load_raw() -> List[str]:
    if not os.path.exists(FAVORITES_PATH):
        return []
    try:
        with open(FAVORITES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x).upper() for x in data]
    except Exception:
        pass
    return []


def _save_raw(items: List[str]) -> None:
    _ensure_dir(FAVORITES_PATH)
    with open(FAVORITES_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(list(dict.fromkeys(items))), f, ensure_ascii=False, indent=2)


def init_favorites() -> List[str]:
    """
    Инициализация файла избранного при старте. Если файла нет — пишем дефолт.
    """
    with _LOCK:
        if not os.path.exists(FAVORITES_PATH):
            _save_raw(_DEFAULT_FAVS)
        # Если файл есть, но пусто — тоже подкидываем дефолт, чтобы не было пустоты
        items = _load_raw()
        if not items:
            _save_raw(_DEFAULT_FAVS)
            items = list(_DEFAULT_FAVS)
        return items


def get_favorites() -> List[str]:
    with _LOCK:
        items = _load_raw()
        if not items:
            # защита от пустоты
            _save_raw(_DEFAULT_FAVS)
            return list(_DEFAULT_FAVS)
        return items


def add_favorite(symbol: str) -> List[str]:
    sym = symbol.strip().upper()
    with _LOCK:
        items = _load_raw()
        if sym not in items:
            items.append(sym)
            _save_raw(items)
        return items


def remove_favorite(symbol: str) -> List[str]:
    sym = symbol.strip().upper()
    with _LOCK:
        items = [s for s in _load_raw() if s != sym]
        _save_raw(items)
        return items
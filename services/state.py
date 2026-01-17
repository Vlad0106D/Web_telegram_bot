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


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _normalize_list(data) -> List[str]:
    """
    Нормализация списка тикеров:
    - только строки
    - trim
    - upper
    - убрать пустые
    - убрать дубли (с сохранением порядка)
    """
    out: List[str] = []
    seen = set()
    if not isinstance(data, list):
        return out
    for x in data:
        s = str(x).strip().upper()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _load_raw() -> List[str]:
    if not os.path.exists(FAVORITES_PATH):
        return []
    try:
        with open(FAVORITES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _normalize_list(data)
    except Exception:
        return []


def _save_raw(items: List[str]) -> None:
    _ensure_dir(FAVORITES_PATH)
    norm = _normalize_list(items)
    with open(FAVORITES_PATH, "w", encoding="utf-8") as f:
        json.dump(norm, f, ensure_ascii=False, indent=2)


def init_favorites() -> List[str]:
    """
    Инициализация файла избранного при старте. Если файла нет — пишем дефолт.
    """
    with _LOCK:
        if not os.path.exists(FAVORITES_PATH):
            _save_raw(_DEFAULT_FAVS)

        items = _load_raw()
        if not items:
            _save_raw(_DEFAULT_FAVS)
            items = list(_DEFAULT_FAVS)
        return items


def get_favorites() -> List[str]:
    with _LOCK:
        items = _load_raw()
        if not items:
            _save_raw(_DEFAULT_FAVS)
            return list(_DEFAULT_FAVS)
        return items


def add_favorite(symbol: str) -> List[str]:
    sym = str(symbol).strip().upper()
    if not sym:
        return get_favorites()

    with _LOCK:
        items = _load_raw()
        if sym not in items:
            items.append(sym)
            _save_raw(items)
        return _load_raw()


def remove_favorite(symbol: str) -> List[str]:
    sym = str(symbol).strip().upper()
    with _LOCK:
        items = [s for s in _load_raw() if s != sym]
        # если удалили всё — не оставляем пустоту, подкидываем дефолт
        if not items:
            items = list(_DEFAULT_FAVS)
        _save_raw(items)
        return _load_raw()
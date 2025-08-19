# config.py
import os

# Токен бота
TOKEN = os.getenv("TOKEN") or os.getenv("BOT_TOKEN", "")

# Вочер: включён ли автозапуск
WATCHER_ENABLED = (os.getenv("WATCHER_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on"))

# Интервал проверки в секундах
WATCHER_INTERVAL_SEC = int(os.getenv("WATCHER_INTERVAL_SEC", "45"))

# Какие таймфреймы смотреть (строка "5m,15m,1h" -> ["5m","15m","1h"])
def _parse_tfs(s: str) -> list[str]:
    return [part.strip() for part in s.replace(" ", "").split(",") if part.strip()]

WATCHER_TFS = _parse_tfs(os.getenv("WATCHER_TFS", "5m,15m,1h"))

# Стартовый список избранного (используется services/state.py)
WATCHLIST = os.getenv("WATCHLIST", "BTCUSDT,ETHUSDT,SOLUSDT")

# Где хранить favorites.json
FAVORITES_PATH = os.getenv("FAVORITES_PATH", "data/favorites.json")

# (опционально) Куда слать алерты, если используешь в watcher
ALERT_CHAT_ID = os.getenv("ALERT_CHAT_ID", "")  # можно оставить пустым
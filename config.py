# config.py
import os

def _get_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")

def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

# Основной токен: берём TELEGRAM_BOT_TOKEN, а также поддерживаем альтернативные имена
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TOKEN = TELEGRAM_BOT_TOKEN or os.getenv("BOT_TOKEN", "").strip() or os.getenv("TOKEN", "").strip()
if not TOKEN:
    raise RuntimeError("Не найден TELEGRAM_BOT_TOKEN (или BOT_TOKEN/TOKEN) в переменных окружения")

# Настройки вотчера
WATCHER_ENABLED = _get_bool("WATCHER_ENABLED", True)
WATCHER_INTERVAL_SEC = _get_int("WATCHER_INTERVAL_SEC", 45)

# Список таймфреймов (пример: 5m,15m,1h)
WATCHER_TFS = [t.strip() for t in os.getenv("WATCHER_TFS", "1h").split(",") if t.strip()]

# Куда слать алерты (опционально)
ALERT_CHAT_ID = os.getenv("ALERT_CHAT_ID", "").strip() or None
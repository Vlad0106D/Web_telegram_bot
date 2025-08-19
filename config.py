# config.py
import os

# ⚠️ Telegram Bot Token теперь только из переменной окружения
TOKEN = os.getenv("TOKEN")
if not TOKEN:
    raise RuntimeError("Не задан TOKEN в переменных окружения!")

# Куда слать алерты
ALERT_CHAT_ID = int(os.getenv("ALERT_CHAT_ID", "776505127"))

# Вочер
WATCHER_ENABLED = os.getenv("WATCHER_ENABLED", "true").lower() == "true"
WATCHER_INTERVAL_SEC = int(os.getenv("WATCHER_INTERVAL_SEC", "45"))

# Параметры сигналов
LOOKBACK_RANGE = int(os.getenv("LOOKBACK_RANGE", "48"))
BB_PERIOD = int(os.getenv("BB_PERIOD", "20"))
BB_STD = float(os.getenv("BB_STD", "2.0"))
BB_SQUEEZE_PCT = float(os.getenv("BB_SQUEEZE_PCT", "2.5"))
PROXIMITY_PCT = float(os.getenv("PROXIMITY_PCT", "0.35"))
BREAK_EPS_PCT = float(os.getenv("BREAK_EPS_PCT", "0.15"))
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", "30"))

# Сеть
HTTP_TIMEOUT_SEC = int(os.getenv("HTTP_TIMEOUT_SEC", "12"))
HTTP_RETRIES = int(os.getenv("HTTP_RETRIES", "3"))
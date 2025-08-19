# config.py
import os

# --- Telegram ---
# Пробуем найти токен в двух вариантах: TOKEN или TELEGRAM_TOKEN
TOKEN = os.getenv("TOKEN") or os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise RuntimeError("Не задан TOKEN или TELEGRAM_TOKEN в переменных окружения!")

# Chat ID для алертов: ALERT_CHAT_ID или ALER… (как у тебя на Render)
ALERT_CHAT_ID = (
    os.getenv("ALERT_CHAT_ID")
    or os.getenv("ALER…")  # подставь точное имя, если Render обрезал
    or "776505127"
)
ALERT_CHAT_ID = int(ALERT_CHAT_ID)

# --- Вочер ---
WATCHER_ENABLED = os.getenv("WATCHER_ENABLED", "true").lower() == "true"
WATCHER_INTERVAL_SEC = int(os.getenv("WATCHER_INTERVAL_SEC", "45"))

# Параметры сигналов «возможен пробой / пробой»
LOOKBACK_RANGE = int(os.getenv("LOOKBACK_RANGE", "48"))
BB_PERIOD = int(os.getenv("BB_PERIOD", "20"))
BB_STD = float(os.getenv("BB_STD", "2.0"))
BB_SQUEEZE_PCT = float(os.getenv("BB_SQUEEZE_PCT", "2.5"))
PROXIMITY_PCT = float(os.getenv("PROXIMITY_PCT", "0.35"))
BREAK_EPS_PCT = float(os.getenv("BREAK_EPS_PCT", "0.15"))
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", "30"))

# --- Сеть ---
HTTP_TIMEOUT_SEC = int(os.getenv("HTTP_TIMEOUT_SEC", "12"))
HTTP_RETRIES = int(os.getenv("HTTP_RETRIES", "3"))
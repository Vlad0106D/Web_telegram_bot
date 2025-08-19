import os

TOKEN = (
    os.getenv("TOKEN")
    or os.getenv("TELEGRAM_TOKEN")
    or os.getenv("TELEGRAM_BOT_TOKEN")  # ✅ твой вариант
)
if not TOKEN:
    raise RuntimeError("Не задан TOKEN / TELEGRAM_TOKEN / TELEGRAM_BOT_TOKEN в переменных окружения!")

ALERT_CHAT_ID = int(os.getenv("ALERT_CHAT_ID", "776505127"))

WATCHER_ENABLED = os.getenv("WATCHER_ENABLED", "true").lower() == "true"
WATCHER_INTERVAL_SEC = int(os.getenv("WATCHER_INTERVAL_SEC", "45"))
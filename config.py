# config.py
import os

# === Телеграм токен ===
# В окружении у тебя переменная называется именно так:
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# === Настройки вочера ===
WATCHER_ENABLED = os.getenv("WATCHER_ENABLED", "1").strip() == "1"

# Период между запусками джобы (в секундах)
WATCHER_INTERVAL_SEC = int(os.getenv("WATCHER_INTERVAL_SEC", "45").strip() or "45")

# Таймфреймы, через запятую. Пример: "5m,15m,1h"
WATCHER_TFS = [
    t.strip()
    for t in os.getenv("WATCHER_TFS", "1h").split(",")
    if t.strip()
]

# Куда слать алерты (chat id). Пример: 776505127
def _read_chat_id() -> int | None:
    raw = os.getenv("ALERT_CHAT_ID", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None

ALERT_CHAT_ID = _read_chat_id()

# --- Breaker settings ---
# сколько свечей образуют диапазон для пробоя
BREAKER_LOOKBACK = int(os.getenv("BREAKER_LOOKBACK", "50").strip() or "50")
# чувствительность пробоя: 0.001 = 0.1% сверх High/ниже Low
BREAKER_EPS = float(os.getenv("BREAKER_EPS", "0.001").strip() or "0.001")
# подавление повторов (сек) на один и тот же {symbol, tf, direction}
BREAKER_COOLDOWN_SEC = int(os.getenv("BREAKER_COOLDOWN_SEC", "900").strip() or "900")
# config.py
import os

# === Телеграм токен ===
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# === Настройки вочера ===
WATCHER_ENABLED = os.getenv("WATCHER_ENABLED", "1").strip() == "1"

# Период между запусками джобы (в секундах)
WATCHER_INTERVAL_SEC = int(os.getenv("WATCHER_INTERVAL_SEC", "45").strip() or "45")

# Таймфреймы для планировщика вочера (как и было: список)
WATCHER_TFS = [
    t.strip()
    for t in os.getenv("WATCHER_TFS", "1h").split(",")
    if t.strip()
]

# Куда слать алерты (chat id)
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
BREAKER_LOOKBACK = int(os.getenv("BREAKER_LOOKBACK", "50").strip() or "50")
BREAKER_EPS = float(os.getenv("BREAKER_EPS", "0.001").strip() or "0.001")
BREAKER_COOLDOWN_SEC = int(os.getenv("BREAKER_COOLDOWN_SEC", "900").strip() or "900")

# --- Strategy (watcher) — MTF и фильтры ---
# ВАЖНО: тут оставляем строку для совместимости с кодом, который делает .split(",")
MTF_TFS = os.getenv("MTF_TFS", "1d,4h,1h,30m,15m,5m").strip()
# А это — удобный список, если где-то нужен уже разобранный формат
MTF_TFS_LIST = [t.strip() for t in MTF_TFS.split(",") if t.strip()]

MTF_PRIMARY_TF = os.getenv("MTF_PRIMARY_TF", "1h").strip()
MTF_TREND_TF = os.getenv("MTF_TREND_TF", "4h").strip()

# Порог уверенности и кулдаун сигналов стратегии
SIGNAL_MIN_CONF = int(os.getenv("SIGNAL_MIN_CONF", "70").strip() or "70")
SIGNAL_COOLDOWN_SEC = int(os.getenv("SIGNAL_COOLDOWN_SEC", "900").strip() or "900")

# --- Настройки уровней и TP/SL ---
LEVEL_LOOKBACK = int(os.getenv("LEVEL_LOOKBACK", "80").strip() or "80")
LEVEL_WINDOW = int(os.getenv("LEVEL_WINDOW", "9").strip() or "9")
TP_BUFFER_PCT = float(os.getenv("TP_BUFFER_PCT", "0.001").strip() or "0.001")  # 0.1%
SL_BUFFER_PCT = float(os.getenv("SL_BUFFER_PCT", "0.002").strip() or "0.002")  # 0.2%
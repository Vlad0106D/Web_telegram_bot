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

# --- Strategy (watcher) — MTF и фильтры сигналов ---
# Набор таймфреймов для мульти‑таймфрейм анализа (читаются/используются по месту в analyze.py)
MTF_TFS = [
    t.strip()
    for t in os.getenv("MTF_TFS", "1d,4h,1h,30m,15m,5m").split(",")
    if t.strip()
]
# Базовый TF для входа (на нём считаются основные индикаторы/уровни)
MTF_PRIMARY_TF = os.getenv("MTF_PRIMARY_TF", "1h").strip()
# Старший TF для фильтра тренда
MTF_TREND_TF = os.getenv("MTF_TREND_TF", "4h").strip()

# Порог уверенности для отправки сигнала от стратегии и кулдаун между повторами
SIGNAL_MIN_CONF = int(os.getenv("SIGNAL_MIN_CONF", "70").strip() or "70")
SIGNAL_COOLDOWN_SEC = int(os.getenv("SIGNAL_COOLDOWN_SEC", "900").strip() or "900")

# --- Настройки уровней и TP/SL (используются в analyze.py) ---
# Сколько баров назад смотреть для поиска экстремумов (уровней)
LEVEL_LOOKBACK = int(os.getenv("LEVEL_LOOKBACK", "80").strip() or "80")
# Окно для локальных максимумов/минимумов (скользящее, центрированное)
LEVEL_WINDOW = int(os.getenv("LEVEL_WINDOW", "9").strip() or "9")
# Микро‑буферы к уровням (чтобы TP чуть не «не добирался», а SL был чуть ниже/выше уровня)
TP_BUFFER_PCT = float(os.getenv("TP_BUFFER_PCT", "0.001").strip() or "0.001")  # 0.1%
SL_BUFFER_PCT = float(os.getenv("SL_BUFFER_PCT", "0.002").strip() or "0.002")  # 0.2%
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

# --- (опционально) Порог и кулдаун для стратегии в вотчере ---
SIGNAL_MIN_CONF = int(os.getenv("SIGNAL_MIN_CONF", "70").strip() or "70")
SIGNAL_COOLDOWN_SEC = int(os.getenv("SIGNAL_COOLDOWN_SEC", "900").strip() or "900")

# --- (опционально) Кулдаун для Reversal ---
REVERSAL_COOLDOWN_SEC = int(os.getenv("REVERSAL_COOLDOWN_SEC", "900").strip() or "900")

# --- Fusion settings ---
FUSION_ENABLED = (os.getenv("FUSION_ENABLED", "1").strip() == "1")
FUSION_COOLDOWN_SEC = int(os.getenv("FUSION_COOLDOWN_SEC", "900").strip() or "900")
FUSION_MIN_CONF = int(os.getenv("FUSION_MIN_CONF", "75").strip() or "75")  # мин. уверенность Strategy, чтобы её голос засчитался
FUSION_REQUIRE_ANY = int(os.getenv("FUSION_REQUIRE_ANY", "2").strip() or "2")  # сколько модулей должны совпасть (2 из 3)

# === FIBO watcher ===
FIBO_ENABLED = (os.getenv("FIBO_ENABLED", "1").strip() == "1")
# По каким ТФ смотреть Фибо. Можно оставить те же, что в вотчере, но тут зададим явно:
FIBO_TFS = [t.strip() for t in os.getenv("FIBO_TFS", "1h,4h,1d").split(",") if t.strip()]
FIBO_COOLDOWN_SEC = int(os.getenv("FIBO_COOLDOWN_SEC", "1200").strip() or "1200")  # 20 min

# Детекция импульса и уровни
FIBO_PIVOT_WINDOW = int(os.getenv("FIBO_PIVOT_WINDOW", "3").strip() or "3")       # свинг-пивот N слева/справа
FIBO_CONFIRM_PULLBACK_PCT = float(os.getenv("FIBO_CONFIRM_PULLBACK_PCT", "0.15").strip() or "0.15")  # 15%

# Зоны и допуски
FIBO_LEVELS_RETR = [23.6, 38.2, 50.0, 61.8, 78.6]
FIBO_LEVELS_EXT  = [127.2, 161.8, 261.8]
FIBO_PROXIMITY_BPS = int(os.getenv("FIBO_PROXIMITY_BPS", "25").strip() or "25")   # 0.25%
FIBO_K_ATR = float(os.getenv("FIBO_K_ATR", "0.35").strip() or "0.35")            # ширина зоны как доля ATR
FIBO_MIN_BODY_FRAC = float(os.getenv("FIBO_MIN_BODY_FRAC", "0.4").strip() or "0.4")  # подтверждение свечой

# Фильтры/приоритизация
FIBO_REQUIRE_TREND_1D = (os.getenv("FIBO_REQUIRE_TREND_1D", "1").strip() == "1")
FIBO_SEND_ONLY_IMPORTANT = (os.getenv("FIBO_SEND_ONLY_IMPORTANT", "0").strip() == "0")  # по умолчанию шлём все подтверждённые
FIBO_IMPORTANT_TAG = os.getenv("FIBO_IMPORTANT_TAG", "important").strip() or "important"
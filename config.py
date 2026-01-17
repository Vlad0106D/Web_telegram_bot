import os

# === Телеграм токен ===
# Поддерживаем оба варианта переменных окружения (на разных деплоях называют по-разному)
TOKEN = (
    os.getenv("TELEGRAM_BOT_TOKEN")
    or os.getenv("TELEGRAM_TOKEN")
    or ""
).strip()

# === Настройки вочера ===
WATCHER_ENABLED = os.getenv("WATCHER_ENABLED", "1").strip() == "1"

# Период между запусками джобы (в секундах)
WATCHER_INTERVAL_SEC = int(os.getenv("WATCHER_INTERVAL_SEC", "45").strip() or "45")

# Таймфреймы для планировщика вочера
# Поддержка: "1h,4h" и "1h 4h"
_raw_tfs = os.getenv("WATCHER_TFS", "1h").strip()
WATCHER_TFS = [
    t.strip()
    for t in _raw_tfs.replace(",", " ").split()
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

# === Настройки Фибоначчи (статические) ===
FIBO_ENABLED = True                   # включить модуль
FIBO_TFS = ["1h", "4h", "1d"]         # ТФ, на которых слать сигнналы Фибо
FIBO_COOLDOWN_SEC = 1200              # кулдаун по повторным сигналам (сек)
FIBO_PIVOT_WINDOW = 3                 # свинг-пивот: N баров слева/справа
FIBO_CONFIRM_PULLBACK_PCT = 0.15      # мин. откат, чтобы импульс считался завершённым (15%)
FIBO_PROXIMITY_BPS = 25               # “толщина” зоны в б.п. (0.25%)
FIBO_K_ATR = 0.35                     # добавочная “толщина” зоны в долях ATR(TF)
FIBO_MIN_BODY_FRAC = 0.4              # подтверждение свечой: тело ≥ 40% диапазона свечи
FIBO_REQUIRE_TREND_1D = True          # фильтр: работать по тренду 1D
FIBO_IMPORTANT_TAG = "important"      # тэг для сильных сигналов
FIBO_LEVELS_RETR = [23.6, 38.2, 50.0, 61.8, 78.6]
FIBO_LEVELS_EXT  = [127.2, 161.8, 261.8]

# === True Trading (реальный счёт) ===
TT_ENABLED = False                 # по умолчанию выкл; включим кнопкой /tt_on
TT_RISK_PCT = 0.01                 # риск на сделку (1% от эквити)
TT_MAX_OPEN_POS = 3                # максимум одновременных позиций
TT_DAILY_LOSS_LIMIT_PCT = 0.03     # дневной лимит просадки (3%) — пауза до завтра
TT_SYMBOL_COOLDOWN_MIN = 30        # кулдаун по символу после входа (мин)
TT_ORDER_SLIPPAGE_BPS = 20         # защита от проскальзывания (0.20%)
TT_REQUIRE_1D_TREND = True         # торговать только в сторону 1D (можно выключить)
TT_MIN_RR_TP1 = 1.6                # минимальный RR до TP1 для допуска сделки

# Доступ к Bybit (реальный счёт) — ключи положи в Render Environment
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "").strip()
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "").strip()
BYBIT_BASE_URL = "https://api.bybit.com"

# === ATTENTION (агрегатор Fibo + Fusion) ===
ATTN_ENABLED = True                 # включить агрегатор
ATTN_FUSION_MIN = 72               # минимальный скор Fusion
ATTN_REQUIRE_1D_TREND = True       # требовать совпадения с трендом 1D
ATTN_MIN_RR_TP1 = 1.6              # минимальный RR до TP1 (из Fibo)
ATTN_COOLDOWN_SEC = 900            # кулдаун по символу/TF/side
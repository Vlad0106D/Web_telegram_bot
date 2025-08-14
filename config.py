# config.py — минимальный конфиг
import os

# Токен из окружения. Никаких хардкодов.
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # обязательно

# Необязательные — можно оставить пустыми/не задавать
ALERT_CHAT_ID = os.getenv("ALERT_CHAT_ID")  # строка (например "-100123...") или None
DEFAULT_PAIRS = os.getenv("WATCHLIST", "BTCUSDT,ETHUSDT,SOLUSDT").replace(" ", "").split(",")

# Биржи по умолчанию (используем OKX, падаем на KuCoin если что)
DEFAULT_EXCHANGE = os.getenv("DEFAULT_EXCHANGE", "okx").lower()  # okx/kucoin

# Форматирование
SYMBOL_SUFFIX = "USDT"
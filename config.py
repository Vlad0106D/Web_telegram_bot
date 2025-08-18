# -*- coding: utf-8 -*-
import os

# === TELEGRAM ===
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # токен из переменных окружения
ALERT_CHAT_ID = os.getenv("ALERT_CHAT_ID")  # строка или None

# === WATCHLIST ===
PAIRS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]

# === INTERVALS ===
BREAKOUT_CHECK_SEC = 10
AUTOSCAN_INTERVAL_MIN = 30

# === INDICATORS / STRATEGY ===
RSI_PERIOD = 14
ADX_PERIOD = 14
BB_PERIOD  = 20
EMA_FAST   = 9
EMA_SLOW   = 21

ADX_STRONG = 25
RSI_OVERBOUGHT = 70
RSI_OVERSOLD   = 30

# === R:R правила (НОВОЕ) ===
# минимальная кратность риска до TP1 (например, 3.0 = 1:3)
MIN_R_MULT   = 3.0
# TP2 как кратность риска
TP2_R_MULT   = 4.5
# множитель ATR для "страховочного" стопа (кроме уровня)
ATR_MULT_SL  = 1.2
# максимальный риск в % от цены (если стоп слишком далёкий — пропускаем)
MAX_RISK_PCT = 0.012  # 1.2%

# === FORMATTING ===
EXCHANGE = "okx"
SYMBOL_SUFFIX = "USDT"
"""
MM v2 configuration (v1)

Scope:
- Symbols: BTC, ETH only
- Purpose:
  BTC -> main market direction
  ETH -> proxy for altcoins
- This file contains ONLY constants.
- No logic, no imports from project code.
"""

# =========================
# Universe
# =========================
SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
]

TFS = [
    "1h",
    "4h",
    "1d",
    "1w",
]

SOURCE = "okx"


# =========================
# TF step (seconds)
# =========================
TF_STEP_SEC = {
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
    "1w": 7 * 24 * 60 * 60,
}


# =========================
# Regime v1 (slow layer)
# =========================
REGIME_CONFIG = {
    "1h": {
        "ma_fast": 20,
        "ma_slow": 50,
        "slope_k": 10,
        "gap_min": 0.0015,
        "gap_strong": 0.0040,
        "slope_min": 0.0003,
        "m_confirm": 3,
    },
    "4h": {
        "ma_fast": 20,
        "ma_slow": 50,
        "slope_k": 8,
        "gap_min": 0.0020,
        "gap_strong": 0.0060,
        "slope_min": 0.0005,
        "m_confirm": 2,
    },
    "1d": {
        "ma_fast": 20,
        "ma_slow": 50,
        "slope_k": 5,
        "gap_min": 0.0030,
        "gap_strong": 0.0100,
        "slope_min": 0.0010,
        "m_confirm": 2,
    },
    "1w": {
        "ma_fast": 10,
        "ma_slow": 20,
        "slope_k": 3,
        "gap_min": 0.0040,
        "gap_strong": 0.0150,
        "slope_min": 0.0020,
        "m_confirm": 1,
    },
}

REGIME_VERSION = "regime_v1"


# =========================
# Phase / State v1 (fast layer)
# =========================
PHASE_CONFIG = {
    "1h": {
        "L": 5,
        "ret_min": 0.0020,
        "oi_min": 0.005,
        "vol_min": 1.2,
        "m_confirm": 2,
    },
    "4h": {
        "L": 3,
        "ret_min": 0.0035,
        "oi_min": 0.008,
        "vol_min": 1.2,
        "m_confirm": 1,
    },
    "1d": {
        "L": 2,
        "ret_min": 0.0070,
        "oi_min": 0.015,
        "vol_min": 1.2,
        "m_confirm": 1,
    },
    "1w": {
        "L": 1,
        "ret_min": 0.0200,
        "oi_min": 0.030,
        "vol_min": 1.2,
        "m_confirm": 1,
    },
}

PHASE_VERSION = "phase_v1"


# =========================
# Phase priority (strict order)
# =========================
PHASE_PRIORITY = [
    "UNWIND",
    "DISTRIBUTION",
    "PRESSURE_UP",
    "PRESSURE_DOWN",
    "WAIT",
]
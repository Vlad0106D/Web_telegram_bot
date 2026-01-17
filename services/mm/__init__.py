# services/mm/__init__.py
"""
MM module (snapshots -> outcomes -> features -> events)

Правила:
- Источник истины: mm_snapshots в БД (Neon)
- Outcomes и любые расчёты: только по данным из БД
- На старте: BTC-USDT и ETH-USDT, TF: H1/H4/D1/W1
"""
# Web Telegram Trading Bot

Telegram-based market analysis system for BTC/ETH. It combines closed-candle
snapshots, market structure, liquidity events, multi-timeframe context,
outcomes, derivatives statistics and versioned market scenarios.

## Scenario engine

The H1 automatic message is a decision-oriented scenario rather than a raw
indicator dump. It separates:

- `Direction` — which side currently has the advantage;
- `Setup` — whether a coherent event sequence exists;
- `Entry` — whether the current price offers acceptable asymmetry;
- primary targets, alternative targets and explicit invalidation.

`/scenario_now [H1|H4|D1|W1]` rebuilds the selected BTC liquidity map and
returns the current scenario. Existing `/mm_report` remains available for the
legacy detailed report.

## Liquidity-zone lifecycle

Zones are persisted in `liquidity_zones`, while every transition is stored in
`liquidity_zone_events`:

`created -> touched -> swept -> reclaimed | accepted | expired`

Pivots become visible only after their right-side confirmation bars. The same
chronological replay is used in live processing and backfill, preventing later
candles from influencing an earlier state.

On startup, `migrations/001_market_scenarios.sql` is applied idempotently.
Normal MM processing reconstructs BTC zones automatically for each configured
timeframe. To rebuild everything explicitly:

```bash
python -m services.mm.zone_backfill
```

Or one series:

```bash
python -m services.mm.zone_backfill --symbol BTC-USDT --tf H1
```

## Required environment

- `TELEGRAM_BOT_TOKEN` (or `TELEGRAM_TOKEN`)
- `DATABASE_URL`
- `ALERT_CHAT_ID`

Optional MM, watcher, Edge and Deriv settings are documented by their defaults
in `config.py` and the corresponding modules.

## Verification

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

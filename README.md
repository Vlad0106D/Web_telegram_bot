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
All SQL files in `migrations/` are applied in lexical order. Migration
`002_ml_foundation.sql` extends the existing schema without replacing raw
snapshots or historical tables.
Normal MM processing reconstructs BTC zones automatically for each configured
timeframe. To rebuild everything explicitly:

```bash
python -m services.mm.zone_backfill
```

Or one series:

```bash
python -m services.mm.zone_backfill --symbol BTC-USDT --tf H1
```

## ML-ready data contract

`mm_snapshots` remains the canonical closed-candle OHLCV source. Production
selects the newest OKX candle explicitly marked `confirm=1` and keeps the
unique `(symbol, tf, ts)` row.

For every persisted live scenario, `mm_features` receives one idempotent
point-in-time feature snapshot containing:

- bar-close and actual availability timestamps;
- immutable algorithm/configuration hash;
- price, ATR, returns, funding and open-interest context;
- scenario, Action Engine, MTF, range and liquidity context;
- data-quality metadata and explicit `live`/`replay` origin.

This is a dual-write foundation only. Telegram decisions continue to use the
existing scenario and Action Engine paths.

## Persistent setup lifecycle

`setup_lifecycle_v1` groups consecutive closed-candle Action Engine readings
into durable episodes instead of treating every candle as a new trade idea.
The additive tables `setup_episodes`, `setup_evaluations`,
`setup_observations`, and `setup_transitions` preserve candidate, watch, ready,
confirmed, cancelled, and expired states. Short-lived score weakness is kept
as an observation, direction changes close the old path, and an Action Engine
fingerprint prevents a confirmed setup from being reopened on every candle.

The lifecycle is an observational data layer. It does not alter Telegram
alerts, Action Engine scores, or execution decisions.

## Setup episode outcomes

`setup_outcome_v1` creates a separate label only after an episode reaches
`confirmed`. Entry is the confirmation close, with a 1 ATR stop, 1.5 ATR
target, and timeframe-specific horizon. Evaluation reads only candles that
were available after confirmation and never changes the source episode or
feature rows.

Resolved labels distinguish `target_hit`, `stop_hit`, `timeout`, and
`ambiguous`. When both levels fall inside one OHLC candle, the result stays
ambiguous instead of inventing an intrabar order. Pending and ambiguous rows
are explicitly unsuitable for directional model training.

## Durable production pipeline

`mm_pipeline_v1` records one persistent completion cursor per symbol and
timeframe in `mm_pipeline_checkpoints`. Restarts therefore do not rebuild D1
and W1 zones every minute or repeat an already completed H1/H4 analysis.
Telegram delivery is tracked separately, so a network timeout retries only the
message and never reruns zones, features, setup lifecycle, or outcomes.

Real candle-processing attempts and per-stage durations are stored in
`mm_pipeline_runs`. Slow synchronous analysis is executed outside the asyncio
event loop, allowing M5 live events and TradFi jobs to remain responsive while
a higher-timeframe candle is processed.

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

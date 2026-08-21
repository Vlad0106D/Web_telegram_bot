from __future__ import annotations

import logging
import os
from pathlib import Path

import psycopg

log = logging.getLogger(__name__)


def ensure_scenario_schema() -> None:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is empty")
    migration = (
        Path(__file__).resolve().parents[2] / "migrations" / "001_market_scenarios.sql"
    )
    with psycopg.connect(url) as conn:
        conn.execute(migration.read_text(encoding="utf-8"))
        conn.commit()
    log.info("Scenario schema is ready")

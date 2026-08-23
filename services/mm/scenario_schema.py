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
    migrations_dir = Path(__file__).resolve().parents[2] / "migrations"
    migrations = sorted(migrations_dir.glob("*.sql"))
    if not migrations:
        raise RuntimeError(f"No migrations found in {migrations_dir}")
    with psycopg.connect(url) as conn:
        for migration in migrations:
            conn.execute(migration.read_text(encoding="utf-8"))
            conn.commit()
            log.info("Applied idempotent migration %s", migration.name)
    log.info(
        "Scenario, ML, setup lifecycle, outcome, and pipeline schemas are ready"
    )

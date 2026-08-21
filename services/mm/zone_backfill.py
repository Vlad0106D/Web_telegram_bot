from __future__ import annotations

import argparse
import logging
from typing import Dict

from services.mm.zone_store import rebuild_zones

log = logging.getLogger(__name__)
DEFAULT_SYMBOLS = ("BTC-USDT", "ETH-USDT")
DEFAULT_TFS = ("H1", "H4", "D1", "W1")


def backfill_all() -> Dict[str, Dict[str, int]]:
    result: Dict[str, Dict[str, int]] = {}
    for symbol in DEFAULT_SYMBOLS:
        for tf in DEFAULT_TFS:
            key = f"{symbol}:{tf}"
            result[key] = rebuild_zones(symbol, tf)
            log.info("zone backfill %s: %s", key, result[key])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chronological liquidity-zone backfill"
    )
    parser.add_argument("--symbol", choices=DEFAULT_SYMBOLS)
    parser.add_argument("--tf", choices=DEFAULT_TFS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if bool(args.symbol) != bool(args.tf):
        parser.error("--symbol and --tf must be provided together")
    print(rebuild_zones(args.symbol, args.tf) if args.symbol else backfill_all())


if __name__ == "__main__":
    main()

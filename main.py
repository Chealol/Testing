"""CLI entry point orchestrating the NFL bot workflow."""
from __future__ import annotations

import argparse
import os
import logging
from datetime import datetime, timezone

from nfl_bot.data import get_week_bundle, auto_week
from nfl_bot.features import (
    build_coach_ref_contexts,
    attach_contexts,
    expand_weather,
    build_implied_totals,
)
from nfl_bot.llm import build_game_packets, llm_picks_via_responses, ask_ai
from logging_config import setup_logging

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NFL bot workflow")
    parser.add_argument(
        "--log-level",
        default=os.getenv("NFL_BOT_LOG_LEVEL", "INFO"),
        help="Logging level",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    level = "DEBUG" if args.verbose else args.log_level
    setup_logging(level)

    today = datetime.now(timezone.utc).date()
    season = today.year
    week = auto_week(season)
    logger.info("Fetching bundle for season=%s, week=%s ...", season, week)
    bundle = get_week_bundle(season, week)

    # Build and attach rolling contexts
    build_coach_ref_contexts(years_back=4)
    bundle = attach_contexts(bundle)

    # Quick data peeks
    logger.info("Schedule head:\n%s", bundle["schedule"].head(3))
    logger.info("Odds events head:\n%s", bundle["odds_events"].head(3))
    logger.info("Weather sample:\n%s", expand_weather(bundle).head(3))
    logger.info("Implied totals head:\n%s", build_implied_totals(bundle).head(3))

    # LLM picks and a sample Q&A
    packets = build_game_packets(bundle)
    picks_df = llm_picks_via_responses(packets, bankroll_units=10.0, debug=False)
    logger.info("LLM picks:\n%s", picks_df.head(10))
    logger.info(
        "%s",
        ask_ai(
            bundle,
            "Which three games this week look most weather-affected or pace-skewed?",
        ),
    )


if __name__ == "__main__":  # pragma: no cover
    main()

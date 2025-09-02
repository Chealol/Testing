"""CLI entry point orchestrating the NFL bot workflow."""
from __future__ import annotations

from datetime import datetime, timezone

from nfl_bot.data import get_week_bundle, auto_week
from nfl_bot.features import (
    build_coach_ref_contexts,
    attach_contexts,
    expand_weather,
    build_implied_totals,
)
from nfl_bot.llm import build_game_packets, llm_picks_via_responses, ask_ai


def main() -> None:
    today = datetime.now(timezone.utc).date()
    season = today.year
    week = auto_week(season)
    print(f"Fetching bundle for season={season}, week={week} ...")
    bundle = get_week_bundle(season, week)

    # Build and attach rolling contexts
    build_coach_ref_contexts(years_back=4)
    bundle = attach_contexts(bundle)

    # Quick data peeks
    print("Schedule head:\n", bundle["schedule"].head(3))
    print("Odds events head:\n", bundle["odds_events"].head(3))
    print("Weather sample:\n", expand_weather(bundle).head(3))
    print("Implied totals head:\n", build_implied_totals(bundle).head(3))

    # LLM picks and a sample Q&A
    packets = build_game_packets(bundle)
    picks_df = llm_picks_via_responses(packets, bankroll_units=10.0, debug=False)
    print("LLM picks:\n", picks_df.head(10))
    print(
        ask_ai(
            bundle,
            "Which three games this week look most weather-affected or pace-skewed?",
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()

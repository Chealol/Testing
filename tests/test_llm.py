import os
os.environ.setdefault("OPENAI_API_KEY", "test")

import pandas as pd

from nfl_bot.llm import build_game_packets


def test_build_game_packets_merges_contexts(
    sample_schedule,
    sample_odds_events,
    sample_odds_long,
    weather_by_game,
    monkeypatch,
):
    bundle = {
        "schedule": sample_schedule,
        "odds_events": sample_odds_events,
        "odds_long": sample_odds_long,
        "weather_by_game": weather_by_game,
    }

    monkeypatch.setattr("nfl_bot.llm.pressure_delta", lambda bundle: pd.DataFrame())
    monkeypatch.setattr("nfl_bot.llm.receiver_vs_secondary", lambda bundle: pd.DataFrame())
    monkeypatch.setattr("nfl_bot.llm.run_fit", lambda bundle: pd.DataFrame())

    packets = build_game_packets(bundle)
    assert isinstance(packets, list)
    assert len(packets) == len(sample_schedule)
    pkt = packets[0]
    assert {
        "game_id",
        "season",
        "week",
        "kickoff",
        "home",
        "away",
        "market",
        "weather",
        "referee",
        "matchups",
    } <= pkt.keys()
    assert {
        "total",
        "home_spread",
        "totals_book",
        "spreads_book",
    } <= pkt["market"].keys()
    assert {
        "temp_f",
        "wind_mph",
        "precip_prob_pct",
        "wind_15_plus",
        "precip_50_plus",
        "temp_below_32",
    } <= pkt["weather"].keys()

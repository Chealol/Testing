import pandas as pd

from nfl_bot.features import build_implied_totals


def test_build_implied_totals_handles_missing_lines():
    schedule = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2024, 2024],
            "week": [1, 1],
            "home_team": ["BUF", "MIA"],
            "away_team": ["NYJ", "NE"],
            "kickoff": [
                pd.Timestamp("2024-09-07T17:00:00Z"),
                pd.Timestamp("2024-09-07T20:00:00Z"),
            ],
        }
    )

    odds_events = pd.DataFrame(
        {
            "event_id": ["evt1", "evt2"],
            "commence_time": [
                "2024-09-07T17:00:00Z",
                "2024-09-07T20:00:00Z",
            ],
            "home_team": ["Buffalo Bills", "Miami Dolphins"],
            "away_team": ["New York Jets", "New England Patriots"],
            "home_code": ["BUF", "MIA"],
            "away_code": ["NYJ", "NE"],
        }
    )

    odds_long = pd.DataFrame(
        [
            {
                "event_id": "evt1",
                "commence_time": "2024-09-07T17:00:00Z",
                "home_team": "Buffalo Bills",
                "away_team": "New York Jets",
                "bookmaker_key": "fanduel",
                "bookmaker": "FanDuel",
                "last_update": "2024-09-01T00:00:00Z",
                "market": "totals",
                "name": "Over",
                "price": -110,
                "point": 44.5,
            },
            {
                "event_id": "evt2",
                "commence_time": "2024-09-07T20:00:00Z",
                "home_team": "Miami Dolphins",
                "away_team": "New England Patriots",
                "bookmaker_key": "fanduel",
                "bookmaker": "FanDuel",
                "last_update": "2024-09-01T00:00:00Z",
                "market": "spreads",
                "name": "Miami Dolphins",
                "price": -110,
                "point": -3.0,
            },
        ]
    )

    bundle = {
        "schedule": schedule,
        "odds_events": odds_events,
        "odds_long": odds_long,
    }

    result = build_implied_totals(bundle)

    g1 = result[result["game_id"] == "g1"].iloc[0]
    assert pd.isna(g1["home_spread"])
    assert pd.isna(g1["home_tt"]) and pd.isna(g1["away_tt"])

    g2 = result[result["game_id"] == "g2"].iloc[0]
    assert pd.isna(g2["total_line"])
    assert pd.isna(g2["home_tt"]) and pd.isna(g2["away_tt"])

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
import pandas as pd


@pytest.fixture
def sample_schedule():
    return pd.DataFrame(
        {
            "game_id": ["g1"],
            "season": [2024],
            "week": [1],
            "home_team": ["BUF"],
            "away_team": ["NYJ"],
            "kickoff": [pd.Timestamp("2024-09-07T17:00:00Z")],
        }
    )


@pytest.fixture
def sample_odds_events():
    return pd.DataFrame(
        {
            "event_id": ["evt1"],
            "commence_time": ["2024-09-07T17:00:00Z"],
            "home_team": ["Buffalo Bills"],
            "away_team": ["New York Jets"],
            "sport_key": ["americanfootball_nfl"],
            "home_code": ["BUF"],
            "away_code": ["NYJ"],
        }
    )


@pytest.fixture
def sample_odds_long():
    return pd.DataFrame(
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
                "event_id": "evt1",
                "commence_time": "2024-09-07T17:00:00Z",
                "home_team": "Buffalo Bills",
                "away_team": "New York Jets",
                "bookmaker_key": "fanduel",
                "bookmaker": "FanDuel",
                "last_update": "2024-09-01T00:00:00Z",
                "market": "spreads",
                "name": "Buffalo Bills",
                "price": -110,
                "point": -7.0,
            },
        ]
    )


@pytest.fixture
def weather_by_game(sample_schedule):
    return pd.DataFrame(
        {
            "game_id": sample_schedule["game_id"],
            "home_team": sample_schedule["home_team"],
            "kickoff": sample_schedule["kickoff"],
            "weather": [
                {
                    "temp_f": 70,
                    "wind_mph": 5,
                    "precip_prob_pct": 10,
                    "time_near_kickoff": "2024-09-07T17:00:00Z",
                    "lat": 0,
                    "lon": 0,
                }
            ],
        }
    )


@pytest.fixture
def sample_weather_json():
    return {
        "hourly": {
            "time": [
                "2024-09-07T00:00",
                "2024-09-07T01:00",
                "2024-09-07T02:00",
            ],
            "temperature_2m": [70, 71, 72],
            "wind_speed_10m": [5, 6, 7],
            "precipitation_probability": [0, 10, 20],
        }
    }

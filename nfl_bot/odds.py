"""Helpers for interacting with The Odds API."""
from __future__ import annotations

from typing import Optional, Tuple
import os
import logging

import requests
import pandas as pd

from . import FULL_TO_CODE
from .utils import http


def call_the_odds_api_odds(
    sport: str = "americanfootball_nfl",
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
    bookmakers: Optional[str] = None,
    commence_from_iso: Optional[str] = None,
    commence_to_iso: Optional[str] = None,
    timeout: int = 30,
) -> Tuple[list, dict]:
    """Call The Odds API and return the JSON payload and useful headers."""
    key = os.getenv("ODDS_API_KEY")
    if not key:
        raise RuntimeError("ODDS_API_KEY missing.")

    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
    params = {
        "apiKey": key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
    }
    if commence_from_iso:
        params["commenceTimeFrom"] = commence_from_iso
    if commence_to_iso:
        params["commenceTimeTo"] = commence_to_iso
    if bookmakers:
        params["bookmakers"] = bookmakers

    try:
        r = http.get(url, params=params, timeout=timeout)
    except requests.RequestException as exc:
        logging.error("The Odds API request failed: %s", exc)
        raise RuntimeError("Failed to fetch odds data") from exc

    headers = {
        "x-requests-remaining": r.headers.get("x-requests-remaining"),
        "x-requests-used": r.headers.get("x-requests-used"),
    }
    return r.json(), headers


def flatten_odds_events(events: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten the events list into event and odds DataFrames."""
    if not events:
        return pd.DataFrame(), pd.DataFrame()
    events_df = pd.json_normalize(events)
    keep = ["id", "commence_time", "home_team", "away_team", "sport_key"]
    for c in keep:
        if c not in events_df.columns:
            events_df[c] = None
    events_df = events_df[keep].rename(columns={"id": "event_id"})

    rows = []
    for e in events:
        eid = e.get("id")
        commence = e.get("commence_time")
        home = e.get("home_team")
        away = e.get("away_team")
        for bk in e.get("bookmakers", []) or []:
            bkey = bk.get("key")
            btitle = bk.get("title")
            updated = bk.get("last_update")
            for mkt in bk.get("markets", []) or []:
                mkey = mkt.get("key")
                for oc in mkt.get("outcomes", []) or []:
                    rows.append(
                        {
                            "event_id": eid,
                            "commence_time": commence,
                            "home_team": home,
                            "away_team": away,
                            "bookmaker_key": bkey,
                            "bookmaker": btitle,
                            "last_update": updated,
                            "market": mkey,
                            "name": oc.get("name"),
                            "price": oc.get("price"),
                            "point": oc.get("point"),
                        }
                    )
    odds_df = pd.DataFrame(rows)
    return events_df, odds_df


def map_events_to_codes(events_df: pd.DataFrame) -> pd.DataFrame:
    """Attach team codes to the odds events DataFrame."""
    out = events_df.copy()
    out["home_code"] = out["home_team"].map(FULL_TO_CODE)
    out["away_code"] = out["away_team"].map(FULL_TO_CODE)
    return out


__all__ = ["call_the_odds_api_odds", "flatten_odds_events", "map_events_to_codes"]

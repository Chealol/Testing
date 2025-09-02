"""Inactives data utilities."""
from __future__ import annotations

from typing import Dict, Any
import pandas as pd


def fetch_inactives(kickoff_ts: pd.Timestamp) -> pd.DataFrame:
    """Fetch the list of inactive players for a given kickoff time.

    This is currently a placeholder that returns an empty frame until a real
    data source is integrated.
    """
    cols = ["team", "player_name", "status", "reason"]
    return pd.DataFrame(columns=cols)


def read_inactives(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Merge inactives data into the provided game bundle."""
    schedule = bundle.get("schedule", pd.DataFrame())
    rows = []
    if not schedule.empty:
        for _, row in schedule.iterrows():
            kdt = row.get("kickoff")
            gid = row.get("game_id")
            df = fetch_inactives(kdt)
            df = df.copy()
            df["game_id"] = gid
            rows.append(df)
    bundle["inactives"] = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["team", "player_name", "status", "reason", "game_id"])
    )
    return bundle


__all__ = ["fetch_inactives", "read_inactives"]

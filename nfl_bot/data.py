"""Data loading utilities for the NFL bot."""
from __future__ import annotations

from typing import List, Dict, Any, Optional
import os
import logging

import pandas as pd
import numpy as np
import nfl_data_py as nfl
from zoneinfo import ZoneInfo

from . import TEAM_CODE_TO_FULL, FULL_TO_CODE, STADIUM_COORDS
from .odds import call_the_odds_api_odds, flatten_odds_events, map_events_to_codes
from .weather import open_meteo_for_game
from .inactives import read_inactives

logger = logging.getLogger(__name__)


def _concat_per_year_safely(import_fn, years: List[int], label: str) -> pd.DataFrame:
    """Helper that concatenates per-year nfl_data_py imports, skipping missing seasons."""
    frames, used = [], []
    for y in years:
        try:
            df = import_fn([y])
            if df is not None and len(df):
                frames.append(df)
                used.append(y)
        except Exception as e:  # pragma: no cover - passthrough logging
            logger.warning("%s: skipping %s -> %s: %s", label, y, type(e).__name__, e)
    if frames:
        out = pd.concat(frames, ignore_index=True)
        logger.info("%s: loaded years %s", label, used)
        return out
    logger.info("%s: no data loaded for %s", label, years)
    return pd.DataFrame()


def load_core_data(years: List[int]) -> Dict[str, pd.DataFrame]:
    """Load core nfl_data_py datasets for the requested seasons."""
    data: Dict[str, pd.DataFrame] = {}
    data["weekly"] = _concat_per_year_safely(nfl.import_weekly_data, years, "weekly")
    data["injuries"] = _concat_per_year_safely(nfl.import_injuries, years, "injuries")
    data["depth_charts"] = _concat_per_year_safely(nfl.import_depth_charts, years, "depth_charts")
    data["schedules"] = _concat_per_year_safely(nfl.import_schedules, years, "schedules")
    data["sc_lines"] = _concat_per_year_safely(nfl.import_sc_lines, years, "sc_lines")
    for stat in ("receiving", "rushing", "passing"):
        try:
            fn = lambda ys, s=stat: nfl.import_ngs_data(s, ys)
            data[f"ngs_{stat}"] = _concat_per_year_safely(fn, years, f"ngs_{stat}")
        except Exception as e:  # pragma: no cover - network errors
            logger.warning("ngs_%s: error -> %s", stat, e)
            data[f"ngs_{stat}"] = pd.DataFrame()
    return data


def latest_injury_snapshot(injuries_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Reduce the injuries table to the latest snapshot for a season."""
    if injuries_df is None or injuries_df.empty:
        return pd.DataFrame()
    df = injuries_df.copy()
    if "season" in df.columns:
        df = df[df["season"] == season].copy()
    if "player_name" not in df.columns:
        src = next((c for c in ["full_name", "player", "display_name", "gsis_name"] if c in df.columns), None)
        if src:
            df["player_name"] = df[src].astype(str)
        elif {"first_name", "last_name"}.issubset(df.columns):
            df["player_name"] = (df["first_name"].fillna("") + " " + df["last_name"].fillna(""))
        else:
            df["player_name"] = df.get("player_id", pd.Series([""] * len(df), index=df.index)).astype(str)
    team_src = next((c for c in ["team", "team_abbr", "club", "club_code", "team_code"] if c in df.columns), None)
    if team_src and team_src != "team":
        df = df.rename(columns={team_src: "team"})
    if "week" not in df.columns:
        wk_src = next((c for c in ["game_week", "week_number"] if c in df.columns), None)
        df = df.rename(columns={wk_src: "week"}) if wk_src else df.assign(week=pd.NA)
    keep = [
        "season", "week", "team", "player_id", "player_name", "position", "practice", "practice_status",
        "report_status", "game_status", "body_part", "injury_notes", "status_date", "report_date", "updated",
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy() if keep else df.copy()
    sort_cols = [c for c in ["week", "team", "player_name"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, na_position="last")
    return out.reset_index(drop=True)


def _parse_kickoff(row: pd.Series) -> Optional[pd.Timestamp]:
    """Best-effort parser for kickoff timestamps."""
    for c in ["kickoff", "datetime", "gamedatetime", "game_start", "start_time", "start_time_et", "game_time"]:
        if c in row and pd.notna(row[c]):
            ts = pd.to_datetime(row[c], utc=False, errors="coerce")
            if pd.notna(ts):
                return ts
    gd, gt = row.get("gameday"), row.get("gametime")
    if pd.notna(gd) and pd.notna(gt):
        try:
            return pd.Timestamp(f"{gd} {gt}").tz_localize(ZoneInfo("America/New_York")).tz_convert(None)
        except Exception:
            ts = pd.to_datetime(f"{gd} {gt}", errors="coerce")
            if pd.notna(ts):
                return ts
    if pd.notna(row.get("gameday")):
        return pd.to_datetime(row["gameday"], errors="coerce")
    if pd.notna(row.get("game_date")):
        return pd.to_datetime(row["game_date"], errors="coerce")
    return None


def get_week_bundle(season: int, week: int, years_back: int = 3, fd_only_fetch: bool = False) -> Dict[str, Any]:
    """Fetch schedules, injuries, odds and weather for a given week."""
    years = list(range(max(2019, season - years_back), season + 1))
    core = load_core_data(years)
    sched = core["schedules"]
    week_games = sched[(sched["season"] == season) & (sched["week"] == week)].copy()
    week_games["kickoff"] = week_games.apply(_parse_kickoff, axis=1)

    bookmakers = "fanduel" if fd_only_fetch else None
    events_json, headers = call_the_odds_api_odds(bookmakers=bookmakers)
    events_df, odds_df = flatten_odds_events(events_json)
    events_df = map_events_to_codes(events_df)

    weather_rows = []
    for _, g in week_games.iterrows():
        home = g.get("home_team")
        kdt = g.get("kickoff")
        wx = open_meteo_for_game(home, kdt) if isinstance(home, str) else None
        weather_rows.append({"season": season, "week": week, "home_team": home, "kickoff": kdt, "weather": wx})

    bundle = {
        "schedule": week_games.reset_index(drop=True),
        "scoring_lines": core.get("sc_lines"),
        "injuries": latest_injury_snapshot(core["injuries"], season),
        "odds_events": events_df,
        "odds_long": odds_df,
        "odds_headers": headers,
        "weather_by_game": pd.DataFrame(weather_rows),
    }
    bundle = read_inactives(bundle)
    return bundle


def auto_week(season: int) -> int:
    """Guess the current NFL week for the given season based on schedule timestamps."""
    core = load_core_data([season])
    s = core.get("schedules", pd.DataFrame()).copy()
    if s.empty:
        return 1

    def _kick(row: pd.Series):
        for c in ["kickoff", "datetime", "gamedatetime", "game_start", "start_time", "start_time_et", "gameday", "game_date"]:
            if c in s.columns and pd.notna(row.get(c)):
                ts = pd.to_datetime(row[c], errors="coerce")
                if pd.notna(ts):
                    return ts
        return pd.NaT

    s["kickoff_ts"] = s.apply(_kick, axis=1)
    future = s[(s["season"] == season) & s["kickoff_ts"].notna() & (s["kickoff_ts"] >= pd.Timestamp.now())]
    if not future.empty:
        return int(future.sort_values("kickoff_ts").iloc[0]["week"])
    wks = s.loc[s["season"] == season, "week"].dropna().astype(int)
    return int(wks.max()) if not wks.empty else 1


def get_coach_for_team(bundle: Dict[str, Any], team_code: str) -> Dict[str, Any]:
    df = bundle.get("coach_ctx", pd.DataFrame())
    if df.empty:
        return {}
    r = df[df["team"] == team_code].dropna(axis=1, how="all").head(1)
    return r.to_dict(orient="records")[0] if not r.empty else {}


def get_ref_for_game(bundle: Dict[str, Any], game_id: str) -> Dict[str, Any]:
    df = bundle.get("ref_ctx", pd.DataFrame())
    if df.empty:
        return {}
    r = df[df["game_id"] == game_id].dropna(axis=1, how="all").head(1)
    return r.to_dict(orient="records")[0] if not r.empty else {}


__all__ = [
    "load_core_data",
    "latest_injury_snapshot",
    "get_week_bundle",
    "auto_week",
    "get_coach_for_team",
    "get_ref_for_game",
]

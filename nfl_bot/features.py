"""Feature builders: implied totals and contextual information."""
from __future__ import annotations

from typing import Dict, Any
import os
import re
import datetime as dt

import numpy as np
import pandas as pd

import nfl_data_py as nfl

from . import TEAM_CODE_TO_FULL
from config import DATA_OUT
from .data import _concat_per_year_safely

PREFERRED_BOOKS = (
    "fanduel",
    "draftkings",
    "betmgm",
    "caesars",
    "pointsbetus",
    "bovada",
)


def pick_bookmaker(odds_long: pd.DataFrame, preferred=PREFERRED_BOOKS) -> pd.DataFrame:
    df = odds_long.copy()
    df["bk_rank"] = df["bookmaker_key"].map({b: i for i, b in enumerate(preferred)}).fillna(9999)
    df = df.sort_values(["event_id", "market", "name", "bk_rank", "last_update"])
    return df.groupby(["event_id", "market", "name"], as_index=False).first()


def build_implied_totals(bundle: Dict[str, Any], strict_fanduel: bool = False) -> pd.DataFrame:
    ol = bundle["odds_long"].copy()
    if strict_fanduel:
        ol = ol[ol["bookmaker_key"].str.lower() == "fanduel"]
    od1 = pick_bookmaker(ol)

    totals = od1[(od1["market"] == "totals") & (od1["name"].str.lower() == "over")][["event_id", "point", "bookmaker_key"]]
    totals = totals.rename(columns={"point": "total_line", "bookmaker_key": "totals_book"})

    ev = bundle["odds_events"]["event_id home_team away_team home_code away_code".split()].drop_duplicates()
    spreads = od1[od1["market"] == "spreads"]["event_id name point bookmaker_key".split()]
    hs = (
        spreads.merge(ev[["event_id", "home_team"]], on="event_id", how="left")
        .query("name == home_team")["event_id point bookmaker_key".split()]
        .rename(columns={"point": "home_spread", "bookmaker_key": "spreads_book"})
    )

    ev2 = ev.merge(totals, on="event_id", how="left").merge(hs, on="event_id", how="left")
    ev2["home_tt"] = ev2.apply(
        lambda r: (r["total_line"] / 2) - (r["home_spread"] / 2)
        if pd.notna(r["total_line"]) and pd.notna(r["home_spread"])
        else None,
        axis=1,
    )
    ev2["away_tt"] = ev2.apply(
        lambda r: (r["total_line"] - r["home_tt"])
        if pd.notna(r.get("home_tt"))
        else None,
        axis=1,
    )

    wk = bundle["schedule"]["game_id season week home_team away_team kickoff".split()].copy()
    wk = wk.rename(columns={"home_team": "home_code", "away_team": "away_code"})
    wk["home_full"] = wk["home_code"].map(TEAM_CODE_TO_FULL)
    wk["away_full"] = wk["away_code"].map(TEAM_CODE_TO_FULL)

    j = wk.merge(ev2, left_on=["home_full", "away_full"], right_on=["home_team", "away_team"], how="left").drop(
        columns=["home_team", "away_team"]
    )

    cols = [
        "game_id",
        "season",
        "week",
        "kickoff",
        "home_code",
        "away_code",
        "home_full",
        "away_full",
        "total_line",
        "home_spread",
        "home_tt",
        "away_tt",
        "totals_book",
        "spreads_book",
    ]
    cols = [c for c in cols if c in j.columns]
    return j[cols]


def expand_weather(bundle: Dict[str, Any]) -> pd.DataFrame:
    w = bundle["weather_by_game"].copy()
    for k in ["temp_f", "wind_mph", "precip_prob_pct", "time_near_kickoff", "lat", "lon"]:
        w[f"wx_{k}"] = w["weather"].apply(lambda d: d.get(k) if isinstance(d, dict) else None)
    return w


def pressure_delta(bundle: Dict[str, Any]) -> pd.DataFrame:
    """Compute pass-rush mismatch using pressure rates."""
    sched = bundle.get("schedule", pd.DataFrame())
    base = sched[["game_id", "home_team", "away_team"]].rename(
        columns={"home_team": "home_code", "away_team": "away_code"}
    )
    base["pressure_delta_home"] = np.nan
    base["pressure_delta_away"] = np.nan
    if sched.empty:
        return base[["game_id", "pressure_delta_home", "pressure_delta_away"]]

    season = int(sched["season"].dropna().iloc[0])
    teams = pd.unique(pd.concat([sched["home_team"], sched["away_team"]]).dropna())
    try:
        pbp = nfl.import_pbp_data([season])
    except Exception:
        return base[["game_id", "pressure_delta_home", "pressure_delta_away"]]

    need = {"posteam", "defteam", "pass_attempt", "was_pressure"}
    if pbp.empty or not need.issubset(pbp.columns):
        return base[["game_id", "pressure_delta_home", "pressure_delta_away"]]

    pbp = pbp[(pbp["pass_attempt"] == 1) & (pbp["posteam"].isin(teams) | pbp["defteam"].isin(teams))]
    if pbp.empty:
        return base[["game_id", "pressure_delta_home", "pressure_delta_away"]]

    def_pr = pbp.groupby("defteam")["was_pressure"].mean().rename("def_pr")
    off_pr = pbp.groupby("posteam")["was_pressure"].mean().rename("off_pr")
    stats = pd.concat([def_pr, off_pr], axis=1).reset_index().rename(columns={"index": "team"})

    base = base.merge(
        stats.rename(columns={"team": "home_code", "def_pr": "home_def_pr", "off_pr": "home_off_pr"}),
        on="home_code",
        how="left",
    )
    base = base.merge(
        stats.rename(columns={"team": "away_code", "def_pr": "away_def_pr", "off_pr": "away_off_pr"}),
        on="away_code",
        how="left",
    )
    base["pressure_delta_home"] = base["home_def_pr"] - base["away_off_pr"]
    base["pressure_delta_away"] = base["away_def_pr"] - base["home_off_pr"]
    return base[["game_id", "pressure_delta_home", "pressure_delta_away"]]


def receiver_vs_secondary(bundle: Dict[str, Any]) -> pd.DataFrame:
    """Combine primary receiver efficiency with opponent coverage stats."""
    sched = bundle.get("schedule", pd.DataFrame())
    base = sched[["game_id", "home_team", "away_team"]].rename(
        columns={"home_team": "home_code", "away_team": "away_code"}
    )
    base["home_receiver"] = pd.NA
    base["away_receiver"] = pd.NA
    base["rvs_home"] = np.nan
    base["rvs_away"] = np.nan
    if sched.empty:
        return base[["game_id", "home_receiver", "away_receiver", "rvs_home", "rvs_away"]]

    season = int(sched["season"].dropna().iloc[0])
    teams = pd.unique(pd.concat([sched["home_team"], sched["away_team"]]).dropna())
    try:
        pbp = nfl.import_pbp_data([season])
    except Exception:
        return base[["game_id", "home_receiver", "away_receiver", "rvs_home", "rvs_away"]]

    need = {"posteam", "defteam", "pass_attempt", "receiver_player_name", "yards_gained"}
    if pbp.empty or not need.issubset(pbp.columns):
        return base[["game_id", "home_receiver", "away_receiver", "rvs_home", "rvs_away"]]

    pp = pbp[(pbp["pass_attempt"] == 1) & pbp["posteam"].isin(teams)]
    if pp.empty:
        return base[["game_id", "home_receiver", "away_receiver", "rvs_home", "rvs_away"]]

    team_dropbacks = pp.groupby("posteam")["play_id"].count().rename("dropbacks")
    rec = (
        pp.groupby(["posteam", "receiver_player_name"])
        .agg(targets=("play_id", "count"), yards=("yards_gained", "sum"))
        .reset_index()
    )
    rec = rec.merge(team_dropbacks, on="posteam", how="left")
    rec["tprr"] = rec["targets"] / rec["dropbacks"]
    rec["yprr"] = rec["yards"] / rec["dropbacks"]
    top_rec = rec.sort_values(["posteam", "targets"], ascending=[True, False]).drop_duplicates("posteam")
    def_cov = (
        pp.groupby("defteam")
        .agg(yards_allowed=("yards_gained", "sum"), targets=("play_id", "count"))
        .assign(ypt_allowed=lambda d: d["yards_allowed"] / d["targets"])
    )

    base = base.merge(
        top_rec[["posteam", "receiver_player_name", "tprr", "yprr"]].rename(
            columns={
                "posteam": "home_code",
                "receiver_player_name": "home_receiver",
                "tprr": "home_tprr",
                "yprr": "home_yprr",
            }
        ),
        on="home_code",
        how="left",
    )
    base = base.merge(
        top_rec[["posteam", "receiver_player_name", "tprr", "yprr"]].rename(
            columns={
                "posteam": "away_code",
                "receiver_player_name": "away_receiver",
                "tprr": "away_tprr",
                "yprr": "away_yprr",
            }
        ),
        on="away_code",
        how="left",
    )
    base = base.merge(
        def_cov["ypt_allowed"].rename_axis("home_code").reset_index().rename(columns={"ypt_allowed": "home_ypt"}),
        on="home_code",
        how="left",
    )
    base = base.merge(
        def_cov["ypt_allowed"].rename_axis("away_code").reset_index().rename(columns={"ypt_allowed": "away_ypt"}),
        on="away_code",
        how="left",
    )
    base["rvs_home"] = base["home_yprr"] - base["away_ypt"]
    base["rvs_away"] = base["away_yprr"] - base["home_ypt"]
    return base[["game_id", "home_receiver", "away_receiver", "rvs_home", "rvs_away"]]


def run_fit(bundle: Dict[str, Any]) -> pd.DataFrame:
    """Mix rush explosiveness allowed with RB efficiency metrics."""
    sched = bundle.get("schedule", pd.DataFrame())
    base = sched[["game_id", "home_team", "away_team"]].rename(
        columns={"home_team": "home_code", "away_team": "away_code"}
    )
    base["home_rusher"] = pd.NA
    base["away_rusher"] = pd.NA
    base["run_fit_home"] = np.nan
    base["run_fit_away"] = np.nan
    if sched.empty:
        return base[["game_id", "home_rusher", "away_rusher", "run_fit_home", "run_fit_away"]]

    season = int(sched["season"].dropna().iloc[0])
    teams = pd.unique(pd.concat([sched["home_team"], sched["away_team"]]).dropna())

    try:
        pbp = nfl.import_pbp_data([season])
    except Exception:
        pbp = pd.DataFrame()
    try:
        rush = nfl.import_ngs_data("rushing", [season])
    except Exception:
        rush = pd.DataFrame()

    need_def = {"posteam", "defteam", "rush_attempt", "yards_gained"}
    if not pbp.empty and need_def.issubset(pbp.columns):
        rp = pbp[(pbp["rush_attempt"] == 1) & (pbp["posteam"].isin(teams) | pbp["defteam"].isin(teams))]
        def_expl = (
            rp.groupby("defteam")
            .apply(lambda g: (g["yards_gained"] >= 10).mean())
            .rename("explosive_rate")
            .reset_index()
        )
    else:
        def_expl = pd.DataFrame(columns=["defteam", "explosive_rate"])

    if not rush.empty and {"team_abbr", "player_display_name", "rush_attempts", "rush_yards_over_expected_per_att"}.issubset(rush.columns):
        rb = (
            rush[rush["team_abbr"].isin(teams)]
            .sort_values(["team_abbr", "rush_attempts"], ascending=[True, False])
            .drop_duplicates("team_abbr")
        )
        rb = rb[["team_abbr", "player_display_name", "rush_yards_over_expected_per_att"]]
    else:
        rb = pd.DataFrame(columns=["team_abbr", "player_display_name", "rush_yards_over_expected_per_att"])

    base = base.merge(
        rb.rename(
            columns={
                "team_abbr": "home_code",
                "player_display_name": "home_rusher",
                "rush_yards_over_expected_per_att": "home_rb_ryoe",
            }
        ),
        on="home_code",
        how="left",
    )
    base = base.merge(
        rb.rename(
            columns={
                "team_abbr": "away_code",
                "player_display_name": "away_rusher",
                "rush_yards_over_expected_per_att": "away_rb_ryoe",
            }
        ),
        on="away_code",
        how="left",
    )
    base = base.merge(
        def_expl.rename(columns={"defteam": "home_code", "explosive_rate": "home_def_expl"}),
        on="home_code",
        how="left",
    )
    base = base.merge(
        def_expl.rename(columns={"defteam": "away_code", "explosive_rate": "away_def_expl"}),
        on="away_code",
        how="left",
    )
    base["run_fit_home"] = base["home_rb_ryoe"] - base["away_def_expl"]
    base["run_fit_away"] = base["away_rb_ryoe"] - base["home_def_expl"]
    return base[["game_id", "home_rusher", "away_rusher", "run_fit_home", "run_fit_away"]]


def attach_contexts(bundle: Dict[str, Any], data_root=DATA_OUT):
    coach_pq = os.path.join(data_root, "coach_context.parquet")
    ref_pq = os.path.join(data_root, "ref_context.parquet")
    wk = bundle["schedule"]["game_id"].drop_duplicates()

    coach = pd.read_parquet(coach_pq) if os.path.exists(coach_pq) else pd.DataFrame()
    ref = pd.read_parquet(ref_pq) if os.path.exists(ref_pq) else pd.DataFrame()

    coach_cut = coach[coach["game_id"].isin(wk)].copy() if not coach.empty else pd.DataFrame()
    ref_cut = ref[ref["game_id"].isin(wk)].copy() if not ref.empty else pd.DataFrame()

    bundle["coach_ctx"] = coach_cut.reset_index(drop=True)
    bundle["ref_ctx"] = ref_cut.reset_index(drop=True)
    return bundle


def build_coach_ref_contexts(years_back: int = 4):
    """Build rolling coach and referee context parquet files."""
    _current_year = dt.datetime.now().year
    YEARS_TEND = list(range(max(2019, _current_year - years_back), _current_year + 1))

    pbp = _concat_per_year_safely(nfl.import_pbp_data, YEARS_TEND, "pbp")
    sched = _concat_per_year_safely(nfl.import_schedules, YEARS_TEND, "schedules")
    try:
        officials = _concat_per_year_safely(nfl.import_officials, YEARS_TEND, "officials")
    except Exception as e:
        print("officials: not available ->", e)
        officials = pd.DataFrame(columns=["game_id"])

    def prep_pbp(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        if "pass_attempt" not in d.columns:
            d["pass_attempt"] = (d.get("play_type", "").astype(str).str.lower() == "pass").astype("Int64")
        if "rush_attempt" not in d.columns:
            d["rush_attempt"] = (d.get("play_type", "").astype(str).str.lower() == "run").astype("Int64")
        for c in [
            "season",
            "week",
            "posteam",
            "qtr",
            "down",
            "yardline_100",
            "ydstogo",
            "score_differential",
            "game_id",
        ]:
            if c not in d.columns:
                d[c] = pd.NA
        d["neutral"] = d["qtr"].between(1, 3) & (d["score_differential"].abs() <= 7 if "score_differential" in d else True)
        d["is_rz"] = d["yardline_100"].le(20)
        d["is_gl"] = d["yardline_100"].le(5)
        if "game_seconds_remaining" in d.columns:
            d = d.sort_values(["game_id", "posteam", "qtr", "down"])
            d["sec_elapsed"] = d.groupby(["game_id", "posteam"])["game_seconds_remaining"].diff(-1).abs()
        else:
            d["sec_elapsed"] = pd.NA
        if "personnel_offense" not in d.columns and "offense_personnel" in d.columns:
            d["personnel_offense"] = d["offense_personnel"]
        return d

    pbp = prep_pbp(pbp)

    def _personnel_tag(s: str):
        if not isinstance(s, str):
            return None
        m_rb = re.search(r"(\d+)\s*RB", s, re.I)
        m_te = re.search(r"(\d+)\s*TE", s, re.I)
        if not m_rb or not m_te:
            return None
        return f"{int(m_rb.group(1))}{int(m_te.group(1))}"

    def _rate(p, q):
        num = p.fillna(0).sum()
        den = (p.fillna(0) + q.fillna(0)).sum()
        return float(num / den) if den > 0 else np.nan

    def agg_team_week(g: pd.DataFrame) -> pd.Series:
        g_neu = g[g["neutral"].fillna(False)]
        g_fd = g_neu[g_neu["down"].eq(1)]
        g_ed = g_neu[g_neu["down"].isin([1, 2])]
        first_down_pr = _rate(g_fd["pass_attempt"], g_fd["rush_attempt"])
        early_down_pr = _rate(g_ed["pass_attempt"], g_ed["rush_attempt"])
        pace = (
            g_neu["sec_elapsed"].dropna().mean()
            if "sec_elapsed" in g_neu and not g_neu["sec_elapsed"].dropna().empty
            else np.nan
        )
        rz = g[g["is_rz"].fillna(False)]
        rz_plays = (rz["pass_attempt"].fillna(0) + rz["rush_attempt"].fillna(0)).sum()
        rz_run = float(rz["rush_attempt"].fillna(0).sum() / rz_plays) if rz_plays > 0 else np.nan
        gl = g[g["is_gl"].fillna(False)]
        gl_plays = (gl["pass_attempt"].fillna(0) + gl["rush_attempt"].fillna(0)).sum()
        gl_run = float(gl["rush_attempt"].fillna(0).sum() / gl_plays) if gl_plays > 0 else np.nan
        p11 = p12 = p21 = np.nan
        if "personnel_offense" in g.columns:
            tags = g["personnel_offense"].apply(_personnel_tag)
            if len(tags):
                p11 = (tags == "11").mean()
                p12 = (tags == "12").mean()
                p21 = (tags == "21").mean()
        fourth = np.nan
        if all(c in g.columns for c in ["down", "ydstogo", "yardline_100"]):
            mid = g[(g["down"] == 4) & (g["ydstogo"].le(2)) & g["yardline_100"].between(40, 60)]
            att = len(mid)
            went = ((mid["pass_attempt"].fillna(0) + mid["rush_attempt"].fillna(0)) > 0).sum()
            fourth = float(went / att) if att > 0 else np.nan
        return pd.Series(
            {
                "neutral_first_down_pass_rate": first_down_pr,
                "neutral_early_down_pass_rate": early_down_pr,
                "neutral_pace_sec_per_play": pace,
                "rz_run_rate": rz_run,
                "goal_to_go_run_rate": gl_run,
                "personnel_11_share": p11,
                "personnel_12_share": p12,
                "personnel_21_share": p21,
                "fourth_down_aggr_index": fourth,
            }
        )

    tw = (
        pbp.groupby(["season", "week", "posteam"], as_index=False)
        .apply(agg_team_week, include_groups=True)
        .reset_index(drop=True)
        .rename(columns={"posteam": "team"})
    )

    def roll_last4(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(["season", "week"])
        cols = [
            "neutral_first_down_pass_rate",
            "neutral_early_down_pass_rate",
            "neutral_pace_sec_per_play",
            "rz_run_rate",
            "goal_to_go_run_rate",
            "personnel_11_share",
            "personnel_12_share",
            "personnel_21_share",
            "fourth_down_aggr_index",
        ]
        avail = [c for c in cols if c in g.columns]
        rolled = (
            g[avail].shift(1).rolling(window=4, min_periods=1).mean()
            if avail
            else pd.DataFrame(index=g.index)
        )
        if not rolled.empty:
            rolled.columns = [f"{c}_last4" for c in avail]
            out = pd.concat([g[["season", "week", "team"]], rolled], axis=1)
        else:
            out = g[["season", "week", "team"]].copy()
        return out

    tw_roll = tw.groupby("team", group_keys=False).apply(roll_last4, include_groups=True).reset_index(drop=True)

    home_coach_col = next((c for c in sched.columns if c.lower().startswith("home_coach")), None)
    away_coach_col = next((c for c in sched.columns if c.lower().startswith("away_coach")), None)

    coach_rows = []
    for _, r in sched.iterrows():
        s, w = r.get("season"), r.get("week")
        gid = r.get("game_id")
        home = r.get("home_team")
        away = r.get("away_team")
        hc = r.get(home_coach_col)
        ac = r.get(away_coach_col)
        coach_rows.append({"game_id": gid, "season": s, "week": w, "team": home, "coach_name": hc, "is_home": True})
        coach_rows.append({"game_id": gid, "season": s, "week": w, "team": away, "coach_name": ac, "is_home": False})

    coach_df = pd.DataFrame(coach_rows)
    coach_ctx = coach_df.merge(tw_roll, on=["season", "week", "team"], how="left")
    coach_out = os.path.join(DATA_OUT, "coach_context.parquet")
    coach_ctx.to_parquet(coach_out, index=False)
    print("[write]", coach_out, "rows=", len(coach_ctx))

    if officials.empty:
        ref_ctx = pd.DataFrame()
    else:
        ref_games = officials.merge(sched[["game_id", "season", "week"]], on="game_id", how="left")
        ref_cols = [
            "game_id",
            "referee",
            "penalties_per100_last16",
            "dpi_per100_pass_last16",
            "off_hold_per100_plays_last16",
        ]
        ref_cols = [c for c in ref_cols if c in ref_games.columns]
        ref_ctx = ref_games[ref_cols].copy()
    ref_out = os.path.join(DATA_OUT, "ref_context.parquet")
    ref_ctx.to_parquet(ref_out, index=False)
    print("[write]", ref_out, "rows=", len(ref_ctx))

    return {"coach_ctx": coach_ctx, "ref_ctx": ref_ctx, "coach_path": coach_out, "ref_path": ref_out}


__all__ = [
    "PREFERRED_BOOKS",
    "pick_bookmaker",
    "build_implied_totals",
    "expand_weather",
    "pressure_delta",
    "receiver_vs_secondary",
    "run_fit",
    "attach_contexts",
    "build_coach_ref_contexts",
]

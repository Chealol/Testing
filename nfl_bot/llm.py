"""LLM powered picks and Q&A helpers."""
from __future__ import annotations

from typing import List, Dict, Any, Optional
import json
import os
import re
import logging

import pandas as pd
import numpy as np
from openai import OpenAI

from . import TEAM_CODE_TO_FULL
from .features import (
    build_implied_totals,
    expand_weather,
    pressure_delta,
    receiver_vs_secondary,
    run_fit,
)

logger = logging.getLogger(__name__)

client = OpenAI()


def build_game_packets(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Merge implied totals, weather, coach & ref context into JSON packets per game."""
    imp = build_implied_totals(bundle, strict_fanduel=False)
    wx = expand_weather(bundle)[
        [
            "home_team",
            "kickoff",
            "wx_temp_f",
            "wx_wind_mph",
            "wx_precip_prob_pct",
            "wx_wind_15_plus",
            "wx_precip_50_plus",
            "wx_temp_below_32",
        ]
    ].rename(columns={"home_team": "home_code"})
    coach = bundle.get("coach_ctx", pd.DataFrame())
    refc = bundle.get("ref_ctx", pd.DataFrame())
    sched = bundle["schedule"]["game_id season week home_team away_team kickoff".split()].rename(
        columns={"home_team": "home_code", "away_team": "away_code"}
    )

    press = pressure_delta(bundle)
    rvs = receiver_vs_secondary(bundle)
    rfit = run_fit(bundle)

    if not coach.empty:
        last4 = [c for c in coach.columns if c.endswith("_last4")]
        h = coach[coach["is_home"] == True][["game_id", "team", "coach_name"] + last4].rename(
            columns={"team": "home_code", "coach_name": "home_coach"}
        )
        a = coach[coach["is_home"] == False][["game_id", "team", "coach_name"] + last4].rename(
            columns={"team": "away_code", "coach_name": "away_coach"}
        )
        for c in last4:
            h.rename(columns={c: f"{c}_home"}, inplace=True)
            a.rename(columns={c: f"{c}_away"}, inplace=True)
        coach_wide = h.merge(a, on="game_id", how="outer")
    else:
        coach_wide = pd.DataFrame()

    if not refc.empty:
        ref_keep = [
            "game_id",
            "ref_name",
            "penalties_per100_last16",
            "dpi_per100_pass_last16",
            "off_hold_per100_plays_last16",
        ]
        ref_small = refc[[c for c in ref_keep if c in refc.columns]].drop_duplicates("game_id")
    else:
        ref_small = pd.DataFrame(columns=["game_id", "ref_name"])

    preferred_keys = ["game_id", "season", "week", "kickoff", "home_code", "away_code"]
    base = sched.merge(imp, on=[k for k in preferred_keys if k in sched.columns and k in imp.columns] or ["game_id"], how="left")

    if {"home_code", "kickoff"}.issubset(base.columns):
        base = base.merge(wx, on=["home_code", "kickoff"], how="left")
    elif "home_code" in base.columns:
        base = base.merge(wx.drop(columns=["kickoff"]), on=["home_code"], how="left")

    if not coach_wide.empty:
        base = base.merge(coach_wide, on="game_id", how="left")
    if not ref_small.empty:
        base = base.merge(ref_small, on="game_id", how="left")
    for extra in (press, rvs, rfit):
        if extra is not None and not extra.empty:
            base = base.merge(extra, on="game_id", how="left")

    def _safe(x, nd=3):
        try:
            if x is None or (isinstance(x, float) and (pd.isna(x) or np.isinf(x))):
                return None
            return round(float(x), nd)
        except Exception:
            return None

    packets = []
    for _, r in base.iterrows():
        home_code = r.get("home_code")
        away_code = r.get("away_code")
        packets.append(
            {
                "game_id": r.get("game_id"),
                "season": int(r["season"]) if pd.notna(r.get("season")) else None,
                "week": int(r["week"]) if pd.notna(r.get("week")) else None,
                "kickoff": str(r.get("kickoff")) if pd.notna(r.get("kickoff")) else None,
                "home": {
                    "code": home_code,
                    "name": TEAM_CODE_TO_FULL.get(home_code, home_code),
                    "implied_total": _safe(r.get("home_tt")),
                    "coach": r.get("home_coach"),
                    "tendencies_last4": {
                        "early_down_pr": _safe(r.get("neutral_early_down_pass_rate_last4_home")),
                        "pace_sec_play": _safe(r.get("neutral_pace_sec_per_play_last4_home")),
                    },
                },
                "away": {
                    "code": away_code,
                    "name": TEAM_CODE_TO_FULL.get(away_code, away_code),
                    "implied_total": _safe(r.get("away_tt")),
                    "coach": r.get("away_coach"),
                    "tendencies_last4": {
                        "early_down_pr": _safe(r.get("neutral_early_down_pass_rate_last4_away")),
                        "pace_sec_play": _safe(r.get("neutral_pace_sec_per_play_last4_away")),
                    },
                },
                "market": {
                    "total": _safe(r.get("total_line")),
                    "home_spread": _safe(r.get("home_spread")),
                    "totals_book": r.get("totals_book"),
                    "spreads_book": r.get("spreads_book"),
                },
                "weather": {
                    "temp_f": _safe(r.get("wx_temp_f")),
                    "wind_mph": _safe(r.get("wx_wind_mph")),
                    "precip_prob_pct": _safe(r.get("wx_precip_prob_pct")),
                    "wind_15_plus": (
                        bool(r.get("wx_wind_15_plus"))
                        if pd.notna(r.get("wx_wind_15_plus"))
                        else None
                    ),
                    "precip_50_plus": (
                        bool(r.get("wx_precip_50_plus"))
                        if pd.notna(r.get("wx_precip_50_plus"))
                        else None
                    ),
                    "temp_below_32": (
                        bool(r.get("wx_temp_below_32"))
                        if pd.notna(r.get("wx_temp_below_32"))
                        else None
                    ),
                },
                "referee": {
                    "name": r.get("ref_name"),
                    "penalties_per100_last16": _safe(r.get("penalties_per100_last16")),
                    "dpi_per100_pass_last16": _safe(r.get("dpi_per100_pass_last16")),
                    "off_hold_per100_plays_last16": _safe(r.get("off_hold_per100_plays_last16")),
                },
                "matchups": {
                    "pressure_delta": {
                        "home": _safe(r.get("pressure_delta_home")),
                        "away": _safe(r.get("pressure_delta_away")),
                    },
                    "receiver_vs_secondary": {
                        "home": {
                            "player": (r.get("home_receiver") if pd.notna(r.get("home_receiver")) else None),
                            "edge": _safe(r.get("rvs_home")),
                        },
                        "away": {
                            "player": (r.get("away_receiver") if pd.notna(r.get("away_receiver")) else None),
                            "edge": _safe(r.get("rvs_away")),
                        },
                    },
                    "run_fit": {
                        "home": {
                            "player": (r.get("home_rusher") if pd.notna(r.get("home_rusher")) else None),
                            "edge": _safe(r.get("run_fit_home")),
                        },
                        "away": {
                            "player": (r.get("away_rusher") if pd.notna(r.get("away_rusher")) else None),
                            "edge": _safe(r.get("run_fit_away")),
                        },
                    },
                },
            }
        )
    return packets


_JSON_SCHEMA = {
    "name": "picks_schema",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "picks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "game_id": {"type": "string"},
                        "market": {"type": "string", "enum": ["spread", "total", "moneyline", "pass"]},
                        "selection": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "units": {"type": "number", "minimum": 0},
                        "edge_note": {"type": "string"},
                    },
                    "required": ["game_id", "market", "selection", "confidence", "units", "edge_note"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["picks"],
        "additionalProperties": False,
    },
}

_SYSTEM_PICKS = (
    "You are an NFL betting analyst. For each game, either select one market (spread/total/moneyline) "
    "with a clear side/number, or return 'pass'. Use implied totals/spread, weather, coach tendencies (last4), "
    "and referee profile (last16). Keep stakes disciplined. Return ONLY JSON that matches the schema."
)


def llm_picks_via_responses(
    packets: List[Dict[str, Any]],
    bankroll_units: float = 10.0,
    model: Optional[str] = None,
    chunk_size: int = 6,
    debug: bool = True,
) -> pd.DataFrame:
    """Get LLM picks; robust to SDK/model capability differences."""
    mdl = model or os.getenv("OPENAI_MODEL", "gpt-5")
    all_picks, raw_snips = [], []
    for i in range(0, len(packets), chunk_size):
        block = packets[i : i + chunk_size]
        prompt = {"bankroll_units": bankroll_units, "games": block}
        raw = None
        try:
            resp = client.responses.create(
                model=mdl,
                reasoning={"effort": "low"},
                instructions=_SYSTEM_PICKS,
                input=json.dumps(prompt, ensure_ascii=False),
                response_format={"type": "json_schema", "json_schema": _JSON_SCHEMA},
                max_output_tokens=800,
            )
            raw = (resp.output_text or "").strip()
        except TypeError:
            resp = client.responses.create(
                model=mdl,
                reasoning={"effort": "low"},
                instructions=_SYSTEM_PICKS + " Schema: " + json.dumps(_JSON_SCHEMA["schema"]),
                input=json.dumps(prompt, ensure_ascii=False),
                max_output_tokens=800,
            )
            raw = (resp.output_text or "").strip()
        except Exception:
            raw = None
        raw_snips.append(raw)
        try:
            js = json.loads(raw) if raw else {"picks": []}
            all_picks.extend(js.get("picks", []))
        except Exception:
            all_picks.extend([])
    df = pd.DataFrame(all_picks)
    if debug:
        for k, snip in enumerate(raw_snips):
            logger.debug("--- Chunk %s ---\n%s\n", k, snip)
    if df.empty:
        return df
    order = ["game_id", "market", "selection", "confidence", "units", "edge_note"]
    order = [c for c in order if c in df.columns]
    return df[order].sort_values(["confidence", "units"], ascending=[False, False]).reset_index(drop=True)


def _find_games_by_text(question: str, sched_df: pd.DataFrame) -> list[str]:
    q = question.lower()
    want = set()
    for code, full in TEAM_CODE_TO_FULL.items():
        if re.search(rf"\b{code.lower()}\b", q) or re.search(rf"\b{full.lower()}\b", q):
            want.add(code)
    if not want:
        return sched_df["game_id"].tolist()
    mask = sched_df["home_team"].isin(want) | sched_df["away_team"].isin(want)
    return sched_df.loc[mask, "game_id"].tolist()


def _limit_context_rows(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    return df.head(n).copy()


def build_qa_context(bundle: dict, question: str) -> dict:
    imp = build_implied_totals(bundle, strict_fanduel=False)
    wx = expand_weather(bundle)[
        [
            "home_team",
            "kickoff",
            "wx_temp_f",
            "wx_wind_mph",
            "wx_precip_prob_pct",
            "wx_wind_15_plus",
            "wx_precip_50_plus",
            "wx_temp_below_32",
        ]
    ]
    coach = bundle.get("coach_ctx", pd.DataFrame())
    refc = bundle.get("ref_ctx", pd.DataFrame())
    sched = bundle["schedule"]["game_id season week home_team away_team kickoff".split()].rename(
        columns={"home_team": "home_code", "away_team": "away_code"}
    )
    game_ids = _find_games_by_text(question, bundle["schedule"])
    sched = sched[sched["game_id"].isin(game_ids)].copy()

    keys = [
        k
        for k in ["game_id", "season", "week", "kickoff", "home_code", "away_code"]
        if k in imp.columns and k in sched.columns
    ] or ["game_id"]
    base = sched.merge(imp, on=keys, how="left")

    if {"home_code", "kickoff"}.issubset(base.columns):
        base = base.merge(wx.rename(columns={"home_team": "home_code"}), on=["home_code", "kickoff"], how="left")
    else:
        base = base.merge(
            wx.rename(columns={"home_team": "home_code"}).drop(columns=["kickoff"], errors="ignore"),
            on="home_code",
            how="left",
        )

    if not coach.empty:
        last4 = [c for c in coach.columns if c.endswith("_last4")]
        h = coach[coach["is_home"] == True][["game_id", "team", "coach_name"] + last4].rename(
            columns={"team": "home_code", "coach_name": "home_coach"}
        )
        a = coach[coach["is_home"] == False][["game_id", "team", "coach_name"] + last4].rename(
            columns={"team": "away_code", "coach_name": "away_coach"}
        )
        for c in last4:
            h.rename(columns={c: f"{c}_home"}, inplace=True)
            a.rename(columns={c: f"{c}_away"}, inplace=True)
        base = base.merge(h, on=["game_id", "home_code"], how="left").merge(a, on=["game_id", "away_code"], how="left")

    if not refc.empty:
        rkeep = [
            "game_id",
            "ref_name",
            "penalties_per100_last16",
            "dpi_per100_pass_last16",
            "off_hold_per100_plays_last16",
        ]
        rsmall = refc[[c for c in rkeep if c in refc.columns]].drop_duplicates("game_id")
        base = base.merge(rsmall, on="game_id", how="left")

    keep = [
        c
        for c in [
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
            "wx_temp_f",
            "wx_wind_mph",
            "wx_precip_prob_pct",
            "wx_wind_15_plus",
            "wx_precip_50_plus",
            "wx_temp_below_32",
            "home_coach",
            "away_coach",
            "neutral_early_down_pass_rate_last4_home",
            "neutral_pace_sec_per_play_last4_home",
            "neutral_early_down_pass_rate_last4_away",
            "neutral_pace_sec_per_play_last4_away",
            "ref_name",
            "penalties_per100_last16",
            "dpi_per100_pass_last16",
            "off_hold_per100_plays_last16",
        ]
        if c in base.columns
    ]
    ctx = _limit_context_rows(base[keep], n=8).copy()

    if "home_full" not in ctx.columns:
        ctx["home_full"] = ctx["home_code"].map(TEAM_CODE_TO_FULL)
    if "away_full" not in ctx.columns:
        ctx["away_full"] = ctx["away_code"].map(TEAM_CODE_TO_FULL)

    games = json.loads(ctx.to_json(orient="records", date_format="iso"))
    return {"question": question, "games": games}


def ask_ai(bundle: dict, question: str, model: Optional[str] = None, max_output_tokens: int = 800) -> str:
    mdl = model or os.getenv("OPENAI_MODEL", "gpt-5")
    context = build_qa_context(bundle, question)
    system = (
        "You are an NFL data assistant. Use ONLY the JSON context provided. "
        "If data is missing, say so briefly. Keep answers concise and actionable."
    )
    resp = client.responses.create(
        model=mdl,
        reasoning={"effort": "low"},
        instructions=system,
        input=json.dumps(context, ensure_ascii=False),
        max_output_tokens=max_output_tokens,
    )
    return (resp.output_text or "").strip()


__all__ = [
    "build_game_packets",
    "llm_picks_via_responses",
    "build_qa_context",
    "ask_ai",
]

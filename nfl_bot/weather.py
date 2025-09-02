"""Open-Meteo integration helpers."""
from __future__ import annotations

from typing import Optional
import logging

import pandas as pd
import requests

from . import STADIUM_COORDS
from .utils import http


def om_forecast_near_kickoff(
    lat: float,
    lon: float,
    kickoff_ts: pd.Timestamp,
    variables: str = "temperature_2m,wind_speed_10m,precipitation_probability",
    timeout: int = 30,
) -> Optional[dict]:
    """Return weather data near the kickoff timestamp from Open-Meteo."""
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": variables,
        "forecast_days": 16,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "auto",
    }
    try:
        r = http.get(base, params=params, timeout=timeout)
    except requests.RequestException as exc:
        logging.error("Open-Meteo request failed: %s", exc)
        raise RuntimeError("Failed to fetch Open-Meteo forecast") from exc
    js = r.json() or {}
    hourly = js.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return None
    target = pd.to_datetime(kickoff_ts)
    tseries = pd.to_datetime(pd.Series(times))
    idx = int((tseries - target).abs().idxmin())

    def _get(field, default=None):
        arr = hourly.get(field) or []
        return arr[idx] if idx < len(arr) else default

    return {
        "lat": lat,
        "lon": lon,
        "time_near_kickoff": times[idx] if idx < len(times) else None,
        "temp_f": _get("temperature_2m"),
        "wind_mph": _get("wind_speed_10m"),
        "precip_prob_pct": _get("precipitation_probability"),
        "raw_hourly_sample": {
            k: (v[idx] if isinstance(v, list) and idx < len(v) else None)
            for k, v in hourly.items()
            if isinstance(v, list)
        },
    }


def open_meteo_for_game(home_code: str, kickoff_ts: Optional[pd.Timestamp]) -> Optional[dict]:
    """Convenience wrapper resolving stadium coordinates for a game."""
    if home_code not in STADIUM_COORDS or kickoff_ts is None or pd.isna(kickoff_ts):
        return None
    lat, lon = STADIUM_COORDS[home_code]
    return om_forecast_near_kickoff(lat, lon, kickoff_ts)


def weather_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple boolean weather flags to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``temp_f``, ``wind_mph`` and ``precip_prob_pct``.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with boolean columns:

        * ``wind_15_plus`` – wind speed at or above 15 mph
        * ``precip_50_plus`` – precipitation probability at or above 50%
        * ``temp_below_32`` – temperature below freezing (32°F)
    """

    out = df.copy()
    if "wind_mph" in out.columns:
        out["wind_15_plus"] = out["wind_mph"].ge(15)
    else:
        out["wind_15_plus"] = pd.NA

    if "precip_prob_pct" in out.columns:
        out["precip_50_plus"] = out["precip_prob_pct"].ge(50)
    else:
        out["precip_50_plus"] = pd.NA

    if "temp_f" in out.columns:
        out["temp_below_32"] = out["temp_f"].lt(32)
    else:
        out["temp_below_32"] = pd.NA

    return out


__all__ = ["om_forecast_near_kickoff", "open_meteo_for_game", "weather_flags"]

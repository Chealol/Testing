import pandas as pd

from nfl_bot.weather import om_forecast_near_kickoff


class DummyResp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def test_om_forecast_near_kickoff_selects_nearest_hour(sample_weather_json, monkeypatch):
    def fake_get(url, params=None, timeout=0):
        return DummyResp(sample_weather_json)

    monkeypatch.setattr("nfl_bot.weather.http.get", fake_get)
    kickoff = pd.Timestamp("2024-09-07T01:20:00")
    res = om_forecast_near_kickoff(0, 0, kickoff)
    assert res["time_near_kickoff"] == "2024-09-07T01:00"
    assert res["temp_f"] == 71
    assert res["wind_mph"] == 6
    assert res["precip_prob_pct"] == 10

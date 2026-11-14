"""
recommender.py - Part 3, Task 5: travel-timing recommendation system.

The dataset covers a single corridor (I-94 westbound), so the system
recommends WHEN to travel rather than which route.

How a recommendation is made
----------------------------
1. Scenario: day (weekday / weekend / holiday / a named day), expected
   weather, temperature, month, the hours the traveller can travel in, and
   the journey length in hours.
2. Model outputs: for every candidate hour the champion regression model
   (Task 1) predicts traffic volume under exactly those conditions, and the
   champion classifier gives the proxy risk probability (bad weather combined
   with heavy traffic).
3. History: the share of comparable past hours (same day and hour) that were
   congested (> 5,500 vehicles), from the Part 2 feature table.
4. Every possible travel window is scored on predicted volume, with penalties
   for historical congestion and proxy risk. The best window, two
   non-overlapping alternatives and the window to avoid are returned with a
   plain-language recommendation.

Command line (from part3_machine_learning):
    python recommendation_system/recommender.py --day weekday
    python recommendation_system/recommender.py --day Friday --weather Snow --from 14 --to 20
    python recommendation_system/recommender.py --day weekend --weather Rain --duration 2
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PART3_DIR = Path(__file__).resolve().parent.parent
for _p in (PART3_DIR, PART3_DIR / "notebooks"):      # shared module common.py lives in notebooks/
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import common  # noqa: E402
from common import FEATURES, MODELS_DIR, WEATHER_TYPES  # noqa: E402

logger = logging.getLogger(__name__)

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEATHER_SEVERITY = {"Clear": 0, "Clouds": 1, "Smoke": 2, "Haze": 3, "Mist": 4, "Fog": 5,
                    "Drizzle": 6, "Rain": 7, "Snow": 8, "Squall": 9, "Thunderstorm": 10}
PRECIPITATION = {"Rain", "Drizzle", "Snow", "Thunderstorm", "Squall"}
LOW_VISIBILITY = {"Mist", "Fog", "Haze", "Smoke"}
AM_PEAK, PM_PEAK = (6, 7, 8), (15, 16, 17)
CONGESTION_PENALTY = 0.5     # +50% score for an hour that was always congested historically
RISK_PENALTY = 1500          # vehicles-equivalent penalty for proxy risk probability 1.0
MAX_ALTERNATIVE_CONGESTION = 0.30   # never suggest an alternative that was often congested


class RecommendationError(ValueError):
    """Invalid recommendation request (unknown day, weather, impossible hours...)."""


@dataclass
class Request:
    day: str = "weekday"              # weekday | weekend | holiday | Monday..Sunday
    weather: str = "Clear"
    temp_c: float | None = None       # default: typical temperature for the month
    month: int = 6
    earliest: int = 6
    latest: int = 21
    duration: int = 1                 # journey length in hours


@dataclass
class Recommendation:
    request: Request
    best_window: tuple[int, int]
    alternatives: list[tuple[int, int]]
    avoid_window: tuple[int, int]
    hourly: pd.DataFrame = field(repr=False)
    message: str = ""

    def to_dict(self) -> dict:
        return {"request": self.request.__dict__, "recommendation": self.message,
                "best_window": f"{self.best_window[0]:02d}:00-{self.best_window[1]:02d}:00",
                "alternatives": [f"{a:02d}:00-{b:02d}:00" for a, b in self.alternatives],
                "avoid_window": f"{self.avoid_window[0]:02d}:00-{self.avoid_window[1]:02d}:00",
                "hourly": [{"hour": int(h), "predicted_volume": round(float(r["pred_volume"])),
                            "historical_congested_share": round(float(r["hist_congested"]), 3),
                            "proxy_risk_probability": round(float(r["risk_prob"]), 3)}
                           for h, r in self.hourly.iterrows()]}


class TravelRecommender:
    def __init__(self, models_dir: Path = MODELS_DIR, history: pd.DataFrame | None = None):
        self.regressor = joblib.load(models_dir / "traffic_regressor_champion.joblib")
        self.classifier = joblib.load(models_dir / "risk_classifier_champion.joblib")
        data = history if history is not None else common.load_modelling_data()
        data = data.assign(day_type=np.where(data["is_holiday"] == 1, "holiday",
                                             np.where(data["is_weekend"] == 1, "weekend", "weekday")))
        agg = dict(hist_mean=("traffic_volume", "mean"), hist_congested=("is_congested", "mean"),
                   hist_hours=("traffic_volume", "size"))
        self.history = data.groupby(["day_type", "hour"]).agg(**agg).reset_index()
        self.history_by_dow = data.groupby(["day_of_week", "hour"]).agg(**agg).reset_index()
        self.month_temp = data.groupby("month")["temp_c"].median().to_dict()
        self.weather_clouds = data.groupby("weather_main")["clouds_all"].median().to_dict()
        self.weather_rain = data.groupby("weather_main")["rain_1h"].median().to_dict()
        self.quartiles = data["traffic_volume"].quantile([0.25, 0.5, 0.75]).round(1).tolist()
        logger.info("Recommender ready: champion models loaded, history from %d hours", len(data))

    # ------------------------------------------------------------ validation
    def validate(self, req: Request) -> Request:
        day = req.day.strip().lower()
        named = [d for d in DAYS if d.lower().startswith(day[:3])] if len(day) >= 3 else []
        if day in ("weekday", "weekend", "holiday"):
            req.day = day
        elif named:
            req.day = named[0]
        else:
            raise RecommendationError(f"Unknown day '{req.day}'. Use weekday, weekend, holiday "
                                      f"or a day name ({', '.join(DAYS)})")
        match = [w for w in WEATHER_TYPES if w.lower() == req.weather.strip().lower()]
        if not match:
            raise RecommendationError(f"Unknown weather '{req.weather}'. Choose from: "
                                      f"{', '.join(WEATHER_TYPES)}")
        req.weather = match[0]
        if not 1 <= req.month <= 12:
            raise RecommendationError(f"Month must be 1-12 (got {req.month})")
        if not (0 <= req.earliest <= 23 and 0 <= req.latest <= 23):
            raise RecommendationError("Hours must be between 0 and 23")
        if not 1 <= req.duration <= 6:
            raise RecommendationError("Journey duration must be between 1 and 6 hours")
        if req.latest - req.earliest + 1 < req.duration:
            raise RecommendationError(f"The window {req.earliest:02d}:00-{req.latest:02d}:59 is "
                                      f"shorter than a {req.duration}-hour journey")
        if req.temp_c is None:
            req.temp_c = round(float(self.month_temp[req.month]), 1)
        elif not -40 <= req.temp_c <= 45:
            raise RecommendationError(f"Temperature {req.temp_c} °C is outside the plausible range")
        return req

    # ------------------------------------------------------------ model inputs
    def scenario_rows(self, req: Request) -> pd.DataFrame:
        if req.day == "weekday":
            dows, holiday = [0, 1, 2, 3, 4], 0
        elif req.day == "weekend":
            dows, holiday = [5, 6], 0
        elif req.day == "holiday":
            dows, holiday = [0], 1               # most US federal holidays are observed on Mondays
        else:
            dows, holiday = [DAYS.index(req.day)], 0
        rows = []
        for dow in dows:                         # loop over every day and hour in the scenario
            for hour in range(req.earliest, req.latest + 1):
                weekend = int(dow >= 5)
                row = {
                    "hour": hour, "day_of_week": dow, "month": req.month, "is_weekend": weekend,
                    "is_peak_hour": int(hour in AM_PEAK + PM_PEAK and not weekend and not holiday),
                    "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
                    "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
                    "month_sin": np.sin(2 * np.pi * (req.month - 1) / 12),
                    "month_cos": np.cos(2 * np.pi * (req.month - 1) / 12),
                    "is_holiday": holiday, "temp_c": req.temp_c,
                    "rain_1h": float(self.weather_rain.get(req.weather, 0.0)), "snow_1h": 0.0,
                    "clouds_all": float(self.weather_clouds.get(req.weather, 40)),
                    "weather_severity": WEATHER_SEVERITY[req.weather],
                    "is_precipitation": int(req.weather in PRECIPITATION),
                    "is_low_visibility": int(req.weather in LOW_VISIBILITY),
                    "is_severe_weather": int(req.weather in common.SEVERE_WEATHER),
                }
                for w in WEATHER_TYPES:
                    row[f"weather_{w.lower()}"] = int(w == req.weather)
                rows.append(row)
        return pd.DataFrame(rows)[FEATURES]

    def feature_row(self, when: pd.Timestamp, weather: str, temp_c: float, is_holiday: int = 0,
                    rain_1h: float | None = None, clouds_all: float | None = None) -> pd.DataFrame:
        """Model features for one specific hour (used by the deployment API)."""
        req = self.validate(Request(day=DAYS[when.dayofweek], weather=weather, temp_c=temp_c,
                                    month=when.month, earliest=when.hour, latest=when.hour))
        row = self.scenario_rows(req)
        row["is_holiday"] = int(is_holiday)
        if is_holiday:
            row["is_peak_hour"] = 0
        if rain_1h is not None:
            row["rain_1h"] = float(rain_1h)
        if clouds_all is not None:
            row["clouds_all"] = float(clouds_all)
        return row

    def congestion_category(self, volume: float) -> str:
        q1, q2, q3 = self.quartiles
        return "Low" if volume <= q1 else "Medium" if volume <= q2 else "High" if volume <= q3 else "Severe"

    def _history(self, req: Request) -> pd.DataFrame:
        if req.day in DAYS:
            h = self.history_by_dow[self.history_by_dow["day_of_week"] == DAYS.index(req.day)]
        else:
            h = self.history[self.history["day_type"] == req.day]
        return h.set_index("hour")[["hist_mean", "hist_congested", "hist_hours"]]

    # ------------------------------------------------------------ recommendation
    def recommend(self, req: Request) -> Recommendation:
        req = self.validate(req)
        X = self.scenario_rows(req)
        X = X.assign(pred_volume=self.regressor.predict(X),
                     risk_prob=self.classifier.predict_proba(X[FEATURES])[:, 1])
        hourly = X.groupby("hour")[["pred_volume", "risk_prob"]].mean().join(self._history(req))
        hourly["hist_congested"] = hourly["hist_congested"].fillna(0.0)
        hourly["score"] = (hourly["pred_volume"] * (1 + CONGESTION_PENALTY * hourly["hist_congested"])
                           + RISK_PENALTY * hourly["risk_prob"])
        logger.debug("Hourly scores: %s", hourly["score"].round(0).to_dict())

        starts = range(req.earliest, req.latest - req.duration + 2)
        window_score = {s: hourly.loc[s:s + req.duration - 1, "score"].mean() for s in starts}
        ranked = sorted(window_score, key=window_score.get)
        best, alternatives = ranked[0], []
        window_congestion = {s: hourly.loc[s:s + req.duration - 1, "hist_congested"].mean() for s in starts}
        for s in ranked[1:]:          # alternatives: not overlapping, not historically congested
            if (window_congestion[s] < MAX_ALTERNATIVE_CONGESTION
                    and all(abs(s - t) >= max(req.duration, 2) for t in [best] + alternatives)):
                alternatives.append(s)
            if len(alternatives) == 2:
                break
        window = lambda s: (s, s + req.duration)
        rec = Recommendation(req, window(best), [window(a) for a in alternatives],
                             window(ranked[-1]), hourly)
        rec.message = self.message(rec)
        logger.info("Recommendation (%s, %s, %.1f°C, month %d, %02d-%02dh, %dh trip): best %02d:00, avoid %02d:00",
                    req.day, req.weather, req.temp_c, req.month, req.earliest, req.latest,
                    req.duration, best, ranked[-1])
        return rec

    @staticmethod
    def message(rec: Recommendation) -> str:
        req, h = rec.request, rec.hourly
        fmt = lambda w: f"{w[0]:02d}:00 and {w[1]:02d}:00"
        span = lambda w: h.loc[w[0]:w[1] - 1]
        best, worst = span(rec.best_window), span(rec.avoid_window)
        reduction = 1 - best["pred_volume"].mean() / worst["pred_volume"].mean()
        day_text = {"weekday": "a weekday", "weekend": "a weekend", "holiday": "a public holiday"}.get(
            req.day, f"a {req.day}")
        weather_text = "" if req.weather in ("Clear", "Clouds") else f" in {req.weather.lower()}"
        trip = "journey" if req.duration == 1 else f"{req.duration}-hour journey"
        msg = (f"For {day_text} {trip}{weather_text}, consider travelling between {fmt(rec.best_window)}, "
               f"when traffic is expected to be around {best['pred_volume'].mean():,.0f} vehicles per hour "
               f"and {best['hist_congested'].mean():.0%} of comparable past hours were congested.")
        if rec.alternatives:
            msg += " Good alternatives: " + "; ".join(fmt(a) for a in rec.alternatives) + "."
        if worst["hist_congested"].mean() >= 0.2 or reduction >= 0.2:
            msg += (f" Avoid {fmt(rec.avoid_window)}: traffic is expected to reach about "
                    f"{worst['pred_volume'].mean():,.0f} vehicles per hour and "
                    f"{worst['hist_congested'].mean():.0%} of comparable past hours were congested, "
                    f"so your recommended window carries about {reduction:.0%} less traffic.")
        if req.weather in common.SEVERE_WEATHER or req.weather in LOW_VISIBILITY:
            msg += (f" {req.weather} reduces visibility or grip, so allow extra time and stay clear "
                    f"of the busiest hours, where the proxy risk is highest.")
        return msg


# ----------------------------------------------------------------- sample output
SAMPLE_SCENARIOS = [
    ("Weekday commute, clear weather (default window 06:00-21:00)", Request(day="weekday")),
    ("Weekday, office hours only (07:00-18:00)", Request(day="weekday", earliest=7, latest=18)),
    ("Friday afternoon in January snow (14:00-20:00)",
     Request(day="Friday", weather="Snow", month=1, earliest=14, latest=20)),
    ("Weekend 2-hour trip in rain", Request(day="weekend", weather="Rain", duration=2)),
    ("Monday morning in fog (05:00-11:00)", Request(day="Monday", weather="Fog", month=11, earliest=5, latest=11)),
    ("Public holiday trip", Request(day="holiday")),
]


def write_samples(path: Path = Path(__file__).resolve().parent / "sample_recommendations.md") -> None:
    engine = TravelRecommender()
    lines = ["# Sample recommendations", "",
             "Generated by `python recommendation_system/recommender.py --write-samples`. Each answer combines "
             "the champion traffic model's prediction for the stated conditions, the historical congestion rate "
             "for comparable hours and the proxy risk model.", ""]
    for title, req in SAMPLE_SCENARIOS:
        rec = engine.recommend(req)
        lines += [f"## {title}", "", f"> {rec.message}", "",
                  "| Hour | Predicted vehicles | Past % congested | Proxy risk |", "|---|---|---|---|"]
        for hour, r in rec.hourly.iterrows():
            tag = " **(best)**" if rec.best_window[0] <= hour < rec.best_window[1] else (
                " *(avoid)*" if rec.avoid_window[0] <= hour < rec.avoid_window[1] else "")
            lines.append(f"| {hour:02d}:00{tag} | {r['pred_volume']:,.0f} | {r['hist_congested']:.0%} | "
                         f"{r['risk_prob']:.2f} |")
        lines.append("")
    path.write_text("\n".join(lines))
    logger.info("Saved %d sample recommendations to %s", len(SAMPLE_SCENARIOS), path.name)


# ----------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Recommend lower-traffic travel windows on I-94 westbound.")
    p.add_argument("--day", default="weekday", help="weekday, weekend, holiday or a day name")
    p.add_argument("--weather", default="Clear", help=f"one of: {', '.join(WEATHER_TYPES)}")
    p.add_argument("--temp", type=float, default=None, help="temperature in °C (default: typical for month)")
    p.add_argument("--month", type=int, default=6, help="month 1-12 (default 6)")
    p.add_argument("--from", dest="earliest", type=int, default=6, help="earliest hour (default 6)")
    p.add_argument("--to", dest="latest", type=int, default=21, help="latest hour (default 21)")
    p.add_argument("--duration", type=int, default=1, help="journey length in hours, 1-6 (default 1)")
    p.add_argument("--write-samples", action="store_true", help="write sample_recommendations.md and exit")
    args = p.parse_args(argv)
    if args.write_samples:
        write_samples()
        return 0
    logger.info("Recommendation requested with arguments: %s", vars(args))
    try:
        rec = TravelRecommender().recommend(Request(args.day, args.weather, args.temp, args.month,
                                                    args.earliest, args.latest, args.duration))
    except RecommendationError as exc:
        logger.error("Invalid recommendation request: %s", exc)
        return 2
    except FileNotFoundError as exc:
        logger.error("Required model or data file missing (run train_supervised.py first): %s", exc)
        return 1
    print(rec.message)                       # answer for the end user
    print()
    print(f"{'Hour':<7}{'Predicted vehicles':>19}{'Past % congested':>18}{'Proxy risk':>12}")
    for hour, r in rec.hourly.iterrows():
        mark = "  <- best" if rec.best_window[0] <= hour < rec.best_window[1] else (
            "  <- avoid" if rec.avoid_window[0] <= hour < rec.avoid_window[1] else "")
        print(f"{hour:02d}:00{r['pred_volume']:>21,.0f}{r['hist_congested']:>17.0%}"
              f"{r['risk_prob']:>12.2f}{mark}")
    return 0


if __name__ == "__main__":
    common.configure_logging()
    logging.getLogger().handlers[0].setLevel(logging.ERROR)   # console: answers + errors only
    sys.exit(main())

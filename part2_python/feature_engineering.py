"""
feature_engineering.py - Part 2, Task 2: ML-ready features with NumPy and pandas.

Called by pipeline.py on the cleaned data (one row per hour). This module only
creates a logger with logging.getLogger(__name__); handlers are configured by
the entry point.

Feature groups
--------------
Time      : hour, day_of_week, day_name, month, year, is_weekend, is_holiday,
            is_peak_hour, time_of_day, and cyclical sin/cos encodings of
            hour, day of week and month.
Weather   : temp_c, one-hot weather_* columns, is_precipitation,
            is_low_visibility, is_severe_weather, weather_severity, temp_band.
Numerical : temp_c_zscore, clouds_minmax, rain_log_minmax (scaled copies).
Target    : congestion_category (data-driven quartiles of traffic_volume:
            Low / Medium / High / Severe) plus is_congested (> 5,500, the
            Part 1 definition) for consistency with the earlier analysis.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Peak hours come from Part 1: on weekdays these hours average above (or
# within 2% of) the 5,500-vehicle congestion threshold.
AM_PEAK_HOURS = (6, 7, 8)
PM_PEAK_HOURS = (15, 16, 17)
PART1_CONGESTION_THRESHOLD = 5500

PRECIPITATION_WEATHER = {"Rain", "Drizzle", "Snow", "Thunderstorm", "Squall"}
LOW_VISIBILITY_WEATHER = {"Mist", "Fog", "Haze", "Smoke"}
SEVERE_WEATHER = {"Thunderstorm", "Squall", "Snow"}
WEATHER_SEVERITY = {
    "Clear": 0, "Clouds": 1, "Smoke": 2, "Haze": 3, "Mist": 4, "Fog": 5,
    "Drizzle": 6, "Rain": 7, "Snow": 8, "Squall": 9, "Thunderstorm": 10,
}
CONGESTION_LABELS = ("Low", "Medium", "High", "Severe")


# ----------------------------------------------------------------- helpers
def cyclical_encode(values: pd.Series, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Map a repeating value onto a circle so that e.g. hour 23 sits next to hour 0."""
    angle = 2 * np.pi * values.to_numpy(dtype=float) / period
    return np.sin(angle), np.cos(angle)


def zscore(values: pd.Series) -> pd.Series:
    mean, std = values.mean(), values.std(ddof=0)
    logger.debug("Scaling '%s' by z-score: mean=%.4f, std=%.4f", values.name, mean, std)
    return (values - mean) / std


def minmax(values: pd.Series, name: str) -> pd.Series:
    low, high = values.min(), values.max()
    logger.debug("Scaling '%s' to 0-1: min=%.4f, max=%.4f", name, low, high)
    span = high - low
    return (values - low) / span if span else values * 0.0


# ----------------------------------------------------------------- feature groups
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["date_time"]
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek                 # 0 = Monday ... 6 = Sunday
    df["day_name"] = dt.dt.day_name()
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df["date"] = dt.dt.normalize()
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # The raw holiday label only appears on the midnight record of each
    # holiday, so spread it to every hour of that date.
    holiday_rows = df.loc[df["holiday"] != "None", ["date", "holiday"]]
    holiday_map = dict(zip(holiday_rows["date"], holiday_rows["holiday"]))
    logger.debug("Holiday propagation: %d holiday dates found in midnight labels",
                 len(holiday_map))
    df["holiday_name"] = df["date"].map(holiday_map).fillna("None")
    df["is_holiday"] = (df["holiday_name"] != "None").astype(int)

    peak = df["hour"].isin(AM_PEAK_HOURS + PM_PEAK_HOURS)
    df["is_peak_hour"] = (peak & (df["is_weekend"] == 0) & (df["is_holiday"] == 0)).astype(int)
    df["time_of_day"] = pd.cut(df["hour"], bins=[-1, 5, 11, 17, 23],
                               labels=["Night", "Morning", "Afternoon", "Evening"]).astype(str)

    df["hour_sin"], df["hour_cos"] = cyclical_encode(df["hour"], 24)
    df["dow_sin"], df["dow_cos"] = cyclical_encode(df["day_of_week"], 7)
    df["month_sin"], df["month_cos"] = cyclical_encode(df["month"] - 1, 12)
    logger.info("Time features added: hour, day_of_week, is_weekend, is_holiday, "
                "is_peak_hour, time_of_day and sin/cos encodings for hour, weekday, month")
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df["temp_c"] = (df["temp"] - 273.15).round(2)

    dummies = pd.get_dummies(df["weather_main"], prefix="weather", dtype=int)
    dummies.columns = [c.lower() for c in dummies.columns]
    logger.debug("One-hot encoding weather_main into %d columns: %s",
                 dummies.shape[1], list(dummies.columns))
    df = pd.concat([df, dummies], axis=1)

    df["is_precipitation"] = (df["weather_main"].isin(PRECIPITATION_WEATHER)
                              | (df["rain_1h"] > 0) | (df["snow_1h"] > 0)).astype(int)
    df["is_low_visibility"] = df["weather_main"].isin(LOW_VISIBILITY_WEATHER).astype(int)
    df["is_severe_weather"] = df["weather_main"].isin(SEVERE_WEATHER).astype(int)
    df["weather_severity"] = df["weather_main"].map(WEATHER_SEVERITY).fillna(0).astype(int)
    df["temp_band"] = pd.cut(df["temp_c"], bins=[-np.inf, 0, 10, 20, np.inf],
                             labels=["Freezing", "Cold", "Mild", "Warm"]).astype(str)
    logger.info("Weather features added: temp_c, %d one-hot weather columns, "
                "is_precipitation, is_low_visibility, is_severe_weather, "
                "weather_severity, temp_band", dummies.shape[1])
    return df


def add_scaled_features(df: pd.DataFrame) -> pd.DataFrame:
    df["temp_c_zscore"] = zscore(df["temp_c"]).round(4)
    df["clouds_minmax"] = minmax(df["clouds_all"], "clouds_all").round(4)
    # Rain is extremely skewed (mostly 0), so log-transform before scaling.
    df["rain_log_minmax"] = minmax(np.log1p(df["rain_1h"]), "log1p(rain_1h)").round(4)
    logger.info("Scaled features added: temp_c_zscore, clouds_minmax, rain_log_minmax "
                "(Part 3 refits scalers on training data only)")
    return df


def add_congestion_target(df: pd.DataFrame) -> pd.DataFrame:
    """Data-driven congestion category from the quartiles of traffic_volume.

    Low    : volume <= Q1
    Medium : Q1 < volume <= Q2 (median)
    High   : Q2 < volume <= Q3
    Severe : volume > Q3
    Quartiles are computed on normal hours only, so flagged sensor faults do
    not distort the thresholds.
    """
    normal = df.loc[df["is_sensor_anomaly"] == 0, "traffic_volume"]
    q1, q2, q3 = normal.quantile([0.25, 0.5, 0.75]).to_numpy()
    logger.debug("Congestion quartile thresholds: Q1=%.1f, Q2=%.1f, Q3=%.1f", q1, q2, q3)

    volume = df["traffic_volume"]
    df["congestion_category"] = np.select(
        [volume <= q1, volume <= q2, volume <= q3], CONGESTION_LABELS[:3],
        default=CONGESTION_LABELS[3])
    df["congestion_level"] = df["congestion_category"].map(
        {label: i for i, label in enumerate(CONGESTION_LABELS)}).astype(int)
    df["is_congested"] = (volume > PART1_CONGESTION_THRESHOLD).astype(int)

    counts = df["congestion_category"].value_counts().reindex(CONGESTION_LABELS)
    logger.debug("Congestion category counts: %s", counts.to_dict())
    logger.info("Target added: congestion_category (quartile-based, 4 classes) and "
                "is_congested (> %d, Part 1 definition: %.1f%% of hours)",
                PART1_CONGESTION_THRESHOLD, 100 * df["is_congested"].mean())
    return df


# ----------------------------------------------------------------- public API
def build_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with all engineered features."""
    logger.info("Feature engineering started: input shape %d rows x %d columns",
                *cleaned.shape)
    df = cleaned.copy()
    for step in (add_time_features, add_weather_features,
                 add_scaled_features, add_congestion_target):
        df = step(df)
    logger.info("Feature engineering complete: output shape %d rows x %d columns "
                "(%d columns added)", *df.shape, df.shape[1] - cleaned.shape[1])
    return df

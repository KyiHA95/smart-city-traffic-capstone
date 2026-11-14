"""
common.py - shared building blocks for every Part 3 script.

Contents
--------
* Paths for data, models, MLflow, results and logs.
* load_modelling_data(): reads the Part 2 feature table, drops flagged sensor
  faults, rebuilds the quartile congestion_category exactly as the brief
  specifies and adds the proxy accident-risk label `high_risk`.
* FEATURES: the common, well-engineered feature set used by BOTH supervised
  models (time features, weather encodings, holiday flag, cyclical encodings
  of hour and day of week).
* time_split(): chronological train/test split (train 2012-2016, test 2017-2018).
* configure_logging(): console + file handlers. It is called ONLY from the
  `if __name__ == "__main__"` block of each entry-point script; library code
  never adds handlers and every module uses logging.getLogger(__name__).
  All scripts append to one shared log, notebooks/part3.log.
* setup_mlflow(): points MLflow at the project's local tracking database.

IMPORTANT: `high_risk` is a documented PROXY built from congestion and
weather. It is not an accident record and must not be read as a prediction
of real accidents.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------- paths
NOTEBOOKS_DIR = Path(__file__).resolve().parent
PART3_DIR = NOTEBOOKS_DIR.parent
REPO_DIR = PART3_DIR.parent
FEATURE_FILE = REPO_DIR / "part2_python" / "output" / "traffic_features.csv.gz"
MODELS_DIR = PART3_DIR / "models"
RESULTS_DIR = NOTEBOOKS_DIR / "outputs"       # metrics (JSON/CSV) and figures
FIGURES_DIR = RESULTS_DIR
LOG_FILE = NOTEBOOKS_DIR / "part3.log"        # one shared log for every script
MLFLOW_DIR = PART3_DIR / "mlflow"
MLFLOW_DB = MLFLOW_DIR / "mlflow.db"
# Model binaries logged to MLflow are duplicates of models/*.joblib, so this
# folder is regenerated locally and excluded from Git (see .gitignore).
MLFLOW_ARTIFACTS = MLFLOW_DIR / "artifacts"

for _folder in (MODELS_DIR, RESULTS_DIR, MLFLOW_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TRAIN_END_YEAR = 2016          # train: 2012-2016, test: 2017-2018 (future data)

# ----------------------------------------------------------------- proxy label definition
SEVERE_WEATHER = {"Thunderstorm", "Squall", "Snow"}
CONGESTION_LABELS = ["Low", "Medium", "High", "Severe"]

# ----------------------------------------------------------------- common feature set
TIME_FEATURES = ["hour", "day_of_week", "month", "is_weekend", "is_peak_hour"]
CYCLICAL_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
HOLIDAY_FEATURES = ["is_holiday"]
WEATHER_NUMERIC = ["temp_c", "rain_1h", "snow_1h", "clouds_all", "weather_severity",
                   "is_precipitation", "is_low_visibility", "is_severe_weather"]
WEATHER_TYPES = ["Clear", "Clouds", "Drizzle", "Fog", "Haze", "Mist", "Rain", "Smoke",
                 "Snow", "Squall", "Thunderstorm"]
WEATHER_ONE_HOT = [f"weather_{w.lower()}" for w in WEATHER_TYPES]
FEATURES = (TIME_FEATURES + CYCLICAL_FEATURES + HOLIDAY_FEATURES
            + WEATHER_NUMERIC + WEATHER_ONE_HOT)

REGRESSION_TARGET = "traffic_volume"
CLASSIFICATION_TARGET = "high_risk"


# ----------------------------------------------------------------- logging (entry points only)
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(module)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> Path:
    """Attach console and file handlers. Call only from an entry-point script."""
    log_path = LOG_FILE
    formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)
    for noisy in ("matplotlib", "PIL", "mlflow", "alembic", "urllib3", "shap",
                  "numba", "jax", "absl", "git", "httpx", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_path


# ----------------------------------------------------------------- data
def add_proxy_label(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild congestion_category from quartiles and add the high_risk proxy (brief's logic)."""
    q1, q2, q3 = df["traffic_volume"].quantile([0.25, 0.5, 0.75]).to_numpy()
    logger.debug("Congestion quartiles: Q1=%.1f Q2=%.1f Q3=%.1f", q1, q2, q3)

    def bucket(v: float) -> str:
        if v <= q1:
            return "Low"
        elif v <= q2:
            return "Medium"
        elif v <= q3:
            return "High"
        return "Severe"

    df["congestion_category"] = df["traffic_volume"].apply(bucket)
    high_congestion = df["congestion_category"].isin(["High", "Severe"])
    risky_weather = (df["weather_main"].isin(SEVERE_WEATHER)
                     | (df["is_low_visibility"] == 1))
    df["high_risk"] = (high_congestion & risky_weather).astype(int)
    logger.info("Proxy label built: high_risk = High/Severe congestion AND severe or "
                "low-visibility weather -> %d positive hours (%.1f%%)",
                int(df["high_risk"].sum()), 100 * df["high_risk"].mean())
    return df


def load_modelling_data(path: Path = FEATURE_FILE) -> pd.DataFrame:
    """Load the Part 2 feature table and prepare it for modelling."""
    try:
        df = pd.read_csv(path, parse_dates=["date_time", "date"])
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Part 2 feature table not found at {path}. Run part2_python/pipeline.py first."
        ) from exc
    logger.info("Loaded Part 2 feature table: %d rows x %d columns", *df.shape)

    faults = df["is_sensor_anomaly"] == 1
    if faults.any():
        logger.warning("Dropped %d hour(s) flagged as sensor faults in Part 2 "
                       "(volume is not a real traffic reading)", int(faults.sum()))
    df = df.loc[~faults].copy()

    for column in WEATHER_ONE_HOT:          # guarantee every one-hot column exists
        if column not in df.columns:
            df[column] = 0
    df = add_proxy_label(df)
    df = df.sort_values("date_time").reset_index(drop=True)
    logger.info("Modelling data ready: %d rows, %d model features", len(df), len(FEATURES))
    return df


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["year"] <= TRAIN_END_YEAR]
    test = df[df["year"] > TRAIN_END_YEAR]
    logger.info("Chronological split: train %d rows (%d-%d), test %d rows (%d-%d)",
                len(train), train["year"].min(), train["year"].max(),
                len(test), test["year"].min(), test["year"].max())
    return train, test


# ----------------------------------------------------------------- MLflow
def setup_mlflow(experiment: str):
    """Use the project-local SQLite tracking store (supports the model registry)."""
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=MLFLOW_ARTIFACTS.as_uri())
    mlflow.set_experiment(experiment)
    logger.info("MLflow tracking at %s, experiment '%s'", MLFLOW_DB.relative_to(REPO_DIR),
                experiment)
    return mlflow


# ----------------------------------------------------------------- small helpers
def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item()
                               if isinstance(o, np.generic) else str(o)))
    logger.info("Saved %s", path.relative_to(REPO_DIR))

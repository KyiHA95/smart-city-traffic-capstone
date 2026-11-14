"""
pipeline.py - Part 2, Task 1: reproducible traffic data pipeline (entry point).

Stages
------
1. Load the raw CSV (with exception handling for file I/O and parsing errors).
2. Validate the schema before any processing.
3. Clean the data, one logged step at a time:
   standardise categories -> coerce numeric types -> parse date/times ->
   remove duplicate rows -> resolve duplicate timestamps -> flag impossible
   temperatures -> flag impossible precipitation -> check cloud cover range ->
   impute flagged values with the monthly median -> validate traffic volume ->
   flag sensor anomalies -> final checks.
4. Engineer ML-ready features            (feature_engineering.py)
5. Create and save the figures            (visualizations.py)

Usage (from the part2_python folder):
    python pipeline.py                 # normal run (INFO and above)
    python pipeline.py --debug         # also show DEBUG intermediate values
    python pipeline.py --input path/to/file.csv --skip-figures

Logging is configured ONLY here (the entry point). Every other module just
calls logging.getLogger(__name__) and never adds handlers.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import WEATHER_SEVERITY, build_features
from visualizations import create_all_figures

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR.parent / "data" / "Metro_Interstate_Traffic_Volume.csv"
OUTPUT_DIR = BASE_DIR / "output"
FIGURE_DIR = BASE_DIR / "figures"
LOG_FILE = BASE_DIR / "pipeline.log"

# ----------------------------------------------------------------- schema
EXPECTED_COLUMNS = {
    "holiday": "text",
    "temp": "numeric",
    "rain_1h": "numeric",
    "snow_1h": "numeric",
    "clouds_all": "numeric",
    "weather_main": "text",
    "weather_description": "text",
    "date_time": "datetime",
    "traffic_volume": "numeric",
}
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# ----------------------------------------------------------------- plausibility rules
# Minneapolis-St Paul record temperatures are about -43 C (230 K) and +42 C
# (315 K); 220-330 K leaves a safety margin while rejecting sensor errors such
# as 0 K (absolute zero).
TEMP_VALID_RANGE_K = (220.0, 330.0)
# The world-record rainfall in one hour is about 305 mm, so anything higher is
# physically impossible (the dataset contains one 9,831.3 mm reading).
PRECIP_MAX_MM_PER_HOUR = 305.0
CLOUDS_VALID_RANGE = (0, 100)
# A busy-period hour counts as a sensor anomaly if its volume is below 5% of
# the typical (median) volume for the same hour and weekday.
ANOMALY_SHARE_OF_TYPICAL = 0.05
ANOMALY_MIN_TYPICAL_VOLUME = 1000
# WEATHER_SEVERITY (imported from feature_engineering) decides which report is
# kept when one hour has several weather rows: the most severe one wins.


# ----------------------------------------------------------------- errors
class PipelineError(Exception):
    """Base class for errors that stop the pipeline."""


class DataLoadError(PipelineError):
    """Raised when the raw file cannot be read or parsed."""


class SchemaValidationError(PipelineError):
    """Raised when expected columns are missing."""


# ----------------------------------------------------------------- logging setup
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(module)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ConsoleFormatter(logging.Formatter):
    """Console format without tracebacks: the full exc_info detail goes to the file."""

    def format(self, record: logging.LogRecord) -> str:
        saved = record.exc_info, record.exc_text
        record.exc_info, record.exc_text = None, None
        try:
            text = super().format(record)
        finally:
            record.exc_info, record.exc_text = saved
        if saved[0] is not None:
            text += f" (full traceback in {LOG_FILE.name})"
        return text


def configure_logging(level: int, log_file: Path) -> None:
    """Send log records to the console and to a log file (entry point only)."""
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(ConsoleFormatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Third-party libraries are noisy at DEBUG; keep them at WARNING.
    for noisy in ("matplotlib", "PIL", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ================================================================= STAGE 1: LOAD
def load_raw_data(path: Path) -> pd.DataFrame:
    """Read the raw CSV, translating I/O and parsing problems into DataLoadError."""
    logger.info("Loading raw data from %s", os.path.relpath(path))
    try:
        # keep_default_na=False stops the holiday text "None" becoming NaN;
        # genuinely empty cells are still treated as missing.
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    except FileNotFoundError as exc:
        raise DataLoadError(f"Raw data file not found: {path}") from exc
    except PermissionError as exc:
        raise DataLoadError(f"No permission to read {path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"Raw data file is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"CSV could not be parsed ({path}): {str(exc).strip()}") from exc
    except UnicodeDecodeError as exc:
        raise DataLoadError(f"File is not valid UTF-8 text: {path}") from exc

    if df.empty:
        raise DataLoadError(f"Raw data file contains a header but no rows: {path}")
    logger.info("Raw file loaded successfully: %d rows x %d columns", *df.shape)
    return df


# ================================================================= STAGE 2: VALIDATE
def validate_schema(df: pd.DataFrame) -> None:
    """Check every expected column is present before any processing happens."""
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"Missing expected column(s): {missing}. Found: {list(df.columns)}")
    extra = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    if extra:
        logger.warning("Schema check: %d unexpected column(s) will be ignored: %s",
                       len(extra), extra)
    logger.info("Schema validation passed: all %d expected columns present",
                len(EXPECTED_COLUMNS))


# ================================================================= STAGE 3: CLEAN
def standardise_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and apply one consistent case per text column."""
    df = df.copy()
    rules = {
        "holiday": lambda s: s.str.strip(),
        "weather_main": lambda s: s.str.strip().str.title(),
        "weather_description": lambda s: s.str.strip().str.lower(),
    }
    for column, rule in rules.items():
        before = df[column].fillna("").astype(str)
        after = rule(before)
        changed = before != after
        n_changed = int(changed.sum())
        if n_changed:
            examples = sorted({f"'{a}' -> '{b}'" for a, b in
                               zip(before[changed], after[changed])})[:3]
            logger.warning(
                "Standardise categories: modified %d value(s) in '%s' to make "
                "case/whitespace consistent (e.g. %s)",
                n_changed, column, ", ".join(examples))
        else:
            logger.info("Standardise categories: '%s' already consistent "
                        "(%d distinct values)", column, after.nunique())
        df[column] = after
    df = df[[c for c in df.columns if c in EXPECTED_COLUMNS]]
    return df


def coerce_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure numeric columns really are numeric; unparseable text becomes NaN."""
    df = df.copy()
    for column, kind in EXPECTED_COLUMNS.items():
        if kind != "numeric":
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        n_bad = int(converted.isna().sum() - df[column].isna().sum())
        if n_bad:
            logger.warning("Numeric types: %d non-numeric value(s) in '%s' set to "
                           "missing (could not be parsed as numbers)", n_bad, column)
        df[column] = converted
    logger.info("Numeric types: temp, rain_1h, snow_1h, clouds_all and "
                "traffic_volume stored as numbers")
    return df


def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date_time with an explicit format and drop rows that fail."""
    df = df.copy()
    parsed = pd.to_datetime(df["date_time"], format=DATETIME_FORMAT, errors="coerce")
    invalid = parsed.isna()
    if invalid.any():
        logger.warning("Date/time parsing: dropped %d row(s) because date_time did "
                       "not match %s", int(invalid.sum()), DATETIME_FORMAT)
    df["date_time"] = parsed
    df = df.loc[~invalid]
    logger.info("Date/time parsing: all remaining timestamps valid, range %s to %s",
                df["date_time"].min(), df["date_time"].max())
    return df


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are exact copies of an earlier row."""
    duplicated = df.duplicated(keep="first")
    n_dup = int(duplicated.sum())
    if n_dup:
        logger.warning("Duplicate rows: dropped %d exact duplicate row(s) "
                       "(identical in every column)", n_dup)
    else:
        logger.info("Duplicate rows: none found")
    return df.loc[~duplicated].copy()


def resolve_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per hour.

    Some hours are recorded several times, once per weather description (for
    example 'mist' and 'light rain' in the same hour). Traffic volume is
    identical in every copy, so keeping all of them would double-count the
    hour. The most severe weather report is kept so that bad-weather hours
    are not lost, and the number of reports is kept as a feature.
    """
    df = df.copy()
    df["n_weather_reports"] = df.groupby("date_time")["date_time"].transform("size")
    df["_severity"] = df["weather_main"].map(WEATHER_SEVERITY).fillna(0)
    df = df.sort_values(["date_time", "_severity"], ascending=[True, False])
    repeated = df.duplicated(subset="date_time", keep="first")
    n_repeated = int(repeated.sum())
    if n_repeated:
        logger.warning(
            "Duplicate timestamps: dropped %d row(s) that repeated an hour already "
            "recorded with a different weather description; kept the most severe "
            "weather report for each of %d affected hours",
            n_repeated, int(df.loc[repeated, "date_time"].nunique()))
    else:
        logger.info("Duplicate timestamps: every hour appears once")
    return df.loc[~repeated].drop(columns="_severity").reset_index(drop=True)


def flag_impossible_temperatures(df: pd.DataFrame) -> pd.DataFrame:
    """Replace physically impossible temperatures with NaN so they can be imputed."""
    df = df.copy()
    low, high = TEMP_VALID_RANGE_K
    bad = ~df["temp"].between(low, high)
    df["temp_was_imputed"] = bad.astype(int)
    if bad.any():
        logger.warning(
            "Impossible temperatures: %d reading(s) outside %.0f-%.0f K (values seen: "
            "%s) set to missing for imputation; 0 K is absolute zero, a sensor error",
            int(bad.sum()), low, high, sorted(df.loc[bad, "temp"].unique().tolist()))
        df.loc[bad, "temp"] = np.nan
    else:
        logger.info("Impossible temperatures: none found")
    return df


def flag_impossible_precipitation(df: pd.DataFrame) -> pd.DataFrame:
    """Replace negative or physically impossible hourly rain/snow totals with NaN."""
    df = df.copy()
    for column in ("rain_1h", "snow_1h"):
        bad = (df[column] < 0) | (df[column] > PRECIP_MAX_MM_PER_HOUR)
        df[f"{column}_was_imputed"] = bad.astype(int)
        if bad.any():
            logger.warning(
                "Impossible precipitation: %d '%s' reading(s) outside 0-%.0f mm/h "
                "(values seen: %s) set to missing for imputation; the world record "
                "for one hour is about 305 mm",
                int(bad.sum()), column, PRECIP_MAX_MM_PER_HOUR,
                sorted(df.loc[bad, column].unique().tolist()))
            df.loc[bad, column] = np.nan
        else:
            logger.info("Impossible precipitation: '%s' within 0-%.0f mm/h",
                        column, PRECIP_MAX_MM_PER_HOUR)
    return df


def check_cloud_cover(df: pd.DataFrame) -> pd.DataFrame:
    """Cloud cover is a percentage; clip anything outside 0-100."""
    df = df.copy()
    low, high = CLOUDS_VALID_RANGE
    bad = ~df["clouds_all"].between(low, high)
    if bad.any():
        logger.warning("Cloud cover: clipped %d value(s) outside %d-%d%%",
                       int(bad.sum()), low, high)
        df["clouds_all"] = df["clouds_all"].clip(low, high)
    else:
        logger.info("Cloud cover: all values within %d-%d%%", low, high)
    return df


def impute_monthly_median(df: pd.DataFrame,
                          columns: tuple[str, ...] = ("temp", "rain_1h", "snow_1h")
                          ) -> pd.DataFrame:
    """Fill missing values with the median of the SAME year-month.

    A single global median would put a summer temperature into a winter month,
    so each month's own median is calculated and applied in an explicit loop.
    If a month has no valid values at all, the median for that calendar month
    across all years is used instead.
    """
    df = df.copy()
    period = df["date_time"].dt.to_period("M")
    calendar_month = df["date_time"].dt.month
    for column in columns:
        missing = df[column].isna()
        if not missing.any():
            logger.info("Monthly median imputation: '%s' has no missing values", column)
            continue
        filled = 0
        for month in sorted(period[missing].unique()):
            in_month = period == month
            month_median = df.loc[in_month, column].median()
            source = f"{month} median"
            if pd.isna(month_median):
                month_median = df.loc[calendar_month == month.month, column].median()
                source = f"all-years month-{month.month} median (no valid {month} data)"
            to_fill = in_month & missing
            df.loc[to_fill, column] = month_median
            filled += int(to_fill.sum())
            logger.debug("Monthly median imputation: '%s' %s = %.3f applied to %d row(s)",
                         column, source, month_median, int(to_fill.sum()))
        logger.warning("Monthly median imputation: imputed %d missing/impossible "
                       "'%s' value(s) using the median of the same month",
                       filled, column)
    return df


def validate_traffic_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose target value is missing or negative (cannot be repaired)."""
    bad = df["traffic_volume"].isna() | (df["traffic_volume"] < 0)
    if bad.any():
        logger.warning("Traffic volume: dropped %d row(s) with missing or negative "
                       "volume (target cannot be imputed)", int(bad.sum()))
    else:
        logger.info("Traffic volume: all values present and non-negative (%d-%d)",
                    int(df["traffic_volume"].min()), int(df["traffic_volume"].max()))
    out = df.loc[~bad].copy()
    out["traffic_volume"] = out["traffic_volume"].astype(int)
    return out


def flag_sensor_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Flag implausibly low counts at normally busy times (likely sensor faults).

    These rows are kept (the volume is the prediction target, so it is not
    imputed) but marked with is_sensor_anomaly = 1 so analyses can exclude them.
    """
    df = df.copy()
    hour = df["date_time"].dt.hour
    weekday = df["date_time"].dt.dayofweek
    typical = df.groupby([hour, weekday])["traffic_volume"].transform("median")
    anomaly = ((typical >= ANOMALY_MIN_TYPICAL_VOLUME)
               & (df["traffic_volume"] < ANOMALY_SHARE_OF_TYPICAL * typical))
    df["is_sensor_anomaly"] = anomaly.astype(int)
    if anomaly.any():
        dates = sorted(df.loc[anomaly, "date_time"].dt.date.astype(str).unique())
        logger.warning(
            "Sensor anomalies: flagged %d hour(s) on %d day(s) with volume below %.0f%% "
            "of the typical value for that hour and weekday; rows kept and marked "
            "is_sensor_anomaly=1", int(anomaly.sum()), len(dates),
            ANOMALY_SHARE_OF_TYPICAL * 100)
        logger.debug("Sensor anomaly dates: %s", ", ".join(dates))
    else:
        logger.info("Sensor anomalies: none detected")
    return df


def final_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Confirm the cleaned table is complete and ordered."""
    df = df.sort_values("date_time").reset_index(drop=True)
    remaining = df[list(EXPECTED_COLUMNS)].isna().sum()
    remaining = remaining[remaining > 0]
    if not remaining.empty:
        raise PipelineError(f"Missing values remain after cleaning: {remaining.to_dict()}")
    if df["date_time"].duplicated().any():
        raise PipelineError("Duplicate timestamps remain after cleaning")
    logger.info("Final checks passed: no missing values, one row per hour, "
                "sorted by date_time")
    return df


CLEANING_STEPS = (
    standardise_categoricals,
    coerce_numeric_types,
    parse_datetimes,
    remove_duplicate_rows,
    resolve_duplicate_timestamps,
    flag_impossible_temperatures,
    flag_impossible_precipitation,
    check_cloud_cover,
    impute_monthly_median,
    validate_traffic_volume,
    flag_sensor_anomalies,
    final_checks,
)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run every cleaning step in order; each step logs its own result."""
    rows_in = len(df)
    for step in CLEANING_STEPS:
        before = len(df)
        df = step(df)
        logger.debug("Step %s: %d -> %d rows", step.__name__, before, len(df))
    logger.info("Cleaning complete: %d rows in, %d rows out (%d removed), %d columns",
                rows_in, len(df), rows_in - len(df), df.shape[1])
    return df


# ================================================================= SAVE
def save_table(df: pd.DataFrame, path: Path, label: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except OSError as exc:
        raise PipelineError(f"Could not write {label} to {path}: {exc}") from exc
    logger.info("Saved %s: %s (%d rows x %d columns)", label, os.path.relpath(path), *df.shape)


# ================================================================= MAIN
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Metro traffic data pipeline")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Path to the raw CSV (default: ../data/...)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Folder for cleaned and feature tables")
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR,
                        help="Folder for saved figures")
    parser.add_argument("--log-file", type=Path, default=LOG_FILE,
                        help="Log file path (default: pipeline.log)")
    parser.add_argument("--debug", action="store_true",
                        help="Log DEBUG-level intermediate values")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Do not create figures")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(logging.DEBUG if args.debug else logging.INFO, args.log_file)
    logger.info("Pipeline started (log level %s)", "DEBUG" if args.debug else "INFO")
    try:
        raw = load_raw_data(args.input)
        validate_schema(raw)
        cleaned = clean_data(raw)
        save_table(cleaned, args.output_dir / "cleaned_traffic.csv.gz", "cleaned data")

        features = build_features(cleaned)
        save_table(features, args.output_dir / "traffic_features.csv.gz", "feature table")

        if args.skip_figures:
            logger.info("Figure creation skipped (--skip-figures)")
        else:
            saved, failed = create_all_figures(features, args.figure_dir)
            if failed:
                raise PipelineError(f"{failed} figure(s) could not be created")
    except PipelineError as exc:
        logger.error("Pipeline stopped: %s", exc, exc_info=True)
        return 1
    except Exception as exc:  # last-resort guard so the run never ends in a raw crash
        logger.error("Pipeline stopped by an unexpected error: %s", exc, exc_info=True)
        return 1

    logger.info("Pipeline finished successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
traffic_cli.py - Part 2, Task 4: mini traffic analytics command-line application.

Queries the processed feature table produced by pipeline.py
(output/traffic_features.csv.gz). Run the pipeline first.

Commands
--------
  summary                         Overview of the processed dataset
  query     --datetime "YYYY-MM-DD HH:MM"
                                  Traffic and weather for a specific hour
  peak      [--top N] [--day-type weekday|weekend|all] [--year YYYY]
                                  Highest-traffic periods (hour of week)
  compare   [--year YYYY]         Weekday vs weekend traffic, by time of day
  recommend --day DAYNAME [--from H] [--to H] [--top N]
                                  Best (least congested) hours to travel
  weather   [--condition NAME]    Traffic under each (or one) weather condition

Examples (from the part2_python folder):
  python cli_app/traffic_cli.py summary
  python cli_app/traffic_cli.py query --datetime "2017-03-14 08:00"
  python cli_app/traffic_cli.py peak --top 5 --day-type weekday
  python cli_app/traffic_cli.py recommend --day Monday --from 6 --to 20

Output design: answers to the user are printed to standard output with
print(); everything about the program's own activity goes through logging.
By default the console shows WARNING and above (so answers stay readable)
and cli_app/cli.log records INFO and above, including every command invoked
and its arguments. Use --verbose to see INFO on screen, --debug for DEBUG.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR.parent / "output" / "traffic_features.csv.gz"
LOG_FILE = APP_DIR / "cli.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(module)s | %(message)s"
DATETIME_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H")
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CONGESTION_THRESHOLD = 5500
EXIT_OK, EXIT_ERROR, EXIT_BAD_INPUT = 0, 1, 2


class InvalidInputError(Exception):
    """User-supplied input that cannot be used (bad date, unknown day, ...)."""


class DataUnavailableError(Exception):
    """The processed data file is missing or unreadable."""


# ----------------------------------------------------------------- logging (entry point)
class ConsoleFormatter(logging.Formatter):
    """Console format without tracebacks; the full detail is kept in cli.log."""

    def format(self, record: logging.LogRecord) -> str:
        saved = record.exc_info, record.exc_text
        record.exc_info, record.exc_text = None, None
        try:
            return super().format(record)
        finally:
            record.exc_info, record.exc_text = saved


def configure_logging(console_level: int) -> None:
    formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level)
    console.setFormatter(ConsoleFormatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(min(console_level, logging.INFO))
    file_handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(min(console_level, logging.INFO))
    root.addHandler(console)
    root.addHandler(file_handler)


# ----------------------------------------------------------------- data
def load_data(path: Path = DATA_FILE) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, parse_dates=["date_time"])
    except FileNotFoundError as exc:
        raise DataUnavailableError(
            f"Processed data not found at {path}. Run 'python pipeline.py' first.") from exc
    except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as exc:
        raise DataUnavailableError(f"Could not read processed data {path}: {exc}") from exc
    logger.debug("Loaded processed data: %d rows x %d columns", *df.shape)
    # Sensor-fault hours are excluded from aggregate statistics.
    return df


def normal_hours(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_sensor_anomaly"] == 0]


# ----------------------------------------------------------------- validation helpers
def parse_user_datetime(text: str) -> datetime:
    for fmt in DATETIME_FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    raise InvalidInputError(
        f"Malformed date/time '{text}'. Use the format YYYY-MM-DD HH:MM, "
        "for example 2017-03-14 08:00")


def parse_day(text: str) -> str:
    match = [d for d in DAYS if d.lower().startswith(text.strip().lower()[:3])]
    if len(text.strip()) < 3 or not match:
        raise InvalidInputError(f"Unknown day '{text}'. Use one of: {', '.join(DAYS)}")
    return match[0]


def check_hour(value: int, name: str) -> None:
    if not 0 <= value <= 23:
        raise InvalidInputError(f"{name} must be an hour between 0 and 23 (got {value})")


def check_year(df: pd.DataFrame, year: int | None) -> None:
    if year is not None and year not in set(df["year"]):
        raise InvalidInputError(
            f"No data for year {year}. Available years: "
            f"{', '.join(map(str, sorted(df['year'].unique())))}")


# ----------------------------------------------------------------- commands
def cmd_summary(df: pd.DataFrame, args: argparse.Namespace) -> None:
    data = normal_hours(df)
    print("I-94 WESTBOUND TRAFFIC - DATASET SUMMARY")
    print(f"  Period covered      : {df['date_time'].min():%d %b %Y} to {df['date_time'].max():%d %b %Y}")
    print(f"  Hours in dataset    : {len(df):,} ({df['is_sensor_anomaly'].sum()} flagged as sensor faults)")
    print(f"  Average volume      : {data['traffic_volume'].mean():,.0f} vehicles/hour")
    print(f"  Busiest hour ever   : {df.loc[df['traffic_volume'].idxmax(), 'date_time']:%a %d %b %Y %H:%M} "
          f"({df['traffic_volume'].max():,} vehicles)")
    print(f"  Congested hours     : {data['is_congested'].mean():.1%} of hours above {CONGESTION_THRESHOLD:,}")
    print("  Congestion classes  : " + ", ".join(
        f"{k} {v:.0%}" for k, v in data["congestion_category"].value_counts(normalize=True)
        .reindex(["Low", "Medium", "High", "Severe"]).items()))


def cmd_query(df: pd.DataFrame, args: argparse.Namespace) -> None:
    when = parse_user_datetime(args.datetime)
    stamp = pd.Timestamp(when).floor("h")
    row = df[df["date_time"] == stamp]
    if row.empty:
        logger.warning("No record for %s; showing the nearest available hours", stamp)
        nearest = df.iloc[(df["date_time"] - stamp).abs().argsort()[:3]].sort_values("date_time")
        print(f"No record for {stamp:%Y-%m-%d %H:00} (the sensor has gaps). Nearest hours:")
        for _, r in nearest.iterrows():
            print(f"  {r['date_time']:%Y-%m-%d %H:00}  {r['traffic_volume']:>5,} vehicles  "
                  f"{r['congestion_category']:<7} {r['weather_description']}")
        return
    r = row.iloc[0]
    typical = normal_hours(df).query("hour == @r.hour and day_of_week == @r.day_of_week")[
        "traffic_volume"].mean()
    print(f"TRAFFIC ON {stamp:%A %d %B %Y, %H:00}")
    print(f"  Traffic volume  : {r['traffic_volume']:,} vehicles "
          f"(typical for this hour and day: {typical:,.0f})")
    print(f"  Congestion      : {r['congestion_category']}"
          f"{'  - above the 5,500 congestion threshold' if r['is_congested'] else ''}")
    print(f"  Weather         : {r['weather_main']} ({r['weather_description']}), "
          f"{r['temp_c']:.1f} °C, cloud cover {r['clouds_all']}%")
    print(f"  Precipitation   : rain {r['rain_1h']} mm, snow {r['snow_1h']} mm")
    if r["is_holiday"]:
        print(f"  Holiday         : {r['holiday_name']}")
    if r["is_sensor_anomaly"]:
        print("  Note            : this hour is flagged as a likely sensor fault")


def cmd_peak(df: pd.DataFrame, args: argparse.Namespace) -> None:
    check_year(df, args.year)
    if args.top < 1:
        raise InvalidInputError("--top must be at least 1")
    data = normal_hours(df)
    if args.year:
        data = data[data["year"] == args.year]
    if args.day_type != "all":
        data = data[data["is_weekend"] == (1 if args.day_type == "weekend" else 0)]
    table = (data.groupby(["day_name", "hour"])
                 .agg(avg=("traffic_volume", "mean"), congested=("is_congested", "mean"))
                 .sort_values("avg", ascending=False).head(args.top))
    scope = f"{args.day_type} hours" + (f", {args.year}" if args.year else ", all years")
    print(f"TOP {args.top} HIGHEST-TRAFFIC PERIODS ({scope})")
    print(f"  {'Rank':<5}{'Day':<11}{'Hour':<14}{'Avg vehicles':>13}{'% congested':>13}")
    for rank, ((day, hour), r) in enumerate(table.iterrows(), start=1):
        print(f"  {rank:<5}{day:<11}{hour:02d}:00-{hour:02d}:59  {r['avg']:>12,.0f}"
              f"{r['congested']:>12.0%}")


def pct_diff(a: float, b: float) -> float:
    """Relative difference a vs b, with tiny values shown as 0 (avoids '-0%')."""
    value = (a - b) / b
    return 0.0 if abs(value) < 0.005 else value


def cmd_compare(df: pd.DataFrame, args: argparse.Namespace) -> None:
    check_year(df, args.year)
    data = normal_hours(df)
    data = data[data["is_holiday"] == 0]
    if args.year:
        data = data[data["year"] == args.year]
    table = data.pivot_table(index="time_of_day", columns="is_weekend",
                             values="traffic_volume", aggfunc="mean")
    table = table.reindex(["Night", "Morning", "Afternoon", "Evening"])
    overall = data.groupby("is_weekend")["traffic_volume"].mean()
    congested = data.groupby("is_weekend")["is_congested"].mean()
    print(f"WEEKDAY vs WEEKEND TRAFFIC ({args.year or 'all years'}, holidays excluded)")
    print(f"  {'Time of day':<24}{'Weekday':>9}{'Weekend':>9}{'Difference':>12}")
    hours = {"Night": "00-05", "Morning": "06-11", "Afternoon": "12-17", "Evening": "18-23"}
    for period, r in table.iterrows():
        print(f"  {period + ' (' + hours[period] + 'h)':<24}{r[0]:>9,.0f}{r[1]:>9,.0f}"
              f"{pct_diff(r[0], r[1]):>+11.0%}")
    print(f"  {'All hours':<24}{overall[0]:>9,.0f}{overall[1]:>9,.0f}"
          f"{pct_diff(overall[0], overall[1]):>+11.0%}")
    print(f"  Congested hours: weekday {congested[0]:.1%}, weekend {congested[1]:.1%}")


def cmd_recommend(df: pd.DataFrame, args: argparse.Namespace) -> None:
    day = parse_day(args.day)
    check_hour(args.start, "--from")
    check_hour(args.end, "--to")
    if args.start > args.end:
        raise InvalidInputError(f"--from ({args.start}) must not be later than --to ({args.end})")
    data = normal_hours(df)
    data = data[(data["day_name"] == day) & (data["is_holiday"] == 0)
                & data["hour"].between(args.start, args.end)]
    table = (data.groupby("hour")
                 .agg(avg=("traffic_volume", "mean"),
                      congested=("is_congested", "mean"),
                      severe=("congestion_category", lambda s: (s == "Severe").mean()))
                 .sort_values(["congested", "avg"]))
    best, worst = table.head(args.top), table.sort_values("avg").tail(1)
    print(f"RECOMMENDED TRAVEL TIMES - {day}, between {args.start:02d}:00 and {args.end:02d}:59")
    print(f"  {'Hour':<14}{'Avg vehicles':>13}{'% congested':>13}{'% severe':>10}")
    for hour, r in best.iterrows():
        print(f"  {hour:02d}:00-{hour:02d}:59  {r['avg']:>12,.0f}{r['congested']:>12.0%}"
              f"{r['severe']:>10.0%}")
    w_hour = worst.index[0]
    print(f"  Avoid: {w_hour:02d}:00-{w_hour:02d}:59, the busiest hour in this window "
          f"({worst['avg'].iloc[0]:,.0f} vehicles, {worst['congested'].iloc[0]:.0%} congested)")


def cmd_weather(df: pd.DataFrame, args: argparse.Namespace) -> None:
    data = normal_hours(df)
    overall = data["traffic_volume"].mean()
    summary = (data.groupby("weather_main")
                   .agg(hours=("traffic_volume", "size"), avg=("traffic_volume", "mean"),
                        congested=("is_congested", "mean"), temp=("temp_c", "mean"))
                   .sort_values("avg", ascending=False))
    if args.condition:
        matches = [w for w in summary.index if w.lower() == args.condition.strip().lower()]
        if not matches:
            raise InvalidInputError(f"Unknown weather condition '{args.condition}'. "
                                    f"Choose from: {', '.join(sorted(summary.index))}")
        summary = summary.loc[matches]
    print(f"TRAFFIC BY WEATHER CONDITION (overall average {overall:,.0f} vehicles/hour)")
    print(f"  {'Weather':<14}{'Hours':>7}{'Avg vehicles':>14}{'vs overall':>12}"
          f"{'% congested':>13}{'Avg temp':>10}")
    for name, r in summary.iterrows():
        flag = "  (small sample)" if r["hours"] < 100 else ""
        print(f"  {name:<14}{int(r['hours']):>7,}{r['avg']:>14,.0f}{pct_diff(r['avg'], overall):>+12.0%}"
              f"{r['congested']:>12.0%}{r['temp']:>8.1f} °C{flag}")


COMMANDS = {"summary": cmd_summary, "query": cmd_query, "peak": cmd_peak,
            "compare": cmd_compare, "recommend": cmd_recommend, "weather": cmd_weather}


# ----------------------------------------------------------------- argument parsing
class LoggingArgumentParser(argparse.ArgumentParser):
    """Logs argument errors as ERROR (instead of only printing usage) and exits cleanly."""

    def error(self, message: str):  # noqa: D401 - argparse API
        logger.error("Invalid command-line input: %s", message)
        self.print_usage(sys.stderr)
        sys.exit(EXIT_BAD_INPUT)


def build_parser() -> argparse.ArgumentParser:
    parser = LoggingArgumentParser(
        prog="traffic_cli", description="Query the processed I-94 traffic dataset.")
    parser.add_argument("--verbose", action="store_true", help="show INFO logs on screen")
    parser.add_argument("--debug", action="store_true", help="show DEBUG logs on screen")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=LoggingArgumentParser)

    sub.add_parser("summary", help="overview of the processed dataset")

    p = sub.add_parser("query", help="traffic and weather for a specific hour")
    p.add_argument("--datetime", required=True, help='e.g. "2017-03-14 08:00"')

    p = sub.add_parser("peak", help="highest-traffic periods")
    p.add_argument("--top", type=int, default=5, help="number of periods (default 5)")
    p.add_argument("--day-type", choices=["weekday", "weekend", "all"], default="all")
    p.add_argument("--year", type=int, help="restrict to one year")

    p = sub.add_parser("compare", help="weekday vs weekend traffic")
    p.add_argument("--year", type=int, help="restrict to one year")

    p = sub.add_parser("recommend", help="least congested hours to travel on a given day")
    p.add_argument("--day", required=True, help="day of week, e.g. Monday or mon")
    p.add_argument("--from", dest="start", type=int, default=6, help="earliest hour (default 6)")
    p.add_argument("--to", dest="end", type=int, default=20, help="latest hour (default 20)")
    p.add_argument("--top", type=int, default=3, help="number of suggestions (default 3)")

    p = sub.add_parser("weather", help="traffic under each weather condition")
    p.add_argument("--condition", help="one condition, e.g. Rain or Snow")
    return parser


def main(argv: list[str] | None = None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--verbose", action="store_true")
    pre.add_argument("--debug", action="store_true")
    flags, _ = pre.parse_known_args(argv)
    configure_logging(logging.DEBUG if flags.debug else
                      logging.INFO if flags.verbose else logging.WARNING)

    args = build_parser().parse_args(argv)
    options = {k: v for k, v in vars(args).items() if k not in ("command", "verbose", "debug")}
    logger.info("Command invoked: %s | arguments: %s", args.command, options)
    try:
        df = load_data()
        COMMANDS[args.command](df, args)
    except InvalidInputError as exc:
        logger.error("Invalid input for '%s': %s", args.command, exc)
        return EXIT_BAD_INPUT
    except DataUnavailableError as exc:
        logger.error("%s", exc)
        return EXIT_ERROR
    except Exception as exc:  # unexpected failure: clear message, full detail in the log file
        logger.error("Command '%s' failed unexpectedly: %s (details in %s)",
                     args.command, exc, LOG_FILE.name, exc_info=True)
        return EXIT_ERROR
    logger.info("Command completed: %s", args.command)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

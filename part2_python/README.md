# Part 2: Reproducible Traffic Analytics Pipeline (Python)

**Author:** Kyi Htoo Aung | Smart City Traffic Intelligence Capstone

A multi-stage Python pipeline that loads, validates and cleans the Metro Interstate Traffic Volume dataset, engineers ML-ready features, saves six Matplotlib figures, and powers a command-line application for querying the processed data. Every stage is logged to the console and to `pipeline.log`.

---

## 1. Project structure

```
part2_python/
├── pipeline.py               Entry point: logging setup, load, validate, clean, then calls the modules below
├── feature_engineering.py    Task 2: time, weather, scaled and target features
├── visualizations.py         Task 3: six Matplotlib figures
├── cli_app/
│   ├── traffic_cli.py        Task 4: command-line application (6 commands)
│   ├── cli.log               Sample CLI log (commands, arguments, errors)
│   └── sample_session.txt    Sample CLI session (inputs and outputs)
├── figures/                  Six saved PNG figures (fig01 ... fig06)
├── output/
│   ├── cleaned_traffic.csv.gz     Cleaned data, one row per hour (40,575 x 14)
│   └── traffic_features.csv.gz    Feature table used by the CLI and Part 3 (40,575 x 54)
├── logs/
│   ├── pipeline_debug_sample.log  Sample of a --debug run (shows DEBUG values)
│   └── pipeline_error_sample.log  Sample of a failed run (missing input file): ERROR with full traceback
├── pipeline.log              Sample log of a normal run
├── part2_report.pdf          1-2 page methodology and findings report
├── requirements.txt
└── README.md
```

The raw dataset is read from the shared `data/` folder at the repository root (`../data/Metro_Interstate_Traffic_Volume.csv`).

## 2. How to run

Python 3.10 or newer is required.

```bash
cd part2_python
pip install -r requirements.txt

python pipeline.py            # full run: clean -> features -> figures (INFO logging)
python pipeline.py --debug    # same, plus DEBUG intermediate values
python pipeline.py --help     # other options: --input, --output-dir, --figure-dir,
                              #   --log-file, --skip-figures
```

The run ends with `Pipeline finished successfully` and exit code 0. If anything prevents completion (missing file, unreadable CSV, missing columns), it logs an ERROR and exits with code 1 instead of crashing.

### Command-line application

Run the pipeline once first, because the CLI reads `output/traffic_features.csv.gz`.

| Command | What it answers | Example |
|---|---|---|
| `summary` | Dataset overview | `python cli_app/traffic_cli.py summary` |
| `query` | Traffic and weather for a specific hour | `python cli_app/traffic_cli.py query --datetime "2017-03-14 08:00"` |
| `peak` | Highest-traffic periods (day and hour) | `python cli_app/traffic_cli.py peak --top 5 --day-type weekday --year 2017` |
| `compare` | Weekday vs weekend traffic by time of day | `python cli_app/traffic_cli.py compare --year 2017` |
| `recommend` | Least congested hours to travel on a given day | `python cli_app/traffic_cli.py recommend --day Friday --from 6 --to 20` |
| `weather` | Traffic under each (or one) weather condition | `python cli_app/traffic_cli.py weather --condition Snow` |

Add `--help` to any command for its options, `--verbose` to see INFO logs on screen, or `--debug` for DEBUG. Invalid input (for example `--datetime "2017-02-30 08:00"`, `--day Funday` or `--year 2030`) produces a clear ERROR message and exit code 2, never a raw traceback. A full example session is in `cli_app/sample_session.txt`.

Example:

```
$ python cli_app/traffic_cli.py peak --top 3 --day-type weekday
TOP 3 HIGHEST-TRAFFIC PERIODS (weekday hours, all years)
  Rank Day        Hour           Avg vehicles  % congested
  1    Wednesday  16:00-16:59         6,414         96%
  2    Tuesday    16:00-16:59         6,389         94%
  3    Thursday   16:00-16:59         6,344         94%
```

## 3. Logging configuration

| Item | Setting |
|---|---|
| Logger per module | Every module creates `logger = logging.getLogger(__name__)`; the root logger is never used directly for messages. |
| Where handlers are configured | Only in the entry points: `configure_logging()` in `pipeline.py` and in `cli_app/traffic_cli.py`. Library modules never add handlers. |
| Pipeline destinations | Console (standard error) **and** `pipeline.log` (overwritten on each run so it always shows the latest run). |
| CLI destinations | Console shows WARNING and above by default so answers stay readable; `cli_app/cli.log` (appended) records INFO and above, including every command and its arguments. |
| Format | `timestamp \| LEVEL \| module \| message`, e.g. `2026-11-13 03:56:33 \| WARNING  \| pipeline \| Duplicate rows: dropped 17 exact duplicate row(s) ...` |
| Errors | `logger.error(..., exc_info=True)`: the log file keeps the full traceback, while the console shows a one-line message. |
| DEBUG mode | `--debug` sets the level to DEBUG. In normal runs DEBUG messages are not emitted at all (compare `pipeline.log` with `logs/pipeline_debug_sample.log`). |
| ERROR example | `logs/pipeline_error_sample.log` was produced by `python pipeline.py --input missing_file.csv`: the failure is logged at ERROR with the full traceback and the run exits with code 1. |
| Output to users | `print()` is used **only** for the CLI's answers. No `print()` reports pipeline status or progress. |

What each level means in this project:

| Level | Used for | Examples |
|---|---|---|
| DEBUG | Fine-grained intermediate values for troubleshooting | Quartile thresholds, monthly medians used for imputation, scaling mean/std/min/max, rows per step, sensor-fault dates |
| INFO | Normal milestones | File loaded (rows x columns), schema passed, each feature group added, shapes before/after feature engineering, each figure saved with its path, command invoked |
| WARNING | Unexpected but recoverable changes to the data | Rows dropped (duplicates), values modified (category standardisation), values imputed (0 K temperatures), anomalies flagged, query hour not found |
| ERROR | Something that stops the run or the command | Missing/unreadable file, missing columns, malformed date, unknown day or weather condition |

## 4. Pipeline stages and results (Task 1)

| # | Step | Result on the raw data | Level |
|---|---|---|---|
| 1 | Load CSV (`try/except` for missing, empty, unreadable and unparseable files; no bare `except`) | 48,204 rows x 9 columns | INFO |
| 2 | Validate schema before any processing | All 9 expected columns present | INFO |
| 3 | Standardise categories (trim, one case per column) | 1,730 `weather_description` values fixed, e.g. "Sky is Clear" to "sky is clear" and "SQUALLS" to "squalls" | WARNING |
| 4 | Coerce numeric types | All numeric columns valid | INFO |
| 5 | Parse and validate `date_time` (explicit format) | All valid, 2012-10-02 to 2018-09-30 | INFO |
| 6 | Remove exact duplicate rows | 17 dropped | WARNING |
| 7 | Resolve duplicate timestamps (one row per hour, keep the most severe weather report) | 7,612 dropped across 5,430 hours | WARNING |
| 8 | Impossible temperatures (outside 220-330 K) set to missing | 10 readings of 0 K | WARNING |
| 9 | Impossible precipitation (above 305 mm/h, the world one-hour record) set to missing | 1 rain reading of 9,831.3 mm | WARNING |
| 10 | Cloud cover range check (0-100%) | All valid | INFO |
| 11 | Impute with the **median of the same month** (explicit loop over months) | 10 temperatures, 1 rain value | WARNING (+ DEBUG per month) |
| 12 | Validate traffic volume | All present, 0 to 7,280 | INFO |
| 13 | Flag sensor anomalies (busy-period hours below 5% of the typical volume for that hour and weekday) | 32 hours on 8 days flagged and kept, `is_sensor_anomaly = 1` | WARNING |
| 14 | Final checks (no missing values, one row per hour, sorted) | Passed: 40,575 rows | INFO |

**Why monthly medians?** The ten 0 K readings occurred in January and February 2014. Their monthly medians are about -13 °C, while the all-data median is about +8 °C, so a global median would have inserted spring-like temperatures into a Minnesota winter.

**Why keep the most severe weather report?** Where one hour has several weather rows (for example "mist" and "light rain"), traffic volume is identical in all of them, so keeping all would double-count the hour. Keeping the most severe report preserves bad-weather hours for the Part 3 risk model; `n_weather_reports` records how many reports existed.

## 5. Features (Task 2)

Shape is logged before (40,575 x 14) and after (40,575 x 54) feature engineering.

| Group | Features |
|---|---|
| Time | `hour`, `day_of_week` (0 = Monday), `day_name`, `month`, `year`, `is_weekend`, `is_holiday` (holiday label spread from the midnight record to all hours of that date), `holiday_name`, `is_peak_hour` (weekday 06-08 and 15-17, from Part 1), `time_of_day` |
| Cyclical encodings | `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`, so that hour 23 is next to hour 0 and December next to January |
| Weather | `temp_c`, one-hot `weather_*` (11 columns), `is_precipitation`, `is_low_visibility` (Mist, Fog, Haze, Smoke), `is_severe_weather` (Thunderstorm, Squall, Snow), `weather_severity` (0-10), `temp_band` |
| Scaled numerical | `temp_c_zscore` (z-score), `clouds_minmax` (0-1), `rain_log_minmax` (log1p then 0-1, because rain is extremely skewed). Part 3 refits scalers on training data only to avoid leakage. |
| Data quality flags | `temp_was_imputed`, `rain_1h_was_imputed`, `snow_1h_was_imputed`, `is_sensor_anomaly`, `n_weather_reports` |
| Target | `congestion_category`, `congestion_level`, `is_congested` (see below) |

**Congestion target logic.** `congestion_category` is data-driven: the quartiles of `traffic_volume` (computed on normal hours, excluding sensor faults) split hours into four equal-sized classes.

| Class | Rule | Range (vehicles/hour) |
|---|---|---|
| Low | volume <= Q1 | up to 1,254 |
| Medium | Q1 < volume <= Q2 | 1,255 to 3,431 |
| High | Q2 < volume <= Q3 | 3,432 to 4,953 |
| Severe | volume > Q3 | above 4,953 |

The thresholds are logged at DEBUG level. `is_congested` keeps the fixed Part 1 definition (volume > 5,500, 15.1% of hours) so Parts 1, 2 and 3 can be compared directly.

## 6. Figures and interpretation (Task 3)

Sensor-fault hours are excluded from all figures. Each saved figure is logged at INFO with its path.

1. **`fig01_hourly_weekday_vs_weekend.png`: traffic demand by hour.** Weekdays show two sharp commuter peaks, about 6,180 vehicles at 07:00 and 6,330 at 16:00, both above the 5,500 congestion line. Weekends have a single gentle midday plateau peaking at about 4,430 around 13:00. Commuting, not general demand, creates congestion.
2. **`fig02_traffic_distribution.png`: distribution of hourly volume.** The distribution is bimodal: a cluster of quiet night hours under about 1,000 and a broad daytime cluster between 4,000 and 6,500. A single average therefore describes almost no real hour; the quartile classes give four equal-sized, interpretable groups.
3. **`fig03_temperature_vs_traffic.png`: temperature vs traffic.** The relationship is weak (r = 0.14). Average volume is flat at about 3,000 to 3,300 between -20 °C and 20 °C, falls in extreme cold (about 2,080 below -25 °C) and rises above 25 °C (4,430 to 4,790). Warm readings occur on summer afternoons, so time of day confounds the link.
4. **`fig04_weather_impact.png`: weather impact.** Among well-sampled conditions, average traffic ranges from Haze (3,677) and Clouds (3,620) down to Fog (2,654). Fog has the lowest average yet the highest share of congested hours (20%), because fog forms in the early morning rush. Weather effects mostly reflect when the weather happens.
5. **`fig05_congestion_heatmap.png`: congestion by hour and weekday.** Congestion is concentrated on weekdays 06:00 to 08:00 and 15:00 to 17:00, reaching 94 to 96% of hours on Tuesday to Thursday at 16:00. Weekends are almost never congested, and on Fridays the afternoon build-up starts earlier.
6. **`fig06_monthly_trend.png`: congestion over time.** Monthly averages show seasonality, lowest in December (3,040) and January (3,060) and highest in late summer (August 3,431). Yearly averages are flat from 2012 to 2016 (3,200 to 3,310), then rise to 3,377 in 2017. The shaded band marks the August 2014 to June 2015 data gap.

## 7. Links to Part 1 and Part 3

- Part 1's data-quality findings (0 K temperatures, the 9,831 mm rain reading, duplicates, repeated timestamps, the July 2016 sensor fault and the 2014-2015 gap) became explicit, logged pipeline steps.
- Part 1's weekday peak hours define `is_peak_hour`, and its 5,500 threshold is kept as `is_congested`.
- Part 2 numbers can differ slightly from Part 1 (for example average volume 3,293 here vs 3,260 there, and r = 0.14 vs 0.13) because Part 2 counts each hour once and excludes sensor faults, whereas Part 1 analysed the raw rows as the brief required.
- `output/traffic_features.csv.gz`, including `is_low_visibility` and `is_severe_weather` for the proxy risk label, is the input to Part 3.

## 8. Limitations

- Data come from a single westbound sensor; gaps (2014-2015) and partial years limit trend analysis.
- The weather category is one label per hour; choosing the most severe report is a documented assumption.
- The sensor-anomaly rule (below 5% of the typical volume) is a heuristic and may miss partial faults.
- The quartile classes describe relative busyness, not road capacity.

## 9. Version control

The work was committed in stages, one commit per task, for example:

```
Add Part 2 data pipeline with schema validation and logging
Add data cleaning steps with monthly median imputation
Add feature engineering module
Add traffic visualisations
Add traffic analytics CLI application
Add Part 2 README, sample logs and report
```

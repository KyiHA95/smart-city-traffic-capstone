"""
visualizations.py - Part 2, Task 3: Matplotlib figures of traffic patterns.

Called by pipeline.py with the feature table. Hours flagged as sensor
anomalies are excluded from every chart so faults do not distort patterns.
Each saved figure is logged at INFO level with its file path.

Figures
-------
fig01_hourly_weekday_vs_weekend.png  Traffic demand by hour, weekday vs weekend
fig02_traffic_distribution.png       Distribution of hourly volume + quartile classes
fig03_temperature_vs_traffic.png     Temperature vs traffic (density + binned mean)
fig04_weather_impact.png             Average traffic and congestion rate by weather
fig05_congestion_heatmap.png         Congestion probability by hour x day of week
fig06_monthly_trend.png              Monthly average volume over time (with data gap)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")  # file output only; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NAVY, ORANGE, GREY, LIGHT = "#1f4e79", "#dd6b20", "#6b7280", "#9ecae1"
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CONGESTION_THRESHOLD = 5500

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 12,
})


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", os.path.relpath(path))
    return path


def plot_hourly_weekday_weekend(df: pd.DataFrame, out_dir: Path) -> Path:
    data = df[df["is_holiday"] == 0]
    profile = data.groupby(["hour", "is_weekend"])["traffic_volume"].agg(
        ["mean", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
    profile.columns = ["mean", "q25", "q75"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for flag, label, colour in ((0, "Weekday", NAVY), (1, "Weekend", ORANGE)):
        p = profile.xs(flag, level="is_weekend")
        ax.plot(p.index, p["mean"], marker="o", ms=4, lw=2, color=colour, label=f"{label} mean")
        ax.fill_between(p.index, p["q25"], p["q75"], color=colour, alpha=0.15,
                        label=f"{label} middle 50%")
    ax.axhline(CONGESTION_THRESHOLD, ls="--", color=GREY, lw=1)
    ax.text(0.2, CONGESTION_THRESHOLD + 120, "Congestion threshold (5,500)", color=GREY, fontsize=8)
    ax.set(title="Traffic demand by hour: weekday vs weekend (holidays excluded)",
           xlabel="Hour of day", ylabel="Vehicles per hour", xticks=range(0, 24))
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    return _save(fig, out_dir / "fig01_hourly_weekday_vs_weekend.png")


def plot_traffic_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    q1, q2, q3 = df["traffic_volume"].quantile([0.25, 0.5, 0.75])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bins = np.arange(0, 7500, 125)
    colours = {"Low": "#9ecae1", "Medium": "#4292c6", "High": "#fd8d3c", "Severe": "#d94801"}
    for label, colour in colours.items():
        ax.hist(df.loc[df["congestion_category"] == label, "traffic_volume"], bins=bins,
                color=colour, label=label, edgecolor="white", linewidth=0.3)
    for q, name in ((q1, "Q1"), (q2, "Median"), (q3, "Q3")):
        ax.axvline(q, color="black", ls=":", lw=1)
        ax.text(q, ax.get_ylim()[1] * 0.95, f" {name}\n {q:,.0f}", fontsize=8, va="top")
    ax.set(title="Distribution of hourly traffic volume with quartile congestion classes",
           xlabel="Vehicles per hour", ylabel="Number of hours")
    ax.legend(title="congestion_category", fontsize=8)
    return _save(fig, out_dir / "fig02_traffic_distribution.png")


def plot_temperature_vs_traffic(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    hb = ax.hexbin(df["temp_c"], df["traffic_volume"], gridsize=45, cmap="Blues", mincnt=1)
    fig.colorbar(hb, ax=ax, label="Hours in cell")
    bins = np.arange(-30, 40, 5)
    binned = df.groupby(pd.cut(df["temp_c"], bins), observed=True)["traffic_volume"].mean()
    centres = [interval.mid for interval in binned.index]
    ax.plot(centres, binned.values, color=ORANGE, marker="o", lw=2, label="Mean per 5 °C band")
    r = np.corrcoef(df["temp_c"], df["traffic_volume"])[0, 1]
    logger.debug("Temperature vs traffic Pearson r (anomalies excluded) = %.4f", r)
    ax.set(title=f"Temperature vs traffic volume (Pearson r = {r:.2f})",
           xlabel="Temperature (°C)", ylabel="Vehicles per hour")
    ax.legend(fontsize=8, loc="upper left")
    return _save(fig, out_dir / "fig03_temperature_vs_traffic.png")


def plot_weather_impact(df: pd.DataFrame, out_dir: Path) -> Path:
    summary = (df.groupby("weather_main")
                 .agg(avg=("traffic_volume", "mean"), congested=("is_congested", "mean"),
                      n=("traffic_volume", "size"))
                 .sort_values("avg"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    colours = [GREY if n < 100 else NAVY for n in summary["n"]]
    ax1.barh(summary.index, summary["avg"], color=colours)
    for i, (avg, n) in enumerate(zip(summary["avg"], summary["n"])):
        ax1.text(avg + 40, i, f"{avg:,.0f} (n={n:,})", va="center", fontsize=8)
    ax1.set(title="Average traffic by weather", xlabel="Vehicles per hour", xlim=(0, 4800))
    ax2.barh(summary.index, summary["congested"] * 100, color=[GREY if n < 100 else ORANGE
                                                            for n in summary["n"]])
    for i, rate in enumerate(summary["congested"] * 100):
        ax2.text(rate + 0.4, i, f"{rate:.1f}%", va="center", fontsize=8)
    ax2.set(title="Share of hours congested (> 5,500)", xlabel="% of hours", xlim=(0, 25))
    fig.suptitle("Weather impact on traffic (grey = fewer than 100 hours, unreliable)",
                 fontsize=10, color=GREY)
    return _save(fig, out_dir / "fig04_weather_impact.png")


def plot_congestion_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    grid = (df.pivot_table(index="day_name", columns="hour", values="is_congested",
                           aggfunc="mean").reindex(DAY_ORDER) * 100)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(24), labels=range(24))
    ax.set_yticks(range(7), labels=DAY_ORDER)
    for (row, col), value in np.ndenumerate(grid.to_numpy()):
        if value >= 50:
            ax.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im, ax=ax, label="% of hours congested (> 5,500)")
    ax.set(title="When does congestion happen? Probability by hour and day of week",
           xlabel="Hour of day")
    return _save(fig, out_dir / "fig05_congestion_heatmap.png")


def plot_monthly_trend(df: pd.DataFrame, out_dir: Path) -> Path:
    monthly = (df.set_index("date_time")["traffic_volume"].resample("MS")
                 .agg(["mean", "size"]))
    monthly.loc[monthly["size"] < 200, "mean"] = np.nan   # too few hours to be reliable
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(monthly.index, monthly["mean"], color=NAVY, marker="o", ms=3, lw=1.8)
    ax.axvspan(pd.Timestamp("2014-08-08"), pd.Timestamp("2015-06-11"), color=GREY, alpha=0.15)
    ax.text(pd.Timestamp("2014-08-20"), monthly["mean"].min(), "No data\n(Aug 2014 - Jun 2015)",
            fontsize=8, color=GREY, va="bottom")
    yearly = df.groupby("year")["traffic_volume"].mean()
    for year, value in yearly.items():
        ax.hlines(value, pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"),
                  color=ORANGE, lw=1, ls="--")
    ax.set(title="Monthly average traffic over time (dashed = yearly average)",
           ylabel="Average vehicles per hour")
    return _save(fig, out_dir / "fig06_monthly_trend.png")


FIGURES: tuple[Callable[[pd.DataFrame, Path], Path], ...] = (
    plot_hourly_weekday_weekend,
    plot_traffic_distribution,
    plot_temperature_vs_traffic,
    plot_weather_impact,
    plot_congestion_heatmap,
    plot_monthly_trend,
)


def create_all_figures(features: pd.DataFrame, out_dir: Path) -> tuple[list[Path], int]:
    """Create every figure; return (saved paths, number of failures)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = features[features["is_sensor_anomaly"] == 0]
    logger.info("Creating %d figures from %d rows (%d sensor-anomaly hours excluded)",
                len(FIGURES), len(data), len(features) - len(data))
    saved, failed = [], 0
    for make_figure in FIGURES:
        try:
            saved.append(make_figure(data, out_dir))
        except (ValueError, KeyError, OSError) as exc:
            failed += 1
            plt.close("all")
            logger.error("Figure %s could not be created: %s", make_figure.__name__, exc,
                         exc_info=True)
    logger.info("Figures complete: %d saved, %d failed", len(saved), failed)
    return saved, failed

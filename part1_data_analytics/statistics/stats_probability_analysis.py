"""
Part 1 - Tasks 2 and 3: Descriptive statistics, correlation and probability.

Usage (from the part1_data_analytics folder):
    python statistics/stats_probability_analysis.py

Outputs (in statistics/):
    stats_probability_results.md   - all numbers with interpretation-ready tables
    stats_probability_results.json - the same numbers in machine-readable form
    figures/*.png                  - supporting charts

Definitions used (from the capstone brief):
    Congestion        : traffic_volume > 5,500 vehicles/hour
    Clear weather     : weather_main == 'Clear'
    Cloudy weather    : weather_main == 'Clouds'
    High temperature  : temp > 292 K (about 18.85 C)
"""
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent
CSV_PATH = BASE.parent / "data" / "Metro_Interstate_Traffic_Volume.csv"
OUT_DIR = BASE / "statistics"
FIG_DIR = OUT_DIR / "figures"

CONGESTION_THRESHOLD = 5500
HIGH_TEMP_K = 292


def load() -> pd.DataFrame:
    # keep_default_na=False stops pandas turning the holiday text "None" into NaN
    df = pd.read_csv(CSV_PATH, keep_default_na=False)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["hour"] = df["date_time"].dt.hour
    df["congestion"] = df["traffic_volume"] > CONGESTION_THRESHOLD
    df["clear"] = df["weather_main"] == "Clear"
    df["clouds"] = df["weather_main"] == "Clouds"
    df["high_temp"] = df["temp"] > HIGH_TEMP_K
    logger.info("Loaded %d rows x %d columns", *df.shape)
    return df


# ---------------------------------------------------------------- Task 2.1
def descriptive_stats(s: pd.Series) -> dict:
    return {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std_sample": float(s.std(ddof=1)),
        "variance_sample": float(s.var(ddof=1)),
        "std_population": float(s.std(ddof=0)),
        "variance_population": float(s.var(ddof=0)),
        "min": float(s.min()),
        "max": float(s.max()),
        "range": float(s.max() - s.min()),
        "q1": float(s.quantile(0.25)),
        "q3": float(s.quantile(0.75)),
        "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
        "coefficient_of_variation_pct": float(100 * s.std(ddof=1) / s.mean()),
        "skewness": float(s.skew()),
        "zero_volume_hours": int((s == 0).sum()),
    }


# ---------------------------------------------------------------- Task 2.2
def correlation(df: pd.DataFrame) -> dict:
    valid = df[df["temp"] > 0]  # 10 rows record 0 K, which is physically impossible
    r_all, p_all = stats.pearsonr(df["temp"], df["traffic_volume"])
    r_val, p_val = stats.pearsonr(valid["temp"], valid["traffic_volume"])
    rho, p_rho = stats.spearmanr(valid["temp"], valid["traffic_volume"])
    # Daytime-only check (06:00-19:59) to remove the night-time effect
    day = valid[valid["hour"].between(6, 19)]
    r_day, _ = stats.pearsonr(day["temp"], day["traffic_volume"])
    return {
        "pearson_r_all_rows": float(r_all), "p_value_all_rows": float(p_all),
        "pearson_r_excl_0K": float(r_val), "p_value_excl_0K": float(p_val),
        "r_squared_excl_0K": float(r_val ** 2),
        "spearman_rho_excl_0K": float(rho), "spearman_p": float(p_rho),
        "pearson_r_daytime_06_19": float(r_day),
        "rows_with_0K_excluded": int((df["temp"] == 0).sum()),
    }


# ---------------------------------------------------------------- Task 3
def probabilities(df: pd.DataFrame) -> dict:
    n = len(df)
    p_c = df["congestion"].mean()
    p_clear = df["clear"].mean()
    p_c_and_clear = (df["congestion"] & df["clear"]).mean()
    p_clear_given_c = p_c_and_clear / p_c
    p_high_given_c = (df["congestion"] & df["high_temp"]).mean() / p_c
    p_high = df["high_temp"].mean()
    p_c_given_clear = p_c_and_clear / p_clear

    # Independence check: P(A and B) vs P(A) x P(B)
    expected_joint = p_c * p_clear

    # 2x2 table, chi-square test of independence (congestion x clear)
    table = pd.crosstab(df["congestion"], df["clear"])
    chi2, chi_p, _, _ = stats.chi2_contingency(table)

    # Odds ratio: odds of congestion in Clear vs Clouds
    def odds(mask):
        c = (df.loc[mask, "congestion"]).sum()
        nc = (~df.loc[mask, "congestion"]).sum()
        return int(c), int(nc), c / nc

    c_clear, nc_clear, odds_clear = odds(df["clear"])
    c_cloud, nc_cloud, odds_cloud = odds(df["clouds"])
    odds_ratio = odds_clear / odds_cloud
    # 95% CI for the odds ratio (Woolf / log method)
    se = np.sqrt(1 / c_clear + 1 / nc_clear + 1 / c_cloud + 1 / nc_cloud)
    ci = (np.exp(np.log(odds_ratio) - 1.96 * se), np.exp(np.log(odds_ratio) + 1.96 * se))

    # Confounder check: stratify by hour and compute a Mantel-Haenszel odds ratio.
    # Clear skies are more common at night, when traffic is always low.
    sub = df[df["clear"] | df["clouds"]]
    num = den = 0.0
    for _, g in sub.groupby("hour"):
        a = (g["clear"] & g["congestion"]).sum()
        b = (g["clear"] & ~g["congestion"]).sum()
        c = (g["clouds"] & g["congestion"]).sum()
        d = (g["clouds"] & ~g["congestion"]).sum()
        t = a + b + c + d
        if t:
            num += a * d / t
            den += b * c / t
    mh_or = num / den if den else float("nan")
    clear_share_night = df.loc[df["hour"].isin([0, 1, 2, 3, 4, 5, 22, 23]), "clear"].mean()
    clear_share_peak = df.loc[df["hour"].isin([7, 8, 15, 16, 17]), "clear"].mean()

    # Congestion rate per weather condition (context for the conclusion)
    by_weather = (df.groupby("weather_main")["congestion"]
                    .agg(["mean", "size"]).sort_values("mean", ascending=False))

    return {
        "n": n,
        "P(Congestion)": p_c,
        "P(Clear)": p_clear,
        "P(Congestion AND Clear)": p_c_and_clear,
        "P(Clear | Congestion)": p_clear_given_c,
        "P(Congestion | Clear)": p_c_given_clear,
        "P(High Temp)": p_high,
        "P(High Temp | Congestion)": p_high_given_c,
        "P(Congestion) x P(Clear)": expected_joint,
        "independence_gap": p_c_and_clear - expected_joint,
        "independence_ratio": p_c_and_clear / expected_joint,
        "chi2": float(chi2), "chi2_p_value": float(chi_p),
        "clear_congested": c_clear, "clear_not_congested": nc_clear,
        "clouds_congested": c_cloud, "clouds_not_congested": nc_cloud,
        "odds_clear": odds_clear, "odds_clouds": odds_cloud,
        "odds_ratio_clear_vs_clouds": odds_ratio,
        "odds_ratio_95ci": [float(ci[0]), float(ci[1])],
        "mantel_haenszel_or_hour_adjusted": float(mh_or),
        "P(Clear) at night (22-05h)": float(clear_share_night),
        "P(Clear) at peaks (7-8h, 15-17h)": float(clear_share_peak),
        "congestion_rate_by_weather": {k: {"rate": float(v["mean"]), "n": int(v["size"])}
                                       for k, v in by_weather.iterrows()},
    }


# ---------------------------------------------------------------- Figures
def save_figures(df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["traffic_volume"], bins=60, color="#2b6cb0", edgecolor="white")
    for x, lab, col in [(df["traffic_volume"].mean(), "Mean", "#c53030"),
                        (df["traffic_volume"].median(), "Median", "#2f855a"),
                        (CONGESTION_THRESHOLD, "Congestion (5,500)", "#dd6b20")]:
        ax.axvline(x, color=col, ls="--", lw=1.5, label=f"{lab}: {x:,.0f}")
    ax.set(title="Distribution of hourly traffic volume (bimodal: night vs day)",
           xlabel="Vehicles per hour", ylabel="Number of hours")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "01_traffic_volume_distribution.png", dpi=150)
    plt.close(fig)

    valid = df[df["temp"] > 0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(valid["temp"] - 273.15, valid["traffic_volume"], s=2, alpha=0.15, color="#2b6cb0")
    m, b = np.polyfit(valid["temp"] - 273.15, valid["traffic_volume"], 1)
    xs = np.linspace(-30, 37, 100)
    ax.plot(xs, m * xs + b, color="#c53030", lw=2, label=f"Linear fit (slope {m:.1f} veh/h per °C)")
    ax.set(title="Temperature vs traffic volume (0 K readings excluded)",
           xlabel="Temperature (°C)", ylabel="Vehicles per hour")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "02_temperature_vs_traffic.png", dpi=150)
    plt.close(fig)

    hourly = df.groupby("hour").agg(cong=("congestion", "mean"), clear=("clear", "mean"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(hourly.index, hourly["cong"] * 100, color="#dd6b20", label="% of hours congested")
    ax2 = ax.twinx()
    ax2.plot(hourly.index, hourly["clear"] * 100, color="#2b6cb0", marker="o", label="% of hours with Clear sky")
    ax.set(title="Congestion and clear skies by hour of day (the confounder)",
           xlabel="Hour of day", ylabel="% congested")
    ax2.set_ylabel("% Clear weather")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "03_congestion_vs_clear_by_hour.png", dpi=150)
    plt.close(fig)
    logger.info("Saved 3 figures to %s", FIG_DIR)


# ---------------------------------------------------------------- Report
def write_markdown(d: dict, c: dict, p: dict) -> None:
    f = lambda x, k=4: f"{x:,.{k}f}"
    lines = [
        "# Task 2 and Task 3 Results\n",
        "Generated by `statistics/stats_probability_analysis.py` on the raw dataset (n = 48,204 hourly records).\n",
        "## 2.1 Traffic volume statistics\n",
        "| Statistic | Value |", "|---|---|",
        f"| Mean | {f(d['mean'],2)} |",
        f"| Median | {f(d['median'],0)} |",
        f"| Standard deviation (sample) | {f(d['std_sample'],2)} |",
        f"| Variance (sample) | {f(d['variance_sample'],2)} |",
        f"| Range (max - min) | {f(d['range'],0)} ({f(d['min'],0)} to {f(d['max'],0)}) |",
        f"| Q1 / Q3 / IQR | {f(d['q1'],0)} / {f(d['q3'],0)} / {f(d['iqr'],0)} |",
        f"| Coefficient of variation | {f(d['coefficient_of_variation_pct'],1)}% |",
        f"| Skewness | {f(d['skewness'],3)} |",
        f"| Hours with zero volume | {d['zero_volume_hours']} |",
        "",
        "## 2.2 Correlation: temperature vs traffic volume\n",
        "| Measure | Value |", "|---|---|",
        f"| Pearson r (all rows) | {f(c['pearson_r_all_rows'])} |",
        f"| Pearson r (excluding {c['rows_with_0K_excluded']} rows at 0 K) | {f(c['pearson_r_excl_0K'])} |",
        f"| R squared (excl. 0 K) | {f(c['r_squared_excl_0K'])} |",
        f"| Spearman rho (excl. 0 K) | {f(c['spearman_rho_excl_0K'])} |",
        f"| Pearson r, daytime hours 06:00-19:59 only | {f(c['pearson_r_daytime_06_19'])} |",
        f"| p-value (excl. 0 K) | {c['p_value_excl_0K']:.2e} |",
        "",
        "## 3.1 Basic probability\n",
        "| Probability | Value |", "|---|---|",
    ]
    for k in ["P(Congestion)", "P(Clear)", "P(Congestion AND Clear)"]:
        lines.append(f"| {k} | {f(p[k])} ({p[k]*100:.2f}%) |")
    lines += ["", "## 3.2 Conditional probability and independence\n",
              "| Measure | Value |", "|---|---|"]
    for k in ["P(Clear | Congestion)", "P(Congestion | Clear)", "P(High Temp)",
              "P(High Temp | Congestion)", "P(Congestion) x P(Clear)"]:
        lines.append(f"| {k} | {f(p[k])} ({p[k]*100:.2f}%) |")
    lines += [
        f"| Observed P(C and Clear) / expected | {f(p['independence_ratio'],3)} |",
        f"| Chi-square (1 df) | {f(p['chi2'],1)}, p = {p['chi2_p_value']:.2e} |",
        "",
        "## 3.2 Odds ratio: congestion in Clear vs Clouds\n",
        "| Weather | Congested | Not congested | Odds |", "|---|---|---|---|",
        f"| Clear | {p['clear_congested']:,} | {p['clear_not_congested']:,} | {f(p['odds_clear'])} |",
        f"| Clouds | {p['clouds_congested']:,} | {p['clouds_not_congested']:,} | {f(p['odds_clouds'])} |",
        "",
        f"**Odds ratio (Clear / Clouds) = {f(p['odds_ratio_clear_vs_clouds'],3)}**, "
        f"95% CI {f(p['odds_ratio_95ci'][0],3)} to {f(p['odds_ratio_95ci'][1],3)}.",
        "",
        f"Hour-adjusted (Mantel-Haenszel) odds ratio = {f(p['mantel_haenszel_or_hour_adjusted'],3)}. "
        f"Clear skies make up {p['P(Clear) at night (22-05h)']*100:.1f}% of night hours but "
        f"{p['P(Clear) at peaks (7-8h, 15-17h)']*100:.1f}% of peak hours.",
        "",
        "## Congestion rate by weather condition\n",
        "| Weather | % congested | Records |", "|---|---|---|",
    ]
    for k, v in p["congestion_rate_by_weather"].items():
        lines.append(f"| {k} | {v['rate']*100:.1f}% | {v['n']:,} |")
    (OUT_DIR / "stats_probability_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = load()
    d = descriptive_stats(df["traffic_volume"])
    c = correlation(df)
    p = probabilities(df)
    save_figures(df)
    write_markdown(d, c, p)
    with open(OUT_DIR / "stats_probability_results.json", "w") as fh:
        json.dump({"descriptive": d, "correlation": c, "probability": p}, fh, indent=2, default=float)
    logger.info("Results written to %s", OUT_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    main()

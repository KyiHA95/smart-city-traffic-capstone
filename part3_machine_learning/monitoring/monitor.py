"""
monitor.py - Part 3, Task 6.4 (monitoring) and 6.5 (alerting).

Simulates production monitoring of the champion traffic-volume model.

* Production batches: each month of 2018 (January-September).
* Check 1, prediction error drift: monthly MAE compared with the MAE for the
  SAME calendar month of 2017, the first out-of-sample year (accepted
  performance; error is seasonal).
      ALERT if MAE exceeds the reference by more than 25%.
* Check 2, feature distribution drift: Population Stability Index (PSI) for
  temperature, cloud cover, weather type and the model's own predictions,
  against the NORMAL for that calendar month over 2016-2017. A single year is
  too narrow (weather varies a lot from one February to the next), while
  2012-2015 used a different recording scheme for cloud cover and weather
  labels (e.g. cloud values 0 and 64, far more "Clouds" hours), so including
  them would flag the change of scheme rather than real drift.
  Cloud cover is recorded as a few discrete values, so it uses fixed bins.
      PSI < 0.10 stable | 0.10-0.25 moderate (noted) | > 0.25 ALERT.
* Simulated incidents prove that the alerts fire:
      sensor_fault : the sensor under-counts by 40% in September 2018
      temperature_shift : September 2018 readings are 10 °C warmer than usual

Output: monitoring_status.json, monitoring_report.md and monitoring_dashboard.png
(PASS / ALERT per batch and an overall system status). Every ALERT is
logged at WARNING level, and the monitoring metrics are logged to MLflow.

Usage (from part3_machine_learning):
    python monitoring/monitor.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

PART3_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PART3_DIR), str(PART3_DIR / "notebooks")]
import common  # noqa: E402
from common import FEATURES, MODELS_DIR  # noqa: E402

logger = logging.getLogger(__name__)

OUT_DIR = PART3_DIR / "monitoring"
MAE_ALERT_RATIO = 1.25
PSI_MODERATE, PSI_ALERT = 0.10, 0.25
NUMERIC_DRIFT = ["temp_c", "clouds_all", "predicted_volume"]
CATEGORICAL_DRIFT = ["weather_main"]


FIXED_BINS = {"clouds_all": [-np.inf, 10, 30, 60, 80, np.inf]}   # discrete coded values


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    if reference.name in FIXED_BINS:
        edges = np.array(FIXED_BINS[reference.name], dtype=float)
    else:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    cats = sorted(set(reference) | set(current))
    ref = reference.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=1e-4)
    cur = current.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=1e-4)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def check_batch(name: str, batch: pd.DataFrame, error_ref: pd.DataFrame, normals: pd.DataFrame) -> dict:
    month = batch["month"].iloc[0]
    ref_month = error_ref[error_ref["month"] == month]
    normal_month = normals[normals["month"] == month]
    ref_mae = mean_absolute_error(ref_month["traffic_volume"], ref_month["predicted_volume"])
    mae = mean_absolute_error(batch["traffic_volume"], batch["predicted_volume"])
    drift = {c: psi_numeric(normal_month[c], batch[c]) for c in NUMERIC_DRIFT}
    drift.update({c: psi_categorical(normal_month[c], batch[c]) for c in CATEGORICAL_DRIFT})

    reasons = []
    if mae > MAE_ALERT_RATIO * ref_mae:
        reasons.append(f"prediction error drift: MAE {mae:.0f} vs reference {ref_mae:.0f} "
                       f"(+{mae / ref_mae - 1:.0%}, limit +{MAE_ALERT_RATIO - 1:.0%})")
    reasons += [f"feature drift: PSI({c}) = {v:.2f} > {PSI_ALERT}" for c, v in drift.items() if v > PSI_ALERT]
    notes = [f"moderate shift: PSI({c}) = {v:.2f}" for c, v in drift.items() if PSI_MODERATE < v <= PSI_ALERT]
    status = "ALERT" if reasons else "PASS"
    error_alert = mae > MAE_ALERT_RATIO * ref_mae
    action = ("Investigate now: accuracy has degraded" if error_alert else
              "Watch: inputs are unusual but accuracy is within limits" if reasons else "None")
    result = {"batch": name, "hours": len(batch), "status": status, "action": action,
              "mean_temp_c": float(batch["temp_c"].mean()),
              "normal_temp_c": float(normal_month["temp_c"].mean()),
              "label": "ALERT / Requires investigation" if reasons else "PASS / Normal",
              "mae": mae, "reference_mae": ref_mae, "mae_ratio": mae / ref_mae,
              **{f"psi_{k}": v for k, v in drift.items()}, "reasons": reasons, "notes": notes}
    if reasons:
        logger.warning("MONITORING ALERT for %s: %s", name, "; ".join(reasons))
    else:
        logger.info("Monitoring PASS for %s: MAE %.0f (reference %.0f), max PSI %.2f%s", name, mae,
                    ref_mae, max(drift.values()), f" ({'; '.join(notes)})" if notes else "")
    return result


def simulate(batch: pd.DataFrame, model, scenario: str) -> pd.DataFrame:
    b = batch.copy()
    if scenario == "sensor_fault":
        b["traffic_volume"] = (b["traffic_volume"] * 0.6).round()       # sensor under-counts by 40%
    elif scenario == "temperature_shift":
        b["temp_c"] = b["temp_c"] + 10.0                                  # heatwave / miscalibrated sensor
        b["predicted_volume"] = model.predict(b[FEATURES])
    return b


def action_key(result: dict) -> str:
    """'Investigate now: ...' -> 'Investigate', 'Watch: ...' -> 'Watch', 'None' -> 'None'."""
    return result["action"].split()[0].rstrip(":")


def dashboard(results: list[dict], overall: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    names = [r["batch"].replace("SIM ", "SIM:\n").replace("_", " ") for r in results]
    palette = {"None": "#2f855a", "Watch": "#e0a100", "Investigate": "#c53030"}
    colours = [palette[action_key(r)] for r in results]
    axes[0].bar(names, [r["mae_ratio"] for r in results], color=colours)
    axes[0].axhline(MAE_ALERT_RATIO, ls="--", color="black", lw=1)
    axes[0].text(len(names) - 0.5, MAE_ALERT_RATIO + 0.03, "alert limit +25%", ha="right", fontsize=8)
    axes[0].set(ylabel="MAE / reference MAE", title=f"Model monitoring dashboard: overall status {overall}")
    psi_cols = [c for c in results[0] if c.startswith("psi_")]
    width = 0.8 / len(psi_cols)
    for i, c in enumerate(psi_cols):
        axes[1].bar(np.arange(len(names)) + i * width - 0.4 + width / 2, [r[c] for r in results], width,
                    label=c.replace("psi_", ""))
    axes[1].axhline(PSI_ALERT, ls="--", color="black", lw=1)
    axes[1].axhline(PSI_MODERATE, ls=":", color="grey", lw=1)
    axes[1].set(ylabel="PSI vs 2016-17 month normal", title="Feature / prediction distribution drift")
    axes[1].legend(fontsize=7, ncol=4)
    short = {"None": "PASS", "Watch": "WATCH", "Investigate": "INVESTIGATE"}
    axes[1].set_xticks(range(len(names)), [f"{n}\n{short[action_key(r)]}"
                                           for n, r in zip(names, results)], fontsize=7)
    for tick, colour in zip(axes[1].get_xticklabels(), colours):
        tick.set_color(colour)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=palette["None"], label="PASS / Normal"),
                            Patch(color=palette["Watch"], label="ALERT: watch (inputs unusual, accuracy OK)"),
                            Patch(color=palette["Investigate"], label="ALERT: investigate (accuracy degraded)")],
                   fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "monitoring_dashboard.png", dpi=150)
    plt.close(fig)
    logger.info("Saved monitoring/monitoring_dashboard.png")


def worst_days(batch: pd.DataFrame, n: int = 3) -> list[str]:
    d = batch.assign(err=(batch["traffic_volume"] - batch["predicted_volume"]).abs())
    g = d.groupby(d["date_time"].dt.date).agg(mae=("err", "mean"), actual=("traffic_volume", "mean"),
                                              predicted=("predicted_volume", "mean"),
                                              weather=("weather_main", lambda s: s.mode().iat[0]))
    return [f"{day}: {r['weather']}, actual {r['actual']:,.0f} vs predicted {r['predicted']:,.0f} vehicles/hour"
            for day, r in g.sort_values("mae", ascending=False).head(n).iterrows()]


def report(results: list[dict], overall: str) -> None:
    lines = ["# Model monitoring report", "",
             f"**Overall system status: {overall}**", "",
             "Model monitored: champion `traffic_volume_regressor` (see `models/MODEL_VERSIONS.md`). "
             "Error reference: the same calendar month of 2017 (first out-of-sample year). Feature reference: "
             "the 2016-2017 normal for that calendar month (earlier years used a different recording scheme "
             "for cloud cover and weather labels). Alert rules: MAE more than 25% above the "
             "reference, or PSI above 0.25 for temperature, cloud cover, weather type or the model's "
             "predictions (PSI 0.10-0.25 is noted as moderate).", "",
             "| Batch | Hours | MAE | Ref. MAE | MAE ratio | PSI temp | PSI clouds | PSI weather | PSI predictions | Status | Action |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['batch']} | {r['hours']} | {r['mae']:.0f} | {r['reference_mae']:.0f} | "
                     f"{r['mae_ratio']:.2f} | {r['psi_temp_c']:.2f} | {r['psi_clouds_all']:.2f} | "
                     f"{r['psi_weather_main']:.2f} | {r['psi_predicted_volume']:.2f} | **{r['label']}** | {r['action']} |")
    lines += ["", "## Alerts and notes", ""]
    for r in results:
        for reason in r["reasons"]:
            lines.append(f"- **{r['batch']}: ALERT**, {reason}")
        for note in r["notes"]:
            lines.append(f"- {r['batch']}: {note} (below the alert limit)")
    real_alerts = [r for r in results if r["status"] == "ALERT" and not r["batch"].startswith("SIM")]
    sims = [r for r in results if r["batch"].startswith("SIM")]
    lines += ["", "## Interpretation", ""]
    passed = sum(r["status"] == "PASS" for r in results if not r["batch"].startswith("SIM"))
    lines.append(f"{passed} of {passed + len(real_alerts)} real 2018 monthly batches passed every check.")
    for r in real_alerts:
        lines.append(f"\n**{r['batch']}** ({r['action']}). Mean temperature {r['mean_temp_c']:.1f} °C vs a "
                     f"2016-2017 normal of {r['normal_temp_c']:.1f} °C. Reasons: {'; '.join(r['reasons'])}. "
                     "Days with the largest errors:")
        lines += [f"- {d}" for d in r.get("worst_days", [])]
    lines.append("\nSimulated incidents (to prove the alerting works): " + "; ".join(
        f"{r['batch'].replace('SIM ', '')} -> {r['status']}" for r in sims) + ".")
    lines += ["", "**Recommended response to an ALERT:** first check the sensor and data feed (a fault looks "
              "like a sudden error spike with normal weather), then check for unusual events such as severe "
              "weather, closures or holidays. Retrain only if the change is real and persists for more than "
              "one batch; a one-off weather event should be documented, not used to retrain immediately."]
    (OUT_DIR / "monitoring_report.md").write_text("\n".join(lines) + "\n")
    logger.info("Saved monitoring/monitoring_report.md")


def main() -> int:
    df = common.load_modelling_data()
    model = joblib.load(MODELS_DIR / "traffic_regressor_champion.joblib")
    df["predicted_volume"] = model.predict(df[FEATURES])
    error_ref = df[df["year"] == 2017]
    normals = df[df["year"].between(2016, 2017)]
    production = df[df["year"] == 2018]
    logger.info("Monitoring champion model: error reference 2017 (%d hours), feature normals 2016-2017 "
                "(%d hours), production 2018 (%d hours)", len(error_ref), len(normals), len(production))

    results = []
    for period, batch in production.groupby(production["date_time"].dt.to_period("M")):
        result = check_batch(str(period), batch, error_ref, normals)
        if result["status"] == "ALERT":
            result["worst_days"] = worst_days(batch)
        results.append(result)
    last = production[production["date_time"].dt.to_period("M") == production["date_time"].dt.to_period("M").max()]
    for scenario in ("sensor_fault", "temperature_shift"):
        results.append(check_batch(f"SIM {scenario}", simulate(last, model, scenario), error_ref, normals))

    real = [r for r in results if not r["batch"].startswith("SIM")]
    overall = "ALERT / Requires investigation" if any(r["status"] == "ALERT" for r in real) else "PASS / Normal"
    logger.info("Overall status (real batches): %s; simulated incidents raised %d alert(s)", overall,
                sum(r["status"] == "ALERT" for r in results if r["batch"].startswith("SIM")))
    dashboard(results, overall)
    report(results, overall)
    common.save_json({"overall_status": overall, "batches": results}, OUT_DIR / "monitoring_status.json")

    mlflow = common.setup_mlflow("traffic_monitoring")
    with mlflow.start_run(run_name="monthly_drift_check_2018"):
        mlflow.set_tags({"task": "monitoring", "overall_status": overall})
        mlflow.log_params({"reference": "2017", "mae_alert_ratio": MAE_ALERT_RATIO, "psi_alert": PSI_ALERT})
        for step, r in enumerate(results):
            mlflow.log_metrics({"mae_ratio": r["mae_ratio"], "max_psi": max(v for k, v in r.items()
                                                                            if k.startswith("psi_"))}, step=step)
        mlflow.log_artifact(str(OUT_DIR / "monitoring_dashboard.png"))
    return 0


if __name__ == "__main__":
    common.configure_logging()
    try:
        sys.exit(main())
    except Exception as exc:
        logger.error("Monitoring run failed: %s", exc, exc_info=True)
        sys.exit(1)

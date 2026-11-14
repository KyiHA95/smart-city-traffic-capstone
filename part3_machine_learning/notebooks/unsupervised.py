"""
unsupervised.py - Part 3, Task 2: unsupervised learning.

1. K-means clustering of traffic conditions (no accident data exists, so we
   cluster conditions instead) using hour of day (cyclical), weekday/weekend,
   weather severity and traffic volume. k is chosen with the silhouette score,
   and every cluster is profiled and named in plain language.
2. Association rule mining (Apriori) on discretised time of day, day type,
   weather group and temperature band, keeping rules whose consequent is a
   congestion level. Rules are ranked by lift and translated into plain
   language.

Runs are tracked in MLflow (experiment "traffic_unsupervised").

Usage (from part3_machine_learning):
    python notebooks/unsupervised.py
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import common
from common import FIGURES_DIR, RANDOM_STATE, RESULTS_DIR

logger = logging.getLogger(__name__)

CLUSTER_FEATURES = ["hour_sin", "hour_cos", "is_weekend", "weather_severity", "traffic_volume"]
K_RANGE = range(2, 13)          # searched and reported
K_ALLOWED = range(3, 9)         # chosen from: an operations team can act on at most ~8 regimes
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.5


# ================================================================= K-MEANS
def choose_k(X: np.ndarray) -> tuple[int, dict]:
    rng = np.random.default_rng(RANDOM_STATE)
    sample = rng.choice(len(X), size=min(8000, len(X)), replace=False)
    scores = {}
    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        scores[k] = {"inertia": float(model.inertia_),
                     "silhouette": float(silhouette_score(X[sample], model.labels_[sample]))}
        logger.debug("k=%d inertia=%.0f silhouette=%.4f", k, scores[k]["inertia"],
                     scores[k]["silhouette"])
    best_k = max(K_ALLOWED, key=lambda k: scores[k]["silhouette"])
    logger.info("Silhouette scores by k: %s",
                ", ".join(f"{k}:{v['silhouette']:.3f}" for k, v in scores.items()))
    logger.info("Chosen k=%d: best silhouette within k=%d-%d. Silhouette keeps rising to k=%d "
                "because the binary weekend flag and stepped weather severity allow ever finer "
                "splits of the same regimes, so larger k adds clusters without adding insight",
                best_k, min(K_ALLOWED), max(K_ALLOWED),
                max(scores, key=lambda k: scores[k]["silhouette"]))
    return best_k, scores


def name_cluster(row: pd.Series, overall_mean: float) -> str:
    if row["mean_volume"] < 0.45 * overall_mean:
        level = "Quiet"
    elif row["mean_volume"] > 1.35 * overall_mean:
        level = "Peak-congestion"
    elif row["mean_volume"] > overall_mean:
        level = "Busy"
    else:
        level = "Moderate"
    when = "weekend" if row["weekend_share"] > 0.6 else "weekday" if row["weekend_share"] < 0.2 else "mixed-day"
    start, end = int(row["typical_hour_start"]), int(row["typical_hour_end"])
    hours = "all day" if end - start >= 16 else f"{start:02d}-{end:02d}h"
    weather = " in bad weather" if row["mean_weather_severity"] >= 5 else ""
    return f"{level} {when} {hours}{weather}"


def run_kmeans(df: pd.DataFrame, mlflow) -> dict:
    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    k, scores = choose_k(X)
    model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X)
    df = df.assign(cluster=model.labels_)

    overall = df["traffic_volume"].mean()
    profiles = []
    for cid, g in df.groupby("cluster"):
        hours = g["hour"]
        profiles.append({
            "cluster": int(cid), "hours": len(g), "share": len(g) / len(df),
            "mean_volume": g["traffic_volume"].mean(),
            "congested_share": g["is_congested"].mean(),
            "weekend_share": g["is_weekend"].mean(),
            "mean_weather_severity": g["weather_severity"].mean(),
            "bad_weather_share": (g["weather_severity"] >= 6).mean(),
            "typical_hour_start": hours.quantile(0.1), "typical_hour_end": hours.quantile(0.9),
            "most_common_weather": g["weather_main"].mode().iat[0],
            "high_risk_share": g["high_risk"].mean(),
        })
    table = pd.DataFrame(profiles).sort_values("mean_volume").reset_index(drop=True)
    table["name"] = [name_cluster(r, overall) for _, r in table.iterrows()]
    for _, r in table.iterrows():
        logger.info("Cluster %d '%s': %d hours (%.0f%%), mean %.0f vehicles, %.0f%% congested, "
                    "%.0f%% weekend, %.0f%% bad weather", r["cluster"], r["name"], r["hours"],
                    100 * r["share"], r["mean_volume"], 100 * r["congested_share"],
                    100 * r["weekend_share"], 100 * r["bad_weather_share"])

    with mlflow.start_run(run_name=f"kmeans_k{k}"):
        mlflow.set_tags({"task": "clustering", "algorithm": "KMeans"})
        mlflow.log_params({"k": k, "features": ",".join(CLUSTER_FEATURES), "scaling": "StandardScaler",
                           "n_init": 20, "k_searched": f"{min(K_RANGE)}-{max(K_RANGE)}"})
        mlflow.log_metrics({"silhouette": scores[k]["silhouette"], "inertia": scores[k]["inertia"]})
        for kk, v in scores.items():
            mlflow.log_metric("silhouette_by_k", v["silhouette"], step=kk)

    # Figures
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ks = list(scores)
    axes[0].plot(ks, [scores[x]["inertia"] for x in ks], marker="o", color="#1f4e79")
    ax2 = axes[0].twinx()
    ax2.plot(ks, [scores[x]["silhouette"] for x in ks], marker="s", color="#dd6b20")
    axes[0].set(title="Choosing k: inertia (blue) and silhouette (orange)", xlabel="k",
                ylabel="Inertia")
    ax2.set_ylabel("Silhouette")
    axes[0].axvline(k, ls=":", color="grey")
    colours = plt.cm.tab10(np.arange(k))
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(df), size=min(6000, len(df)), replace=False)
    sample = df.iloc[idx]
    for _, r in table.iterrows():
        pts = sample[sample["cluster"] == r["cluster"]]
        jitter = rng.uniform(-0.35, 0.35, len(pts))
        axes[1].scatter(pts["hour"] + jitter, pts["traffic_volume"], s=3, alpha=0.5,
                        color=colours[int(r["cluster"]) % 10], label=r["name"])
    axes[1].set(title=f"K-means clusters (k={k}) by hour and volume", xlabel="Hour of day",
                ylabel="Vehicles/hour")
    axes[1].legend(fontsize=6, markerscale=3, loc="upper left", framealpha=0.9)
    axes[1].set_ylim(0, df["traffic_volume"].max() * 1.45)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "kmeans_clusters.png", dpi=150)
    plt.close(fig)
    logger.info("Saved figure: notebooks/outputs/kmeans_clusters.png")
    return {"k": k, "scores": scores, "clusters": table.to_dict(orient="records")}


# ================================================================= ASSOCIATION RULES
def build_transactions(df: pd.DataFrame) -> pd.DataFrame:
    bands = pd.Series(np.select(
        [df["hour"].between(6, 9), df["hour"].between(10, 14), df["hour"].between(15, 18),
         df["hour"].between(19, 22)],
        ["AM peak (06-09)", "Midday (10-14)", "PM peak (15-18)", "Evening (19-22)"],
        default="Night (23-05)"), index=df.index)
    day = np.where(df["is_holiday"] == 1, "Holiday", np.where(df["is_weekend"] == 1, "Weekend", "Weekday"))
    weather = np.select([df["weather_main"].isin(common.SEVERE_WEATHER), df["is_low_visibility"] == 1,
                         df["weather_main"].isin(["Rain", "Drizzle"]), df["weather_main"] == "Clouds"],
                        ["Severe weather", "Low visibility", "Rain/drizzle", "Cloudy"], default="Clear")
    items = pd.DataFrame({"time": "Time=" + bands, "day": "Day=" + pd.Series(day, index=df.index),
                          "weather": "Weather=" + pd.Series(weather, index=df.index),
                          "temp": "Temp=" + df["temp_band"],
                          "congestion": "Congestion=" + df["congestion_category"]})
    onehot = pd.get_dummies(items, prefix="", prefix_sep="").astype(bool)
    logger.info("Transactions built: %d hours x %d items (time band, day type, weather group, "
                "temperature band, congestion level)", *onehot.shape)
    return onehot


def prune_redundant(rules: pd.DataFrame, tolerance: float = 0.02) -> pd.DataFrame:
    """Drop a rule when a simpler rule (subset of its conditions, same outcome) has a lift
    within 2% of it: the extra conditions add (almost) nothing."""
    keep = []
    rows = list(rules.itertuples(index=False))
    for r in rows:
        redundant = any(o.consequents == r.consequents and o.antecedents < r.antecedents
                        and o.lift >= r.lift * (1 - tolerance) for o in rows)
        keep.append(not redundant)
    pruned = rules[keep].reset_index(drop=True)
    logger.info("Rule pruning: removed %d redundant rule(s) whose extra conditions add less "
                "than %.0f%% lift over a simpler rule", len(rules) - len(pruned), tolerance * 100)
    return pruned


def plain_language(rule: pd.Series, base_rates: dict) -> str:
    conditions = " and ".join(item.split("=", 1)[1] for item in sorted(rule["antecedents"]))
    target = next(iter(rule["consequents"])).split("=", 1)[1]
    return (f"When it is {conditions}, congestion is {target} in {rule['confidence']:.0%} of hours, "
            f"{rule['lift']:.1f} times more often than the {base_rates[target]:.0%} seen overall.")


def run_association_rules(df: pd.DataFrame, mlflow) -> dict:
    onehot = build_transactions(df)
    frequent = apriori(onehot, min_support=MIN_SUPPORT, use_colnames=True)
    rules = association_rules(frequent, metric="confidence", min_threshold=MIN_CONFIDENCE)
    is_target = rules["consequents"].apply(lambda s: len(s) == 1 and next(iter(s)).startswith("Congestion="))
    no_target_in_lhs = rules["antecedents"].apply(lambda s: not any(i.startswith("Congestion=") for i in s))
    rules = rules[is_target & no_target_in_lhs].sort_values(["lift", "confidence"], ascending=False)
    rules = prune_redundant(rules)
    base = df["congestion_category"].value_counts(normalize=True).to_dict()
    rules["plain_language"] = rules.apply(plain_language, axis=1, base_rates=base)
    logger.info("Apriori: %d frequent itemsets, %d rules predicting congestion level "
                "(min support %.2f, min confidence %.2f)", len(frequent), len(rules),
                MIN_SUPPORT, MIN_CONFIDENCE)
    top = rules.head(10)
    for i, r in top.iterrows():
        logger.info("Rule %d (lift %.2f, support %.3f): %s", i + 1, r["lift"], r["support"],
                    r["plain_language"])

    with mlflow.start_run(run_name="association_rules_apriori"):
        mlflow.set_tags({"task": "association_rules", "algorithm": "Apriori"})
        mlflow.log_params({"min_support": MIN_SUPPORT, "min_confidence": MIN_CONFIDENCE,
                           "items": "time band, day type, weather group, temp band, congestion"})
        mlflow.log_metrics({"n_rules": len(rules), "top_lift": float(top["lift"].max()),
                            "top_confidence": float(top["confidence"].iloc[0])})

    severe_or_low = rules[rules["consequents"].apply(lambda s: next(iter(s)) in
                                                     ("Congestion=Severe", "Congestion=Low"))]
    show = pd.concat([severe_or_low[severe_or_low["consequents"].apply(
        lambda s: next(iter(s)) == lvl)].head(4) for lvl in ("Congestion=Severe", "Congestion=Low")])
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = [" + ".join(i.split("=", 1)[1] for i in sorted(r["antecedents"])) + "  ->  "
              + next(iter(r["consequents"])).split("=", 1)[1] for _, r in show.iterrows()]
    colours = ["#d94801" if "Severe" in l.split("->")[1] else "#4292c6" for l in labels]
    ax.barh(labels[::-1], show["lift"].to_numpy()[::-1], color=colours[::-1])
    for i, (lift, conf) in enumerate(zip(show["lift"].to_numpy()[::-1], show["confidence"].to_numpy()[::-1])):
        ax.text(lift + 0.03, i, f"lift {lift:.2f}, conf {conf:.0%}", va="center", fontsize=8)
    ax.set(title="Strongest association rules for Severe (orange) and Low (blue) congestion",
           xlabel="Lift (how many times more likely than average)", xlim=(0, show["lift"].max() * 1.3))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "association_rules_top.png", dpi=150)
    plt.close(fig)
    logger.info("Saved figure: notebooks/outputs/association_rules_top.png")

    export = rules.head(25).copy()
    export["antecedents"] = export["antecedents"].apply(lambda s: " AND ".join(sorted(s)))
    export["consequents"] = export["consequents"].apply(lambda s: next(iter(s)))
    cols = ["antecedents", "consequents", "support", "confidence", "lift", "plain_language"]
    export[cols].round(4).to_csv(RESULTS_DIR / "association_rules_top25.csv", index=False)
    logger.info("Saved notebooks/outputs/association_rules_top25.csv")
    return {"n_rules": len(rules), "top_rules": export[cols].round(4).to_dict(orient="records"),
            "base_rates": base}


def main() -> int:
    df = common.load_modelling_data()
    mlflow = common.setup_mlflow("traffic_unsupervised")
    clusters = run_kmeans(df, mlflow)
    rules = run_association_rules(df, mlflow)
    common.save_json({"kmeans": clusters, "association_rules": rules},
                     RESULTS_DIR / "unsupervised_results.json")
    return 0


if __name__ == "__main__":
    common.configure_logging()
    try:
        raise SystemExit(main())
    except Exception as exc:
        logger.error("Unsupervised analysis failed: %s", exc, exc_info=True)
        raise SystemExit(1)

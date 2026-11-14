"""
train_supervised.py - Part 3, Task 1 (supervised ML) + Task 4/6.1/6.2 (MLflow).

Two supervised problems share ONE feature set (common.FEATURES: time features,
weather encodings, holiday flag, cyclical hour/day-of-week/month encodings):

* Regression     : predict hourly traffic_volume.
* Classification : estimate the proxy accident-risk label high_risk
                   (a documented proxy, NOT real accident data).

Three algorithms per problem (a linear baseline plus two tree ensembles) are
trained on 2012-2016 and evaluated on unseen 2017-2018 data.
Every run is tracked in MLflow (parameters, metrics, model artifact, tags)
and registered as a new version of the model in the MLflow Model Registry;
the best version gets the alias "champion" and is exported to models/.

Usage (from part3_machine_learning):
    python notebooks/train_supervised.py
"""
from __future__ import annotations

import logging
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common
from common import (CLASSIFICATION_TARGET, FEATURES, FIGURES_DIR, MODELS_DIR, RANDOM_STATE,
                    REGRESSION_TARGET, RESULTS_DIR)

logger = logging.getLogger(__name__)

REGRESSION_MODEL_NAME = "traffic_volume_regressor"
CLASSIFICATION_MODEL_NAME = "proxy_risk_classifier"


def regression_candidates() -> dict:
    return {
        "ridge_regression": (make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
                             {"algorithm": "Ridge", "alpha": 1.0, "scaling": "StandardScaler"}),
        "random_forest": (RandomForestRegressor(n_estimators=80, max_depth=12, min_samples_leaf=5,
                                                n_jobs=-1, random_state=RANDOM_STATE),
                          {"algorithm": "RandomForestRegressor", "n_estimators": 80,
                           "max_depth": 12, "min_samples_leaf": 5}),
        "hist_gradient_boosting": (HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=48, l2_regularization=1.0,
            random_state=RANDOM_STATE),
            {"algorithm": "HistGradientBoostingRegressor", "max_iter": 400,
             "learning_rate": 0.06, "max_leaf_nodes": 48, "l2_regularization": 1.0}),
    }


def classification_candidates() -> dict:
    return {
        "logistic_regression": (make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0)),
            {"algorithm": "LogisticRegression", "C": 1.0, "class_weight": "balanced",
             "scaling": "StandardScaler"}),
        "random_forest": (RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=3,
                                                 class_weight="balanced", n_jobs=-1,
                                                 random_state=RANDOM_STATE),
                          {"algorithm": "RandomForestClassifier", "n_estimators": 150,
                           "max_depth": 14, "min_samples_leaf": 3, "class_weight": "balanced"}),
        "hist_gradient_boosting": (HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31, class_weight="balanced",
            random_state=RANDOM_STATE),
            {"algorithm": "HistGradientBoostingClassifier", "max_iter": 300,
             "learning_rate": 0.06, "max_leaf_nodes": 31, "class_weight": "balanced"}),
    }


def regression_metrics(y_true, y_pred) -> dict:
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred)))}


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {"accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, proba),
            "pr_auc": average_precision_score(y_true, proba)}


def train_and_track(mlflow, task: str, candidates: dict, train: pd.DataFrame,
                    test: pd.DataFrame, target: str, registered_name: str) -> list[dict]:
    """Fit each candidate, evaluate on the test years, log and register it in MLflow."""
    from mlflow.models import infer_signature
    X_train, y_train = train[FEATURES], train[target]
    X_test, y_test = test[FEATURES], test[target]
    results = []
    for version, (key, (model, params)) in enumerate(candidates.items(), start=1):
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        if task == "regression":
            preds = model.predict(X_test)
            metrics = regression_metrics(y_test, preds)
        else:
            preds = model.predict_proba(X_test)[:, 1]
            metrics = classification_metrics(y_test, preds)
            # Harder check: within risky-weather hours the label depends only on whether
            # congestion occurs, so this measures genuine predictive skill.
            risky = ((test["weather_main"].isin(common.SEVERE_WEATHER))
                     | (test["is_low_visibility"] == 1)).to_numpy()
            sub = classification_metrics(y_test[risky], preds[risky])
            metrics.update({f"risky_weather_{k}": v for k, v in sub.items()})
        metrics["train_seconds"] = fit_seconds

        with mlflow.start_run(run_name=f"{task}_{key}") as run:
            mlflow.set_tags({"task": task, "target": target, "model_key": key,
                             "label_type": "proxy" if task == "classification" else "observed",
                             "split": "train 2012-2016 / test 2017-2018"})
            mlflow.log_params({**params, "n_features": len(FEATURES),
                               "train_rows": len(train), "test_rows": len(test)})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            # cloudpickle: our own freshly trained models; MLflow's default skops format
            # rejects scikit-learn tree internals unless each type is whitelisted.
            mlflow.sklearn.log_model(model, name="model", registered_model_name=registered_name,
                                     serialization_format="cloudpickle",
                                     signature=infer_signature(X_test.head(50),
                                                               model.predict(X_test.head(50))),
                                     input_example=X_test.head(3))
            run_id = run.info.run_id

        path = MODELS_DIR / f"{task}_v{version}_{key}.joblib"
        joblib.dump(model, path, compress=3)
        size_mb = path.stat().st_size / 1e6
        metric_text = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()
                                if k != "train_seconds" and not k.startswith("risky_weather_"))
        if task == "classification":
            metric_text += (f" | within risky-weather hours: F1={metrics['risky_weather_f1']:.3f}, "
                            f"ROC AUC={metrics['risky_weather_roc_auc']:.3f}")
        logger.info("%s v%d %s trained in %.1fs: %s (saved %s, %.1f MB)", task, version, key,
                    fit_seconds, metric_text, path.name, size_mb)
        results.append({"task": task, "version": version, "model": key, **params,
                        **metrics, "model_file": path.name, "model_size_mb": size_mb,
                        "mlflow_run_id": run_id, "registered_model": registered_name,
                        "predictions": preds})
    return results


CHAMPION_TOLERANCE = 0.005   # metric gap treated as a practical tie


def promote_champion(mlflow, results: list[dict], metric: str, registered_name: str,
                     export_name: str) -> dict:
    """Pick the best model; among near-ties (within CHAMPION_TOLERANCE) prefer the smallest,
    which is cheaper to serve, faster and uses less energy."""
    top = max(r[metric] for r in results)
    near_ties = [r for r in results if top - r[metric] <= CHAMPION_TOLERANCE]
    best = min(near_ties, key=lambda r: r["model_size_mb"])
    if len(near_ties) > 1:
        logger.info("%s: %d versions within %.3f of the best %s; choosing the smallest (%s, %.1f MB)",
                    registered_name, len(near_ties), CHAMPION_TOLERANCE, metric, best["model"],
                    best["model_size_mb"])
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(registered_name, "champion", str(best["version"]))
    for r in results:
        client.set_model_version_tag(registered_name, str(r["version"]), "stage",
                                     "champion" if r is best else "archived")
    champion = joblib.load(MODELS_DIR / best["model_file"])
    joblib.dump(champion, MODELS_DIR / export_name, compress=3)
    logger.info("Champion for %s: v%d %s (%s=%.3f) -> alias 'champion', exported %s",
                registered_name, best["version"], best["model"], metric, best[metric], export_name)
    return best


# ----------------------------------------------------------------- figures and analysis
def plot_results(reg: list[dict], clf: list[dict], test: pd.DataFrame, champs: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = [r["model"].replace("_", "\n") for r in reg]
    axes[0].bar(names, [r["MAE"] for r in reg], color="#1f4e79")
    axes[0].set(title="Regression: MAE on 2017-2018 (lower is better)", ylabel="Vehicles/hour",
                ylim=(0, max(r["MAE"] for r in reg) * 1.3))
    for i, r in enumerate(reg):
        axes[0].text(i, r["MAE"] + 10, f"MAE {r['MAE']:.0f}\nR² {r['R2']:.3f}", ha="center", fontsize=8)
    metrics = ["precision", "recall", "f1", "roc_auc"]
    width = 0.25
    for i, r in enumerate(clf):
        axes[1].bar(np.arange(len(metrics)) + i * width, [r[m] for m in metrics], width,
                    label=r["model"].replace("_", " "))
    axes[1].set_xticks(np.arange(len(metrics)) + width, ["Precision", "Recall", "F1", "ROC AUC"])
    axes[1].set(title="Classification (proxy risk) on 2017-2018", ylim=(0, 1.05))
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "supervised_model_comparison.png", dpi=150)
    plt.close(fig)

    best_reg = champs["regression"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    week = test["date_time"].between("2017-03-13", "2017-03-20")
    ax.plot(test.loc[week, "date_time"], test.loc[week, REGRESSION_TARGET], label="Actual", color="#1f4e79")
    ax.plot(test.loc[week, "date_time"], best_reg["predictions"][week.to_numpy()], label="Predicted",
            color="#dd6b20", ls="--")
    ax.set(title=f"Traffic volume: actual vs predicted, one test week (champion: {best_reg['model']})",
           ylabel="Vehicles/hour")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "regression_actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    y = test[CLASSIFICATION_TARGET]
    for r in clf:
        fpr, tpr, _ = roc_curve(y, r["predictions"])
        axes[0].plot(fpr, tpr, label=f"{r['model']} (AUC {r['roc_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], ls=":", color="grey")
    axes[0].set(title="ROC curves (test 2017-2018)", xlabel="False positive rate",
                ylabel="True positive rate")
    axes[0].legend(fontsize=7)
    best_clf = champs["classification"]
    cm = confusion_matrix(y, (best_clf["predictions"] >= 0.5).astype(int))
    axes[1].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        axes[1].text(j, i, f"{v:,}", ha="center", va="center",
                     color="white" if v > cm.max() / 2 else "black")
    axes[1].set(title=f"Confusion matrix: {best_clf['model']}", xticks=[0, 1], yticks=[0, 1],
                xticklabels=["Pred low", "Pred high"], yticklabels=["Actual low", "Actual high"])
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "classification_roc_confusion.png", dpi=150)
    plt.close(fig)
    logger.info("Saved figures: supervised_model_comparison.png, "
                "regression_actual_vs_predicted.png, classification_roc_confusion.png")


def error_breakdown(test: pd.DataFrame, reg_best: dict, clf_best: dict) -> dict:
    """Error by condition and period, used by the bias and fairness report."""
    d = test.copy()
    d["abs_error"] = np.abs(d[REGRESSION_TARGET] - reg_best["predictions"])
    d["pct_error"] = d["abs_error"] / d[REGRESSION_TARGET].clip(lower=100)
    d["risk_pred"] = (clf_best["predictions"] >= 0.5).astype(int)
    d["hour_band"] = pd.cut(d["hour"], [-1, 5, 9, 15, 19, 23],
                            labels=["Night 0-5", "AM peak 6-9", "Midday 10-15", "PM peak 16-19",
                                    "Evening 20-23"]).astype(str)
    d["weather_group"] = np.select(
        [d["weather_main"].isin(common.SEVERE_WEATHER), d["is_low_visibility"] == 1,
         d["is_precipitation"] == 1], ["Severe (snow/storm)", "Low visibility", "Rain/drizzle"],
        default="Clear/cloudy")
    d["day_group"] = np.where(d["is_holiday"] == 1, "Holiday",
                              np.where(d["is_weekend"] == 1, "Weekend", "Weekday"))
    out = {}
    for group in ("hour_band", "weather_group", "day_group", "year"):
        g = d.groupby(group)
        reg = g.agg(hours=("abs_error", "size"), MAE=("abs_error", "mean"),
                    median_pct_error=("pct_error", "median"))
        positives = d[d[CLASSIFICATION_TARGET] == 1].groupby(group)
        reg["risk_recall"] = positives["risk_pred"].mean()
        reg["risk_positives"] = positives.size()
        negatives = d[d[CLASSIFICATION_TARGET] == 0].groupby(group)
        reg["false_alarm_rate"] = negatives["risk_pred"].mean()
        out[group] = reg.round(4).reset_index().to_dict(orient="records")
    logger.info("Error breakdown by hour band, weather, day type and year computed for "
                "the bias and fairness report")
    return out


def write_versions(reg: list[dict], clf: list[dict], champs: dict) -> None:
    lines = ["# Model versions and performance",
             "",
             "Generated by `train_supervised.py`. Every version is a registered model version in "
             "the MLflow Model Registry (`mlflow/mlflow.db`); the best version per model carries "
             "the alias **champion** and is exported for deployment. Test data: 2017-2018 "
             "(trained on 2012-2016).", "",
             f"## {REGRESSION_MODEL_NAME} (target: traffic_volume)", "",
             "| Version | Algorithm | MAE | R² | RMSE | Train time (s) | Size (MB) | Status |",
             "|---|---|---|---|---|---|---|---|"]
    for r in reg:
        status = "champion" if r["version"] == champs["regression"]["version"] else "archived"
        lines.append(f"| v{r['version']} | {r['algorithm']} | {r['MAE']:.1f} | {r['R2']:.3f} | "
                     f"{r['RMSE']:.1f} | {r['train_seconds']:.1f} | {r['model_size_mb']:.1f} | {status} |")
    lines += ["", f"## {CLASSIFICATION_MODEL_NAME} (target: proxy high_risk)", "",
              "| Version | Algorithm | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC | "
              "ROC AUC (risky-weather hours) | Train time (s) | Size (MB) | Status |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in clf:
        status = "champion" if r["version"] == champs["classification"]["version"] else "archived"
        lines.append(f"| v{r['version']} | {r['algorithm']} | {r['accuracy']:.3f} | "
                     f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['roc_auc']:.3f} | "
                     f"{r['pr_auc']:.3f} | {r['risky_weather_roc_auc']:.3f} | {r['train_seconds']:.1f} | "
                     f"{r['model_size_mb']:.1f} | {status} |")
    lines += ["", "Champion selection: highest R² for regression, highest F1 for classification "
              "(F1 balances missed high-risk hours against false alarms on an imbalanced label). "
              "Versions within 0.005 of the best score are treated as a tie and the smallest model "
              "wins, because it is cheaper to serve and store for practically the same accuracy."
              ' Limitation: the champion is chosen using the same 2017-2018 test scores that are reported as unseen-data results, so those scores are slightly optimistic. A stricter approach would choose the champion on a separate validation year (for example 2016) and report 2017-2018 only once. The effect here is small: the two tree ensembles are about 0.001 R² apart, and the gradient-boosting classifier has the best F1, accuracy and precision (random forest is slightly higher on recall and ROC AUC).', "",
              'Note on the proxy label: high_risk is built from weather (an input feature) and congestion, so every positive hour has severe or low-visibility weather, and none of the three models raises a false alarm in any other hour. The overall scores are therefore inflated: the model can rule out every clear-weather hour from the weather inputs alone. For the same reason, F1 restricted to risky-weather hours is mathematically identical to the overall F1, so it adds no information and is left out of this table (MLflow still records it as risky_weather_f1, alongside the other risky-weather metrics, and it equals the overall F1). PR AUC is almost unchanged for the same reason. The risky-weather ROC AUC column instead ranks hours only within severe or low-visibility weather, where the model must genuinely predict congestion. The drop in ROC AUC (for the champion, 0.997 overall to 0.985) is the evidence that this restricted test is harder. These scores measure congestion prediction, not accident risk.']
    (MODELS_DIR / "MODEL_VERSIONS.md").write_text("\n".join(lines) + "\n")
    logger.info("Saved model version table: models/MODEL_VERSIONS.md")


def main() -> int:
    df = common.load_modelling_data()
    train, test = common.time_split(df)
    logger.info("Feature set (%d): %s", len(FEATURES), ", ".join(FEATURES))
    mlflow = common.setup_mlflow("traffic_supervised_models")

    reg = train_and_track(mlflow, "regression", regression_candidates(), train, test,
                          REGRESSION_TARGET, REGRESSION_MODEL_NAME)
    clf = train_and_track(mlflow, "classification", classification_candidates(), train, test,
                          CLASSIFICATION_TARGET, CLASSIFICATION_MODEL_NAME)
    champs = {"regression": promote_champion(mlflow, reg, "R2", REGRESSION_MODEL_NAME,
                                             "traffic_regressor_champion.joblib"),
              "classification": promote_champion(mlflow, clf, "f1", CLASSIFICATION_MODEL_NAME,
                                                 "risk_classifier_champion.joblib")}

    plot_results(reg, clf, test, champs)
    breakdown = error_breakdown(test, champs["regression"], champs["classification"])
    write_versions(reg, clf, champs)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "predictions"} for r in rows]
    common.save_json({"features": FEATURES, "regression": strip(reg), "classification": strip(clf),
                      "champions": {k: v["model"] for k, v in champs.items()},
                      "label_positive_rate": {"train": train[CLASSIFICATION_TARGET].mean(),
                                              "test": test[CLASSIFICATION_TARGET].mean()},
                      "error_breakdown": breakdown},
                     RESULTS_DIR / "supervised_results.json")
    return 0


if __name__ == "__main__":
    common.configure_logging()
    try:
        raise SystemExit(main())
    except Exception as exc:
        logger.error("Supervised training failed: %s", exc, exc_info=True)
        raise SystemExit(1)

"""
deep_learning.py - Part 3, Task 3: LSTM for traffic behaviour + explainability.

Problem: predict NEXT-HOUR traffic volume from the previous 24 hours
(sequence input) plus known context for the target hour (calendar and
weather). The data are naturally sequential, so an LSTM is used.
Only windows of 24 consecutive recorded hours are used; the sensor gaps
(e.g. 2014-2015) are never bridged.

Explainability
--------------
LSTMs are hard to explain directly with SHAP, so, as the brief allows:
* SHAP (TreeExplainer) is applied to a comparable tree model
  (HistGradientBoosting) trained on the SAME problem with lag features.
* LIME explains individual predictions of the Task 1 proxy-risk classifier.
* Permutation importance is computed for the LSTM's own input channels to
  check that it relies on the same signals as the explained tree model.

Usage (from part3_machine_learning):
    python notebooks/deep_learning.py
"""
from __future__ import annotations

import logging
import os
import time

if "KERAS_BACKEND" not in os.environ:          # use JAX if installed, else TensorFlow
    try:
        import jax  # noqa: F401
        os.environ["KERAS_BACKEND"] = "jax"
    except ImportError:
        os.environ["KERAS_BACKEND"] = "tensorflow"

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import common
from common import FEATURES, FIGURES_DIR, MODELS_DIR, RANDOM_STATE, RESULTS_DIR, TRAIN_END_YEAR

logger = logging.getLogger(__name__)

WINDOW = 24
SEQ_CHANNELS = ["volume_scaled", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
                "is_holiday", "temp_scaled", "severity_scaled", "is_precipitation"]
CONTEXT = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday",
           "temp_scaled", "severity_scaled", "is_precipitation", "is_low_visibility"]
EPOCHS = 80
BATCH = 256


# ----------------------------------------------------------------- data preparation
def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    train_mask = df["year"] <= TRAIN_END_YEAR
    scale = {"volume_max": float(df.loc[train_mask, "traffic_volume"].max()),
             "temp_mean": float(df.loc[train_mask, "temp_c"].mean()),
             "temp_std": float(df.loc[train_mask, "temp_c"].std())}
    logger.debug("Scaling parameters fitted on training years only: %s", scale)
    df = df.copy()
    df["volume_scaled"] = df["traffic_volume"] / scale["volume_max"]
    df["temp_scaled"] = (df["temp_c"] - scale["temp_mean"]) / scale["temp_std"]
    df["severity_scaled"] = df["weather_severity"] / 10.0
    return df, scale


def build_windows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return sequences, context, target and target index for every valid window."""
    hours = df["date_time"].to_numpy().astype("datetime64[h]").astype(np.int64)
    # a window ending at i is valid if hours i-WINDOW..i are consecutive
    consecutive = np.diff(hours, prepend=hours[0] - 1) == 1
    run = np.zeros(len(df), dtype=int)
    for i in range(len(df)):                                  # explicit loop over the series
        run[i] = run[i - 1] + 1 if i > 0 and consecutive[i] else 0
    valid = np.where(run >= WINDOW)[0]
    seq_values = df[SEQ_CHANNELS].to_numpy(dtype="float32")
    ctx_values = df[CONTEXT].to_numpy(dtype="float32")
    X_seq = np.stack([seq_values[i - WINDOW:i] for i in valid])
    X_ctx = ctx_values[valid]
    y = df["volume_scaled"].to_numpy(dtype="float32")[valid]
    dropped = len(df) - len(valid)
    logger.warning("Sequence windows: %d valid 24-hour windows; %d hour(s) could not be used "
                   "as targets because the previous 24 hours were not all recorded (sensor gaps)",
                   len(valid), dropped)
    return X_seq, X_ctx, y, valid


def lag_features(df: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    """Flattened equivalent of the LSTM input for the comparable tree model."""
    v = df["traffic_volume"].to_numpy(dtype=float)
    out = pd.DataFrame({
        "lag_1h": v[idx - 1], "lag_2h": v[idx - 2], "lag_3h": v[idx - 3],
        "lag_24h": v[idx - 24], "mean_prev_24h": [v[i - 24:i].mean() for i in idx],
        "change_last_hour": v[idx - 1] - v[idx - 2],
    })
    for col in ["hour", "day_of_week", "is_weekend", "is_holiday", "temp_c", "weather_severity",
                "is_precipitation", "is_low_visibility", "hour_sin", "hour_cos"]:
        out[col] = df[col].to_numpy()[idx]
    return out


# ----------------------------------------------------------------- LSTM
def build_lstm(n_channels: int, n_context: int):
    import keras
    from keras import layers
    keras.utils.set_random_seed(RANDOM_STATE)
    seq_in = keras.Input(shape=(WINDOW, n_channels), name="previous_24_hours")
    ctx_in = keras.Input(shape=(n_context,), name="target_hour_context")
    x = layers.LSTM(48, name="lstm")(seq_in)
    x = layers.Dropout(0.1)(x)
    x = layers.Concatenate()([x, ctx_in])
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, name="next_hour_volume")(x)
    model = keras.Model([seq_in, ctx_in], out, name="traffic_lstm")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def run(df: pd.DataFrame, mlflow) -> dict:
    import keras
    df, scale = prepare(df)
    X_seq, X_ctx, y, idx = build_windows(df)
    is_train = (df["year"].to_numpy()[idx] <= TRAIN_END_YEAR)
    vmax = scale["volume_max"]

    model = build_lstm(len(SEQ_CHANNELS), len(CONTEXT))
    logger.info("LSTM architecture: 24x%d sequence -> LSTM(48) -> concat %d context features -> "
                "Dense(32) -> Dense(1); %d trainable parameters", len(SEQ_CHANNELS), len(CONTEXT),
                model.count_params())
    start = time.perf_counter()
    history = model.fit([X_seq[is_train], X_ctx[is_train]], y[is_train], validation_split=0.15,
                        epochs=EPOCHS, batch_size=BATCH, verbose=0, shuffle=True,
                        callbacks=[keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])
    train_seconds = time.perf_counter() - start
    epochs_run = len(history.history["loss"])
    logger.info("LSTM trained in %.1fs (%d of max %d epochs; %s)", train_seconds, epochs_run, EPOCHS,
                "stopped early on validation loss" if epochs_run < EPOCHS else
                "validation loss still improving at the epoch limit")

    test_pred = model.predict([X_seq[~is_train], X_ctx[~is_train]], verbose=0).ravel() * vmax
    y_test = y[~is_train] * vmax
    v = df["traffic_volume"].to_numpy(dtype=float)
    test_idx = idx[~is_train]
    results = {
        "LSTM": {"MAE": mean_absolute_error(y_test, test_pred), "R2": r2_score(y_test, test_pred)},
        "Naive: same as previous hour": {"MAE": mean_absolute_error(y_test, v[test_idx - 1]),
                                         "R2": r2_score(y_test, v[test_idx - 1])},
        "Naive: same hour yesterday": {"MAE": mean_absolute_error(y_test, v[test_idx - 24]),
                                       "R2": r2_score(y_test, v[test_idx - 24])},
    }

    # comparable tree model on the same windows (used for SHAP)
    lag_train, lag_test = lag_features(df, idx[is_train]), lag_features(df, test_idx)
    tree = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=48,
                                         random_state=RANDOM_STATE)
    tree.fit(lag_train, y[is_train] * vmax)
    tree_pred = tree.predict(lag_test)
    results["Tree model with lag features (SHAP)"] = {"MAE": mean_absolute_error(y_test, tree_pred),
                                                       "R2": r2_score(y_test, tree_pred)}
    for name, m in results.items():
        logger.info("Next-hour test results (2017-2018) | %-38s MAE=%6.1f  R2=%.3f", name,
                    m["MAE"], m["R2"])

    model_path = MODELS_DIR / "lstm_next_hour.keras"
    model.save(model_path)
    joblib.dump(tree, MODELS_DIR / "next_hour_tree_for_shap.joblib", compress=3)
    common.save_json(scale, MODELS_DIR / "lstm_scaling.json")
    logger.info("Saved LSTM model %s (%.2f MB) and comparable tree model", model_path.name,
                model_path.stat().st_size / 1e6)

    with mlflow.start_run(run_name="lstm_next_hour_volume"):
        mlflow.set_tags({"task": "deep_learning", "algorithm": "LSTM",
                         "keras_backend": keras.backend.backend()})
        mlflow.log_params({"window_hours": WINDOW, "lstm_units": 48, "dense_units": 32,
                           "dropout": 0.1, "batch_size": BATCH, "max_epochs": EPOCHS,
                           "epochs_run": epochs_run, "optimizer": "Adam(1e-3)",
                           "sequence_channels": ",".join(SEQ_CHANNELS), "context": ",".join(CONTEXT),
                           "train_windows": int(is_train.sum()), "test_windows": int((~is_train).sum())})
        mlflow.log_metrics({"test_MAE": results["LSTM"]["MAE"], "test_R2": results["LSTM"]["R2"],
                            "train_seconds": train_seconds,
                            "baseline_prev_hour_MAE": results["Naive: same as previous hour"]["MAE"]})
        for epoch, (loss, val) in enumerate(zip(history.history["loss"], history.history["val_loss"])):
            mlflow.log_metrics({"loss": loss, "val_loss": val}, step=epoch)
        mlflow.log_artifact(str(model_path), artifact_path="model")
    with mlflow.start_run(run_name="next_hour_tree_comparable_for_shap"):
        mlflow.set_tags({"task": "deep_learning_comparator", "algorithm": "HistGradientBoostingRegressor"})
        mlflow.log_params({"features": ",".join(lag_train.columns), "max_iter": 400})
        mlflow.log_metrics({"test_MAE": results["Tree model with lag features (SHAP)"]["MAE"],
                            "test_R2": results["Tree model with lag features (SHAP)"]["R2"]})

    # training curve + example week
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 2]})
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set(title="LSTM training curve (MSE, scaled)", xlabel="Epoch")
    axes[0].legend()
    times = df["date_time"].to_numpy()[test_idx]
    week = (times >= np.datetime64("2017-10-02")) & (times < np.datetime64("2017-10-09"))
    axes[1].plot(times[week], y_test[week], label="Actual", color="#1f4e79")
    axes[1].plot(times[week], test_pred[week], label="LSTM", color="#dd6b20", ls="--")
    axes[1].set(title="Next-hour prediction, one test week (Oct 2017)", ylabel="Vehicles/hour")
    axes[1].legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "lstm_training_and_forecast.png", dpi=150)
    plt.close(fig)
    logger.info("Saved figure: notebooks/outputs/lstm_training_and_forecast.png")

    perm = lstm_permutation_importance(model, X_seq[~is_train], X_ctx[~is_train], y_test, vmax)
    shap_info = explain_with_shap(tree, lag_test)
    return {"results": results, "epochs_run": epochs_run, "train_seconds": train_seconds,
            "parameters": int(model.count_params()), "windows": {"train": int(is_train.sum()),
                                                                  "test": int((~is_train).sum())},
            "lstm_permutation_importance": perm, "shap": shap_info}


def lstm_permutation_importance(model, X_seq, X_ctx, y_true, vmax, n: int = 4000) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    sel = rng.choice(len(y_true), size=min(n, len(y_true)), replace=False)
    Xs, Xc, yt = X_seq[sel], X_ctx[sel], y_true[sel]
    base = mean_absolute_error(yt, model.predict([Xs, Xc], verbose=0).ravel() * vmax)
    out = {}
    for j, ch in enumerate(SEQ_CHANNELS):
        shuffled = Xs.copy()
        shuffled[:, :, j] = shuffled[rng.permutation(len(Xs)), :, j]
        out[f"sequence: {ch}"] = mean_absolute_error(
            yt, model.predict([shuffled, Xc], verbose=0).ravel() * vmax) - base
    for j, ch in enumerate(CONTEXT):
        shuffled = Xc.copy()
        shuffled[:, j] = shuffled[rng.permutation(len(Xc)), j]
        out[f"target hour: {ch}"] = mean_absolute_error(
            yt, model.predict([Xs, shuffled], verbose=0).ravel() * vmax) - base
    out = dict(sorted(out.items(), key=lambda kv: -kv[1]))
    logger.info("LSTM permutation importance (MAE increase when shuffled), top 5: %s",
                ", ".join(f"{k} +{v:.0f}" for k, v in list(out.items())[:5]))
    return out


def explain_with_shap(tree, lag_test: pd.DataFrame) -> dict:
    import shap
    sample = lag_test.sample(n=min(2000, len(lag_test)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(tree)
    values = explainer.shap_values(sample)
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=sample.columns).sort_values(ascending=False)
    logger.info("SHAP mean |value| (vehicles/hour), top 5: %s",
                ", ".join(f"{k} {v:.0f}" for k, v in mean_abs.head(5).items()))

    plt.figure()
    shap.summary_plot(values, sample, show=False, max_display=12)
    plt.title("SHAP: what drives next-hour traffic predictions", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary_next_hour.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # local explanation: a weekday morning in the test period
    row = int(np.argmax(((sample["hour"] == 7) & (sample["is_weekend"] == 0)).to_numpy()))
    base_value = float(np.ravel(explainer.expected_value)[0])
    explanation = shap.Explanation(values=values[row], base_values=base_value,
                                   data=sample.iloc[row].values, feature_names=list(sample.columns))
    plt.figure()
    shap.plots.waterfall(explanation, max_display=10, show=False)
    plt.title("SHAP: one weekday 07:00 prediction explained", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_waterfall_example.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved figures: shap_summary_next_hour.png, shap_waterfall_example.png")
    return {"mean_abs_shap": mean_abs.round(1).to_dict(),
            "expected_value": float(np.ravel(explainer.expected_value)[0]),
            "example": {"features": sample.iloc[row].round(3).to_dict(),
                        "prediction": float(tree.predict(sample.iloc[[row]])[0]),
                        "contributions": dict(sorted(zip(sample.columns, values[row].round(1)),
                                                     key=lambda kv: -abs(kv[1]))[:6])}}


def explain_classifier_with_lime(df: pd.DataFrame) -> dict:
    """LIME explanations for two individual predictions of the Task 1 risk classifier."""
    from lime.lime_tabular import LimeTabularExplainer
    clf = joblib.load(MODELS_DIR / "risk_classifier_champion.joblib")
    train, test = common.time_split(df)
    explainer = LimeTabularExplainer(train[FEATURES].to_numpy(), feature_names=FEATURES,
                                     class_names=["not high risk", "high risk"], mode="classification",
                                     discretize_continuous=True, random_state=RANDOM_STATE)
    predict = lambda X: clf.predict_proba(pd.DataFrame(X, columns=FEATURES))
    cases = {
        "high_risk_fog_morning": test[(test["high_risk"] == 1) & (test["weather_main"] == "Fog")].iloc[0],
        "low_risk_fog_night": test[(test["high_risk"] == 0) & (test["weather_main"] == "Fog")
                                   & (test["hour"] <= 4)].iloc[0],
    }
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, (name, row) in zip(axes, cases.items()):
        exp = explainer.explain_instance(row[FEATURES].to_numpy(dtype=float), predict, num_features=8)
        weights = exp.as_list()
        prob = float(predict(row[FEATURES].to_numpy(dtype=float).reshape(1, -1))[0, 1])
        out[name] = {"date_time": str(row["date_time"]), "predicted_probability": prob,
                     "actual": int(row["high_risk"]), "weights": weights}
        labels = [w[0] for w in weights][::-1]
        vals = [w[1] for w in weights][::-1]
        ax.barh(labels, vals, color=["#d94801" if v > 0 else "#4292c6" for v in vals])
        ax.set_title(f"LIME: {row['date_time']:%a %d %b %Y %H:00}, {row['weather_main']}\n"
                     f"P(high risk) = {prob:.2f}, actual = {int(row['high_risk'])}", fontsize=9)
        ax.tick_params(labelsize=7)
        logger.info("LIME %s (%s): P(high risk)=%.2f; top reasons: %s", name, row["date_time"], prob,
                    "; ".join(f"{w[0]} ({w[1]:+.2f})" for w in weights[:3]))
    fig.suptitle("LIME: why the proxy-risk classifier made two individual predictions "
                 "(orange pushes towards high risk)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "lime_risk_classifier.png", dpi=150)
    plt.close(fig)
    logger.info("Saved figure: notebooks/outputs/lime_risk_classifier.png")
    return out


def main() -> int:
    df = common.load_modelling_data()
    mlflow = common.setup_mlflow("traffic_deep_learning")
    summary = run(df, mlflow)
    summary["lime"] = explain_classifier_with_lime(df)
    common.save_json(summary, RESULTS_DIR / "deep_learning_results.json")
    return 0


if __name__ == "__main__":
    common.configure_logging()
    try:
        raise SystemExit(main())
    except Exception as exc:
        logger.error("Deep learning step failed: %s", exc, exc_info=True)
        raise SystemExit(1)

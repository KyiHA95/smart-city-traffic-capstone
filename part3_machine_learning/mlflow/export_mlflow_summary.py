"""
export_mlflow_summary.py - write a human-readable summary of every MLflow
experiment, run, parameter, metric and registered model version
(mlflow/MLFLOW_SUMMARY.md), so the tracking results can be reviewed on
GitHub without starting the MLflow UI.

Usage (from part3_machine_learning):
    python mlflow/export_mlflow_summary.py

To browse the full tracking data interactively (from part3_machine_learning):
    mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
then open http://127.0.0.1:5000
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import common  # noqa: E402
from common import MLFLOW_DIR

logger = logging.getLogger(__name__)
KEY_METRICS = ["MAE", "R2", "RMSE", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
               "risky_weather_f1", "silhouette", "n_rules", "top_lift", "test_MAE", "test_R2",
               "train_seconds", "mae_ratio", "max_psi"]


def main() -> int:
    mlflow = common.setup_mlflow("traffic_supervised_models")
    client = mlflow.tracking.MlflowClient()
    lines = ["# MLflow experiment tracking summary", "",
             "Exported from `mlflow/mlflow.db` by `export_mlflow_summary.py`. Browse interactively with "
             "`mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db` (run from `part3_machine_learning`).", ""]
    experiments = [e for e in client.search_experiments() if e.name != "Default"]
    total_runs = 0
    for exp in sorted(experiments, key=lambda e: e.experiment_id):
        runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time ASC"])
        total_runs += len(runs)
        lines += [f"## Experiment: `{exp.name}` ({len(runs)} runs)", "",
                  "| Run | Key parameters | Key metrics |", "|---|---|---|"]
        for run in runs:
            params = {k: v for k, v in run.data.params.items()
                      if k in ("algorithm", "k", "n_estimators", "max_depth", "max_iter", "learning_rate",
                               "C", "alpha", "lstm_units", "window_hours", "epochs_run", "min_support",
                               "min_confidence", "reference", "class_weight")}
            metrics = {k: v for k, v in run.data.metrics.items() if k in KEY_METRICS}
            lines.append(f"| {run.info.run_name} | " + ", ".join(f"{k}={v}" for k, v in params.items())
                         + " | " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()) + " |")
        lines.append("")
    lines += ["## Model registry", "", "| Registered model | Version | Alias / stage | Source run |",
              "|---|---|---|---|"]
    for rm in client.search_registered_models():
        aliases = {v: a for a, v in rm.aliases.items()}
        for mv in sorted(client.search_model_versions(f"name='{rm.name}'"), key=lambda m: int(m.version)):
            stage = aliases.get(mv.version, mv.tags.get("stage", ""))
            lines.append(f"| {rm.name} | v{mv.version} | {stage} | {mv.run_id[:8]} |")
    (MLFLOW_DIR / "MLFLOW_SUMMARY.md").write_text("\n".join(lines) + "\n")
    logger.info("Exported %d experiments and %d runs to mlflow/MLFLOW_SUMMARY.md", len(experiments), total_runs)
    return 0


if __name__ == "__main__":
    common.configure_logging()
    sys.exit(main())

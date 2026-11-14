"""
run_part3.py - rebuild every Part 3 output from scratch, in order.

Steps: supervised models (Task 1) -> unsupervised learning (Task 2) ->
LSTM + SHAP/LIME (Task 3) -> sample recommendations (Task 5) -> monitoring
(Task 6.4/6.5) -> API tests (Task 6.3) -> MLflow summary export (Task 4/6.2). The MLflow database is reset first so
model versions are numbered v1-v3 again.

Usage (from part3_machine_learning):
    python notebooks/run_part3.py
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time

import common

logger = logging.getLogger(__name__)

STEPS = [["notebooks/train_supervised.py"], ["notebooks/unsupervised.py"], ["notebooks/deep_learning.py"],
         ["recommendation_system/recommender.py", "--write-samples"], ["monitoring/monitor.py"],
         ["deployment/test_api.py"], ["mlflow/export_mlflow_summary.py"]]


def main() -> int:
    for stale in (common.MLFLOW_DB, common.MLFLOW_ARTIFACTS):
        if stale.is_dir():
            shutil.rmtree(stale)
        elif stale.exists():
            stale.unlink()
    logger.warning("Reset MLflow tracking store (mlflow.db and artifacts) so the rebuild starts "
                   "from model version 1")
    timings = {}
    for command in STEPS:
        step = " ".join(command)
        start = time.perf_counter()
        logger.info("Running %s", step)
        result = subprocess.run([sys.executable, *command], cwd=common.PART3_DIR)
        timings[step] = time.perf_counter() - start
        if result.returncode != 0:
            logger.error("%s failed with exit code %d; stopping (see logs/)", step, result.returncode)
            return 1
        logger.info("Finished %s in %.1fs", step, timings[step])
    total = sum(timings.values())
    logger.info("Part 3 rebuilt successfully in %.1f minutes (%.0f seconds of compute)",
                total / 60, total)
    return 0


if __name__ == "__main__":
    if common.LOG_FILE.exists():
        common.LOG_FILE.unlink()          # a full rebuild starts a fresh log
    common.configure_logging()
    sys.exit(main())

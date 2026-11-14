# MLflow experiment tracking summary

Exported from `mlflow/mlflow.db` by `export_mlflow_summary.py`. Browse interactively with `mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db` (run from `part3_machine_learning`).

## Experiment: `traffic_supervised_models` (6 runs)

| Run | Key parameters | Key metrics |
|---|---|---|
| regression_ridge_regression | algorithm=Ridge, alpha=1.0 | MAE=548.953, R2=0.872, RMSE=707.572, train_seconds=0.060 |
| regression_random_forest | algorithm=RandomForestRegressor, n_estimators=80, max_depth=12 | MAE=228.410, R2=0.964, RMSE=374.022, train_seconds=4.690 |
| regression_hist_gradient_boosting | algorithm=HistGradientBoostingRegressor, max_iter=400, learning_rate=0.06 | MAE=232.914, R2=0.963, RMSE=379.047, train_seconds=2.547 |
| classification_logistic_regression | algorithm=LogisticRegression, C=1.0, class_weight=balanced | accuracy=0.979, precision=0.817, recall=0.987, f1=0.894, roc_auc=0.998, pr_auc=0.966, risky_weather_f1=0.894, train_seconds=0.088 |
| classification_random_forest | algorithm=RandomForestClassifier, n_estimators=150, max_depth=14, class_weight=balanced | accuracy=0.989, precision=0.899, recall=0.992, f1=0.943, roc_auc=0.998, pr_auc=0.975, risky_weather_f1=0.943, train_seconds=1.289 |
| classification_hist_gradient_boosting | algorithm=HistGradientBoostingClassifier, max_iter=300, learning_rate=0.06, class_weight=balanced | accuracy=0.991, precision=0.924, recall=0.976, f1=0.949, roc_auc=0.997, pr_auc=0.978, risky_weather_f1=0.949, train_seconds=1.002 |

## Experiment: `traffic_unsupervised` (2 runs)

| Run | Key parameters | Key metrics |
|---|---|---|
| kmeans_k8 | k=8 | silhouette=0.371 |
| association_rules_apriori | min_support=0.01, min_confidence=0.5 | n_rules=42.000, top_lift=3.680 |

## Experiment: `traffic_deep_learning` (2 runs)

| Run | Key parameters | Key metrics |
|---|---|---|
| lstm_next_hour_volume | window_hours=24, lstm_units=48, epochs_run=80 | test_MAE=180.069, test_R2=0.981, train_seconds=42.069 |
| next_hour_tree_comparable_for_shap | max_iter=400 | test_MAE=153.964, test_R2=0.986 |

## Experiment: `traffic_monitoring` (1 runs)

| Run | Key parameters | Key metrics |
|---|---|---|
| monthly_drift_check_2018 | reference=2017 | mae_ratio=0.975, max_psi=2.341 |

## Model registry

| Registered model | Version | Alias / stage | Source run |
|---|---|---|---|
| proxy_risk_classifier | v1 | archived | 4f73389f |
| proxy_risk_classifier | v2 | archived | 64cdaf3d |
| proxy_risk_classifier | v3 | champion | 8340ddd5 |
| traffic_volume_regressor | v1 | archived | 7ab4cfc6 |
| traffic_volume_regressor | v2 | archived | 66353dd9 |
| traffic_volume_regressor | v3 | champion | 3b059ac4 |

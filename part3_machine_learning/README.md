# Part 3: Machine Learning and AI (Intelligent Mobility Solution)

**Author:** Kyi Htoo Aung | Smart City Traffic Intelligence Capstone

Part 3 turns the cleaned data and features from Part 2 into an AI solution: supervised and unsupervised models, an LSTM with SHAP/LIME explanations, MLflow experiment tracking and model registry, a travel-timing recommendation system, a FastAPI deployment mock-up, drift monitoring with PASS/ALERT alerting, and a responsible AI report.

## Accident data: documented proxy, not a real accident dataset

**No real accident dataset was sourced.** As instructed in the brief, the classification task uses a **documented proxy label**:

```
congestion_category = quartiles of traffic_volume (Low / Medium / High / Severe)
high_risk = 1  if congestion_category is High or Severe
               AND (weather is Thunderstorm, Squall or Snow  OR  is_low_visibility == 1 [Mist, Fog, Haze, Smoke])
```

`high_risk` marks 7.7% of hours. It demonstrates the classification workflow only and **must not be interpreted as a prediction of actual accidents** (see `responsible_ai_report.pdf`).

## Project structure

```
part3_machine_learning/
├── notebooks/
│   ├── part3_walkthrough.ipynb   Executed notebook presenting every task, with live demos
│   ├── common.py                 Shared: paths, data loading, proxy label, feature set, logging, MLflow setup
│   ├── train_supervised.py       Task 1 (+ Task 4, 6.1, 6.2): 3 regressors + 3 classifiers, tracked and registered in MLflow
│   ├── unsupervised.py           Task 2: K-means clustering and Apriori association rules
│   ├── deep_learning.py          Task 3: LSTM next-hour forecast; SHAP (comparable tree model), LIME, permutation importance
│   ├── run_part3.py              Rebuilds everything from scratch in order (about 2 minutes)
│   ├── outputs/                  Metrics (JSON/CSV) and the 9 figures produced by the scripts
│   └── part3.log                 Log of the full rebuild (every script appends to it)
├── models/
│   ├── MODEL_VERSIONS.md         Task 6.1: every model version and its performance
│   ├── traffic_regressor_champion.joblib, risk_classifier_champion.joblib   (served by the API)
│   ├── regression_v1..v3_*.joblib, classification_v1..v3_*.joblib          (all versions)
│   └── lstm_next_hour.keras, lstm_scaling.json, next_hour_tree_for_shap.joblib
├── mlflow/
│   ├── screenshots/              MLflow UI: runs table, metric charts, model registry, version history
│   ├── mlflow.db                 MLflow tracking store and model registry (SQLite)
│   ├── MLFLOW_SUMMARY.md         All experiments, runs, parameters, metrics and registered versions
│   └── export_mlflow_summary.py  Writes MLFLOW_SUMMARY.md from the database
├── deployment/
│   ├── app.py                    Task 6.3: FastAPI service (/health, /predict/traffic, /predict/risk, /recommend)
│   ├── test_api.py               Calls every endpoint, including invalid inputs
│   └── sample_api_responses.json
├── recommendation_system/
│   ├── recommender.py            Task 5: travel-timing recommender (module + command-line tool)
│   └── sample_recommendations.md
├── monitoring/
│   ├── monitor.py                Task 6.4-6.5: error drift + feature drift (PSI), PASS / ALERT
│   ├── monitoring_report.md, monitoring_status.json
│   └── monitoring_dashboard.png
├── responsible_ai_report.pdf     Task 7: bias and fairness, governance and sustainability
├── requirements.txt              Pinned library versions
└── README.md
```

The final capstone report covering all three parts is `final_capstone_report.pdf` at the repository root.

## How to run

Part 3 reads the Part 2 feature table (`part2_python/output/traffic_features.csv.gz`), so Part 2 must have been run first.

```bash
cd part3_machine_learning
pip install -r requirements.txt     # exact library versions used to produce the committed results

python notebooks/run_part3.py           # rebuild everything (resets MLflow; about 2 minutes on one CPU)

# or run tasks individually
python notebooks/train_supervised.py    # Task 1 + MLflow tracking/registry
python notebooks/unsupervised.py        # Task 2
python notebooks/deep_learning.py       # Task 3 (run train_supervised.py first: LIME explains its classifier)
python monitoring/monitor.py            # Task 6.4-6.5
python deployment/test_api.py           # Task 6.3 checks
python mlflow/export_mlflow_summary.py

# Recommendation system (Task 5)
python recommendation_system/recommender.py --day weekday
python recommendation_system/recommender.py --day Friday --weather Snow --month 1 --from 14 --to 20
python recommendation_system/recommender.py --day weekend --weather Rain --duration 2

# API (Task 6.3): start the server, then open http://127.0.0.1:8000/docs
uvicorn deployment.app:app --reload

# MLflow UI (Task 4 / 6.2): then open http://127.0.0.1:5000
python -m mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
```

The MLflow database holds every run, parameter, metric and registered model version; `mlflow/screenshots/` shows them in the MLflow UI without running anything. The model binaries that MLflow logs are duplicates of the files in `models/`, so `mlflow/artifacts/` is not committed; `python notebooks/run_part3.py` regenerates it.

## Results summary

All models are trained on 2012-2016 and tested on unseen 2017-2018 data (a chronological split, so there is no look-ahead leakage). Both supervised models share one 31-feature set: time features, cyclical encodings of hour, day of week and month, the holiday flag and weather encodings.

| Task | Result |
|---|---|
| 1. Regression (traffic volume) | Ridge baseline MAE 549, R² 0.872. Random forest MAE 228, R² 0.964. **Gradient boosting (champion) MAE 233, R² 0.963**: within 0.005 R² of the forest at one fifth of the size |
| 1. Classification (proxy risk) | Logistic regression F1 0.894. Random forest F1 0.943. **Gradient boosting (champion) F1 0.949**, accuracy 0.991, precision 0.924, recall 0.976, ROC AUC 0.997. Scores are inflated because the label is built partly from weather inputs. Ranked only within severe/low-visibility hours, ROC AUC falls to 0.985, evidence that this restricted test is harder (F1 there is identical to overall F1 by construction, so it is not used) |
| 2. K-means | k = 8 (best silhouette, 0.371, within an interpretable range of 3-8). Regimes include quiet weekday nights, weekday morning and afternoon peaks (35% and 48% congested) and bad-weather peak hours (only 28% congested) |
| 2. Association rules | 42 non-redundant rules. Strongest: weekday + AM peak (+ cold, mild, low visibility or rain) leads to Severe congestion in 88-92% of hours (lift 3.5-3.7) |
| 3. LSTM + explainability | Next-hour MAE 180, R² 0.981, against 561-592 for naive baselines. SHAP is applied to a comparable tree model trained on the same problem (MAE 154), because SHAP is hard to apply reliably to an LSTM: last hour's volume, time of day and the latest change dominate, and the LSTM's own permutation importance agrees. LIME explains individual risk predictions |
| 4. MLflow (advanced technique) | 4 experiments, 11 runs, 6 registered model versions, champion aliases. Why, how, value and limitations are explained in the final report |
| 5. Recommender | Example: weekday office hours: travel 10:00-11:00 (alternatives 18:00-19:00, 13:00-14:00), avoid 16:00-17:00 (94% of comparable hours congested) |
| 6. Deployment and monitoring | FastAPI serves both champions and the recommender. 2018 by month: 4 PASS, 4 ALERT graded "watch" (unusual weather, accuracy fine), 1 ALERT graded "investigate" (April 2018 snowstorm, error +56%). Both simulated incidents are detected |
| 7. Responsible AI | Errors are 57% higher in snow/storms and 2.4 times higher on holidays; see `responsible_ai_report.pdf` |

## Logging

Every module creates `logger = logging.getLogger(__name__)`. Handlers are attached only when a script runs as an entry point (`common.configure_logging`), writing to the console and to `notebooks/part3.log`. INFO marks milestones (data loaded, model trained or saved, figure saved); WARNING marks rows dropped (sensor faults, gap-broken sequences), rejected API requests and every monitoring ALERT; ERROR marks failures. `print()` is used only for the recommender's answers on the command line.

## Limitations

- The risk label is a proxy derived partly from model inputs, so its scores overstate what real accident prediction would achieve.
- Single westbound sensor, a 10-month gap (2014-2015), and a change in how cloud cover and weather labels were recorded around 2015.
- Errors are larger for severe weather, holidays and weekends (details in `responsible_ai_report.pdf`).
- Champion models are chosen on the same 2017-2018 test scores that are reported, so those scores are slightly optimistic; a separate validation year would be stricter. The effect is small because the top models are near-tied (see `models/MODEL_VERSIONS.md`).
- The deployment is a local mock-up without authentication, scaling or a live data feed.

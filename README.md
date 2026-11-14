# Smart City Traffic Intelligence: From Data Analytics to AI-Powered Mobility

**Author:** Kyi Htoo Aung | NUS SoC AI, ML and Data Science Programme, Capstone Project

An end-to-end traffic intelligence solution for a Smart City Mobility Analytics Team, built on the Metro Interstate Traffic Volume dataset: 48,204 hourly records of westbound I-94 traffic near Minneapolis-St Paul (October 2012 to September 2018) with weather and US holiday information.

The project has three connected parts. Each part reuses and extends the data preparation, findings and features of the part before it. The full write-up is in **[`final_capstone_report.pdf`](final_capstone_report.pdf)**.

> **Accident data:** no real accident dataset was sourced. The Part 3 classification task uses the brief's **documented proxy label** (`high_risk` = High/Severe congestion during severe or low-visibility weather). It demonstrates the workflow only and is not a prediction of actual accidents.

## Repository structure

```
smart-city-traffic-capstone/
├── data/
│   └── Metro_Interstate_Traffic_Volume.csv   Raw dataset (shared by all parts)
│
├── part1_data_analytics/                     Part 1: SQL, statistics, probability, Power BI
│   ├── sql/                traffic_analysis.sql (load + all queries), traffic.db
│   ├── statistics/         statistical_probability_analysis.md, stats_probability_analysis.py (reproduces Tasks 2-3)
│   ├── powerbi/            traffic_dashboard.pbix / .pdf, data-quality screenshots, powerbi_notes.md
│   └── insights_report.pdf 2-page Data Analytics Insights Report
│
├── part2_python/                             Part 2: reproducible Python pipeline
│   ├── pipeline.py         Entry point: logging, load, validate, clean
│   ├── feature_engineering.py, visualizations.py
│   ├── cli_app/            traffic_cli.py (6 commands), cli.log, sample_session.txt
│   ├── figures/            6 Matplotlib figures
│   ├── output/             cleaned and feature tables (input to Part 3)
│   ├── logs/, pipeline.log Sample logs (DEBUG run and normal run)
│   ├── part2_report.pdf    Methodology and findings report
│   └── README.md
│
├── part3_machine_learning/                   Part 3: machine learning and AI
│   ├── notebooks/          walkthrough notebook, training scripts, outputs (metrics, figures), part3.log
│   ├── models/             all model versions, champions, LSTM, MODEL_VERSIONS.md
│   ├── mlflow/             tracking database and model registry, MLFLOW_SUMMARY.md
│   ├── deployment/         FastAPI app, API tests, sample responses
│   ├── recommendation_system/   recommender and sample recommendations
│   ├── monitoring/         drift monitor, report, PASS/ALERT dashboard
│   ├── responsible_ai_report.pdf
│   └── README.md
│
├── final_capstone_report.pdf                 Methodology and findings across all three parts
├── .gitignore
└── README.md
```

## Parts at a glance

| Part | Focus | Key outputs | Details |
|---|---|---|---|
| 1. Data Analytics | SQL (SQLite), descriptive statistics, correlation, probability, Power BI | SQL analysis, statistics write-up, interactive dashboard, insights report | [`part1_data_analytics`](part1_data_analytics) |
| 2. Python | Logged data pipeline, cleaning, feature engineering, Matplotlib, CLI app | Clean feature table, 6 figures, CLI with 6 commands, pipeline.log | [`part2_python/README.md`](part2_python/README.md) |
| 3. Machine Learning and AI | Supervised, unsupervised and deep learning, SHAP/LIME, MLflow, recommender, FastAPI, monitoring, responsible AI | 6 registered model versions, LSTM, recommender, API, PASS/ALERT monitoring, responsible AI report | [`part3_machine_learning/README.md`](part3_machine_learning/README.md) |

## Key findings

- **Congestion is a weekday commuting problem.** Traffic exceeds 5,500 vehicles/hour mainly on weekdays at 07:00-08:00 and 15:00-17:00, reaching 94-96% of Tuesday-Thursday 16:00 hours. Weekends are almost never congested.
- **Time of day matters far more than weather.** Temperature correlates only weakly with volume (r = 0.13); the apparent weather effect on congestion disappears once hour of day is controlled for (odds ratio 1.006); and SHAP values, clusters and association rules in Part 3 confirm it.
- **Traffic is highly predictable.** Gradient boosting reaches R² 0.963 (MAE 233 vehicles/hour) on unseen 2017-2018 data from calendar and weather alone; an LSTM predicts the next hour with MAE 180, three times better than naive baselines.
- **The raw data needed careful cleaning:** inconsistent labels, 17 duplicate rows, 7,612 repeated hourly records, ten 0 K temperatures, a 9,831 mm rainfall reading, sensor outages and a 10-month gap.
- **Monitoring works on real events:** the monthly drift check flags April 2018, when a snowstorm pushed traffic 54% below forecast.

## How to run

Requirements: Python 3.12+ (the pinned library versions in each part's `requirements.txt` need it); Power BI Desktop (Windows) to open the dashboard; the `sqlite3` shell or DB Browser for SQLite for the SQL file.

```bash
# Part 1: rebuild the SQLite database and run every query
cd part1_data_analytics/sql
sqlite3 traffic.db < traffic_analysis.sql

# Part 1: reproduce the statistics and probability results (Tasks 2-3)
cd ../statistics
pip install -r requirements.txt
python stats_probability_analysis.py

# Part 2: run the full pipeline, then query it with the CLI
cd part2_python
pip install -r requirements.txt
python pipeline.py
python cli_app/traffic_cli.py summary

# Part 3: rebuild all models, MLflow runs, monitoring and API tests (about 2 minutes)
cd part3_machine_learning
python notebooks/run_part3.py              # packages are listed in part3_machine_learning/README.md
python recommendation_system/recommender.py --day Friday --weather Snow --month 1
uvicorn deployment.app:app --reload          # API docs at http://127.0.0.1:8000/docs
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
```

Logging: every Python module uses `logging.getLogger(__name__)`; handlers are configured only in entry-point scripts and write to the console and a log file. Details are in each part's README.

## Tools

SQLite, Python (pandas, NumPy, SciPy, scikit-learn, Keras, Matplotlib, SHAP, LIME, mlxtend, MLflow, FastAPI), Power BI Desktop (Power Query and DAX), Git and GitHub.

## Limitations

Data come from a single westbound sensor, with a 10-month gap and partial first and last years. The risk label is a proxy, not accident data. Model errors are larger in severe weather and on holidays. Results describe associations, not causal effects. See `part3_machine_learning/responsible_ai_report.pdf`.

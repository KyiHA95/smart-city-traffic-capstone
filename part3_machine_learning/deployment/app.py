"""
app.py - Part 3, Task 6.3: FastAPI deployment mock-up.

Serves the champion models exported by train_supervised.py (the versions
carrying the MLflow alias "champion") and the recommendation engine.

Endpoints
---------
GET  /health             service status and the model versions being served
POST /predict/traffic    predicted hourly traffic volume + congestion category
POST /predict/risk       proxy risk probability (NOT a real accident prediction)
POST /recommend          best travel window with a plain-language recommendation
GET  /docs               interactive Swagger documentation (auto-generated)

Run locally (from part3_machine_learning):
    uvicorn deployment.app:app --reload
then open http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request as HttpRequest
from pydantic import BaseModel, Field

PART3_DIR = Path(__file__).resolve().parent.parent
for _p in (PART3_DIR, PART3_DIR / "notebooks"):      # shared module common.py lives in notebooks/
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import common  # noqa: E402
from recommendation_system.recommender import (Request, RecommendationError,  # noqa: E402
                                               TravelRecommender)

logger = logging.getLogger(__name__)

VERSIONS_FILE = common.RESULTS_DIR / "supervised_results.json"
PROXY_NOTE = ("high_risk is a documented proxy (heavy congestion during severe or low-visibility "
              "weather), not a prediction of real accidents")

app = FastAPI(title="I-94 Traffic Intelligence API (mock deployment)", version="1.0.0",
              description="Serves the champion traffic-volume regressor, the proxy-risk classifier "
                          "and the travel-timing recommender. " + PROXY_NOTE)
_engine: TravelRecommender | None = None


def engine() -> TravelRecommender:
    global _engine
    if _engine is None:
        _engine = TravelRecommender()
    return _engine


def served_versions() -> dict:
    try:
        info = json.loads(VERSIONS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {"note": "version metadata unavailable"}
    champ = {}
    for task, key in (("regression", "regression"), ("classification", "classification")):
        best = next(r for r in info[key] if r["model"] == info["champions"][task])
        champ[task] = {"registered_model": best["registered_model"], "version": best["version"],
                       "algorithm": best["algorithm"]}
    return champ


# ----------------------------------------------------------------- schemas
class HourInput(BaseModel):
    datetime: str = Field(..., examples=["2018-10-02 08:00"], description="date and hour to predict")
    weather: str = Field("Clear", examples=["Rain"], description="Clear, Clouds, Rain, Snow, Fog, ...")
    temp_c: float = Field(..., ge=-40, le=45, examples=[12.5])
    is_holiday: bool = False
    rain_1h: float | None = Field(None, ge=0, le=305)
    clouds_all: float | None = Field(None, ge=0, le=100)


class RecommendInput(BaseModel):
    day: str = Field("weekday", examples=["Friday"])
    weather: str = Field("Clear", examples=["Snow"])
    temp_c: float | None = Field(None, ge=-40, le=45)
    month: int = Field(6, ge=1, le=12)
    earliest: int = Field(6, ge=0, le=23)
    latest: int = Field(21, ge=0, le=23)
    duration: int = Field(1, ge=1, le=6)


def build_row(inp: HourInput) -> pd.DataFrame:
    try:
        when = pd.Timestamp(inp.datetime)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, f"Malformed datetime '{inp.datetime}'. Use YYYY-MM-DD HH:MM") from exc
    if pd.isna(when):  # an empty string parses to NaT (no timestamp) instead of raising
        raise HTTPException(400, "Missing datetime. Use YYYY-MM-DD HH:MM")
    try:
        return engine().feature_row(when, inp.weather, inp.temp_c, int(inp.is_holiday),
                                    inp.rain_1h, inp.clouds_all)
    except RecommendationError as exc:
        raise HTTPException(400, str(exc)) from exc


# ----------------------------------------------------------------- middleware
@app.middleware("http")
async def log_requests(request: HttpRequest, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    level = logging.WARNING if response.status_code >= 400 else logging.INFO
    logger.log(level, "%s %s -> %d (%.0f ms)", request.method, request.url.path,
               response.status_code, (time.perf_counter() - start) * 1000)
    return response


# ----------------------------------------------------------------- endpoints
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models": served_versions()}


@app.post("/predict/traffic")
def predict_traffic(inp: HourInput) -> dict:
    row = build_row(inp)
    volume = float(engine().regressor.predict(row)[0])
    return {"datetime": inp.datetime, "predicted_volume": round(volume),
            "congestion_category": engine().congestion_category(volume),
            "congested_over_5500": volume > 5500, "model": served_versions().get("regression")}


@app.post("/predict/risk")
def predict_risk(inp: HourInput) -> dict:
    row = build_row(inp)
    prob = float(engine().classifier.predict_proba(row)[0, 1])
    return {"datetime": inp.datetime, "proxy_risk_probability": round(prob, 3),
            "high_risk": prob >= 0.5, "note": PROXY_NOTE,
            "model": served_versions().get("classification")}


@app.post("/recommend")
def recommend(inp: RecommendInput) -> dict:
    try:
        rec = engine().recommend(Request(**inp.model_dump()))
    except RecommendationError as exc:
        raise HTTPException(400, str(exc)) from exc
    return rec.to_dict()


if __name__ == "__main__":
    import uvicorn
    common.configure_logging()
    uvicorn.run(app, host="127.0.0.1", port=8000)

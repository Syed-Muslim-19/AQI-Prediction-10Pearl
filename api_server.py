from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, Query

from aqi_serving import (
    DEFAULT_FEATURE_STORE_PATH,
    DEFAULT_MODEL_REGISTRY_DIR,
    aqi_alert_level,
    eda_summary,
    forecast_recursive,
    load_feature_store,
    load_latest_model_artifacts,
    shap_explanations,
    statistical_baselines,
)


app = FastAPI(title="AQI Forecast API", version="1.0.0")


@lru_cache(maxsize=1)
def _load_artifacts():
    return load_latest_model_artifacts(DEFAULT_MODEL_REGISTRY_DIR)


def _load_table():
    return load_feature_store(DEFAULT_FEATURE_STORE_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/latest")
def latest() -> dict:
    frame = _load_table()
    if frame.empty:
        return {"message": "No feature rows found."}

    row = frame.sort_values("observed_at_utc").iloc[-1].to_dict()
    aqi_value = float(row.get("aqi", 0.0)) if row.get("aqi") is not None else 0.0
    row["alert_level"] = aqi_alert_level(aqi_value)
    return row


@app.get("/forecast")
def forecast(horizon_hours: int = Query(72, ge=1, le=168)) -> dict:
    artifacts = _load_artifacts()
    frame = _load_table()

    ml_forecast = forecast_recursive(frame, artifacts, horizon_hours=horizon_hours)
    baselines = statistical_baselines(frame, horizon_hours=horizon_hours)

    return {
        "model_name": artifacts.model_name,
        "target_column": artifacts.target_column,
        "trained_at_utc": artifacts.trained_at_utc,
        "horizon_hours": horizon_hours,
        "ml_forecast": ml_forecast,
        "baselines": baselines,
    }


@app.get("/eda")
def eda() -> dict:
    frame = _load_table()
    return eda_summary(frame)


@app.get("/explain")
def explain(top_k: int = Query(10, ge=3, le=30)) -> dict:
    artifacts = _load_artifacts()
    frame = _load_table()
    return shap_explanations(frame, artifacts, top_k=top_k)


@app.post("/reload")
def reload_models() -> dict[str, str]:
    _load_artifacts.cache_clear()
    _load_artifacts()
    return {"status": "reloaded"}

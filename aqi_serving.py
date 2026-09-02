from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


DEFAULT_FEATURE_STORE_PATH = Path("data/feature_store/aqi_feature_table.csv")
DEFAULT_MODEL_REGISTRY_DIR = Path("data/model_registry")


@dataclass
class ModelArtifacts:
    model: Any
    preprocessor: Any
    feature_columns: list[str]
    model_name: str
    target_column: str
    trained_at_utc: str
    registry_path: Path


def _as_path(value: str, base_dir: Path) -> Path:
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def aqi_label_from_value(value: float) -> str:
    if value < 1.5:
        return "Good"
    if value < 2.5:
        return "Fair"
    if value < 3.5:
        return "Moderate"
    if value < 4.5:
        return "Poor"
    return "Very Poor"


def aqi_alert_level(value: float) -> str:
    if value >= 4.0:
        return "hazard"
    if value >= 3.0:
        return "warning"
    return "normal"


def load_feature_store(feature_store_path: Path = DEFAULT_FEATURE_STORE_PATH) -> pd.DataFrame:
    if not feature_store_path.exists():
        raise FileNotFoundError(f"Feature store not found: {feature_store_path}")
    frame = pd.read_csv(feature_store_path)
    if "observed_at_utc" in frame.columns:
        frame["observed_at_utc"] = pd.to_datetime(frame["observed_at_utc"], utc=True, errors="coerce")
    frame = frame.sort_values("observed_at_utc").reset_index(drop=True)
    return frame


def load_latest_model_artifacts(model_registry_dir: Path = DEFAULT_MODEL_REGISTRY_DIR) -> ModelArtifacts:
    latest_path = model_registry_dir / "latest_model.json"
    if not latest_path.exists():
        raise FileNotFoundError(f"Model pointer not found: {latest_path}")

    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    registry_path = _as_path(payload["path"], Path.cwd())

    model_bundle = joblib.load(registry_path / "model.joblib")
    feature_columns = json.loads((registry_path / "features.json").read_text(encoding="utf-8"))
    return ModelArtifacts(
        model=model_bundle["model"],
        preprocessor=model_bundle["preprocessor"],
        feature_columns=feature_columns,
        model_name=str(payload.get("model_name", "unknown")),
        target_column=str(payload.get("target_column", "target_aqi_t_plus_1h")),
        trained_at_utc=str(payload.get("trained_at_utc", "")),
        registry_path=registry_path,
    )


def _prepare_input_frame(row: pd.Series, feature_columns: list[str]) -> pd.DataFrame:
    row_dict = row.to_dict()
    prepared = {column: row_dict.get(column, np.nan) for column in feature_columns}
    return pd.DataFrame([prepared])


def predict_one_step(row: pd.Series, artifacts: ModelArtifacts) -> float:
    X_row = _prepare_input_frame(row, artifacts.feature_columns)
    transformed = artifacts.preprocessor.transform(X_row)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed_df = pd.DataFrame(transformed, columns=artifacts.preprocessor.get_feature_names_out())
    prediction = artifacts.model.predict(transformed_df)
    return float(np.asarray(prediction).reshape(-1)[0])


def _build_next_row(current_row: pd.Series, next_timestamp: pd.Timestamp, predicted_aqi: float, aqi_history: list[float]) -> pd.Series:
    next_row = current_row.copy()
    next_row["observed_at_utc"] = next_timestamp
    next_row["hour"] = next_timestamp.hour
    next_row["day"] = next_timestamp.day
    next_row["month"] = next_timestamp.month
    next_row["day_of_week"] = next_timestamp.weekday()
    next_row["is_weekend"] = int(next_timestamp.weekday() >= 5)

    lag_1 = aqi_history[-1] if len(aqi_history) >= 1 else np.nan
    lag_24 = aqi_history[-24] if len(aqi_history) >= 24 else np.nan

    next_row["aqi"] = predicted_aqi
    next_row["aqi_label"] = aqi_label_from_value(predicted_aqi)
    next_row["aqi_lag_1"] = lag_1
    next_row["aqi_lag_24"] = lag_24
    next_row["aqi_change_rate_1h"] = predicted_aqi - lag_1 if pd.notna(lag_1) else np.nan
    next_row["aqi_pct_change_1h"] = ((predicted_aqi - lag_1) / lag_1) if pd.notna(lag_1) and lag_1 != 0 else np.nan

    window_3 = aqi_history[-3:] if len(aqi_history) >= 3 else aqi_history
    window_24 = aqi_history[-24:] if len(aqi_history) >= 24 else aqi_history
    next_row["aqi_3h_mean"] = float(np.mean(window_3)) if len(window_3) else np.nan
    next_row["aqi_24h_mean"] = float(np.mean(window_24)) if len(window_24) else np.nan
    return next_row


def forecast_recursive(feature_table: pd.DataFrame, artifacts: ModelArtifacts, horizon_hours: int = 72) -> list[dict[str, Any]]:
    if feature_table.empty:
        return []

    current_row = feature_table.sort_values("observed_at_utc").iloc[-1].copy()
    aqi_history = feature_table["aqi"].dropna().astype(float).tolist()

    forecasts: list[dict[str, Any]] = []
    for step in range(1, horizon_hours + 1):
        predicted = predict_one_step(current_row, artifacts)
        next_timestamp = pd.to_datetime(current_row["observed_at_utc"], utc=True) + pd.Timedelta(hours=1)
        forecasts.append(
            {
                "step_hour": step,
                "timestamp_utc": next_timestamp.isoformat(),
                "aqi_pred": predicted,
                "aqi_label": aqi_label_from_value(predicted),
                "alert_level": aqi_alert_level(predicted),
                "model": artifacts.model_name,
            }
        )
        current_row = _build_next_row(current_row, next_timestamp, predicted, aqi_history)
        aqi_history.append(predicted)

    return forecasts


def statistical_baselines(feature_table: pd.DataFrame, horizon_hours: int = 72) -> dict[str, list[dict[str, Any]]]:
    if feature_table.empty:
        return {"persistence": [], "moving_average_24h": []}

    ordered = feature_table.sort_values("observed_at_utc")
    last_timestamp = pd.to_datetime(ordered.iloc[-1]["observed_at_utc"], utc=True)
    last_aqi = float(ordered["aqi"].dropna().iloc[-1])
    moving_avg = float(ordered["aqi"].dropna().tail(24).mean())

    persistence = []
    moving_average = []
    for step in range(1, horizon_hours + 1):
        ts = (last_timestamp + pd.Timedelta(hours=step)).isoformat()
        persistence.append({"step_hour": step, "timestamp_utc": ts, "aqi_pred": last_aqi, "model": "persistence"})
        moving_average.append({"step_hour": step, "timestamp_utc": ts, "aqi_pred": moving_avg, "model": "moving_average_24h"})
    return {"persistence": persistence, "moving_average_24h": moving_average}


def eda_summary(feature_table: pd.DataFrame) -> dict[str, Any]:
    ordered = feature_table.sort_values("observed_at_utc").copy()
    ordered = ordered.dropna(subset=["aqi"])
    if ordered.empty:
        return {"row_count": 0, "hourly_avg": [], "daily_avg": [], "trend_slope": 0.0}

    ordered["hour"] = pd.to_numeric(ordered["hour"], errors="coerce")
    ordered["date"] = pd.to_datetime(ordered["observed_at_utc"], utc=True, errors="coerce").dt.date

    hourly = ordered.groupby("hour", dropna=True)["aqi"].mean().reset_index().sort_values("hour")
    daily = ordered.groupby("date", dropna=True)["aqi"].mean().reset_index().sort_values("date")

    x = np.arange(len(ordered), dtype=float)
    y = ordered["aqi"].to_numpy(dtype=float)
    slope = float(np.polyfit(x, y, 1)[0]) if len(ordered) > 1 else 0.0

    return {
        "row_count": int(len(ordered)),
        "aqi_min": float(np.min(y)),
        "aqi_max": float(np.max(y)),
        "aqi_mean": float(np.mean(y)),
        "trend_slope": slope,
        "hourly_avg": hourly.to_dict(orient="records"),
        "daily_avg": daily.to_dict(orient="records"),
    }


def shap_explanations(feature_table: pd.DataFrame, artifacts: ModelArtifacts, top_k: int = 10) -> dict[str, Any]:
    try:
        import shap
    except Exception:
        return {"enabled": False, "message": "SHAP is not installed in this environment."}

    if feature_table.empty:
        return {"enabled": False, "message": "Feature table is empty."}

    sample = feature_table.sort_values("observed_at_utc").tail(min(200, len(feature_table))).copy()
    sample = sample.fillna(np.nan)
    sample_features = sample.reindex(columns=artifacts.feature_columns, fill_value=np.nan)
    transformed = artifacts.preprocessor.transform(sample_features)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    latest = transformed[-1:]
    background = transformed[:-1] if len(transformed) > 1 else transformed
    feature_names = list(artifacts.preprocessor.get_feature_names_out())

    estimator = artifacts.model
    if hasattr(estimator, "named_steps") and "model" in estimator.named_steps:
        estimator = estimator.named_steps["model"]

    # Model types that shap.TreeExplainer natively supports (fast, exact SHAP
    # values via each library's tree-structure API). Add new tree-based model
    # class-name prefixes here as they're introduced in train_candidate_models.
    tree_model_prefixes = ("randomforest", "xgb", "lgbm", "gradientboosting", "extratrees", "catboost")

    try:
        estimator_name = estimator.__class__.__name__.lower()
        if estimator_name.startswith(tree_model_prefixes):
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(latest)
            values = np.asarray(shap_values).reshape(-1)
        elif estimator_name.startswith("ridge"):
            explainer = shap.LinearExplainer(estimator, background)
            shap_values = explainer.shap_values(latest)
            values = np.asarray(shap_values).reshape(-1)
        else:
            return {"enabled": False, "message": "SHAP explainer is not configured for this model type."}
    except Exception as exc:
        return {"enabled": False, "message": f"Failed to compute SHAP values: {exc}"}

    ranked_idx = np.argsort(np.abs(values))[::-1][:top_k]
    ranked = [
        {
            "feature": feature_names[int(idx)],
            "shap_value": float(values[int(idx)]),
            "abs_shap": float(abs(values[int(idx)])),
        }
        for idx in ranked_idx
    ]
    return {"enabled": True, "top_features": ranked}
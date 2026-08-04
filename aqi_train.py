"""Train AQI forecasting models from the local feature store.

The script reads the curated feature table, trains tabular regression models,
evaluates them on a time-based holdout split, selects the best model by RMSE,
and stores the winner in a local model registry folder.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_FEATURE_STORE_PATH = Path("data/feature_store/aqi_feature_table.csv")
DEFAULT_MODEL_REGISTRY_DIR = Path("data/model_registry")
DEFAULT_TARGET_COLUMN = "target_aqi_t_plus_1h"
DEFAULT_TIME_COLUMN = "observed_at_utc"
DEFAULT_GROUP_COLUMN = "location_label"


@dataclass(frozen=True)
class TrainSettings:
    feature_store_path: Path
    model_registry_dir: Path
    target_column: str
    time_column: str
    group_column: str
    test_fraction: float
    min_rows: int
    random_state: int
    include_tensorflow: bool


@dataclass(frozen=True)
class ModelResult:
    name: str
    rmse: float
    mae: float
    r2: float
    trained_at_utc: str
    target_column: str
    row_count: int
    train_rows: int
    test_rows: int
    features: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AQI models from the feature store.")
    parser.add_argument("--feature-store-path", default=None)
    parser.add_argument("--model-registry-dir", default=None)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--min-rows", type=int, default=24)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--include-tensorflow", action="store_true")
    return parser.parse_args()


def load_settings(args: argparse.Namespace) -> TrainSettings:
    feature_store_path = Path(args.feature_store_path or DEFAULT_FEATURE_STORE_PATH)
    model_registry_dir = Path(args.model_registry_dir or DEFAULT_MODEL_REGISTRY_DIR)
    target_column = args.target_column or DEFAULT_TARGET_COLUMN
    time_column = args.time_column or DEFAULT_TIME_COLUMN
    group_column = args.group_column or DEFAULT_GROUP_COLUMN

    if not 0 < args.test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    return TrainSettings(
        feature_store_path=feature_store_path,
        model_registry_dir=model_registry_dir,
        target_column=target_column,
        time_column=time_column,
        group_column=group_column,
        test_fraction=args.test_fraction,
        min_rows=args.min_rows,
        random_state=args.random_state,
        include_tensorflow=bool(args.include_tensorflow),
    )


def load_feature_table(path: Path, target_column: str, time_column: str, group_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature store file not found: {path}")

    frame = pd.read_csv(path)
    required_columns = {target_column, time_column, group_column}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Feature table is missing required columns: {missing}")

    frame[time_column] = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    frame = frame.dropna(subset=[time_column, target_column]).copy()
    frame = frame.sort_values([group_column, time_column]).reset_index(drop=True)
    return frame


def add_time_numeric_features(frame: pd.DataFrame, time_column: str) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["obs_ts"] = enriched[time_column].astype("int64") // 10**9
    return enriched


def build_feature_matrix(frame: pd.DataFrame, target_column: str, time_column: str, group_column: str) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], list[str]]:
    prepared = add_time_numeric_features(frame, time_column)
    y = pd.to_numeric(prepared[target_column], errors="coerce")
    prepared = prepared.loc[y.notna()].copy()
    y = y.loc[y.notna()].astype(float)

    drop_columns = {target_column, "obs_ts"}
    feature_frame = prepared.drop(columns=[column for column in drop_columns if column in prepared.columns])

    categorical_columns = [
        column
        for column in [group_column, "weather_main", "weather_description", "aqi_label", "air_pollution_source"]
        if column in feature_frame.columns
    ]
    numeric_columns = [column for column in feature_frame.columns if column not in categorical_columns and column != time_column]
    feature_columns = [column for column in feature_frame.columns if column != time_column]
    return feature_frame, y, numeric_columns, categorical_columns, feature_columns


def make_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_columns),
            ("categorical", categorical_transformer, categorical_columns),
        ],
        remainder="drop",
    )


def temporal_train_test_split(frame: pd.DataFrame, y: pd.Series, test_fraction: float, time_column: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    ordered_index = frame.sort_values(time_column).index
    test_size = max(1, int(math.ceil(len(ordered_index) * test_fraction)))
    test_index = ordered_index[-test_size:]
    train_index = ordered_index[:-test_size]
    if len(train_index) == 0:
        raise ValueError("Not enough rows to create a temporal train/test split.")
    return frame.loc[train_index], frame.loc[test_index], y.loc[train_index], y.loc[test_index]


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) < 2:
        r2_value = float("nan")
    else:
        r2_value = float(r2_score(y_true, y_pred))
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": r2_value,
    }


def try_train_tensorflow_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> tuple[Any, np.ndarray] | None:
    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception:
        return None

    tf.random.set_seed(42)
    train_matrix = np.asarray(X_train, dtype=np.float32)
    test_matrix = np.asarray(X_test, dtype=np.float32)
    y_train_array = np.asarray(y_train, dtype=np.float32)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(train_matrix.shape[1],)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.15),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.fit(train_matrix, y_train_array, validation_split=0.2 if len(train_matrix) > 10 else 0.0, epochs=80, batch_size=min(16, len(train_matrix)), verbose=0)
    predictions = model.predict(test_matrix, verbose=0).reshape(-1)
    return model, predictions


def train_candidate_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, include_tensorflow: bool) -> dict[str, tuple[Any, np.ndarray]]:
    candidates: dict[str, tuple[Any, np.ndarray]] = {}

    ridge = Pipeline(
        steps=[
            ("model", Ridge(alpha=1.0, random_state=42)),
        ]
    )
    ridge.fit(X_train, y_train)
    candidates["ridge"] = (ridge, ridge.predict(X_test))

    random_forest = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    random_forest.fit(X_train, y_train)
    candidates["random_forest"] = (random_forest, random_forest.predict(X_test))

    if include_tensorflow:
        tf_result = try_train_tensorflow_model(X_train, y_train, X_test)
        if tf_result is not None:
            tf_model, tf_predictions = tf_result
            candidates["tensorflow_dense"] = (tf_model, tf_predictions)

    return candidates


def derive_reportable_frame(frame: pd.DataFrame, feature_columns: list[str], preprocessor: ColumnTransformer) -> pd.DataFrame:
    transformed = preprocessor.fit_transform(frame[feature_columns])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed_columns = preprocessor.get_feature_names_out()
    return pd.DataFrame(transformed, columns=transformed_columns, index=frame.index)


def train_and_evaluate(settings: TrainSettings) -> tuple[ModelResult, Path, list[dict[str, float]]]:
    frame = load_feature_table(settings.feature_store_path, settings.target_column, settings.time_column, settings.group_column)
    if len(frame) < settings.min_rows:
        raise ValueError(f"Need at least {settings.min_rows} rows, but found {len(frame)}.")

    feature_frame, y, numeric_columns, categorical_columns, feature_columns = build_feature_matrix(
        frame,
        settings.target_column,
        settings.time_column,
        settings.group_column,
    )

    X_train_raw, X_test_raw, y_train, y_test = temporal_train_test_split(
        feature_frame,
        y,
        settings.test_fraction,
        settings.time_column,
    )

    usable_feature_columns = [column for column in feature_columns if X_train_raw[column].notna().any()]
    usable_numeric_columns = [column for column in numeric_columns if column in usable_feature_columns]
    usable_categorical_columns = [column for column in categorical_columns if column in usable_feature_columns]

    if not usable_feature_columns:
        raise ValueError("No usable feature columns were found in the training split.")

    preprocessor = make_preprocessor(usable_numeric_columns, usable_categorical_columns)
    X_train = derive_reportable_frame(X_train_raw, usable_feature_columns, preprocessor)
    X_test_matrix = preprocessor.transform(X_test_raw[usable_feature_columns])
    if hasattr(X_test_matrix, "toarray"):
        X_test_matrix = X_test_matrix.toarray()
    X_test = pd.DataFrame(X_test_matrix, columns=preprocessor.get_feature_names_out(), index=X_test_raw.index)

    candidate_models = train_candidate_models(X_train, y_train, X_test, settings.include_tensorflow)

    evaluation_rows = []
    best_name = None
    best_result: dict[str, float] | None = None
    best_model: Any = None
    best_predictions: np.ndarray | None = None

    for name, (model, predictions) in candidate_models.items():
        metrics = evaluate_predictions(y_test, predictions)
        evaluation_rows.append({"model": name, **metrics})
        if best_result is None or metrics["rmse"] < best_result["rmse"]:
            best_name = name
            best_result = metrics
            best_model = model
            best_predictions = predictions

    assert best_name is not None and best_result is not None and best_model is not None and best_predictions is not None

    trained_at_utc = datetime.now(timezone.utc).isoformat()
    result = ModelResult(
        name=best_name,
        rmse=best_result["rmse"],
        mae=best_result["mae"],
        r2=best_result["r2"],
        trained_at_utc=trained_at_utc,
        target_column=settings.target_column,
        row_count=len(feature_frame),
        train_rows=len(X_train_raw),
        test_rows=len(X_test_raw),
        features=usable_feature_columns,
    )

    registry_path = persist_model(
        settings=settings,
        model_name=best_name,
        model=best_model,
        preprocessor=preprocessor,
        report=result,
        evaluation_rows=evaluation_rows,
        feature_columns=usable_feature_columns,
    )
    return result, registry_path, evaluation_rows


def persist_model(
    settings: TrainSettings,
    model_name: str,
    model: Any,
    preprocessor: ColumnTransformer,
    report: ModelResult,
    evaluation_rows: list[dict[str, float]],
    feature_columns: list[str],
) -> Path:
    settings.model_registry_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    registry_path = settings.model_registry_dir / f"aqi_{report.target_column}_{model_name}_{timestamp}"
    registry_path.mkdir(parents=True, exist_ok=False)

    joblib.dump({"model": model, "preprocessor": preprocessor}, registry_path / "model.joblib")
    (registry_path / "metrics.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    (registry_path / "evaluation.json").write_text(json.dumps(evaluation_rows, indent=2), encoding="utf-8")
    (registry_path / "features.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    (registry_path / "manifest.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "target_column": report.target_column,
                "created_at_utc": report.trained_at_utc,
                "source_feature_store": str(settings.feature_store_path),
                "row_count": report.row_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    latest_pointer = settings.model_registry_dir / "latest_model.json"
    latest_pointer.write_text(
        json.dumps(
            {
                "path": str(registry_path),
                "model_name": model_name,
                "target_column": report.target_column,
                "trained_at_utc": report.trained_at_utc,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return registry_path


def main() -> None:
    args = parse_args()
    settings = load_settings(args)
    result, registry_path, evaluation_rows = train_and_evaluate(settings)

    print("Model comparison:")
    for row in evaluation_rows:
        print(f"- {row['model']}: RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}, R2={row['r2']:.4f}")

    print("Selected best model:")
    print(f"- model: {result.name}")
    print(f"- target: {result.target_column}")
    print(f"- rmse: {result.rmse:.4f}")
    print(f"- mae: {result.mae:.4f}")
    print(f"- r2: {result.r2:.4f}")
    print(f"- registry_path: {registry_path}")


if __name__ == "__main__":
    main()
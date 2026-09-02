from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


DEFAULT_FEATURE_STORE_PATH = Path("data/feature_store/aqi_feature_table.csv")
DEFAULT_MODEL_REGISTRY_DIR = Path("data/model_registry")
DEFAULT_TARGET_COLUMN = "target_aqi_t_plus_1h"
DEFAULT_TIME_COLUMN = "observed_at_utc"
DEFAULT_GROUP_COLUMN = "location_label"

# Columns that are only ever knowable in hindsight relative to *any* target
# horizon. All of these get stripped before training, regardless of which
# target you're fitting -- keep the active target out of this list, it's
# removed separately.
TARGET_COLUMN_PREFIX = "target_aqi_t_plus_"

# High-cardinality free-text columns that duplicate lower-cardinality
# columns already in the table (weather_main covers the same signal as
# weather_description with a fraction of the one-hot columns). Excluded by
# default to avoid overfitting a ~200-row dataset; override with
# --keep-high-cardinality if you want them back once you have more data.
DEFAULT_EXCLUDED_CATEGORICAL = ["weather_description"]


@dataclass(frozen=True)
class TrainSettings:
    feature_store_path: Path
    model_registry_dir: Path
    target_column: str
    time_column: str
    group_column: str
    test_fraction: float
    min_test_rows: int
    max_test_fraction: float
    min_rows: int
    random_state: int
    include_tensorflow: bool
    keep_high_cardinality: bool
    cv_splits: int
    search_iterations: int
    target_mode: str
    current_value_column: str
    backtest_folds: int
    skip_diagnostics: bool
    selection_metric: str
    min_changed_rows_for_selection: int
    round_predictions: bool
    no_ensemble: bool


@dataclass(frozen=True)
class ModelResult:
    name: str
    rmse: float
    mae: float
    r2: float
    baseline_rmse: float
    baseline_mae: float
    beats_baseline: bool
    trained_at_utc: str
    target_column: str
    row_count: int
    train_rows: int
    test_rows: int
    test_fraction_used: float
    features: list[str]
    best_params: dict[str, Any]
    target_mode: str
    current_value_column: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AQI models from the feature store.")
    parser.add_argument("--feature-store-path", default=None)
    parser.add_argument("--model-registry-dir", default=None)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--group-column", default=None)
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Target fraction of rows to hold out once the dataset is large enough to support it.",
    )
    parser.add_argument(
        "--min-test-rows",
        type=int,
        default=5,
        help="Smallest acceptable holdout size in rows, regardless of dataset size.",
    )
    parser.add_argument(
        "--max-test-fraction",
        type=float,
        default=0.3,
        help="Upper bound on holdout fraction, so small datasets don't lose too much training data.",
    )
    parser.add_argument("--min-rows", type=int, default=24)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--include-tensorflow", action="store_true")
    parser.add_argument(
        "--keep-high-cardinality",
        action="store_true",
        help="Keep free-text columns like weather_description as features instead of excluding them.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=3,
        help="Number of TimeSeriesSplit folds used for hyperparameter search within the training rows.",
    )
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=25,
        help="Number of RandomizedSearchCV draws per model during hyperparameter search.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["delta", "absolute"],
        default="delta",
        help=(
            "'delta' (default) trains models to predict the CHANGE from the current AQI "
            "reading and reconstructs the absolute forecast at evaluation time. This makes "
            "the persistence baseline the model's zero-point, so a model can never score "
            "worse than the baseline -- it can only add value on top of it. 'absolute' "
            "reverts to predicting the raw future AQI directly (the original behavior)."
        ),
    )
    parser.add_argument(
        "--current-value-column",
        default="aqi",
        help="Column holding the current AQI reading, used as the reference point for --target-mode delta.",
    )
    parser.add_argument(
        "--backtest-folds",
        type=int,
        default=5,
        help="Number of rolling time-based folds to sanity-check stability with (0 disables). "
        "This is separate from the single reported holdout metric -- it tells you how much "
        "that number would swing under a different split.",
    )
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip the duplicate-row diagnostic printed before training.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["rmse", "changed_rmse"],
        default="rmse",
        help=(
            "How to pick the 'best' model. 'rmse' (default) uses aggregate RMSE across all test "
            "rows, which rewards a model for being conservative on the majority of rows where AQI "
            "doesn't change -- it can pick a model that's actually worse at anticipating real "
            "changes. 'changed_rmse' selects based on RMSE restricted to rows where AQI actually "
            "moved, which is arguably the metric that matters, but only reliable once you have "
            "enough changed rows in the holdout (the script warns and falls back to 'rmse' if "
            "there are fewer than --min-changed-rows-for-selection)."
        ),
    )
    parser.add_argument(
        "--min-changed-rows-for-selection",
        type=int,
        default=15,
        help="Minimum changed rows in the holdout before --selection-metric changed_rmse is trusted "
        "over aggregate RMSE. Below this, selection silently falls back to 'rmse' and a note is printed.",
    )
    parser.add_argument(
        "--round-predictions",
        action="store_true",
        help="Round predicted AQI to the nearest integer before scoring, matching how AQI is actually "
        "reported (a 1-5 category). Can reduce noise-driven RMSE from continuous model outputs.",
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Skip the automatic ensemble candidate (mean of xgboost/random_forest/elastic_net delta "
        "predictions). The ensemble is included and eligible for selection by default.",
    )
    return parser.parse_args()


def load_settings(args: argparse.Namespace) -> TrainSettings:
    feature_store_path = Path(args.feature_store_path or DEFAULT_FEATURE_STORE_PATH)
    model_registry_dir = Path(args.model_registry_dir or DEFAULT_MODEL_REGISTRY_DIR)
    target_column = args.target_column or DEFAULT_TARGET_COLUMN
    time_column = args.time_column or DEFAULT_TIME_COLUMN
    group_column = args.group_column or DEFAULT_GROUP_COLUMN

    if not 0 < args.test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if not 0 < args.max_test_fraction < 1:
        raise ValueError("max_test_fraction must be between 0 and 1.")
    if args.min_test_rows < 1:
        raise ValueError("min_test_rows must be at least 1.")
    if args.cv_splits < 2:
        raise ValueError("cv_splits must be at least 2.")

    return TrainSettings(
        feature_store_path=feature_store_path,
        model_registry_dir=model_registry_dir,
        target_column=target_column,
        time_column=time_column,
        group_column=group_column,
        test_fraction=args.test_fraction,
        min_test_rows=args.min_test_rows,
        max_test_fraction=args.max_test_fraction,
        min_rows=args.min_rows,
        random_state=args.random_state,
        include_tensorflow=bool(args.include_tensorflow),
        keep_high_cardinality=bool(args.keep_high_cardinality),
        cv_splits=args.cv_splits,
        search_iterations=args.search_iterations,
        target_mode=args.target_mode,
        current_value_column=args.current_value_column,
        backtest_folds=args.backtest_folds,
        skip_diagnostics=bool(args.skip_diagnostics),
        selection_metric=args.selection_metric,
        min_changed_rows_for_selection=args.min_changed_rows_for_selection,
        round_predictions=bool(args.round_predictions),
        no_ensemble=bool(args.no_ensemble),
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

    # Cyclic encoding of hour-of-day. A raw 0-23 integer makes hour 23 and
    # hour 0 look maximally distant even though they're one hour apart --
    # this matters for linear models (ElasticNet) trying to capture a daily
    # pollution cycle (rush hour, overnight inversion, etc.), and costs
    # nothing for tree models that don't need it.
    if "hour" in enriched.columns:
        hour_radians = 2 * np.pi * enriched["hour"].astype(float) / 24.0
        enriched["hour_sin"] = np.sin(hour_radians)
        enriched["hour_cos"] = np.cos(hour_radians)
    return enriched


def build_feature_matrix(
    frame: pd.DataFrame,
    target_column: str,
    time_column: str,
    group_column: str,
    keep_high_cardinality: bool,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], list[str]]:
    prepared = add_time_numeric_features(frame, time_column)
    y = pd.to_numeric(prepared[target_column], errors="coerce")
    prepared = prepared.loc[y.notna()].copy()
    y = y.loc[y.notna()].astype(float)

    # Drop the active target plus every *other* horizon's target column --
    # those are equally unknown at inference time and leaving them in is
    # leakage, even though the original script only dropped the active one.
    other_target_columns = [
        column
        for column in prepared.columns
        if column.startswith(TARGET_COLUMN_PREFIX) and column != target_column
    ]
    drop_columns = {target_column, "obs_ts", *other_target_columns}
    feature_frame = prepared.drop(columns=[column for column in drop_columns if column in prepared.columns])

    if not keep_high_cardinality:
        feature_frame = feature_frame.drop(
            columns=[column for column in DEFAULT_EXCLUDED_CATEGORICAL if column in feature_frame.columns]
        )

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


def compute_adaptive_test_size(
    n_rows: int,
    target_fraction: float,
    min_test_rows: int,
    max_test_fraction: float,
) -> int:
    """Derive a holdout size in rows purely from the current dataset size.

    No absolute row-count thresholds are hardcoded. Behavior scales automatically:
      - Tiny datasets: the floor (`min_test_rows`) dominates, so evaluation still
        has enough points to be non-degenerate (avoids the classic n_test=1 -> R2=0 trap).
      - Mid-size datasets: the target fraction (e.g. 20%) applies directly.
      - The `max_test_fraction` ceiling stops the holdout from ever eating too much
        of a still-small dataset's training signal.
    At least 1 row is always reserved for training.
    """
    if n_rows < 2:
        raise ValueError("Need at least 2 usable rows to form any train/test split.")

    fraction_based = math.ceil(n_rows * target_fraction)
    test_size = max(fraction_based, min_test_rows)

    ceiling = max(1, math.floor(n_rows * max_test_fraction))
    test_size = min(test_size, ceiling)

    # Never let the test split consume the entire dataset.
    test_size = min(test_size, n_rows - 1)
    test_size = max(test_size, 1)
    return test_size


def temporal_train_test_split(
    frame: pd.DataFrame,
    y: pd.Series,
    time_column: str,
    target_fraction: float,
    min_test_rows: int,
    max_test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, float]:
    ordered_index = frame.sort_values(time_column).index
    n_rows = len(ordered_index)

    test_size = compute_adaptive_test_size(
        n_rows=n_rows,
        target_fraction=target_fraction,
        min_test_rows=min_test_rows,
        max_test_fraction=max_test_fraction,
    )

    test_index = ordered_index[-test_size:]
    train_index = ordered_index[:-test_size]
    if len(train_index) == 0:
        raise ValueError("Not enough rows to create a temporal train/test split.")

    test_fraction_used = test_size / n_rows
    return frame.loc[train_index], frame.loc[test_index], y.loc[train_index], y.loc[test_index], test_fraction_used


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


DEFAULT_STALENESS_CHECK_COLUMNS = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3", "aqi"]


def diagnose_duplicate_rows(
    frame: pd.DataFrame,
    time_column: str,
    group_column: str,
    check_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Flag stale scrapes: consecutive rows whose pollutant readings are
    byte-identical to the previous scrape for the same location.

    This is a common artifact of polling a weather/pollution API faster than
    its source data actually refreshes -- you get the same underlying
    reading twice under two different timestamps, even though other fields
    (temperature, wind, etc.) may still tick over. Rows like that aren't
    independent pollution observations; if enough of them exist, holdout
    metrics look better than they'd be on genuinely fresh readings, because
    the model (or even the naive baseline) is effectively being tested on
    data it has already seen a near-copy of.
    """
    columns = [c for c in (check_columns or DEFAULT_STALENESS_CHECK_COLUMNS) if c in frame.columns]
    if not columns:
        return {"total_rows": len(frame), "stale_pollutant_rows": 0, "stale_pollutant_pct": 0.0, "checked_columns": []}

    sorted_frame = frame.sort_values([group_column, time_column])
    shifted = sorted_frame.groupby(group_column)[columns].shift(1)
    is_stale = (sorted_frame[columns].to_numpy() == shifted.to_numpy()).all(axis=1) & shifted.notna().all(axis=1).to_numpy()
    stale_count = int(is_stale.sum())
    total = len(sorted_frame)
    return {
        "total_rows": total,
        "stale_pollutant_rows": stale_count,
        "stale_pollutant_pct": round(100 * stale_count / total, 1) if total else 0.0,
        "checked_columns": columns,
    }


def rolling_origin_backtest(
    frame: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    time_column: str,
    current_value_column: str,
    target_mode: str,
    n_folds: int,
    random_state: int,
) -> list[dict[str, float]]:
    """Evaluate a fixed, lightly-regularized model across several time-based
    folds instead of one holdout, so you get a distribution of RMSE rather
    than a single number that could just be a lucky (or unlucky) split.
    Uses a modest, fixed RandomForest config -- this is a sanity check on
    stability, not a hyperparameter search.
    """
    ordered_index = frame.sort_values(time_column).index
    n_rows = len(ordered_index)
    fold_size = max(5, n_rows // (n_folds + 3))
    results = []

    for fold in range(n_folds):
        test_end = n_rows - fold * fold_size
        test_start = test_end - fold_size
        train_end = test_start
        if train_end < fold_size * 2 or test_start < 0:
            break
        train_idx = ordered_index[:train_end]
        test_idx = ordered_index[test_start:test_end]

        X_train_raw = frame.loc[train_idx]
        X_test_raw = frame.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]

        preprocessor = make_preprocessor(numeric_columns, categorical_columns)
        X_train = preprocessor.fit_transform(X_train_raw[feature_columns])
        X_test = preprocessor.transform(X_test_raw[feature_columns])
        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()
            X_test = X_test.toarray()

        if target_mode == "delta":
            current_train = X_train_raw[current_value_column].astype(float)
            current_test = X_test_raw[current_value_column].astype(float)
            y_train_model = y_train - current_train
        else:
            current_test = None
            y_train_model = y_train

        model = RandomForestRegressor(
            n_estimators=150, max_depth=3, min_samples_leaf=5, random_state=random_state, n_jobs=-1
        )
        model.fit(X_train, y_train_model)
        raw_pred = model.predict(X_test)
        abs_pred = current_test.to_numpy() + raw_pred if target_mode == "delta" else raw_pred

        baseline_pred = X_test_raw[current_value_column].astype(float).to_numpy()
        results.append(
            {
                "fold": fold,
                "test_start": str(X_test_raw[time_column].min()),
                "test_end": str(X_test_raw[time_column].max()),
                "n_test": len(test_idx),
                "model_rmse": float(np.sqrt(mean_squared_error(y_test, abs_pred))),
                "baseline_rmse": float(np.sqrt(mean_squared_error(y_test, baseline_pred))),
            }
        )
    return results


def evaluate_changed_rows_only(
    y_true: pd.Series, y_pred: np.ndarray, current_values: pd.Series
) -> dict[str, Any]:
    """Restrict evaluation to rows where the AQI actually moved.

    Aggregate RMSE on a mostly-static target is dominated by the easy rows
    where nothing happened -- both the model and the naive baseline get
    those "free." The rows that matter (and that a persistence baseline is
    incapable of getting right) are the ones where AQI actually changed.
    This reports model vs. baseline error on exactly that subset, which is
    the honest test of whether there's learnable signal at all.
    """
    actual_delta = y_true.to_numpy() - current_values.to_numpy()
    changed_mask = actual_delta != 0
    n_changed = int(changed_mask.sum())
    if n_changed == 0:
        return {"n_changed": 0, "model_rmse_changed": float("nan"), "baseline_rmse_changed": float("nan")}

    y_true_changed = y_true.to_numpy()[changed_mask]
    y_pred_changed = np.asarray(y_pred)[changed_mask]
    baseline_pred_changed = current_values.to_numpy()[changed_mask]

    return {
        "n_changed": n_changed,
        "n_total": len(y_true),
        "changed_pct": round(100 * n_changed / len(y_true), 1),
        "model_rmse_changed": float(np.sqrt(mean_squared_error(y_true_changed, y_pred_changed))),
        "baseline_rmse_changed": float(np.sqrt(mean_squared_error(y_true_changed, baseline_pred_changed))),
    }


def compute_persistence_baseline(
    frame_with_current_aqi: pd.DataFrame, y_test: pd.Series, current_aqi_column: str = "aqi"
) -> dict[str, float]:
    """Naive forecast: assume the target equals the most recently observed AQI.

    This is the bar every model needs to clear. AQI is autocorrelated hour to
    hour, so "predict no change" is a deceptively strong baseline on small,
    fairly stable series -- if a model can't beat it, it isn't adding value.
    """
    if current_aqi_column not in frame_with_current_aqi.columns:
        return {"rmse": float("nan"), "mae": float("nan")}
    naive_pred = frame_with_current_aqi.loc[y_test.index, current_aqi_column].astype(float).to_numpy()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, naive_pred))),
        "mae": float(mean_absolute_error(y_test, naive_pred)),
    }


def build_search_spaces(random_state: int) -> dict[str, tuple[Any, dict[str, Any]]]:
    """Model + hyperparameter distribution pairs, sized for small tabular data.

    Ranges are deliberately conservative (fewer trees, shallower depth,
    stronger regularization ceilings) since a ~200-row training set will
    overfit fast with the defaults a larger dataset could tolerate.
    """
    xgboost_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )
    xgboost_space = {
        "n_estimators": randint(50, 250),
        "max_depth": randint(2, 5),
        "learning_rate": uniform(0.01, 0.19),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_alpha": uniform(0.0, 1.0),
        "reg_lambda": uniform(0.5, 2.5),
        "min_child_weight": randint(1, 6),
    }

    random_forest_model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    random_forest_space = {
        "n_estimators": randint(100, 400),
        "max_depth": randint(2, 8),
        "min_samples_leaf": randint(1, 8),
        "max_features": uniform(0.4, 0.6),
    }

    elastic_net_model = ElasticNet(random_state=random_state, max_iter=5000)
    elastic_net_space = {
        "alpha": uniform(0.001, 2.0),
        "l1_ratio": uniform(0.0, 1.0),
    }

    return {
        "xgboost": (xgboost_model, xgboost_space),
        "random_forest": (random_forest_model, random_forest_space),
        "elastic_net": (elastic_net_model, elastic_net_space),
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
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.fit(
        train_matrix,
        y_train_array,
        validation_split=0.2 if len(train_matrix) > 10 else 0.0,
        epochs=80,
        batch_size=min(16, len(train_matrix)),
        verbose=0,
    )
    predictions = model.predict(test_matrix, verbose=0).reshape(-1)
    return model, predictions


class MeanEnsembleRegressor:
    """Averages predictions from a set of already-fitted regressors.

    Kept intentionally trivial -- this is not a stacked/weighted ensemble,
    just an unweighted mean, which is enough to reduce prediction variance
    when the underlying models disagree without adding any new overfitting
    risk of its own.
    """

    def __init__(self, models: list[Any]) -> None:
        self.models = models

    def predict(self, X: Any) -> np.ndarray:
        return np.mean([model.predict(X) for model in self.models], axis=0)


def train_candidate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    include_tensorflow: bool,
    cv_splits: int,
    search_iterations: int,
    random_state: int,
    skip_ensemble: bool = False,
) -> dict[str, tuple[Any, np.ndarray, dict[str, Any]]]:
    candidates: dict[str, tuple[Any, np.ndarray, dict[str, Any]]] = {}

    # TimeSeriesSplit keeps every CV fold's validation fold *after* its
    # training fold in time, matching how the model will actually be used.
    # n_splits is capped so each fold still has enough rows to be meaningful
    # on small datasets.
    effective_splits = max(2, min(cv_splits, len(X_train) // 10 or 2))
    cv = TimeSeriesSplit(n_splits=effective_splits)

    for name, (model, param_distributions) in build_search_spaces(random_state).items():
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=search_iterations,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        predictions = best_model.predict(X_test)
        candidates[name] = (best_model, predictions, search.best_params_)

    if not skip_ensemble and len(candidates) >= 2:
        # Simple mean-of-predictions ensemble across whatever candidates were
        # just trained (in delta-space, same as the individual models). Free
        # variance reduction when the models disagree somewhat -- no new
        # training cost, just averages predictions already computed above.
        # Store a real predictable object (not just the averaged array) so
        # the ensemble can still be used for inference if it's selected and
        # persisted to the registry.
        sub_models = [model for model, _, _ in candidates.values() if model is not None]
        ensemble_model = MeanEnsembleRegressor(sub_models)
        stacked = np.mean([predictions for _, predictions, _ in candidates.values()], axis=0)
        candidates["ensemble_mean"] = (ensemble_model, stacked, {"members": list(candidates.keys())})

    if include_tensorflow:
        tf_result = try_train_tensorflow_model(X_train, y_train, X_test)
        if tf_result is not None:
            tf_model, tf_predictions = tf_result
            candidates["tensorflow_dense"] = (tf_model, tf_predictions, {})

    return candidates


def derive_reportable_frame(frame: pd.DataFrame, feature_columns: list[str], preprocessor: ColumnTransformer) -> pd.DataFrame:
    transformed = preprocessor.fit_transform(frame[feature_columns])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed_columns = preprocessor.get_feature_names_out()
    return pd.DataFrame(transformed, columns=transformed_columns, index=frame.index)


def train_and_evaluate(settings: TrainSettings) -> tuple[ModelResult, Path, list[dict[str, Any]]]:
    frame = load_feature_table(settings.feature_store_path, settings.target_column, settings.time_column, settings.group_column)
    if len(frame) < settings.min_rows:
        raise ValueError(f"Need at least {settings.min_rows} rows, but found {len(frame)}.")

    feature_frame, y, numeric_columns, categorical_columns, feature_columns = build_feature_matrix(
        frame,
        settings.target_column,
        settings.time_column,
        settings.group_column,
        settings.keep_high_cardinality,
    )

    X_train_raw, X_test_raw, y_train, y_test, test_fraction_used = temporal_train_test_split(
        feature_frame,
        y,
        settings.time_column,
        target_fraction=settings.test_fraction,
        min_test_rows=settings.min_test_rows,
        max_test_fraction=settings.max_test_fraction,
    )

    baseline_metrics = compute_persistence_baseline(feature_frame, y_test)

    if settings.target_mode == "delta":
        if settings.current_value_column not in X_train_raw.columns:
            raise ValueError(
                f"--current-value-column '{settings.current_value_column}' not found in the feature table; "
                "pass --target-mode absolute or point at the right column."
            )
        current_values_train = X_train_raw[settings.current_value_column].astype(float)
        current_values_test = X_test_raw[settings.current_value_column].astype(float)
        y_train_for_model = y_train - current_values_train
    else:
        current_values_test = None
        y_train_for_model = y_train

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

    candidate_models = train_candidate_models(
        X_train,
        y_train_for_model,
        X_test,
        settings.include_tensorflow,
        settings.cv_splits,
        settings.search_iterations,
        settings.random_state,
        skip_ensemble=settings.no_ensemble,
    )

    evaluation_rows = []
    best_name = None
    best_result: dict[str, float] | None = None
    best_model: Any = None
    best_params: dict[str, Any] = {}

    n_changed_available = None
    effective_selection_metric = settings.selection_metric
    if settings.selection_metric == "changed_rmse":
        # Determine changed-row count once, from y_test, to decide if the fallback applies.
        if settings.target_mode == "delta":
            n_changed_available = int((y_test.to_numpy() - current_values_test.to_numpy() != 0).sum())
        if not n_changed_available or n_changed_available < settings.min_changed_rows_for_selection:
            print(
                f"Note: --selection-metric changed_rmse requested but only {n_changed_available or 0} changed "
                f"rows in the holdout (need >= {settings.min_changed_rows_for_selection}). Falling back to "
                "aggregate 'rmse' for model selection -- there isn't enough signal to trust changed-only "
                "comparisons yet."
            )
            effective_selection_metric = "rmse"

    for name, (model, predictions, params) in candidate_models.items():
        # predictions are in delta-space when target_mode == "delta"; add the
        # current AQI back on to get an absolute forecast that's directly
        # comparable to the persistence baseline and to y_test.
        if settings.target_mode == "delta":
            absolute_predictions = current_values_test.to_numpy() + np.asarray(predictions)
        else:
            absolute_predictions = np.asarray(predictions)
        if settings.round_predictions:
            absolute_predictions = np.round(absolute_predictions)
        metrics = evaluate_predictions(y_test, absolute_predictions)
        changed_metrics = (
            evaluate_changed_rows_only(y_test, absolute_predictions, current_values_test)
            if settings.target_mode == "delta"
            else {}
        )
        beats_baseline = (
            metrics["rmse"] < baseline_metrics["rmse"] if not math.isnan(baseline_metrics["rmse"]) else None
        )
        evaluation_rows.append(
            {"model": name, **metrics, "beats_baseline": beats_baseline, "best_params": params, **changed_metrics}
        )

        if effective_selection_metric == "changed_rmse" and not math.isnan(changed_metrics.get("model_rmse_changed", float("nan"))):
            selection_score = changed_metrics["model_rmse_changed"]
        else:
            selection_score = metrics["rmse"]

        if best_result is None or selection_score < best_result.get("_selection_score", float("inf")):
            best_name = name
            best_result = {**metrics, "_selection_score": selection_score}
            best_model = model
            best_params = params

    assert best_name is not None and best_result is not None and best_model is not None

    trained_at_utc = datetime.now(timezone.utc).isoformat()
    beats_baseline_final = (
        best_result["rmse"] < baseline_metrics["rmse"] if not math.isnan(baseline_metrics["rmse"]) else False
    )
    result = ModelResult(
        name=best_name,
        rmse=best_result["rmse"],
        mae=best_result["mae"],
        r2=best_result["r2"],
        baseline_rmse=baseline_metrics["rmse"],
        baseline_mae=baseline_metrics["mae"],
        beats_baseline=beats_baseline_final,
        trained_at_utc=trained_at_utc,
        target_column=settings.target_column,
        row_count=len(feature_frame),
        train_rows=len(X_train_raw),
        test_rows=len(X_test_raw),
        test_fraction_used=test_fraction_used,
        features=usable_feature_columns,
        best_params=best_params,
        target_mode=settings.target_mode,
        current_value_column=settings.current_value_column,
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
    evaluation_rows: list[dict[str, Any]],
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
                "test_fraction_used": report.test_fraction_used,
                "target_mode": report.target_mode,
                "current_value_column": report.current_value_column,
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

    if not settings.skip_diagnostics:
        raw_frame = load_feature_table(
            settings.feature_store_path, settings.target_column, settings.time_column, settings.group_column
        )
        dup_report = diagnose_duplicate_rows(raw_frame, settings.time_column, settings.group_column)
        print(
            f"Data quality check: {dup_report['stale_pollutant_rows']}/{dup_report['total_rows']} rows "
            f"({dup_report['stale_pollutant_pct']}%) have pollutant readings ({', '.join(dup_report['checked_columns'])}) "
            "byte-identical to the immediately preceding scrape for the same location. High values here mean "
            "your source (e.g. OpenWeather) is refreshing slower than you're polling -- those rows are stale "
            "copies, not independent observations, and can make holdout metrics look better than they'd be "
            "on truly fresh readings."
        )
        print()

    result, registry_path, evaluation_rows = train_and_evaluate(settings)

    print(f"Persistence baseline: RMSE={result.baseline_rmse:.4f}, MAE={result.baseline_mae:.4f}")
    print("Model comparison:")
    for row in evaluation_rows:
        beats = row.get("beats_baseline")
        beats_str = "beats baseline" if beats else ("below baseline" if beats is False else "baseline n/a")
        print(f"- {row['model']}: RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}, R2={row['r2']:.4f} ({beats_str})")
        if row.get("n_changed"):
            print(
                f"    on rows where AQI actually changed ({row['n_changed']}/{row['n_total']} = {row['changed_pct']}%): "
                f"model_rmse={row['model_rmse_changed']:.4f} vs baseline_rmse={row['baseline_rmse_changed']:.4f}"
            )

    print("Selected best model:")
    print(f"- model: {result.name}")
    print(f"- target: {result.target_column}")
    print(f"- rmse: {result.rmse:.4f}")
    print(f"- mae: {result.mae:.4f}")
    print(f"- r2: {result.r2:.4f}")
    print(f"- beats persistence baseline: {result.beats_baseline}")
    print(f"- target_mode: {result.target_mode} (current_value_column: {result.current_value_column})")
    print(f"- best_params: {result.best_params}")
    print(f"- rows: {result.row_count} (train={result.train_rows}, test={result.test_rows}, test_fraction_used={result.test_fraction_used:.3f})")
    print(f"- registry_path: {registry_path}")

    if settings.backtest_folds > 0:
        frame = load_feature_table(
            settings.feature_store_path, settings.target_column, settings.time_column, settings.group_column
        )
        feature_frame, y, numeric_columns, categorical_columns, feature_columns = build_feature_matrix(
            frame, settings.target_column, settings.time_column, settings.group_column, settings.keep_high_cardinality
        )
        backtest_rows = rolling_origin_backtest(
            feature_frame,
            y,
            feature_columns,
            numeric_columns,
            categorical_columns,
            settings.time_column,
            settings.current_value_column,
            settings.target_mode,
            settings.backtest_folds,
            settings.random_state,
        )
        if backtest_rows:
            model_rmses = [row["model_rmse"] for row in backtest_rows]
            baseline_rmses = [row["baseline_rmse"] for row in backtest_rows]
            wins = sum(1 for m, b in zip(model_rmses, baseline_rmses) if m < b)
            print()
            print(
                f"Rolling backtest ({len(backtest_rows)} folds, fixed light RandomForest -- "
                "this checks STABILITY, not the tuned model above):"
            )
            for row in backtest_rows:
                verdict = "beats baseline" if row["model_rmse"] < row["baseline_rmse"] else "below baseline"
                print(
                    f"  fold {row['fold']} [{row['test_start']} -> {row['test_end']}, n={row['n_test']}]: "
                    f"model_rmse={row['model_rmse']:.4f} baseline_rmse={row['baseline_rmse']:.4f} ({verdict})"
                )
            print(
                f"  summary: model beat baseline in {wins}/{len(backtest_rows)} folds | "
                f"mean model RMSE={np.mean(model_rmses):.4f} (std={np.std(model_rmses):.4f}) | "
                f"mean baseline RMSE={np.mean(baseline_rmses):.4f} (std={np.std(baseline_rmses):.4f})"
            )
            print(
                "  If wins are inconsistent across folds or the std is large relative to the RMSE gap, "
                "treat the single holdout number above as noisy rather than a confirmed win."
            )

            win_rate = wins / len(backtest_rows)
            mean_model = float(np.mean(model_rmses))
            mean_baseline = float(np.mean(baseline_rmses))
            gap_pct = (mean_baseline - mean_model) / mean_baseline * 100 if mean_baseline else 0.0

            print()
            print("=" * 70)
            if win_rate >= 0.6 and mean_model < mean_baseline:
                verdict = (
                    f"VERDICT: Model shows a real edge -- won {wins}/{len(backtest_rows)} folds and mean RMSE "
                    f"is {gap_pct:.1f}% below baseline. Worth trusting, still worth re-checking as data grows."
                )
            elif win_rate <= 0.4 or mean_model >= mean_baseline:
                verdict = (
                    f"VERDICT: NOT a confirmed win yet. Model beat baseline in only {wins}/{len(backtest_rows)} "
                    f"folds and mean RMSE is {'above' if mean_model >= mean_baseline else 'about equal to'} the "
                    "persistence baseline. Don't deploy this as-is. The fix is more data (keep scraping), not "
                    "more model tuning -- re-run this script periodically and watch the win rate and changed-row "
                    "count (currently low) trend upward."
                )
            else:
                verdict = (
                    f"VERDICT: Inconclusive ({wins}/{len(backtest_rows)} folds, {gap_pct:.1f}% RMSE gap). "
                    "Too close to call with this much data. Keep collecting and re-run."
                )
            print(verdict)
            print("=" * 70)


if __name__ == "__main__":
    main()
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aqi_train import TrainSettings, train_and_evaluate


def test_trainer_smoke(tmp_path):
    """Smoke test: run trainer on the project feature store and ensure it produces a model registry entry."""
    feature_store = Path("data/feature_store/aqi_feature_table.csv")
    assert feature_store.exists(), "Feature store not found: data/feature_store/aqi_feature_table.csv"

    model_registry_dir = tmp_path / "model_registry_test"
    settings = TrainSettings(
        feature_store_path=feature_store,
        model_registry_dir=model_registry_dir,
        target_column="target_aqi_t_plus_1h",
        time_column="observed_at_utc",
        group_column="location_label",
        test_fraction=0.2,
        min_test_rows=5,
        max_test_fraction=0.3,
        min_rows=24,
        random_state=42,
        include_tensorflow=False,
        keep_high_cardinality=False,
        cv_splits=2,
        search_iterations=2,
        target_mode="delta",
        current_value_column="aqi",
        backtest_folds=0,
        skip_diagnostics=True,
        selection_metric="rmse",
        min_changed_rows_for_selection=5,
        round_predictions=False,
        no_ensemble=True,
    )

    result, registry_path, evaluation_rows = train_and_evaluate(settings)

    # Basic assertions
    assert registry_path.exists(), "Model registry path was not created"
    assert result.rmse >= 0
    assert isinstance(evaluation_rows, list) and len(evaluation_rows) >= 1

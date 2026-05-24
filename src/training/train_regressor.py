"""Train regression models for next-step p95 latency prediction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.utils.constants import DEFAULT_TARGET_REGRESSION, SERVICE_COLUMNS_TO_DROP, SOURCE_COLUMNS_TO_DROP
from src.utils.dataframe import align_columns, split_features_target
from src.utils.io import load_yaml, read_manifest_index, read_table, save_json, save_table
from src.utils.paths import get_config_path, get_manifest_files_index_path, get_models_dir, get_predictions_dir, get_reports_dir, make_dir, to_path
from src.utils.validation import validate_non_empty_dataframe, validate_target_column


@dataclass(frozen=True)
class TrainRegressorConfig:
    target_col: str = DEFAULT_TARGET_REGRESSION
    sample_frac: float | None = 0.10
    max_rows_per_file: int | None = None
    random_state: int = 42
    models: tuple[str, ...] = ("dummy", "random_forest", "hist_gradient_boosting")


def _filter_manifest(df: pd.DataFrame, role: str, scenario: str | None) -> pd.DataFrame:
    result = df[df["file_role"] == role].copy()
    if scenario and scenario != "all":
        mask = result["relative_path_from_part"].astype(str).str.contains(f"/{scenario}/|^{scenario}/", regex=True)
        result = result[mask].copy()
    return result


def load_role_dataset(manifest_df: pd.DataFrame, role: str, scenario: str | None = None, sample_frac: float | None = None, max_rows_per_file: int | None = None, random_state: int = 42) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, row in _filter_manifest(manifest_df, role, scenario).iterrows():
        path = Path(row["absolute_path"])
        try:
            df = read_table(path)
        except Exception:
            continue
        if max_rows_per_file is not None and len(df) > max_rows_per_file:
            df = df.head(max_rows_per_file).copy()
        if sample_frac is not None and 0 < sample_frac < 1 and len(df) > 1:
            df = df.sample(frac=sample_frac, random_state=random_state).sort_index()
        df["dataset_source_path"] = str(path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No readable files for role={role}, scenario={scenario}")
    result = pd.concat(frames, ignore_index=True)
    validate_non_empty_dataframe(result, f"{role} dataset")
    return result


def build_regression_models(random_state: int) -> dict[str, Pipeline]:
    return {
        "dummy": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyRegressor(strategy="median"))]),
        "random_forest": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=3, n_jobs=-1, random_state=random_state))]),
        "hist_gradient_boosting": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(max_iter=180, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=0.05, random_state=random_state))]),
    }


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {"mae": float(mean_absolute_error(y_true, y_pred)), "rmse": float(np.sqrt(mse)), "r2": float(r2_score(y_true, y_pred))}


def prepare_xy(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    validate_target_column(df, target_col, "regression dataset")
    other_targets = [c for c in df.columns if c.startswith("target_") and c != target_col]
    drop_cols = SOURCE_COLUMNS_TO_DROP + SERVICE_COLUMNS_TO_DROP + other_targets + ["dataset_source_path"]
    return split_features_target(df, target_col, drop_columns=drop_cols, numeric_only=True)


def train_and_evaluate_regressors(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, config: TrainRegressorConfig) -> tuple[dict[str, Any], dict[str, Pipeline], dict[str, pd.DataFrame]]:
    x_train, y_train = prepare_xy(train_df, config.target_col)
    x_val, y_val = prepare_xy(val_df, config.target_col)
    x_test, y_test = prepare_xy(test_df, config.target_col)
    x_val = align_columns(x_train, x_val)
    x_test = align_columns(x_train, x_test)
    metrics: dict[str, Any] = {"target_col": config.target_col, "feature_count": x_train.shape[1], "train_rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df), "models": {}}
    fitted: dict[str, Pipeline] = {}
    preds: dict[str, pd.DataFrame] = {}
    for name, model in build_regression_models(config.random_state).items():
        if name not in config.models:
            continue
        model.fit(x_train, y_train)
        metrics["models"][name] = {
            "train": regression_metrics(y_train, model.predict(x_train)),
            "val": regression_metrics(y_val, model.predict(x_val)),
            "test": regression_metrics(y_test, model.predict(x_test)),
        }
        y_pred = model.predict(x_test)
        preds[name] = pd.DataFrame({"split": "test", "y_true": y_test.to_numpy(), "y_pred": y_pred})
        fitted[name] = model
    return metrics, fitted, preds


def run_training(config_path: str | Path | None = None, scenario: str | None = None) -> dict[str, Any]:
    config = load_yaml(config_path or get_config_path())
    training = config.get("training", {})
    random_state = int(config.get("project", {}).get("random_state", 42))
    scenario = scenario or training.get("default_scenario", "all")
    train_config = TrainRegressorConfig(
        target_col=config.get("target", {}).get("regression", DEFAULT_TARGET_REGRESSION),
        sample_frac=training.get("sample_frac", 0.10),
        max_rows_per_file=training.get("max_rows_per_file"),
        random_state=random_state,
        models=tuple(training.get("models_regression", ["dummy", "random_forest", "hist_gradient_boosting"])),
    )
    manifest_df = read_manifest_index(get_manifest_files_index_path(config))
    train_df = load_role_dataset(manifest_df, "train", scenario, train_config.sample_frac, train_config.max_rows_per_file, random_state)
    val_df = load_role_dataset(manifest_df, "val", scenario, train_config.sample_frac, train_config.max_rows_per_file, random_state)
    test_df = load_role_dataset(manifest_df, "test", scenario, train_config.sample_frac, train_config.max_rows_per_file, random_state)
    metrics, models, predictions = train_and_evaluate_regressors(train_df, val_df, test_df, train_config)
    metrics["scenario"] = scenario
    models_dir = make_dir(get_models_dir(config) / "regression" / str(scenario))
    pred_dir = make_dir(get_predictions_dir(config) / "regression" / str(scenario))
    report_dir = make_dir(get_reports_dir(config) / "regression" / str(scenario))
    for name, model in models.items():
        joblib.dump(model, models_dir / f"{name}.joblib")
        save_table(predictions[name], pred_dir / f"{name}_test_predictions.parquet")
    save_json(metrics, report_dir / "regression_metrics.json")
    rows = [{"model": m, "split": s, **vals} for m, splits in metrics["models"].items() for s, vals in splits.items()]
    save_table(pd.DataFrame(rows), report_dir / "regression_metrics.csv")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(get_config_path()))
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()
    print(run_training(args.config, args.scenario))


if __name__ == "__main__":
    main()

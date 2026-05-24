"""Train classifiers for next-step performance anomaly prediction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.training.train_regressor import load_role_dataset
from src.utils.constants import DEFAULT_TARGET_CLASSIFICATION, SERVICE_COLUMNS_TO_DROP, SOURCE_COLUMNS_TO_DROP
from src.utils.dataframe import align_columns, split_features_target
from src.utils.io import load_yaml, read_manifest_index, save_json, save_table
from src.utils.paths import get_config_path, get_manifest_files_index_path, get_models_dir, get_predictions_dir, get_reports_dir, make_dir
from src.utils.validation import validate_target_column


@dataclass(frozen=True)
class TrainClassifierConfig:
    target_col: str = DEFAULT_TARGET_CLASSIFICATION
    sample_frac: float | None = 0.10
    max_rows_per_file: int | None = None
    random_state: int = 42
    models: tuple[str, ...] = ("dummy", "random_forest", "hist_gradient_boosting")


def build_classification_models(random_state: int) -> dict[str, Pipeline]:
    return {
        "dummy": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="most_frequent"))]),
        "random_forest": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=160, max_depth=14, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=random_state))]),
        "hist_gradient_boosting": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingClassifier(max_iter=180, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=0.05, random_state=random_state))]),
    }


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict[str, float]:
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    result["roc_auc"] = float(roc_auc_score(y_true, y_score)) if y_score is not None and len(pd.Series(y_true).unique()) > 1 else float("nan")
    return result


def prepare_xy(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    validate_target_column(df, target_col, "classification dataset")
    other_targets = [c for c in df.columns if c.startswith("target_") and c != target_col]
    drop_cols = SOURCE_COLUMNS_TO_DROP + SERVICE_COLUMNS_TO_DROP + other_targets + ["dataset_source_path"]
    x, y = split_features_target(df, target_col, drop_columns=drop_cols, numeric_only=True)
    return x, y.astype(int)


def _score(model: Pipeline, x: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.shape[1] == 2:
            return proba[:, 1]
    return None


def train_and_evaluate_classifiers(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, config: TrainClassifierConfig) -> tuple[dict[str, Any], dict[str, Pipeline], dict[str, pd.DataFrame]]:
    x_train, y_train = prepare_xy(train_df, config.target_col)
    x_val, y_val = prepare_xy(val_df, config.target_col)
    x_test, y_test = prepare_xy(test_df, config.target_col)
    x_val = align_columns(x_train, x_val)
    x_test = align_columns(x_train, x_test)
    metrics: dict[str, Any] = {"target_col": config.target_col, "feature_count": x_train.shape[1], "train_rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df), "positive_share_train": float(y_train.mean()) if len(y_train) else 0.0, "models": {}}
    fitted: dict[str, Pipeline] = {}
    preds: dict[str, pd.DataFrame] = {}
    for name, model in build_classification_models(config.random_state).items():
        if name not in config.models:
            continue
        model.fit(x_train, y_train)
        metrics["models"][name] = {}
        for split, x_part, y_part in [("train", x_train, y_train), ("val", x_val, y_val), ("test", x_test, y_test)]:
            y_pred = model.predict(x_part)
            metrics["models"][name][split] = classification_metrics(y_part, y_pred, _score(model, x_part))
        y_pred = model.predict(x_test)
        y_score = _score(model, x_test)
        preds[name] = pd.DataFrame({"split": "test", "y_true": y_test.to_numpy(), "y_pred": y_pred, "anomaly_probability": y_score if y_score is not None else y_pred})
        fitted[name] = model
    return metrics, fitted, preds


def run_training(config_path: str | Path | None = None, scenario: str | None = None) -> dict[str, Any]:
    config = load_yaml(config_path or get_config_path())
    training = config.get("training", {})
    random_state = int(config.get("project", {}).get("random_state", 42))
    scenario = scenario or training.get("default_scenario", "all")
    train_config = TrainClassifierConfig(
        target_col=config.get("target", {}).get("main", DEFAULT_TARGET_CLASSIFICATION),
        sample_frac=training.get("sample_frac", 0.10),
        max_rows_per_file=training.get("max_rows_per_file"),
        random_state=random_state,
        models=tuple(training.get("models_classification", ["dummy", "random_forest", "hist_gradient_boosting"])),
    )
    manifest_df = read_manifest_index(get_manifest_files_index_path(config))
    train_df = load_role_dataset(manifest_df, "train", scenario, train_config.sample_frac, train_config.max_rows_per_file, random_state)
    val_df = load_role_dataset(manifest_df, "val", scenario, train_config.sample_frac, train_config.max_rows_per_file, random_state)
    test_df = load_role_dataset(manifest_df, "test", scenario, train_config.sample_frac, train_config.max_rows_per_file, random_state)
    metrics, models, predictions = train_and_evaluate_classifiers(train_df, val_df, test_df, train_config)
    metrics["scenario"] = scenario
    models_dir = make_dir(get_models_dir(config) / "classification" / str(scenario))
    pred_dir = make_dir(get_predictions_dir(config) / "classification" / str(scenario))
    report_dir = make_dir(get_reports_dir(config) / "classification" / str(scenario))
    for name, model in models.items():
        joblib.dump(model, models_dir / f"{name}.joblib")
        save_table(predictions[name], pred_dir / f"{name}_test_predictions.parquet")
    save_json(metrics, report_dir / "classification_metrics.json")
    rows = [{"model": m, "split": s, **vals} for m, splits in metrics["models"].items() for s, vals in splits.items()]
    save_table(pd.DataFrame(rows), report_dir / "classification_metrics.csv")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(get_config_path()))
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()
    print(run_training(args.config, args.scenario))


if __name__ == "__main__":
    main()

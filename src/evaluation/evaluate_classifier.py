"""Evaluate saved classifier prediction files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.utils.io import load_yaml, read_table, save_json, save_table
from src.utils.paths import get_config_path, get_reports_dir, make_dir, to_path


def evaluate_prediction_frame(df: pd.DataFrame) -> dict[str, float]:
    required = {"y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Prediction file is missing columns: {sorted(missing)}")
    y_true = df["y_true"].astype(int)
    y_pred = df["y_pred"].astype(int)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    result["roc_auc"] = float(roc_auc_score(y_true, df["anomaly_probability"])) if "anomaly_probability" in df.columns and len(y_true.unique()) > 1 else float("nan")
    return result


def evaluate_predictions_dir(predictions_dir: str | Path) -> pd.DataFrame:
    predictions_dir = to_path(predictions_dir)
    files = sorted(list(predictions_dir.glob("*.parquet")) + list(predictions_dir.glob("*.csv")))
    if not files:
        raise FileNotFoundError(f"No prediction files found in {predictions_dir}")
    rows = []
    for path in files:
        rows.append({"prediction_file": path.name, **evaluate_prediction_frame(read_table(path))})
    return pd.DataFrame(rows)


def run_evaluation(predictions_dir: str | Path, config_path: str | Path | None = None, output_name: str = "classification_prediction_metrics") -> pd.DataFrame:
    config = load_yaml(config_path or get_config_path())
    reports_dir = make_dir(get_reports_dir(config) / "evaluation")
    metrics_df = evaluate_predictions_dir(predictions_dir)
    save_table(metrics_df, reports_dir / f"{output_name}.csv")
    save_json(metrics_df.to_dict(orient="records"), reports_dir / f"{output_name}.json")
    return metrics_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--config", default=str(get_config_path()))
    parser.add_argument("--output-name", default="classification_prediction_metrics")
    args = parser.parse_args()
    print(run_evaluation(args.predictions_dir, args.config, args.output_name))


if __name__ == "__main__":
    main()

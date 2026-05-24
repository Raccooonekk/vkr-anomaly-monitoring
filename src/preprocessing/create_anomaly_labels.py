"""Create performance anomaly flags from aggregated monitoring metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import read_table, save_metadata, save_table
from src.utils.paths import make_dir, to_path


@dataclass(frozen=True)
class AnomalyLabelConfig:
    """Configuration for threshold-based labeling."""

    anomaly_quantile: float = 0.95
    train_frac_for_thresholds: float = 0.70
    save_parquet: bool = True
    save_csv: bool = False
    csv_encoding: str = "utf-8-sig"


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def chronological_train_part(df: pd.DataFrame, train_frac: float = 0.70) -> pd.DataFrame:
    if not 0 < train_frac <= 1:
        raise ValueError(f"train_frac must be in (0, 1]. Got: {train_frac}")
    train_size = max(1, int(len(df) * train_frac))
    return df.iloc[:train_size].copy()


def compute_anomaly_thresholds(df: pd.DataFrame, config: AnomalyLabelConfig | None = None) -> dict[str, Any]:
    config = config or AnomalyLabelConfig()
    train_df = chronological_train_part(df, config.train_frac_for_thresholds)
    latency_col = _first_existing(train_df, ["p95_latency", "max_latency", "mean_latency"])
    cpu_col = _first_existing(train_df, ["cpu_total", "cpu_max", "cpu_mean"])
    memory_col = _first_existing(train_df, ["memory_total", "memory_max", "memory_mean"])
    thresholds: dict[str, Any] = {
        "anomaly_quantile": float(config.anomaly_quantile),
        "train_frac_for_thresholds": float(config.train_frac_for_thresholds),
        "latency_column": latency_col,
        "cpu_column": cpu_col,
        "memory_column": memory_col,
        "latency_threshold": None,
        "cpu_threshold": None,
        "memory_threshold": None,
    }
    if latency_col:
        thresholds["latency_threshold"] = float(train_df[latency_col].quantile(config.anomaly_quantile))
    if cpu_col:
        thresholds["cpu_threshold"] = float(train_df[cpu_col].quantile(config.anomaly_quantile))
    if memory_col:
        thresholds["memory_threshold"] = float(train_df[memory_col].quantile(config.anomaly_quantile))
    return thresholds


def apply_anomaly_labels(df: pd.DataFrame, thresholds: dict[str, Any]) -> pd.DataFrame:
    result = df.copy()
    for name in ["latency", "cpu", "memory"]:
        source_column = thresholds.get(f"{name}_column")
        threshold = thresholds.get(f"{name}_threshold")
        output_column = f"{name}_anomaly"
        if source_column and source_column in result.columns and threshold is not None:
            result[output_column] = (result[source_column] >= threshold).astype(int)
        else:
            result[output_column] = 0
    result["system_anomaly"] = (
        (result["latency_anomaly"] == 1)
        | (result["cpu_anomaly"] == 1)
        | (result["memory_anomaly"] == 1)
    ).astype(int)
    return result


def summarize_labels(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"row_count": int(len(df))}
    for column in ["latency_anomaly", "cpu_anomaly", "memory_anomaly", "system_anomaly"]:
        if column in df.columns:
            count = int(df[column].sum())
            summary[f"{column}_count"] = count
            summary[f"{column}_share"] = float(count / len(df)) if len(df) else 0.0
    return summary


def create_anomaly_state(aggregated_df: pd.DataFrame, config: AnomalyLabelConfig | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = config or AnomalyLabelConfig()
    thresholds = compute_anomaly_thresholds(aggregated_df, config)
    labeled_df = apply_anomaly_labels(aggregated_df, thresholds)
    return labeled_df, {"thresholds": thresholds, "label_summary": summarize_labels(labeled_df)}


def create_anomaly_labels_file(input_path: str | Path, output_dir: str | Path, config: AnomalyLabelConfig | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = config or AnomalyLabelConfig()
    output_dir = make_dir(output_dir)
    labeled_df, metadata = create_anomaly_state(read_table(to_path(input_path)), config)
    if config.save_parquet:
        save_table(labeled_df, output_dir / "marked_anomaly_state.parquet")
    if config.save_csv:
        save_table(labeled_df, output_dir / "marked_anomaly_state.csv", encoding=config.csv_encoding)
    save_metadata(metadata, output_dir / "anomaly_label_metadata.json")
    return labeled_df, metadata

"""Build predictive datasets for anomaly monitoring."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.preprocessing.aggregate_system_state import AggregationConfig, aggregate_file, is_relevant_source_table
from src.preprocessing.create_anomaly_labels import AnomalyLabelConfig, create_anomaly_state
from src.utils.io import load_yaml, save_metadata, save_table
from src.utils.paths import get_config_path, get_output_root, get_processed_dataset_root, make_dir, to_path
from src.utils.validation import validate_config

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictiveDatasetConfig:
    n_lags: int = 3
    forecast_horizon: int = 1
    min_rows_after_processing: int = 10
    save_parquet: bool = True
    save_csv: bool = False
    csv_encoding: str = "utf-8-sig"
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(asctime)s | %(levelname)s | %(message)s")


def discover_source_files(input_root: str | Path) -> list[Path]:
    input_root = to_path(input_root)
    return sorted(path for path in input_root.rglob("*.csv") if is_relevant_source_table(path))


def add_lag_features(df: pd.DataFrame, feature_columns: list[str], n_lags: int) -> pd.DataFrame:
    result = df.copy()
    for lag in range(1, n_lags + 1):
        shifted = df[feature_columns].shift(lag)
        shifted.columns = [f"{column}_lag_{lag}" for column in feature_columns]
        result = pd.concat([result, shifted], axis=1)
    return result


def add_future_targets(df: pd.DataFrame, forecast_horizon: int) -> pd.DataFrame:
    result = df.copy()
    mapping = {
        "system_anomaly": "target_anomaly_t_plus_1",
        "latency_anomaly": "target_latency_anomaly_t_plus_1",
        "cpu_anomaly": "target_cpu_anomaly_t_plus_1",
        "memory_anomaly": "target_memory_anomaly_t_plus_1",
        "p95_latency": "target_p95_latency_t_plus_1",
        "label_trace": "target_label_trace_t_plus_1",
    }
    for source, target in mapping.items():
        if source in result.columns:
            result[target] = result[source].shift(-forecast_horizon)
    return result


def chronological_split(df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.0")
    train_end = int(len(df) * train_frac)
    val_end = train_end + int(len(df) * val_frac)
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def _lag_base_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"window_id", "trace_id", "label_trace", "latency_anomaly", "cpu_anomaly", "memory_anomaly", "system_anomaly"}
    return [c for c in df.select_dtypes(include=["number", "bool"]).columns if c not in excluded and not c.startswith("target_")]


def build_predictive_dataset(labeled_df: pd.DataFrame, config: PredictiveDatasetConfig | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = config or PredictiveDatasetConfig()
    working = labeled_df.sort_values("window_id").reset_index(drop=True) if "window_id" in labeled_df.columns else labeled_df.reset_index(drop=True)
    lag_base_columns = _lag_base_columns(working)
    predictive = add_lag_features(working, lag_base_columns, config.n_lags)
    predictive = add_future_targets(predictive, config.forecast_horizon)
    target_columns = [c for c in predictive.columns if c.startswith("target_")]
    lag_columns = [c for c in predictive.columns if "_lag_" in c]
    predictive = predictive.dropna(subset=target_columns + lag_columns).copy()
    for column in ["target_anomaly_t_plus_1", "target_latency_anomaly_t_plus_1", "target_cpu_anomaly_t_plus_1", "target_memory_anomaly_t_plus_1"]:
        if column in predictive.columns:
            predictive[column] = predictive[column].astype(int)
    metadata = {"row_count_labeled": int(len(labeled_df)), "row_count_predictive": int(len(predictive)), "n_lags": config.n_lags, "forecast_horizon": config.forecast_horizon, "target_columns": target_columns, "lag_columns_count": len(lag_columns)}
    return predictive.reset_index(drop=True), metadata


def _out_dir(input_path: Path, input_root: Path, output_root: Path, part_name: str) -> Path:
    rel = input_path.relative_to(input_root)
    scenario = rel.parts[0] if rel.parts else "unknown"
    parent_tail = rel.parent.parts[1:] if len(rel.parent.parts) > 1 else tuple()
    return output_root / "predictive_processed_dataset" / part_name / scenario / Path(*parent_tail) / input_path.stem


def save_predictive_outputs(predictive_df: pd.DataFrame, output_dir: str | Path, config: PredictiveDatasetConfig) -> dict[str, int]:
    output_dir = make_dir(output_dir)
    train, val, test = chronological_split(predictive_df, config.train_frac, config.val_frac, config.test_frac)
    if config.save_parquet:
        save_table(predictive_df, output_dir / "predictive_p95_dataset.parquet")
        save_table(train, output_dir / "train.parquet")
        save_table(val, output_dir / "val.parquet")
        save_table(test, output_dir / "test.parquet")
    if config.save_csv:
        save_table(predictive_df, output_dir / "predictive_p95_dataset.csv", encoding=config.csv_encoding)
        save_table(train, output_dir / "train.csv", encoding=config.csv_encoding)
        save_table(val, output_dir / "val.csv", encoding=config.csv_encoding)
        save_table(test, output_dir / "test.csv", encoding=config.csv_encoding)
    return {"train_rows": len(train), "val_rows": len(val), "test_rows": len(test)}


def process_one_file(input_path: str | Path, input_root: str | Path, output_root: str | Path, part_name: str, aggregation_config: AggregationConfig, anomaly_config: AnomalyLabelConfig, predictive_config: PredictiveDatasetConfig) -> dict[str, Any]:
    input_path = to_path(input_path)
    input_root = to_path(input_root)
    output_root = to_path(output_root)
    rel = input_path.relative_to(input_root)
    output_dir = _out_dir(input_path, input_root, output_root, part_name)
    aggregated, aggregation_meta = aggregate_file(input_path, output_dir, rel, aggregation_config)
    labeled, label_meta = create_anomaly_state(aggregated, anomaly_config)
    if predictive_config.save_parquet:
        save_table(labeled, output_dir / "marked_anomaly_state.parquet")
    if predictive_config.save_csv:
        save_table(labeled, output_dir / "marked_anomaly_state.csv", encoding=predictive_config.csv_encoding)
    predictive, predictive_meta = build_predictive_dataset(labeled, predictive_config)
    if len(predictive) < predictive_config.min_rows_after_processing:
        raise ValueError(f"Too few rows after predictive processing for {input_path}: {len(predictive)}")
    saved_meta = save_predictive_outputs(predictive, output_dir, predictive_config)
    metadata = {"source_file": input_path.name, "source_relpath": rel.as_posix(), "output_dir": str(output_dir), "processed_at_utc": datetime.now(timezone.utc).isoformat(), "aggregation": aggregation_meta, "anomaly_labels": label_meta, "predictive_dataset": predictive_meta, "saved": saved_meta}
    save_metadata(metadata, output_dir / "metadata.json")
    return metadata


def _configs(config: dict) -> tuple[AggregationConfig, AnomalyLabelConfig, PredictiveDatasetConfig]:
    prep = config.get("preprocessing", {})
    out = config.get("output", {})
    split = config.get("split", {})
    return (
        AggregationConfig(int(prep.get("rolling_window", 3)), int(prep.get("min_rows_after_processing", 10)), bool(out.get("save_parquet", True)), bool(out.get("save_csv", False)), str(out.get("csv_encoding", "utf-8-sig"))),
        AnomalyLabelConfig(float(prep.get("anomaly_quantile", 0.95)), float(split.get("train_frac", 0.70)), bool(out.get("save_parquet", True)), bool(out.get("save_csv", False)), str(out.get("csv_encoding", "utf-8-sig"))),
        PredictiveDatasetConfig(int(prep.get("n_lags", 3)), int(prep.get("forecast_horizon", 1)), int(prep.get("min_rows_after_processing", 10)), bool(out.get("save_parquet", True)), bool(out.get("save_csv", False)), str(out.get("csv_encoding", "utf-8-sig")), float(split.get("train_frac", 0.70)), float(split.get("val_frac", 0.15)), float(split.get("test_frac", 0.15))),
    )


def run_preprocessing(config_path: str | Path | None = None, batch_start: int | None = None, batch_size: int | None = None) -> dict[str, Any]:
    config = load_yaml(config_path or get_config_path())
    validate_config(config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    input_root = get_processed_dataset_root(config)
    output_root = get_output_root(config)
    execution = config.get("execution", {})
    batch_start = int(batch_start if batch_start is not None else execution.get("batch_start", 0))
    raw_size = batch_size if batch_size is not None else execution.get("batch_size")
    batch_size = int(raw_size) if raw_size is not None else None
    files = discover_source_files(input_root)
    selected = files[batch_start:] if batch_size is None else files[batch_start:batch_start + batch_size]
    part_name = f"part_{batch_start:04d}_{batch_start + len(selected) - 1:04d}"
    agg_cfg, anomaly_cfg, pred_cfg = _configs(config)
    run_meta: dict[str, Any] = {"input_root": str(input_root), "output_root": str(output_root), "part_name": part_name, "total_source_files": len(files), "selected_files": len(selected), "processed": [], "failed": []}
    for path in selected:
        try:
            run_meta["processed"].append(process_one_file(path, input_root, output_root, part_name, agg_cfg, anomaly_cfg, pred_cfg))
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to process %s", path)
            run_meta["failed"].append({"source_file": str(path), "error": repr(exc)})
    run_meta["processed_count"] = len(run_meta["processed"])
    run_meta["failed_count"] = len(run_meta["failed"])
    save_metadata(run_meta, make_dir(output_root / "metadata") / f"preprocessing_{part_name}.json")
    if run_meta["processed_count"] == 0:
        raise RuntimeError(f"Preprocessing completed without processed files: {run_meta['failed'][:3]}")
    return run_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(get_config_path()))
    parser.add_argument("--batch-start", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    print(run_preprocessing(args.config, args.batch_start, args.batch_size))


if __name__ == "__main__":
    main()

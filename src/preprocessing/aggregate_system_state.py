"""
Агрегация исходных метрик микросервисной архитектуры в системное состояние.

Модуль не привязан к фиксированному количеству сервисов: колонки сервисов
обнаруживаются динамически по шаблонам имен. Результат используется далее для
разметки аномалий, формирования лагов и обучения ML-моделей.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.dataframe import reduce_memory_usage
from src.utils.io import read_table, save_metadata, save_table
from src.utils.paths import make_dir, to_path

SERVICE_RE = re.compile(r"^(?P<service_id>\d+)_(?P<metric>.+)$")


@dataclass(frozen=True)
class AggregationConfig:
    """Параметры агрегации системного состояния."""

    rolling_window: int = 3
    min_rows_after_processing: int = 10
    save_parquet: bool = True
    save_csv: bool = False
    csv_encoding: str = "utf-8-sig"


def _sort_key(column: str) -> tuple[int, str]:
    match = SERVICE_RE.match(str(column))
    return (int(match.group("service_id")), column) if match else (10**9, column)


def get_service_metric_columns(df: pd.DataFrame, metric_suffix: str) -> list[str]:
    """Возвращает колонки вида ``<service_id>_<metric_suffix>``."""
    result = []
    for column in df.columns:
        match = SERVICE_RE.match(str(column))
        if match and match.group("metric") == metric_suffix:
            result.append(str(column))
    return sorted(result, key=_sort_key)


def get_columns_containing(df: pd.DataFrame, tokens: list[str]) -> list[str]:
    """Возвращает числовые сервисные колонки, содержащие все заданные токены."""
    lowered = [token.lower() for token in tokens]
    result = []
    for column in df.columns:
        column_text = str(column)
        if not SERVICE_RE.match(column_text):
            continue
        if all(token in column_text.lower() for token in lowered):
            if pd.api.types.is_numeric_dtype(df[column]):
                result.append(column_text)
    return sorted(result, key=_sort_key)


def _num(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=df.index)
    return df[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _std(values: pd.DataFrame) -> pd.Series:
    if values.shape[1] <= 1:
        return pd.Series(0.0, index=values.index)
    return values.std(axis=1, skipna=True).fillna(0.0)


def add_resource_aggregates(result: pd.DataFrame, source: pd.DataFrame, columns: list[str], prefix: str) -> pd.DataFrame:
    """Добавляет sum/max/mean/std для группы ресурсных метрик."""
    values = _num(source, columns)
    if values.empty:
        result[f"{prefix}_available"] = 0
        return result
    result[f"{prefix}_total"] = values.sum(axis=1, skipna=True)
    result[f"{prefix}_max"] = values.max(axis=1, skipna=True)
    result[f"{prefix}_mean"] = values.mean(axis=1, skipna=True)
    result[f"{prefix}_std"] = _std(values)
    result[f"{prefix}_service_count"] = values.notna().sum(axis=1)
    result[f"{prefix}_available"] = 1
    return result


def safe_divide(num: pd.Series, den: pd.Series, eps: float = 1e-9) -> pd.Series:
    """Безопасное деление признаков с заменой некорректных значений на 0."""
    return (num / (den.replace(0, np.nan) + eps)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_dynamic_features(df: pd.DataFrame, rolling_window: int = 3) -> pd.DataFrame:
    """Добавляет diff, pct_change и rolling-признаки для ключевых метрик."""
    candidates = [
        "p95_latency", "mean_latency", "max_latency", "cpu_total", "cpu_max",
        "memory_total", "memory_max", "resource_pressure", "net_rx_total", "net_tx_total", "net_total",
    ]
    for column in [c for c in candidates if c in df.columns]:
        series = pd.to_numeric(df[column], errors="coerce")
        df[f"{column}_diff_1"] = series.diff().fillna(0.0)
        df[f"{column}_pct_change_1"] = series.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df[f"{column}_rolling_mean_{rolling_window}"] = series.rolling(rolling_window, min_periods=1).mean()
        df[f"{column}_rolling_std_{rolling_window}"] = series.rolling(rolling_window, min_periods=1).std().fillna(0.0)
    return df


def build_system_state(
    source_df: pd.DataFrame,
    source_file: str | Path | None = None,
    source_relpath: str | Path | None = None,
    config: AggregationConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Формирует агрегированное состояние системы из исходной таблицы."""
    config = config or AggregationConfig()
    result = pd.DataFrame(index=source_df.index)
    result["window_id"] = source_df["window_id"] if "window_id" in source_df.columns else np.arange(len(source_df))

    for column in ["trace_id", "label_trace"]:
        if column in source_df.columns:
            result[column] = source_df[column]

    latency_cols = get_service_metric_columns(source_df, "latency")
    latency = _num(source_df, latency_cols)
    if not latency.empty:
        result["p50_latency"] = latency.quantile(0.50, axis=1)
        result["p90_latency"] = latency.quantile(0.90, axis=1)
        result["p95_latency"] = latency.quantile(0.95, axis=1)
        result["mean_latency"] = latency.mean(axis=1, skipna=True)
        result["max_latency"] = latency.max(axis=1, skipna=True)
        result["min_latency"] = latency.min(axis=1, skipna=True)
        result["std_latency"] = _std(latency)
        result["latency_service_count"] = latency.notna().sum(axis=1)

    start_cols = get_service_metric_columns(source_df, "start")
    start = _num(source_df, start_cols)
    if not start.empty:
        result["mean_start"] = start.mean(axis=1, skipna=True)
        result["max_start"] = start.max(axis=1, skipna=True)
        result["min_start"] = start.min(axis=1, skipna=True)
        result["std_start"] = _std(start)
        result["span_start"] = result["max_start"] - result["min_start"]

    cpu_cols = get_columns_containing(source_df, ["container_cpu_usage_seconds_total"])
    memory_cols = get_columns_containing(source_df, ["container_memory_usage_bytes"])
    net_rx_cols = get_columns_containing(source_df, ["container_network_receive_bytes_total"])
    net_tx_cols = get_columns_containing(source_df, ["container_network_transmit_bytes_total"])

    for prefix, columns in [("cpu", cpu_cols), ("memory", memory_cols), ("net_rx", net_rx_cols), ("net_tx", net_tx_cols)]:
        result = add_resource_aggregates(result, source_df, columns, prefix)

    if {"p95_latency", "cpu_total"}.issubset(result.columns):
        result["latency_to_cpu_ratio"] = safe_divide(result["p95_latency"], result["cpu_total"])
    if {"p95_latency", "memory_total"}.issubset(result.columns):
        result["latency_to_memory_ratio"] = safe_divide(result["p95_latency"], result["memory_total"])
    if {"p95_latency", "net_rx_total"}.issubset(result.columns):
        result["latency_to_net_rx_ratio"] = safe_divide(result["p95_latency"], result["net_rx_total"])
    if {"p95_latency", "net_tx_total"}.issubset(result.columns):
        result["latency_to_net_tx_ratio"] = safe_divide(result["p95_latency"], result["net_tx_total"])
    if {"cpu_total", "memory_total"}.issubset(result.columns):
        result["resource_pressure"] = result["cpu_total"].fillna(0.0) + result["memory_total"].fillna(0.0)
    if {"net_rx_total", "net_tx_total"}.issubset(result.columns):
        result["net_total"] = result["net_rx_total"].fillna(0.0) + result["net_tx_total"].fillna(0.0)

    result = add_dynamic_features(result, config.rolling_window)

    if source_file is not None:
        result["source_file"] = Path(source_file).name
    if source_relpath is not None:
        relpath = Path(source_relpath)
        result["source_relpath"] = relpath.as_posix()
        for idx in range(3):
            result[f"source_level_{idx + 1}"] = relpath.parts[idx] if idx < len(relpath.parts) else ""

    numeric = result.select_dtypes(include=["number", "bool"]).columns
    result[numeric] = result[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result = reduce_memory_usage(result)

    metadata = {
        "source_file": str(source_file) if source_file is not None else None,
        "source_relpath": str(source_relpath) if source_relpath is not None else None,
        "row_count_source": int(len(source_df)),
        "row_count_aggregated": int(len(result)),
        "latency_columns": latency_cols,
        "start_columns": start_cols,
        "cpu_columns": cpu_cols,
        "memory_columns": memory_cols,
        "net_rx_columns": net_rx_cols,
        "net_tx_columns": net_tx_cols,
    }
    return result, metadata


def is_relevant_source_table(path: str | Path) -> bool:
    """Проверяет, что файл является CSV-таблицей исходного dataset."""
    path = to_path(path)
    return path.is_file() and path.suffix.lower() == ".csv"


def aggregate_file(input_path: str | Path, output_dir: str | Path, source_relpath: str | Path | None = None, config: AggregationConfig | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Читает один CSV, агрегирует состояние и сохраняет результат."""
    config = config or AggregationConfig()
    input_path = to_path(input_path)
    output_dir = make_dir(output_dir)
    source = read_table(input_path)
    aggregated, metadata = build_system_state(source, input_path, source_relpath, config)
    if len(aggregated) < config.min_rows_after_processing:
        raise ValueError(f"Too few rows after aggregation for {input_path}: {len(aggregated)}")
    if config.save_parquet:
        save_table(aggregated, output_dir / "aggregated_system_state.parquet")
    if config.save_csv:
        save_table(aggregated, output_dir / "aggregated_system_state.csv", encoding=config.csv_encoding)
    save_metadata(metadata, output_dir / "aggregation_metadata.json")
    return aggregated, metadata

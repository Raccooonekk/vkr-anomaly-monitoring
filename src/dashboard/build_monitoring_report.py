"""Build an HTML report for the anomaly monitoring experiment."""

from __future__ import annotations

import argparse
import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import load_json, load_yaml, read_table
from src.utils.paths import get_config_path, get_output_root, make_dir, to_path


def _metric_files(output_root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in ["reports/**/classification_metrics.json", "reports/**/regression_metrics.json"]:
        files.extend(output_root.glob(pattern))
    return sorted(files)


def _prediction_files(output_root: Path) -> list[Path]:
    return sorted(output_root.glob("predictions/**/*_test_predictions.parquet"))


def flatten_training_metrics(metric_json: dict[str, Any], metric_file: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    task = "classification" if "classification" in metric_file.as_posix() else "regression"
    scenario = metric_json.get("scenario", metric_file.parent.name)
    for model_name, split_data in metric_json.get("models", {}).items():
        for split_name, values in split_data.items():
            rows.append({"task": task, "scenario": scenario, "model": model_name, "split": split_name, **values})
    return pd.DataFrame(rows)


def build_prediction_summary(prediction_files: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in prediction_files:
        try:
            df = read_table(path)
        except Exception:
            continue
        row: dict[str, Any] = {"task": "classification" if "classification" in path.as_posix() else "regression", "file": path.name, "rows": int(len(df))}
        if {"y_true", "y_pred"}.issubset(df.columns):
            error = pd.to_numeric(df["y_pred"], errors="coerce") - pd.to_numeric(df["y_true"], errors="coerce")
            row["mean_error"] = float(error.mean())
            row["mean_abs_error"] = float(error.abs().mean())
        if "anomaly_probability" in df.columns:
            prob = pd.to_numeric(df["anomaly_probability"], errors="coerce")
            row["mean_anomaly_probability"] = float(prob.mean())
            row["high_risk_count_0_8"] = int((prob >= 0.8).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def dataframe_to_html_table(df: pd.DataFrame, empty_message: str) -> str:
    if df.empty:
        return f"<p>{html.escape(empty_message)}</p>"
    return df.to_html(index=False, border=0, classes="data-table", escape=True)


def build_html_report(output_root: str | Path, report_path: str | Path | None = None) -> Path:
    output_root = to_path(output_root)
    report_dir = make_dir(output_root / "reports" / "dashboard")
    report_path = to_path(report_path or report_dir / "monitoring_report.html")
    frames = []
    for path in _metric_files(output_root):
        metric_json = load_json(path)
        frame = flatten_training_metrics(metric_json, path)
        if not frame.empty:
            frames.append(frame)
    metrics_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    prediction_summary_df = build_prediction_summary(_prediction_files(output_root))
    created_at = datetime.now(timezone.utc).isoformat()
    html_text = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Отчет системы предиктивного мониторинга</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
    h1, h2 {{ color: #1f2937; }}
    .card {{ border: 1px solid #d0d7de; border-radius: 10px; padding: 16px; margin: 16px 0; }}
    .data-table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    .data-table th, .data-table td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; }}
    .data-table th {{ background: #f6f8fa; }}
    .muted {{ color: #57606a; }}
  </style>
</head>
<body>
  <h1>Отчет системы предиктивного мониторинга аномалий производительности</h1>
  <p class="muted">Сформировано: {html.escape(created_at)}</p>
  <div class="card"><h2>Назначение отчета</h2><p>Отчет агрегирует результаты обучения, оценки качества и тестовых предсказаний моделей.</p></div>
  <div class="card"><h2>Метрики моделей</h2>{dataframe_to_html_table(metrics_df, 'Файлы метрик обучения не найдены.')}</div>
  <div class="card"><h2>Сводка по предсказаниям</h2>{dataframe_to_html_table(prediction_summary_df, 'Файлы предсказаний не найдены.')}</div>
</body>
</html>
"""
    report_path.write_text(html_text, encoding="utf-8")
    return report_path


def run_report_build(config_path: str | Path | None = None) -> Path:
    config = load_yaml(config_path or get_config_path())
    return build_html_report(get_output_root(config))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(get_config_path()))
    args = parser.parse_args()
    print(f"Report saved to: {run_report_build(args.config)}")


if __name__ == "__main__":
    main()

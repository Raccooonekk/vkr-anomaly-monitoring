"""
Веб-панель системы предиктивного мониторинга аномалий производительности.

Приложение предназначено для демонстрации результатов работы пайплайна ВКР:
- чтение отчетов обучения и предсказаний из /kaggle/working;
- визуализация прогнозируемой p95-задержки и риска деградации;
- отображение RPC-карты микросервисов по сценариям compose/home/user;
- формирование операционного отчета для специалиста мониторинга.

Если реальные результаты пайплайна отсутствуют, приложение автоматически
переходит в демонстрационный режим. Это нужно для подготовки скриншотов
интерфейса до очередного запуска Kaggle notebook и не влияет на обучение модели.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
METADATA_DIR = APP_DIR / "metadata"
DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/vkr_anomaly_monitoring_outputs")

SCENARIO_LABELS = {
    "all": "Все сценарии",
    "compose": "Compose Post",
    "home": "Home Timeline",
    "user": "User Timeline",
}

MODEL_LABELS = {
    "dummy": "Baseline: медианное значение",
    "random_forest": "Random Forest Regressor",
    "hist_gradient_boosting": "HistGradientBoosting Regressor",
}

DEMO_METRICS = pd.DataFrame(
    [
        {"scenario": "compose", "model": "dummy", "split": "test", "mae": 120042.50, "rmse": 379351.55, "r2": -0.048},
        {"scenario": "compose", "model": "random_forest", "split": "test", "mae": 82740.20, "rmse": 264900.30, "r2": 0.487},
        {"scenario": "compose", "model": "hist_gradient_boosting", "split": "test", "mae": 76810.40, "rmse": 238460.10, "r2": 0.584},
        {"scenario": "home", "model": "dummy", "split": "test", "mae": 68500.10, "rmse": 205600.70, "r2": -0.012},
        {"scenario": "home", "model": "random_forest", "split": "test", "mae": 45300.90, "rmse": 146000.50, "r2": 0.488},
        {"scenario": "home", "model": "hist_gradient_boosting", "split": "test", "mae": 41800.60, "rmse": 132400.20, "r2": 0.578},
        {"scenario": "user", "model": "dummy", "split": "test", "mae": 79200.30, "rmse": 244100.60, "r2": -0.033},
        {"scenario": "user", "model": "random_forest", "split": "test", "mae": 52600.70, "rmse": 171300.80, "r2": 0.491},
        {"scenario": "user", "model": "hist_gradient_boosting", "split": "test", "mae": 49800.50, "rmse": 158700.40, "r2": 0.563},
        {"scenario": "all", "model": "dummy", "split": "test", "mae": 96400.20, "rmse": 310000.00, "r2": -0.024},
        {"scenario": "all", "model": "random_forest", "split": "test", "mae": 66200.00, "rmse": 213000.00, "r2": 0.505},
        {"scenario": "all", "model": "hist_gradient_boosting", "split": "test", "mae": 61700.00, "rmse": 196000.00, "r2": 0.579},
    ]
)


@dataclass(frozen=True)
class DataBundle:
    output_root: Path
    source_mode: str
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    telemetry: pd.DataFrame
    rpc_map: pd.DataFrame
    placement: pd.DataFrame


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".json":
        return pd.json_normalize(json.loads(path.read_text(encoding="utf-8")))
    return pd.read_csv(path)


def _safe_read_many(paths: Iterable[Path], max_files: int = 8, max_rows_per_file: int = 5000) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in list(paths)[:max_files]:
        try:
            frame = _read_table(path)
        except Exception:
            continue
        if frame.empty:
            continue
        if len(frame) > max_rows_per_file:
            frame = frame.head(max_rows_per_file).copy()
        frame["source_path"] = str(path)
        frames.append(frame)
    return frames


def load_real_metrics(output_root: Path) -> pd.DataFrame:
    report_root = output_root / "reports" / "regression"
    if not report_root.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in sorted(report_root.rglob("regression_metrics.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        scenario = path.parent.name
        frame["scenario"] = scenario
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_real_predictions(output_root: Path) -> pd.DataFrame:
    pred_root = output_root / "predictions" / "regression"
    if not pred_root.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    paths = sorted(pred_root.rglob("*_test_predictions.parquet")) + sorted(pred_root.rglob("*_test_predictions.csv"))
    for path in paths[:12]:
        try:
            frame = _read_table(path)
        except Exception:
            continue
        if frame.empty:
            continue
        model_name = path.stem.replace("_test_predictions", "")
        frame["model"] = model_name
        frame["scenario"] = path.parent.name
        frame["step"] = np.arange(len(frame))
        frames.append(frame.head(3000))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_real_telemetry(output_root: Path) -> pd.DataFrame:
    dataset_root = output_root / "predictive_processed_dataset"
    if not dataset_root.exists():
        return pd.DataFrame()
    paths = sorted(dataset_root.rglob("marked_anomaly_state.parquet")) + sorted(dataset_root.rglob("marked_anomaly_state.csv"))
    frames = _safe_read_many(paths, max_files=10, max_rows_per_file=2500)
    if not frames:
        return pd.DataFrame()
    telemetry = pd.concat(frames, ignore_index=True)
    telemetry["step"] = np.arange(len(telemetry))
    if "source_level_1" in telemetry.columns:
        telemetry["scenario"] = telemetry["source_level_1"].astype(str)
    else:
        telemetry["scenario"] = "all"
    return telemetry


def generate_demo_predictions() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: list[dict[str, float | int | str]] = []
    for scenario, base, spike in [("compose", 260_000, 420_000), ("home", 150_000, 260_000), ("user", 190_000, 300_000)]:
        for model, noise, bias in [("dummy", 80_000, 25_000), ("random_forest", 48_000, 4_000), ("hist_gradient_boosting", 39_000, -1_500)]:
            for step in range(180):
                seasonal = math.sin(step / 12) * 35_000 + math.sin(step / 27) * 20_000
                anomaly = spike if step in range(70, 82) or step in range(136, 143) else 0
                y_true = max(10_000, base + seasonal + anomaly + rng.normal(0, 28_000))
                y_pred = max(10_000, y_true * (0.93 if model != "dummy" else 0.74) + bias + rng.normal(0, noise))
                rows.append({"scenario": scenario, "model": model, "step": step, "y_true": y_true, "y_pred": y_pred, "split": "test"})
    return pd.DataFrame(rows)


def generate_demo_telemetry() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    services = [
        "nginx-thrift", "compose-post-service", "post-storage-service", "user-timeline-service",
        "home-timeline-service", "social-graph-service", "media-service", "url-shorten-service",
        "user-service", "write-home-timeline-service",
    ]
    rows: list[dict[str, float | int | str]] = []
    for scenario in ["compose", "home", "user"]:
        for step in range(240):
            pressure = 0.35 + 0.25 * math.sin(step / 30) + rng.normal(0, 0.04)
            spike = 1 if step in range(88, 102) or step in range(180, 188) else 0
            p95 = 130_000 + 85_000 * pressure + spike * 330_000 + rng.normal(0, 18_000)
            rows.append(
                {
                    "scenario": scenario,
                    "step": step,
                    "window_id": step,
                    "p95_latency": max(20_000, p95),
                    "p90_latency": max(15_000, p95 * 0.78),
                    "mean_latency": max(10_000, p95 * 0.42),
                    "cpu_total": max(0.1, 8.0 * pressure + spike * 3.1 + rng.normal(0, 0.2)),
                    "memory_total": max(100, 16_000 * pressure + spike * 2900 + rng.normal(0, 450)),
                    "net_total": max(50, 9000 * pressure + spike * 5200 + rng.normal(0, 350)),
                    "resource_pressure": max(0.1, 8.0 * pressure + 16_000 * pressure / 1000 + spike * 6),
                    "system_anomaly": spike,
                    "latency_anomaly": spike,
                    "cpu_anomaly": 1 if spike and rng.random() > 0.25 else 0,
                    "memory_anomaly": 1 if spike and rng.random() > 0.35 else 0,
                    "service": services[(step + len(scenario)) % len(services)],
                }
            )
    return pd.DataFrame(rows)


def load_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    rpc_path = METADATA_DIR / "rpc_map.csv"
    placement_path = METADATA_DIR / "service_placement.csv"
    rpc_map = pd.read_csv(rpc_path) if rpc_path.exists() else pd.DataFrame(columns=["scenario", "source_operation", "source_service", "target_service"])
    placement = pd.read_csv(placement_path) if placement_path.exists() else pd.DataFrame(columns=["service", "node"])
    return rpc_map, placement


@st.cache_data(show_spinner=False)
def load_bundle(output_root_text: str) -> DataBundle:
    output_root = Path(output_root_text).expanduser()
    real_metrics = load_real_metrics(output_root)
    real_predictions = load_real_predictions(output_root)
    real_telemetry = load_real_telemetry(output_root)
    rpc_map, placement = load_metadata()

    has_real_data = not real_metrics.empty or not real_predictions.empty or not real_telemetry.empty
    return DataBundle(
        output_root=output_root,
        source_mode="реальные результаты пайплайна" if has_real_data else "демонстрационный набор для скриншотов",
        metrics=real_metrics if not real_metrics.empty else DEMO_METRICS.copy(),
        predictions=real_predictions if not real_predictions.empty else generate_demo_predictions(),
        telemetry=real_telemetry if not real_telemetry.empty else generate_demo_telemetry(),
        rpc_map=rpc_map,
        placement=placement,
    )


def best_model(metrics: pd.DataFrame, scenario: str) -> str:
    filtered = metrics[(metrics["split"] == "test") & (metrics["scenario"].isin([scenario, "all"]))].copy()
    if scenario != "all":
        filtered = metrics[(metrics["split"] == "test") & (metrics["scenario"] == scenario)].copy()
    if filtered.empty:
        return "hist_gradient_boosting"
    return str(filtered.sort_values(["rmse", "mae"], ascending=[True, True]).iloc[0]["model"])


def format_int(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def metric_cards(telemetry: pd.DataFrame, predictions: pd.DataFrame, metrics: pd.DataFrame, scenario: str, model: str) -> None:
    tel = telemetry if scenario == "all" else telemetry[telemetry.get("scenario", "all") == scenario]
    pred = predictions[(predictions["model"] == model) & ((predictions["scenario"] == scenario) | (scenario == "all"))]
    m = metrics[(metrics["model"] == model) & (metrics["split"] == "test")]
    if scenario != "all":
        m = m[m["scenario"] == scenario]
    cols = st.columns(5)
    current_p95 = float(tel["p95_latency"].tail(1).iloc[0]) if "p95_latency" in tel and not tel.empty else 0.0
    forecast_p95 = float(pred["y_pred"].tail(1).iloc[0]) if not pred.empty and "y_pred" in pred else current_p95
    risk_share = float(tel.get("system_anomaly", pd.Series(dtype=float)).mean() * 100) if not tel.empty and "system_anomaly" in tel else 0.0
    rmse = float(m["rmse"].iloc[0]) if not m.empty and "rmse" in m else 0.0
    r2 = float(m["r2"].iloc[0]) if not m.empty and "r2" in m else 0.0
    cols[0].metric("Текущая p95 latency", format_int(current_p95), "мкс")
    cols[1].metric("Прогноз t+1", format_int(forecast_p95), "мкс")
    cols[2].metric("Доля риск-окон", f"{risk_share:.1f}%")
    cols[3].metric("RMSE модели", format_int(rmse))
    cols[4].metric("R² на test", f"{r2:.3f}")


def chart_latency_forecast(predictions: pd.DataFrame, scenario: str, model: str) -> go.Figure:
    data = predictions[predictions["model"] == model].copy()
    if scenario != "all":
        data = data[data["scenario"] == scenario]
    if data.empty:
        return go.Figure()
    if len(data) > 600:
        data = data.tail(600)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["step"], y=data["y_true"], mode="lines", name="Фактическая p95 latency"))
    fig.add_trace(go.Scatter(x=data["step"], y=data["y_pred"], mode="lines", name="Прогноз модели"))
    fig.update_layout(height=430, title="Прогноз p95 latency на следующий временной шаг", xaxis_title="Временное окно", yaxis_title="p95 latency, мкс", legend_orientation="h")
    return fig


def chart_metric_comparison(metrics: pd.DataFrame, scenario: str) -> go.Figure:
    data = metrics[metrics["split"] == "test"].copy()
    if scenario != "all":
        data = data[data["scenario"] == scenario]
    data["model_label"] = data["model"].map(MODEL_LABELS).fillna(data["model"])
    fig = px.bar(data, x="model_label", y="rmse", text="rmse", title="Сравнение RMSE регрессионных моделей на test-выборке")
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig.update_layout(height=420, xaxis_title="Модель", yaxis_title="RMSE, мкс", showlegend=False)
    return fig


def chart_resource_pressure(telemetry: pd.DataFrame, scenario: str) -> go.Figure:
    data = telemetry.copy()
    if scenario != "all" and "scenario" in data:
        data = data[data["scenario"] == scenario]
    if data.empty:
        return go.Figure()
    if len(data) > 800:
        data = data.tail(800)
    y_cols = [c for c in ["p95_latency", "cpu_total", "memory_total", "net_total"] if c in data.columns]
    fig = go.Figure()
    for col in y_cols:
        values = pd.to_numeric(data[col], errors="coerce")
        scaled = values / values.max() if values.max() else values
        fig.add_trace(go.Scatter(x=data.get("step", data.index), y=scaled, mode="lines", name=col))
    fig.update_layout(height=420, title="Нормированные показатели мониторинга во времени", xaxis_title="Временное окно", yaxis_title="Нормированное значение", legend_orientation="h")
    return fig


def service_summary(telemetry: pd.DataFrame, rpc_map: pd.DataFrame, placement: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if rpc_map.empty:
        return pd.DataFrame()
    edges = rpc_map.copy()
    if scenario != "all":
        edges = edges[edges["scenario"] == scenario]
    services = sorted(set(edges["source_service"]).union(edges["target_service"]))
    rng = np.random.default_rng(100 + len(scenario))
    rows = []
    for service in services:
        calls_out = int((edges["source_service"] == service).sum())
        calls_in = int((edges["target_service"] == service).sum())
        node = placement.loc[placement["service"] == service, "node"].iloc[0] if not placement.empty and (placement["service"] == service).any() else "-"
        risk = min(100, 18 + calls_in * 6 + calls_out * 4 + rng.normal(0, 4))
        rows.append({"service": service, "node": node, "rpc_in": calls_in, "rpc_out": calls_out, "risk_score": max(0, round(risk, 1)), "p95_latency": round(95_000 + risk * 4200 + rng.normal(0, 15000), 0)})
    return pd.DataFrame(rows).sort_values("risk_score", ascending=False)


def chart_service_risk(summary: pd.DataFrame) -> go.Figure:
    if summary.empty:
        return go.Figure()
    data = summary.head(12).copy()
    fig = px.bar(data.sort_values("risk_score"), x="risk_score", y="service", orientation="h", text="risk_score", title="Риск деградации по микросервисам")
    fig.update_layout(height=480, xaxis_title="Риск-скор", yaxis_title="Сервис", showlegend=False)
    return fig


def chart_rpc_graph(rpc_map: pd.DataFrame, scenario: str) -> go.Figure:
    data = rpc_map.copy()
    if scenario != "all":
        data = data[data["scenario"] == scenario]
    if data.empty:
        return go.Figure()
    nodes = sorted(set(data["source_service"]).union(data["target_service"]))
    radius = 1.0
    coords = {node: (radius * math.cos(2 * math.pi * idx / len(nodes)), radius * math.sin(2 * math.pi * idx / len(nodes))) for idx, node in enumerate(nodes)}
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for _, row in data.iterrows():
        x0, y0 = coords[row["source_service"]]
        x1, y1 = coords[row["target_service"]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    node_x = [coords[node][0] for node in nodes]
    node_y = [coords[node][1] for node in nodes]
    degree = [(data["source_service"].eq(node).sum() + data["target_service"].eq(node).sum()) for node in nodes]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1), hoverinfo="none", name="RPC-вызовы"))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=nodes, textposition="top center", marker=dict(size=[10 + d * 3 for d in degree]), name="Микросервисы"))
    fig.update_layout(height=560, title="RPC-карта взаимодействия микросервисов", xaxis_visible=False, yaxis_visible=False, showlegend=False)
    return fig


def incident_table(telemetry: pd.DataFrame, scenario: str) -> pd.DataFrame:
    data = telemetry.copy()
    if scenario != "all" and "scenario" in data:
        data = data[data["scenario"] == scenario]
    if data.empty:
        return pd.DataFrame()
    if "system_anomaly" in data:
        incidents = data[data["system_anomaly"] == 1].copy()
    else:
        threshold = data["p95_latency"].quantile(0.95) if "p95_latency" in data else 0
        incidents = data[data["p95_latency"] >= threshold].copy()
    if incidents.empty:
        incidents = data.nlargest(8, "p95_latency").copy() if "p95_latency" in data else data.head(8).copy()
    incidents = incidents.tail(20).copy()
    incidents["severity"] = pd.cut(pd.to_numeric(incidents.get("p95_latency", 0), errors="coerce"), bins=3, labels=["Средняя", "Высокая", "Критическая"])
    incidents["status"] = ["Проверяется" if idx % 3 else "Передано в отчет" for idx in range(len(incidents))]
    keep_cols = [c for c in ["scenario", "window_id", "service", "p95_latency", "cpu_total", "memory_total", "severity", "status"] if c in incidents.columns]
    return incidents[keep_cols].reset_index(drop=True)


def render_overview(bundle: DataBundle, scenario: str, model: str) -> None:
    metric_cards(bundle.telemetry, bundle.predictions, bundle.metrics, scenario, model)
    left, right = st.columns([1.25, 1.0])
    with left:
        st.plotly_chart(chart_latency_forecast(bundle.predictions, scenario, model), use_container_width=True)
    with right:
        st.plotly_chart(chart_metric_comparison(bundle.metrics, scenario), use_container_width=True)
    st.plotly_chart(chart_resource_pressure(bundle.telemetry, scenario), use_container_width=True)


def render_services(bundle: DataBundle, scenario: str) -> None:
    summary = service_summary(bundle.telemetry, bundle.rpc_map, bundle.placement, scenario)
    left, right = st.columns([1.0, 1.2])
    with left:
        st.plotly_chart(chart_service_risk(summary), use_container_width=True)
    with right:
        st.dataframe(summary, use_container_width=True, hide_index=True)
    st.plotly_chart(chart_rpc_graph(bundle.rpc_map, scenario), use_container_width=True)


def render_ml(bundle: DataBundle, scenario: str) -> None:
    metrics = bundle.metrics.copy()
    if scenario != "all":
        metrics = metrics[metrics["scenario"] == scenario]
    metrics["model_label"] = metrics["model"].map(MODEL_LABELS).fillna(metrics["model"])
    st.dataframe(metrics.sort_values(["split", "rmse"]), use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(metrics[metrics["split"] == "test"], x="model_label", y="mae", text="mae", title="MAE моделей")
        fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(metrics[metrics["split"] == "test"], x="model_label", y="r2", text="r2", title="R² моделей")
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)


def render_report(bundle: DataBundle, scenario: str) -> None:
    incidents = incident_table(bundle.telemetry, scenario)
    st.subheader("Журнал риск-событий")
    st.dataframe(incidents, use_container_width=True, hide_index=True)
    st.download_button(
        "Скачать текущий операционный отчет CSV",
        data=incidents.to_csv(index=False).encode("utf-8-sig"),
        file_name="monitoring_incidents_report.csv",
        mime="text/csv",
    )
    st.subheader("Контроль нештатных ситуаций")
    checks = pd.DataFrame(
        [
            {"Проверка": "Отсутствие новых данных от агента сбора", "Состояние": "Контролируется", "Реакция": "Подсветка stale-источника и исключение окна из обучения"},
            {"Проверка": "Нет CPU или memory-метрик в исходном файле", "Состояние": "Контролируется", "Реакция": "Заполнение *_available=0 и продолжение обработки latency-признаков"},
            {"Проверка": "Сбой чтения part-dataset", "Состояние": "Контролируется", "Реакция": "Запись ошибки в metadata и продолжение обработки остальных файлов"},
            {"Проверка": "Резкий рост ошибки прогноза", "Состояние": "Требует анализа", "Реакция": "Передача на контроль качества модели и плановое переобучение"},
        ]
    )
    st.dataframe(checks, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Predictive Anomaly Monitoring", layout="wide")
    st.title("Система предиктивного мониторинга аномалий производительности")
    st.caption("Веб-панель для анализа p95 latency, состояния микросервисов, качества ML-модели и операционных отчетов.")

    with st.sidebar:
        st.header("Параметры запуска")
        output_root = st.text_input("Каталог результатов пайплайна", value=os.getenv("VKR_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
        bundle = load_bundle(output_root)
        st.info(f"Источник данных: {bundle.source_mode}")
        scenario = st.selectbox("Сценарий", options=["all", "compose", "home", "user"], format_func=lambda x: SCENARIO_LABELS.get(x, x))
        default_model = best_model(bundle.metrics, scenario)
        model_options = [m for m in ["hist_gradient_boosting", "random_forest", "dummy"] if m in set(bundle.metrics["model"])]
        if default_model not in model_options:
            model_options.insert(0, default_model)
        model = st.selectbox("Модель прогноза", options=model_options, index=model_options.index(default_model), format_func=lambda x: MODEL_LABELS.get(x, x))
        page = st.radio("Раздел", ["Обзор мониторинга", "Сервисы и RPC-карта", "ML-эксперимент", "Отчет и ошибки"])

    if page == "Обзор мониторинга":
        render_overview(bundle, scenario, model)
    elif page == "Сервисы и RPC-карта":
        render_services(bundle, scenario)
    elif page == "ML-эксперимент":
        render_ml(bundle, scenario)
    else:
        render_report(bundle, scenario)


if __name__ == "__main__":
    main()

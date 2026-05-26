"""Streamlit application for access to anomaly monitoring results.

The application is designed for the operator/SRE specialist. It reads already
calculated reports from /kaggle/working/vkr_anomaly_monitoring_outputs and
shows experiment status, model metrics, manifest integrity and monitoring
widgets. The application does not write to /kaggle/input.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/vkr_anomaly_monitoring_outputs")


@st.cache_data(show_spinner=False)
def read_csv_if_exists(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def metric_card(label: str, value: str) -> None:
    st.metric(label=label, value=value)


def load_tables(output_root: Path) -> dict[str, pd.DataFrame]:
    return {
        "regression": read_csv_if_exists(str(output_root / "reports/regression/all/regression_metrics.csv")),
        "classification": read_csv_if_exists(str(output_root / "reports/classification/all/classification_metrics.csv")),
        "role_index": read_csv_if_exists(str(output_root / "merged_predictive_manifest/merged_role_index.csv")),
        "samples_index": read_csv_if_exists(str(output_root / "merged_predictive_manifest/merged_samples_index.csv")),
    }


def render_experiment_overview(tables: dict[str, pd.DataFrame]) -> None:
    st.subheader("Состояние экспериментального запуска")
    samples = tables["samples_index"]
    role_index = tables["role_index"]
    complete_count = int(samples["is_complete_for_training"].sum()) if not samples.empty else 0
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Логических объектов", str(len(samples)))
    with col2:
        metric_card("Полных объектов", str(complete_count))
    with col3:
        metric_card("Файлов manifest", str(int(role_index["file_count"].sum()) if not role_index.empty else 0))
    with col4:
        metric_card("Сценарий", ", ".join(samples["scenario"].dropna().unique()) if not samples.empty else "нет данных")

    if not role_index.empty:
        fig = px.bar(role_index, x="file_role", y="file_count", title="Состав файлов логического manifest")
        st.plotly_chart(fig, use_container_width=True)


def render_model_metrics(tables: dict[str, pd.DataFrame]) -> None:
    st.subheader("Качество ML-моделей")
    regression = tables["regression"]
    classification = tables["classification"]
    tab1, tab2 = st.tabs(["Регрессия p95_latency(t+1)", "Классификация аномалии(t+1)"])

    with tab1:
        if regression.empty:
            st.warning("Файл regression_metrics.csv не найден.")
        else:
            st.dataframe(regression, use_container_width=True)
            test_df = regression[regression["split"] == "test"]
            st.plotly_chart(px.bar(test_df, x="model", y="rmse", title="RMSE на test-выборке"), use_container_width=True)
            st.plotly_chart(px.bar(test_df, x="model", y="r2", title="R2 на test-выборке"), use_container_width=True)

    with tab2:
        if classification.empty:
            st.warning("Файл classification_metrics.csv не найден.")
        else:
            st.dataframe(classification, use_container_width=True)
            test_df = classification[classification["split"] == "test"]
            st.plotly_chart(px.bar(test_df, x="model", y="f1", title="F1-score на test-выборке"), use_container_width=True)
            st.plotly_chart(px.bar(test_df, x="model", y="roc_auc", title="ROC-AUC на test-выборке"), use_container_width=True)


def render_monitoring_view(tables: dict[str, pd.DataFrame]) -> None:
    st.subheader("Оперативная панель мониторинга")
    st.info(
        "Панель имитирует рабочее место специалиста мониторинга: отображает "
        "состояние обработки, риск аномального состояния, качество модели и "
        "целостность обработанных наборов данных."
    )
    classification = tables["classification"]
    if not classification.empty:
        test_hgb = classification[(classification["model"] == "hist_gradient_boosting") & (classification["split"] == "test")]
        if not test_hgb.empty:
            row = test_hgb.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{row['accuracy']:.3f}")
            c2.metric("Precision", f"{row['precision']:.3f}")
            c3.metric("Recall", f"{row['recall']:.3f}")
            c4.metric("F1-score", f"{row['f1']:.3f}")

    samples = tables["samples_index"]
    if not samples.empty:
        st.write("Контроль целостности обработанных объектов")
        st.dataframe(samples[["logical_sample_key", "has_train", "has_val", "has_test", "is_complete_for_training"]], use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="VKR Anomaly Monitoring", layout="wide")
    st.title("Система предиктивного мониторинга аномалий производительности")
    st.caption("Приложение доступа к результатам обработки, обучения моделей и мониторинга состояния микросервисной архитектуры")

    output_root = Path(st.sidebar.text_input("Путь к результатам", str(DEFAULT_OUTPUT_ROOT)))
    tables = load_tables(output_root)

    page = st.sidebar.radio("Раздел", ["Обзор запуска", "Метрики моделей", "Панель мониторинга"])
    if page == "Обзор запуска":
        render_experiment_overview(tables)
    elif page == "Метрики моделей":
        render_model_metrics(tables)
    else:
        render_monitoring_view(tables)


if __name__ == "__main__":
    main()

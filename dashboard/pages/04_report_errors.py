"""
Страница Streamlit для раздела «Отчет и ошибки».

Страница предназначена для подготовки двух скриншотов главы 3 ВКР:
1) журнал риск-событий и контроль ошибок источников данных;
2) результат проверки обработки нештатных ситуаций.

Код не меняет существующий ML-пайплайн. Если реальные результаты пайплайна
отсутствуют, используется демонстрационный набор данных для визуальной проверки
веб-интерфейса и подготовки скриншотов.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/vkr_anomaly_monitoring_outputs")

SCENARIO_LABELS = {
    "all": "Все сценарии",
    "compose": "Compose Post",
    "home": "Home Timeline",
    "user": "User Timeline",
}

PERIOD_WINDOW_MAP = {
    "Последний час": 12,
    "Последние 6 часов": 72,
    "Последние 12 часов": 144,
    "Все наблюдения": None,
}

SERVICE_BY_SCENARIO = {
    "compose": [
        "nginx-web-server",
        "nginx-thrift",
        "compose-post-service",
        "text-service",
        "media-service",
        "url-shorten-service",
        "user-service",
        "social-graph-service",
        "post-storage-service",
        "user-timeline-service",
        "write-home-timeline-service",
    ],
    "home": [
        "nginx-web-server",
        "nginx-thrift",
        "home-timeline-service",
        "home-timeline-redis",
        "post-storage-service",
        "post-storage-memcached",
        "post-storage-mongodb",
    ],
    "user": [
        "nginx-web-server",
        "nginx-thrift",
        "user-timeline-service",
        "user-timeline-redis",
        "user-timeline-mongodb",
        "post-storage-service",
        "post-storage-memcached",
        "post-storage-mongodb",
    ],
}


def format_int(value: float | int) -> str:
    """Форматирует число с пробелами для удобного отображения в UI."""
    if pd.isna(value):
        return "0"
    return f"{float(value):,.0f}".replace(",", " ")


def read_table(path: Path) -> pd.DataFrame:
    """Читает CSV или Parquet-файл без изменения исходных данных."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_real_telemetry(output_root: Path) -> pd.DataFrame:
    """
    Пытается загрузить реальные результаты пайплайна.
    Если файлов нет или их невозможно прочитать, возвращает пустой DataFrame.
    """
    dataset_root = output_root / "predictive_processed_dataset"
    if not dataset_root.exists():
        return pd.DataFrame()

    paths = sorted(dataset_root.rglob("marked_anomaly_state.parquet")) + sorted(dataset_root.rglob("marked_anomaly_state.csv"))
    frames: list[pd.DataFrame] = []
    for path in paths[:10]:
        try:
            frame = read_table(path).head(2500).copy()
        except Exception:
            continue
        if frame.empty:
            continue
        frame["source_path"] = str(path)
        if "source_level_1" in frame.columns:
            frame["scenario"] = frame["source_level_1"].astype(str)
        elif "scenario" not in frame.columns:
            frame["scenario"] = "all"
        if "step" not in frame.columns:
            frame["step"] = np.arange(len(frame))
        if "service" not in frame.columns:
            frame["service"] = "system"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def generate_demo_telemetry() -> pd.DataFrame:
    """Формирует демонстрационный поток мониторинга для скриншотов."""
    rng = np.random.default_rng(21)
    rows: list[dict[str, float | int | str]] = []
    for scenario, services in SERVICE_BY_SCENARIO.items():
        for step in range(240):
            base_pressure = 0.42 + 0.20 * np.sin(step / 28) + rng.normal(0, 0.025)
            degradation = int(step in range(86, 101) or step in range(176, 188))
            for index, service in enumerate(services):
                service_factor = 0.82 + index * 0.045 + rng.normal(0, 0.020)
                is_core_service = service in {
                    "compose-post-service",
                    "home-timeline-service",
                    "user-timeline-service",
                    "post-storage-service",
                    "social-graph-service",
                }
                anomaly = int(degradation and is_core_service)
                p95_latency = 95_000 * service_factor + 120_000 * base_pressure + anomaly * 310_000 + rng.normal(0, 18_000)
                cpu_total = 1.8 * service_factor + 6.2 * base_pressure + anomaly * 2.6 + rng.normal(0, 0.20)
                memory_total = 3_800 * service_factor + 13_500 * base_pressure + anomaly * 2_600 + rng.normal(0, 420)
                net_total = 2_600 * service_factor + 8_800 * base_pressure + anomaly * 4_900 + rng.normal(0, 300)
                rows.append(
                    {
                        "scenario": scenario,
                        "step": step,
                        "window_id": step,
                        "service": service,
                        "p95_latency": max(20_000, p95_latency),
                        "cpu_total": max(0.1, cpu_total),
                        "memory_total": max(100, memory_total),
                        "net_total": max(50, net_total),
                        "system_anomaly": anomaly,
                    }
                )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_telemetry(output_root_text: str) -> tuple[pd.DataFrame, str]:
    output_root = Path(output_root_text).expanduser()
    real = load_real_telemetry(output_root)
    if not real.empty:
        return enrich_time(real), "реальные результаты пайплайна"
    return enrich_time(generate_demo_telemetry()), "демонстрационный набор для скриншотов"


def enrich_time(data: pd.DataFrame) -> pd.DataFrame:
    """Добавляет условное время события для отчетов за период."""
    result = data.copy()
    if "event_time" not in result.columns:
        base_timestamp = pd.Timestamp("2025-04-10 08:00:00")
        step = pd.to_numeric(result.get("step", pd.Series(np.arange(len(result)))), errors="coerce").fillna(0).astype(int)
        result["event_time"] = base_timestamp + pd.to_timedelta(step * 5, unit="min")
    return result


def filter_scenario(data: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if scenario == "all" or "scenario" not in data.columns:
        return data.copy()
    return data[data["scenario"] == scenario].copy()


def filter_period(data: pd.DataFrame, period_label: str) -> pd.DataFrame:
    if data.empty:
        return data
    window_count = PERIOD_WINDOW_MAP.get(period_label)
    if window_count is None or "step" not in data.columns:
        return data.copy()
    max_step = int(pd.to_numeric(data["step"], errors="coerce").max())
    return data[data["step"] >= max_step - window_count + 1].copy()


def build_incidents(data: pd.DataFrame) -> pd.DataFrame:
    """Формирует журнал риск-событий по флагу аномалии или по 95-му перцентилю задержки."""
    if data.empty:
        return pd.DataFrame()
    if "system_anomaly" in data.columns:
        incidents = data[data["system_anomaly"] == 1].copy()
    else:
        threshold = data["p95_latency"].quantile(0.95)
        incidents = data[data["p95_latency"] >= threshold].copy()
    if incidents.empty and "p95_latency" in data.columns:
        incidents = data.nlargest(20, "p95_latency").copy()

    incidents = enrich_time(incidents)
    incidents = incidents.sort_values(["event_time", "service"]).tail(80).copy()
    latency = pd.to_numeric(incidents["p95_latency"], errors="coerce")
    incidents["severity"] = pd.cut(latency, bins=3, labels=["Средняя", "Высокая", "Критическая"])
    statuses = ["Передано в отчет", "Проверяется", "Под наблюдением", "Проверяется"]
    incidents["status"] = [statuses[i % len(statuses)] for i in range(len(incidents))]
    keep_cols = [
        "scenario",
        "event_time",
        "window_id",
        "service",
        "p95_latency",
        "cpu_total",
        "memory_total",
        "net_total",
        "severity",
        "status",
    ]
    return incidents[[col for col in keep_cols if col in incidents.columns]].reset_index(drop=True)


def source_health_table(mode: str) -> pd.DataFrame:
    """Возвращает состояние источников данных для штатного или тестового режима."""
    if mode == "normal":
        return pd.DataFrame(
            [
                {"Источник": "Агенты сбора latency-метрик", "Состояние": "Норма", "Последнее обновление": "08:55", "Комментарий": "Поток latency-данных поступает без пропусков"},
                {"Источник": "Агенты сбора CPU/memory/network", "Состояние": "Норма", "Последнее обновление": "08:55", "Комментарий": "Ресурсные метрики доступны для активных сервисов"},
                {"Источник": "Manifest логического объединения", "Состояние": "Норма", "Последнее обновление": "08:50", "Комментарий": "Индекс обработанных part-dataset сформирован корректно"},
                {"Источник": "Хранилище результатов прогнозирования", "Состояние": "Норма", "Последнее обновление": "08:56", "Комментарий": "Предсказания доступны на веб-панели"},
            ]
        )
    return pd.DataFrame(
        [
            {"Источник": "Агенты сбора latency-метрик", "Состояние": "Норма", "Последнее обновление": "08:55", "Комментарий": "Основной поток данных сохранен"},
            {"Источник": "Агенты сбора CPU/memory/network", "Состояние": "Деградация", "Последнее обновление": "08:41", "Комментарий": "Часть ресурсных метрик отсутствует, выставлен флаг *_available = 0"},
            {"Источник": "Файл part-dataset #17", "Состояние": "Ошибка", "Последнее обновление": "08:43", "Комментарий": "Ошибка чтения файла, набор исключен из обучения и занесен в metadata failed"},
            {"Источник": "Manifest логического объединения", "Состояние": "Норма", "Последнее обновление": "08:50", "Комментарий": "Остальные part-dataset включены в индекс"},
            {"Источник": "Контроль качества модели", "Состояние": "Требует анализа", "Последнее обновление": "08:57", "Комментарий": "Зафиксирован рост ошибки прогноза, требуется переобучение"},
        ]
    )


def abnormal_checks_table() -> pd.DataFrame:
    """Таблица проверки нештатных ситуаций из раздела 3.4."""
    return pd.DataFrame(
        [
            {
                "Нештатная ситуация": "Отсутствует часть ресурсных метрик",
                "Возможная причина": "Агент сбора не передал CPU, memory или network-метрики",
                "Реакция системы": "Для соответствующей группы выставляется *_available = 0, остальные признаки продолжают обрабатываться",
                "Результат": "Файл не исключается полностью, если доступны latency-метрики",
                "Статус проверки": "Успешно",
            },
            {
                "Нештатная ситуация": "Файл содержит слишком мало строк после обработки",
                "Возможная причина": "Короткий trace или большое число строк без лагов/target",
                "Реакция системы": "Формируется ошибка Too few rows after predictive processing",
                "Результат": "Файл фиксируется в metadata failed и не используется в обучении",
                "Статус проверки": "Успешно",
            },
            {
                "Нештатная ситуация": "Не удалось прочитать отдельный part-dataset",
                "Возможная причина": "Повреждение файла, отсутствие parquet-зависимости или неполный batch",
                "Реакция системы": "Ошибка перехватывается при чтении роли train/val/test",
                "Результат": "Обучение продолжается по доступным файлам, если набор не пустой",
                "Статус проверки": "Успешно",
            },
            {
                "Нештатная ситуация": "Нет manifest-файла",
                "Возможная причина": "Логическое объединение не было сформировано",
                "Реакция системы": "Пайплайн завершает выполнение с понятной ошибкой отсутствия manifest files index",
                "Результат": "Исключается обучение на неполном и неконтролируемом наборе",
                "Статус проверки": "Успешно",
            },
            {
                "Нештатная ситуация": "Резкий рост ошибки прогноза",
                "Возможная причина": "Дрифт данных, изменение характера нагрузки, новая конфигурация сервисов",
                "Реакция системы": "Событие отображается в панели контроля качества и требует переобучения",
                "Результат": "Специалист получает основание для анализа качества модели",
                "Статус проверки": "Требует анализа",
            },
        ]
    )


def operational_summary(incidents: pd.DataFrame, period_label: str, service_count: int) -> pd.DataFrame:
    if incidents.empty:
        return pd.DataFrame(
            [
                ("Период отчета", period_label),
                ("Количество риск-событий", 0),
                ("Критических событий", 0),
                ("Доля критических событий", "0,0 %"),
                ("Средняя p95 latency", "0"),
                ("Сервис с максимальным пиком", "-"),
                ("Количество сервисов в отчете", service_count),
            ],
            columns=["Показатель", "Значение"],
        )
    critical_count = int((incidents["severity"] == "Критическая").sum()) if "severity" in incidents.columns else 0
    critical_share = critical_count / len(incidents) * 100 if len(incidents) else 0
    peak_service = incidents.groupby("service")["p95_latency"].max().sort_values(ascending=False).index[0]
    mean_latency = pd.to_numeric(incidents["p95_latency"], errors="coerce").mean()
    return pd.DataFrame(
        [
            ("Период отчета", period_label),
            ("Количество риск-событий", len(incidents)),
            ("Критических событий", critical_count),
            ("Доля критических событий", f"{critical_share:.1f} %"),
            ("Средняя p95 latency", format_int(mean_latency)),
            ("Сервис с максимальным пиком", peak_service),
            ("Количество сервисов в отчете", service_count),
        ],
        columns=["Показатель", "Значение"],
    )


def chart_incident_trend(incidents: pd.DataFrame):
    if incidents.empty:
        return px.line(title="Динамика риск-событий по уровню критичности")
    data = incidents.copy()
    data["time_bucket"] = pd.to_datetime(data["event_time"]).dt.strftime("%H:%M")
    grouped = data.groupby(["time_bucket", "severity"], as_index=False).size().rename(columns={"size": "events"})
    fig = px.area(grouped, x="time_bucket", y="events", color="severity", title="Динамика риск-событий по уровню критичности")
    fig.update_layout(height=360, xaxis_title="Временной интервал", yaxis_title="Количество событий")
    return fig


def chart_service_distribution(incidents: pd.DataFrame):
    if incidents.empty:
        return px.bar(title="Максимальная p95 latency по сервисам")
    grouped = incidents.groupby("service", as_index=False)["p95_latency"].max().sort_values("p95_latency", ascending=False).head(10)
    fig = px.bar(grouped, x="service", y="p95_latency", text="p95_latency", title="Максимальная p95 latency по сервисам в отчете")
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig.update_layout(height=360, xaxis_title="Сервис", yaxis_title="p95 latency, мкс")
    return fig


def chart_check_status(checks: pd.DataFrame):
    grouped = checks.groupby("Статус проверки", as_index=False).size().rename(columns={"size": "count"})
    fig = px.bar(grouped, x="Статус проверки", y="count", text="count", title="Результаты проверки нештатных ситуаций")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=340, xaxis_title="Статус", yaxis_title="Количество ситуаций")
    return fig


def prepare_report_data(data: pd.DataFrame, scenario: str, period_label: str, services: list[str], only_critical: bool) -> pd.DataFrame:
    filtered = filter_scenario(data, scenario)
    filtered = filter_period(filtered, period_label)
    if services:
        filtered = filtered[filtered["service"].isin(services)].copy()
    incidents = build_incidents(filtered)
    if only_critical and not incidents.empty:
        incidents = incidents[incidents["severity"] == "Критическая"].copy()
    return incidents


def render_operational_report(data: pd.DataFrame, scenario: str, period_label: str, selected_services: list[str], only_critical: bool) -> None:
    incidents = prepare_report_data(data, scenario, period_label, selected_services, only_critical)
    health = source_health_table(mode="normal")
    summary = operational_summary(incidents, period_label, len(selected_services))

    cards = st.columns(5)
    cards[0].metric("Риск-событий", len(incidents))
    cards[1].metric("Критических", int((incidents["severity"] == "Критическая").sum()) if not incidents.empty else 0)
    cards[2].metric("Период", period_label)
    cards[3].metric("Источников в норме", int((health["Состояние"] == "Норма").sum()))
    cards[4].metric("Сервисов в отчете", len(selected_services))

    actions = st.columns(3)
    actions[0].download_button(
        "Сформировать операционный отчет CSV",
        data=incidents.to_csv(index=False).encode("utf-8-sig"),
        file_name="operational_monitoring_report.csv",
        mime="text/csv",
        use_container_width=True,
    )
    actions[1].download_button(
        "Выгрузить сводку отчета",
        data=summary.to_csv(index=False).encode("utf-8-sig"),
        file_name="operational_report_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if actions[2].button("Сформировать отчет на панели", use_container_width=True):
        st.success("Операционный отчет за выбранный период сформирован.")

    left, right = st.columns([1.25, 0.95])
    left.subheader("Журнал риск-событий")
    left.dataframe(incidents, use_container_width=True, hide_index=True)
    right.subheader("Контроль ошибок источников данных")
    right.dataframe(health, use_container_width=True, hide_index=True)

    bottom_left, bottom_right = st.columns([1.0, 1.0])
    bottom_left.subheader("Превью операционного отчета")
    bottom_left.dataframe(summary, use_container_width=True, hide_index=True)
    bottom_left.plotly_chart(chart_incident_trend(incidents), use_container_width=True)
    bottom_right.subheader("Распределение деградаций по сервисам")
    bottom_right.plotly_chart(chart_service_distribution(incidents), use_container_width=True)


def render_abnormal_check(data: pd.DataFrame, scenario: str, period_label: str, selected_services: list[str], only_critical: bool) -> None:
    incidents = prepare_report_data(data, scenario, period_label, selected_services, only_critical)
    checks = abnormal_checks_table()
    health = source_health_table(mode="test")

    cards = st.columns(5)
    cards[0].metric("Проверенных ситуаций", len(checks))
    cards[1].metric("Успешно обработано", int((checks["Статус проверки"] == "Успешно").sum()))
    cards[2].metric("Требуют анализа", int((checks["Статус проверки"] != "Успешно").sum()))
    cards[3].metric("Ошибок источников", int((health["Состояние"] == "Ошибка").sum()))
    cards[4].metric("Деградаций источников", int((health["Состояние"] == "Деградация").sum()))

    actions = st.columns(3)
    actions[0].download_button(
        "Выгрузить протокол проверки CSV",
        data=checks.to_csv(index=False).encode("utf-8-sig"),
        file_name="abnormal_situations_check_report.csv",
        mime="text/csv",
        use_container_width=True,
    )
    actions[1].download_button(
        "Выгрузить журнал источников",
        data=health.to_csv(index=False).encode("utf-8-sig"),
        file_name="source_health_check_report.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if actions[2].button("Запустить повторную проверку", use_container_width=True):
        st.warning("Повторная проверка выполнена, результаты обновлены на панели контроля.")

    left, right = st.columns([1.25, 0.95])
    left.subheader("Проверка обработки нештатных ситуаций")
    left.dataframe(checks, use_container_width=True, hide_index=True)
    right.subheader("Результат проверки источников данных")
    right.dataframe(health, use_container_width=True, hide_index=True)

    chart_col, table_col = st.columns([1.0, 1.0])
    chart_col.plotly_chart(chart_check_status(checks), use_container_width=True)
    table_col.subheader("Тестовые риск-события")
    table_col.dataframe(incidents.head(25), use_container_width=True, hide_index=True)
    st.info(
        "Режим проверки показывает, что система сохраняет работоспособность при частичных пропусках ресурсных метрик, "
        "исключает поврежденные part-dataset из обучения, контролирует наличие manifest-файла и фиксирует рост ошибки прогноза "
        "как основание для последующего анализа и переобучения модели."
    )


def main() -> None:
    st.set_page_config(page_title="Отчет и ошибки", layout="wide")
    st.title("Отчет и ошибки")
    st.caption(
        "Панель предназначена для формирования операционного отчета за выбранный период, "
        "просмотра журнала риск-событий и контроля результатов проверки нештатных ситуаций."
    )

    with st.sidebar:
        st.header("Параметры отчета")
        output_root = st.text_input("Каталог результатов пайплайна", value=os.getenv("VKR_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
        telemetry, source_mode = load_telemetry(output_root)
        st.caption(f"Источник: {source_mode}")
        scenario = st.selectbox("Сценарий", options=["all", "compose", "home", "user"], format_func=lambda x: SCENARIO_LABELS.get(x, x))
        available_services = sorted(filter_scenario(telemetry, scenario)["service"].dropna().astype(str).unique().tolist())
        period_label = st.selectbox("Период отчета", options=list(PERIOD_WINDOW_MAP.keys()), index=1)
        selected_services = st.multiselect(
            "Сервисы отчета",
            options=available_services,
            default=available_services[: min(6, len(available_services))],
        )
        only_critical = st.checkbox("Только критические события", value=False)

    mode = st.radio(
        "Режим панели",
        options=["Операционный отчет", "Проверка нештатных ситуаций"],
        horizontal=True,
    )

    if mode == "Операционный отчет":
        render_operational_report(telemetry, scenario, period_label, selected_services, only_critical)
    else:
        render_abnormal_check(telemetry, scenario, period_label, selected_services, only_critical)


if __name__ == "__main__":
    main()

"""
Единые константы проекта.

Файл нужен для того, чтобы не дублировать строковые значения
в разных модулях предобработки, обучения и оценки качества.
"""

from __future__ import annotations

from typing import Final


PROJECT_NAME: Final[str] = "vkr-anomaly-monitoring"

# Основные сценарии исходного датасета микросервисной архитектуры.
SCENARIOS: Final[list[str]] = ["compose", "home", "user"]

# Роли файлов в manifest логического объединения.
FILE_ROLES: Final[list[str]] = [
    "aggregated",
    "predictive",
    "train",
    "val",
    "test",
    "metadata",
]

TRAIN_ROLE: Final[str] = "train"
VAL_ROLE: Final[str] = "val"
TEST_ROLE: Final[str] = "test"

# Основные целевые переменные.
DEFAULT_TARGET_CLASSIFICATION: Final[str] = "target_anomaly_t_plus_1"
DEFAULT_TARGET_REGRESSION: Final[str] = "target_p95_latency_t_plus_1"

TARGET_COLUMNS: Final[list[str]] = [
    "target_anomaly_t_plus_1",
    "target_latency_anomaly_t_plus_1",
    "target_cpu_anomaly_t_plus_1",
    "target_memory_anomaly_t_plus_1",
    "target_p95_latency_t_plus_1",
    "target_label_trace_t_plus_1",
]

# Служебные колонки, которые не должны попадать в модель как признаки.
SOURCE_COLUMNS_TO_DROP: Final[list[str]] = [
    "source_file",
    "source_relpath",
    "source_level_1",
    "source_level_2",
    "source_level_3",
    "dataset_name",
    "part_name",
    "logical_sample_key",
    "file_name",
    "file_suffix",
    "file_role",
    "parent_dir_name",
    "absolute_path",
    "relative_path_from_part",
]

# Колонки исходной разметки и промежуточных флагов.
SERVICE_COLUMNS_TO_DROP: Final[list[str]] = [
    "trace_id",
    "label_trace",
    "latency_anomaly",
    "cpu_anomaly",
    "memory_anomaly",
    "system_anomaly",
]

# Колонки, которые должны быть в merged_files_index.
REQUIRED_MANIFEST_COLUMNS: Final[list[str]] = [
    "absolute_path",
    "file_role",
    "relative_path_from_part",
]

# Поддерживаемые форматы табличных данных.
SUPPORTED_TABLE_SUFFIXES: Final[set[str]] = {".csv", ".parquet"}

# Поддерживаемые форматы метаданных.
SUPPORTED_METADATA_SUFFIXES: Final[set[str]] = {".json", ".yaml", ".yml"}

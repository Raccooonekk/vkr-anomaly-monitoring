"""
Утилиты для чтения и сохранения файлов.

Модуль используется во всех частях pipeline:
- предобработка;
- логическое объединение;
- обучение;
- оценка качества;
- сохранение metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.utils.constants import REQUIRED_MANIFEST_COLUMNS
from src.utils.paths import make_dir, require_file, to_path
from src.utils.validation import validate_manifest_columns


def load_yaml(path: str | Path) -> dict:
    """
    Загружает YAML-конфиг.

    Если файл пустой, возвращает пустой словарь.
    """
    path = require_file(path, description="YAML file")

    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return data if data is not None else {}


def save_yaml(data: dict, path: str | Path) -> None:
    """
    Сохраняет словарь в YAML-файл.
    """
    path = to_path(path)
    make_dir(path.parent)

    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            allow_unicode=True,
            sort_keys=False,
        )


def load_json(path: str | Path) -> dict:
    """
    Загружает JSON-файл.
    """
    path = require_file(path, description="JSON file")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: dict | list, path: str | Path, indent: int = 2) -> None:
    """
    Сохраняет данные в JSON-файл.
    """
    path = to_path(path)
    make_dir(path.parent)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=indent,
        )


def read_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """
    Читает таблицу CSV или Parquet.

    Поддерживаемые форматы:
    - .csv
    - .parquet
    """
    path = require_file(path, description="table file")
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)

    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)

    raise ValueError(
        f"Unsupported table format: {suffix}. "
        "Supported formats: .csv, .parquet"
    )


def safe_read_table(path: str | Path, **kwargs: Any) -> pd.DataFrame | None:
    """
    Безопасно читает таблицу.

    Если файл отсутствует или формат не поддерживается, возвращает None.
    Используется там, где отсутствие отдельного файла не должно ломать весь pipeline.
    """
    path = to_path(path)

    if not path.exists() or not path.is_file():
        return None

    try:
        return read_table(path, **kwargs)
    except Exception:
        return None


def save_table(
    df: pd.DataFrame,
    path: str | Path,
    save_index: bool = False,
    **kwargs: Any,
) -> None:
    """
    Сохраняет DataFrame в CSV или Parquet.

    Родительская директория создается автоматически.
    """
    path = to_path(path)
    make_dir(path.parent)

    suffix = path.suffix.lower()

    if suffix == ".csv":
        encoding = kwargs.pop("encoding", "utf-8-sig")
        df.to_csv(path, index=save_index, encoding=encoding, **kwargs)
        return

    if suffix == ".parquet":
        df.to_parquet(path, index=save_index, **kwargs)
        return

    raise ValueError(
        f"Unsupported table format: {suffix}. "
        "Supported formats: .csv, .parquet"
    )


def save_metadata(metadata: dict, path: str | Path) -> None:
    """
    Сохраняет metadata в JSON.
    """
    path = to_path(path)

    if path.suffix.lower() != ".json":
        raise ValueError(
            f"Metadata must be saved as .json file. Got: {path}"
        )

    save_json(metadata, path)


def read_manifest_index(
    path: str | Path,
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Читает manifest index и проверяет обязательные колонки.

    По умолчанию проверяются колонки:
    - absolute_path
    - file_role
    - relative_path_from_part
    """
    required_columns = required_columns or REQUIRED_MANIFEST_COLUMNS
    manifest_df = read_table(path)

    validate_manifest_columns(
        manifest_df,
        required_columns=required_columns,
    )

    return manifest_df


def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    """
    Преобразует DataFrame в список словарей.

    Удобно для сохранения небольших таблиц в JSON metadata.
    """
    return df.to_dict(orient="records")


def save_dataframe_preview(
    df: pd.DataFrame,
    path: str | Path,
    rows: int = 20,
) -> None:
    """
    Сохраняет первые строки DataFrame для быстрой проверки результата.
    """
    preview = df.head(rows).copy()
    save_table(preview, path)

"""
Базовые операции с pandas DataFrame.

Функции из этого файла используются при подготовке признаков
и передаче данных в модели машинного обучения.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def drop_columns_if_exist(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Удаляет только те колонки, которые реально есть в DataFrame.
    """
    existing_columns = [column for column in columns if column in df.columns]
    return df.drop(columns=existing_columns)


def select_existing_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Возвращает DataFrame только с теми колонками из списка,
    которые реально присутствуют в таблице.
    """
    existing_columns = [column for column in columns if column in df.columns]
    return df[existing_columns].copy()


def get_numeric_columns(
    df: pd.DataFrame,
    exclude_columns: list[str] | None = None,
) -> list[str]:
    """
    Возвращает список числовых колонок.

    exclude_columns используется для исключения target или служебных полей.
    """
    exclude_columns = set(exclude_columns or [])

    numeric_columns = df.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    return [
        column for column in numeric_columns
        if column not in exclude_columns
    ]


def remove_non_numeric_features(
    df: pd.DataFrame,
    keep_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Удаляет нечисловые признаки.

    keep_columns позволяет сохранить target или другие нужные колонки,
    даже если они не попали в числовой список.
    """
    keep_columns = keep_columns or []
    numeric_columns = df.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    selected_columns = list(dict.fromkeys(numeric_columns + keep_columns))
    selected_columns = [
        column for column in selected_columns
        if column in df.columns
    ]

    return df[selected_columns].copy()


def reduce_memory_usage(
    df: pd.DataFrame,
    exclude_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Уменьшает потребление памяти DataFrame.

    На Kaggle это важно, потому что большие CSV/Parquet-файлы
    могут привести к перезапуску notebook из-за нехватки памяти.

    Функция:
    - int64 пытается привести к int32/int16/int8;
    - float64 приводит к float32;
    - исключенные колонки не изменяет.
    """
    exclude_columns = set(exclude_columns or [])
    optimized_df = df.copy()

    for column in optimized_df.columns:
        if column in exclude_columns:
            continue

        col_data = optimized_df[column]

        if pd.api.types.is_integer_dtype(col_data):
            col_min = col_data.min()
            col_max = col_data.max()

            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                optimized_df[column] = col_data.astype(np.int8)
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                optimized_df[column] = col_data.astype(np.int16)
            elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                optimized_df[column] = col_data.astype(np.int32)

        elif pd.api.types.is_float_dtype(col_data):
            optimized_df[column] = col_data.astype(np.float32)

    return optimized_df


def get_memory_usage_mb(df: pd.DataFrame) -> float:
    """
    Возвращает объем памяти DataFrame в мегабайтах.
    """
    return float(df.memory_usage(deep=True).sum() / 1024**2)


def split_features_target(
    df: pd.DataFrame,
    target_col: str,
    drop_columns: list[str] | None = None,
    numeric_only: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Разделяет DataFrame на X и y.

    Parameters
    ----------
    df:
        Исходная таблица.
    target_col:
        Название целевой переменной.
    drop_columns:
        Колонки, которые нужно исключить из признаков.
    numeric_only:
        Если True, в X остаются только числовые признаки.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")

    drop_columns = drop_columns or []

    y = df[target_col].copy()

    columns_to_drop = list(dict.fromkeys(drop_columns + [target_col]))
    X = drop_columns_if_exist(df, columns_to_drop)

    if numeric_only:
        X = remove_non_numeric_features(X)

    return X, y


def align_columns(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Выравнивает набор колонок other_df под train_df.

    Используется для val/test, чтобы порядок и состав признаков
    совпадал с train.
    """
    missing_columns = [
        column for column in train_df.columns
        if column not in other_df.columns
    ]

    aligned_df = other_df.copy()

    for column in missing_columns:
        aligned_df[column] = 0

    return aligned_df[train_df.columns].copy()

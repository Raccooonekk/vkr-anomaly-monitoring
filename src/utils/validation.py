"""
Функции проверки конфигурации, таблиц и manifest-файлов.

Проверки нужны для того, чтобы pipeline падал не с неясной ошибкой pandas,
а с понятным сообщением о проблеме в данных или настройках.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def validate_config(config: dict) -> None:
    """
    Проверяет наличие основных секций config.yaml.
    """
    required_sections = [
        "project",
        "kaggle",
        "dataset",
        "output",
        "preprocessing",
        "split",
        "target",
    ]

    missing = [section for section in required_sections if section not in config]

    if missing:
        raise KeyError(
            "Config is missing required sections: "
            + ", ".join(missing)
        )

    validate_split_fractions(config["split"])


def validate_split_fractions(split_config: dict, tolerance: float = 1e-6) -> None:
    """
    Проверяет корректность долей train/val/test.

    Сумма должна быть равна 1.0.
    """
    train_frac = float(split_config.get("train_frac", 0))
    val_frac = float(split_config.get("val_frac", 0))
    test_frac = float(split_config.get("test_frac", 0))

    total = train_frac + val_frac + test_frac

    if abs(total - 1.0) > tolerance:
        raise ValueError(
            "Split fractions must sum to 1.0. "
            f"Got train={train_frac}, val={val_frac}, "
            f"test={test_frac}, total={total}"
        )

    for name, value in {
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
    }.items():
        if value < 0 or value > 1:
            raise ValueError(
                f"{name} must be between 0 and 1. Got: {value}"
            )


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    object_name: str = "DataFrame",
) -> None:
    """
    Проверяет наличие обязательных колонок в DataFrame.
    """
    required_columns = list(required_columns)
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise KeyError(
            f"{object_name} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def validate_manifest_columns(
    manifest_df: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    """
    Проверяет обязательные колонки manifest index.
    """
    validate_required_columns(
        manifest_df,
        required_columns=required_columns,
        object_name="Manifest index",
    )


def validate_non_empty_dataframe(
    df: pd.DataFrame,
    object_name: str = "DataFrame",
) -> None:
    """
    Проверяет, что DataFrame не пустой.
    """
    if df.empty:
        raise ValueError(f"{object_name} is empty.")


def validate_target_column(
    df: pd.DataFrame,
    target_col: str,
    object_name: str = "DataFrame",
) -> None:
    """
    Проверяет наличие целевой переменной.
    """
    if target_col not in df.columns:
        possible_targets = [
            column for column in df.columns
            if column.startswith("target_")
        ]

        raise KeyError(
            f"Target column '{target_col}' not found in {object_name}. "
            f"Available target-like columns: {possible_targets}"
        )


def validate_file_role(
    file_role: str,
    allowed_roles: Iterable[str],
) -> None:
    """
    Проверяет, что роль файла входит в допустимый список.
    """
    allowed_roles = list(allowed_roles)

    if file_role not in allowed_roles:
        raise ValueError(
            f"Unsupported file role: {file_role}. "
            f"Allowed roles: {allowed_roles}"
        )


def validate_scenario(
    scenario: str,
    allowed_scenarios: Iterable[str],
) -> None:
    """
    Проверяет, что сценарий входит в допустимый список.
    """
    allowed_scenarios = list(allowed_scenarios)

    if scenario not in allowed_scenarios:
        raise ValueError(
            f"Unsupported scenario: {scenario}. "
            f"Allowed scenarios: {allowed_scenarios}"
        )


def validate_probability_values(
    series: pd.Series,
    object_name: str = "probability column",
) -> None:
    """
    Проверяет, что значения вероятностей находятся в диапазоне [0, 1].
    """
    invalid_mask = (series < 0) | (series > 1)

    if invalid_mask.any():
        invalid_count = int(invalid_mask.sum())
        raise ValueError(
            f"{object_name} contains {invalid_count} values "
            "outside the [0, 1] range."
        )

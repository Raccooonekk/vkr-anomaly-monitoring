import yaml
import pandas as pd
from pathlib import Path


def load_yaml(path: Path) -> dict:
    """
    Загружает YAML-конфиг.
    """
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_table(path: Path) -> pd.DataFrame:
    """
    Читает CSV или Parquet.
    """
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file format: {suffix}")


def save_table(df: pd.DataFrame, path: Path, save_index: bool = False) -> None:
    """
    Сохраняет DataFrame в CSV или Parquet.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(path, index=save_index, encoding="utf-8-sig")
        return

    if suffix == ".parquet":
        df.to_parquet(path, index=save_index)
        return

    raise ValueError(f"Unsupported file format: {suffix}")

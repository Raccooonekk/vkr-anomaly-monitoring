"""
Утилиты для работы с путями проекта.

Модуль учитывает особенности Kaggle:
- /kaggle/input используется только для чтения;
- /kaggle/working используется для сохранения результатов;
- локальные Windows-пути не используются.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


def get_project_root() -> Path:
    """
    Возвращает корень проекта.

    Для файла src/utils/paths.py корень находится на два уровня выше:
    src/utils/paths.py -> src/utils -> src -> project_root.
    """
    return Path(__file__).resolve().parents[2]


def get_config_path() -> Path:
    """
    Возвращает путь к основному YAML-конфигу проекта.
    """
    return get_project_root() / "configs" / "config.yaml"


def to_path(path: str | Path) -> Path:
    """
    Преобразует строку или Path в объект Path.
    """
    return path if isinstance(path, Path) else Path(path)


def make_dir(path: str | Path) -> Path:
    """
    Создает директорию, если она отсутствует.

    Возвращает Path созданной или уже существующей директории.
    """
    path = to_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_exists(path: str | Path, description: str = "path") -> Path:
    """
    Проверяет существование файла или директории.

    Если путь отсутствует, выбрасывает понятную ошибку.
    """
    path = to_path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Required {description} does not exist: {path}"
        )

    return path


def require_file(path: str | Path, description: str = "file") -> Path:
    """
    Проверяет, что путь существует и является файлом.
    """
    path = require_exists(path, description=description)

    if not path.is_file():
        raise FileNotFoundError(
            f"Required {description} is not a file: {path}"
        )

    return path


def require_dir(path: str | Path, description: str = "directory") -> Path:
    """
    Проверяет, что путь существует и является директорией.
    """
    path = require_exists(path, description=description)

    if not path.is_dir():
        raise NotADirectoryError(
            f"Required {description} is not a directory: {path}"
        )

    return path


def resolve_first_existing_path(
    candidates: Iterable[str | Path],
    description: str = "path",
) -> Path:
    """
    Возвращает первый существующий путь из списка кандидатов.

    Это полезно для Kaggle, где путь к датасету может отличаться
    в зависимости от способа подключения dataset.
    """
    checked_paths: list[Path] = []

    for candidate in candidates:
        path = to_path(candidate)
        checked_paths.append(path)

        if path.exists():
            return path

    checked = "\n".join(f"- {path}" for path in checked_paths)
    raise FileNotFoundError(
        f"Could not find existing {description}. Checked paths:\n{checked}"
    )


def is_subpath(path: str | Path, parent: str | Path) -> bool:
    """
    Проверяет, находится ли path внутри parent.

    Используется для защиты от записи в /kaggle/input.
    """
    path = to_path(path).resolve(strict=False)
    parent = to_path(parent).resolve(strict=False)

    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def ensure_writable_output_path(
    path: str | Path,
    kaggle_input_root: str | Path = "/kaggle/input",
) -> Path:
    """
    Проверяет, что путь не находится внутри /kaggle/input.

    В Kaggle директория /kaggle/input доступна только для чтения.
    Все результаты нужно сохранять в /kaggle/working.
    """
    path = to_path(path)

    if is_subpath(path, kaggle_input_root):
        raise PermissionError(
            "Writing to /kaggle/input is forbidden. "
            f"Use /kaggle/working for outputs. Got path: {path}"
        )

    return path


def get_output_root(config: dict) -> Path:
    """
    Возвращает корневую директорию для результатов из config.
    """
    output_root = Path(config["output"]["root"])
    ensure_writable_output_path(
        output_root,
        kaggle_input_root=config.get("kaggle", {}).get("input_root", "/kaggle/input"),
    )
    return make_dir(output_root)


def get_logs_dir(config: dict) -> Path:
    """
    Возвращает директорию для логов.
    """
    output_root = get_output_root(config)
    logs_dir = config.get("output", {}).get("logs_dir", "logs")
    return make_dir(output_root / logs_dir)


def get_reports_dir(config: dict) -> Path:
    """
    Возвращает директорию для отчетов и таблиц результатов.
    """
    output_root = get_output_root(config)
    reports_dir = config.get("output", {}).get("reports_dir", "reports")
    return make_dir(output_root / reports_dir)


def get_models_dir(config: dict) -> Path:
    """
    Возвращает директорию для сохранения моделей.
    """
    output_root = get_output_root(config)
    models_dir = config.get("output", {}).get("models_dir", "models")
    return make_dir(output_root / models_dir)


def get_predictions_dir(config: dict) -> Path:
    """
    Возвращает директорию для сохранения предсказаний модели.
    """
    output_root = get_output_root(config)
    predictions_dir = config.get("output", {}).get("predictions_dir", "predictions")
    return make_dir(output_root / predictions_dir)


def get_metadata_dir(config: dict) -> Path:
    """
    Возвращает директорию для сохранения metadata-файлов.
    """
    output_root = get_output_root(config)
    metadata_dir = config.get("output", {}).get("metadata_dir", "metadata")
    return make_dir(output_root / metadata_dir)


def get_dataset_root(config: dict) -> Path:
    """
    Возвращает существующий путь к исходному датасету.

    Сначала проверяется список dataset_root_candidates.
    Если он отсутствует, используется dataset.dataset_root.
    """
    dataset_config = config.get("dataset", {})
    candidates = dataset_config.get("dataset_root_candidates")

    if candidates:
        return resolve_first_existing_path(
            candidates,
            description="dataset root",
        )

    return require_dir(
        dataset_config["dataset_root"],
        description="dataset root",
    )


def get_processed_dataset_root(config: dict) -> Path:
    """
    Возвращает путь к processed_dataset внутри исходного датасета.
    """
    dataset_root = get_dataset_root(config)
    processed_dir = config.get("dataset", {}).get(
        "processed_dataset_dir",
        "processed_dataset",
    )

    return require_dir(
        dataset_root / processed_dir,
        description="processed dataset directory",
    )


def get_manifest_root(config: dict) -> Path:
    """
    Возвращает существующий путь к директории manifest.

    Сначала проверяется manifest.root_candidates.
    Если он отсутствует, используется manifest.root.
    """
    manifest_config = config.get("manifest", {})
    candidates = manifest_config.get("root_candidates")

    if candidates:
        return resolve_first_existing_path(
            candidates,
            description="manifest root",
        )

    return require_dir(
        manifest_config["root"],
        description="manifest root",
    )


def get_manifest_files_index_path(config: dict) -> Path:
    """
    Возвращает путь к merged_files_index.
    """
    manifest_root = get_manifest_root(config)
    file_name = config.get("manifest", {}).get(
        "files_index",
        "merged_files_index.parquet",
    )

    return require_file(
        manifest_root / file_name,
        description="manifest files index",
    )


def get_manifest_samples_index_path(config: dict) -> Path:
    """
    Возвращает путь к merged_samples_index.
    """
    manifest_root = get_manifest_root(config)
    file_name = config.get("manifest", {}).get(
        "samples_index",
        "merged_samples_index.parquet",
    )

    return require_file(
        manifest_root / file_name,
        description="manifest samples index",
    )


def get_manifest_role_index_path(config: dict) -> Path:
    """
    Возвращает путь к merged_role_index.
    """
    manifest_root = get_manifest_root(config)
    file_name = config.get("manifest", {}).get(
        "role_index",
        "merged_role_index.parquet",
    )

    return require_file(
        manifest_root / file_name,
        description="manifest role index",
    )


def get_optional_path(path: Optional[str | Path]) -> Optional[Path]:
    """
    Возвращает Path или None, если путь не задан.
    """
    if path is None:
        return None

    return to_path(path)

"""
Единая настройка логирования проекта.

Логи нужны для:
- Kaggle Notebook;
- GitHub Actions;
- отладки batch-обработки;
- анализа падений pipeline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from src.utils.paths import make_dir, to_path


_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "vkr_anomaly_monitoring",
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    reset_handlers: bool = True,
) -> logging.Logger:
    """
    Создает и настраивает logger.

    Parameters
    ----------
    name:
        Имя logger.
    level:
        Уровень логирования: DEBUG, INFO, WARNING, ERROR.
    log_file:
        Необязательный путь к файлу логов.
    reset_handlers:
        Если True, старые handlers удаляются.
        Это защищает от дублирования логов в Kaggle Notebook.
    """
    logger = logging.getLogger(name)

    if isinstance(level, str):
        log_level = getattr(logging, level.upper(), logging.INFO)
    else:
        log_level = level

    logger.setLevel(log_level)
    logger.propagate = False

    if reset_handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file = to_path(log_file)
        make_dir(log_file.parent)

        file_handler = logging.FileHandler(
            log_file,
            mode="a",
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "vkr_anomaly_monitoring") -> logging.Logger:
    """
    Возвращает logger по имени.
    """
    return logging.getLogger(name)


def log_section(logger: logging.Logger, title: str) -> None:
    """
    Выводит визуальный разделитель этапов pipeline.
    """
    separator = "=" * 80
    logger.info(separator)
    logger.info(title)
    logger.info(separator)


def log_config(logger: logging.Logger, config: dict[str, Any]) -> None:
    """
    Логирует ключевые параметры конфигурации без вывода токенов и секретов.
    """
    safe_keys = [
        "project",
        "kaggle",
        "dataset",
        "manifest",
        "output",
        "preprocessing",
        "split",
        "target",
        "training",
    ]

    logger.info("Loaded configuration:")

    for key in safe_keys:
        if key in config:
            logger.info("%s: %s", key, config[key])


def setup_logger_from_config(config: dict) -> logging.Logger:
    """
    Настраивает logger на основе config.yaml.
    """
    logging_config = config.get("logging", {})
    output_config = config.get("output", {})

    level = logging_config.get("level", "INFO")
    log_to_file = logging_config.get("log_to_file", True)
    log_filename = logging_config.get("log_filename", "pipeline.log")

    log_file = None

    if log_to_file:
        output_root = Path(output_config.get("root", "/kaggle/working"))
        logs_dir = output_config.get("logs_dir", "logs")
        log_file = output_root / logs_dir / log_filename

    return setup_logger(
        name=config.get("project", {}).get("name", "vkr_anomaly_monitoring"),
        level=level,
        log_file=log_file,
    )

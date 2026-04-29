from pathlib import Path


def get_project_root() -> Path:
    """
    Возвращает корень проекта.
    """
    return Path(__file__).resolve().parents[2]


def get_config_path() -> Path:
    """
    Путь к основному config.yaml.
    """
    return get_project_root() / "configs" / "config.yaml"


def make_dir(path: Path) -> Path:
    """
    Создает папку, если ее нет.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
"""Build logical manifest for processed predictive datasets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import load_yaml, save_json, save_table
from src.utils.paths import get_config_path, get_output_root, make_dir, to_path

ROLE_BY_STEM = {
    "aggregated_system_state": "aggregated",
    "marked_anomaly_state": "marked",
    "predictive_p95_dataset": "predictive",
    "train": "train",
    "val": "val",
    "test": "test",
    "metadata": "metadata",
}


def detect_file_role(path: Path) -> str | None:
    """Detects file role from its stem."""
    return ROLE_BY_STEM.get(path.stem)


def find_predictive_roots(search_roots: list[str | Path]) -> list[Path]:
    """Finds predictive_processed_dataset directories."""
    roots: list[Path] = []
    for search_root in search_roots:
        root = to_path(search_root)
        if root.exists():
            roots.extend(path for path in root.rglob("predictive_processed_dataset") if path.is_dir())
    return sorted(set(roots))


def build_files_index(predictive_roots: list[str | Path]) -> pd.DataFrame:
    """Builds a file-level manifest index."""
    rows: list[dict[str, Any]] = []
    for root in predictive_roots:
        root = to_path(root)
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".parquet", ".csv", ".json"}:
                continue
            role = detect_file_role(path)
            if role is None:
                continue
            rel = path.relative_to(root).as_posix()
            parts = Path(rel).parts
            rows.append({
                "absolute_path": str(path),
                "file_role": role,
                "relative_path_from_part": rel,
                "predictive_root": str(root),
                "part_name": parts[0] if parts else "",
                "scenario": parts[1] if len(parts) > 1 else "",
                "logical_sample_key": "/".join(parts[:-1]) if len(parts) > 1 else path.stem,
                "file_name": path.name,
                "file_suffix": path.suffix.lower(),
                "parent_dir_name": path.parent.name,
            })
    if not rows:
        raise FileNotFoundError("No processed predictive files found for manifest build.")
    return pd.DataFrame(rows).sort_values(["part_name", "scenario", "logical_sample_key", "file_role"])


def build_samples_index(files_index: pd.DataFrame) -> pd.DataFrame:
    """Builds one row per logical processed source object."""
    rows: list[dict[str, Any]] = []
    for sample_key, group in files_index.groupby("logical_sample_key"):
        roles = set(group["file_role"].tolist())
        rows.append({
            "logical_sample_key": sample_key,
            "part_name": group["part_name"].iloc[0],
            "scenario": group["scenario"].iloc[0],
            "file_count": int(len(group)),
            "has_predictive": "predictive" in roles,
            "has_train": "train" in roles,
            "has_val": "val" in roles,
            "has_test": "test" in roles,
            "has_metadata": "metadata" in roles,
            "is_complete_for_training": {"train", "val", "test"}.issubset(roles),
        })
    return pd.DataFrame(rows).sort_values(["part_name", "scenario", "logical_sample_key"])


def build_role_index(files_index: pd.DataFrame) -> pd.DataFrame:
    """Builds role-level statistics."""
    return files_index.groupby(["file_role", "file_suffix"], as_index=False).agg(file_count=("absolute_path", "count"))


def write_manifest(files_index: pd.DataFrame, output_dir: str | Path) -> dict[str, Any]:
    """Writes manifest parquet, csv and json files."""
    output_dir = make_dir(output_dir)
    samples_index = build_samples_index(files_index)
    role_index = build_role_index(files_index)
    save_table(files_index, output_dir / "merged_files_index.parquet")
    save_table(samples_index, output_dir / "merged_samples_index.parquet")
    save_table(role_index, output_dir / "merged_role_index.parquet")
    save_table(files_index, output_dir / "merged_files_index.csv")
    save_table(samples_index, output_dir / "merged_samples_index.csv")
    save_table(role_index, output_dir / "merged_role_index.csv")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": int(len(files_index)),
        "sample_count": int(len(samples_index)),
        "complete_training_sample_count": int(samples_index["is_complete_for_training"].sum()),
        "roles": role_index.to_dict(orient="records"),
        "scenarios": sorted(files_index["scenario"].dropna().unique().tolist()),
    }
    save_json(manifest, output_dir / "merged_manifest.json")
    return manifest


def run_manifest_build(config_path: str | Path | None = None, search_roots: list[str | Path] | None = None, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Runs logical manifest build."""
    config = load_yaml(config_path or get_config_path())
    if search_roots is None:
        search_roots = config.get("manifest_build", {}).get("search_roots") or [config.get("kaggle", {}).get("input_root", "/kaggle/input"), str(get_output_root(config))]
    roots = find_predictive_roots(search_roots)
    files_index = build_files_index(roots)
    output_dir = output_dir or config.get("manifest_build", {}).get("output_dir", str(get_output_root(config) / "merged_predictive_manifest"))
    return write_manifest(files_index, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(get_config_path()))
    parser.add_argument("--search-root", action="append", dest="search_roots")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    print(json.dumps(run_manifest_build(args.config, args.search_roots, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

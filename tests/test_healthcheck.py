from __future__ import annotations

import json
from pathlib import Path

from topper_perception.healthcheck import check_paths, load_path_config, package_versions


def test_load_and_check_existing_path(tmp_path: Path) -> None:
    config_path = tmp_path / "paths.json"
    config_path.write_text(
        json.dumps({"existing": str(tmp_path)}),
        encoding="utf-8",
    )

    paths = load_path_config(config_path)
    result = check_paths(paths)

    assert result["existing"]["exists"] is True
    assert result["existing"]["is_dir"] is True


def test_missing_path_is_reported_without_creation(tmp_path: Path) -> None:
    missing = tmp_path / "not-created"

    result = check_paths({"missing": missing})

    assert result["missing"]["exists"] is False
    assert missing.exists() is False


def test_data_analysis_packages_are_installed() -> None:
    assert all(version != "MISSING" for version in package_versions().values())


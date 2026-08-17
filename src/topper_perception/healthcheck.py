"""Environment and external-path health checks for the team workbench."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Mapping


PACKAGE_NAMES = (
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "scikit-learn",
    "seaborn",
    "pytest",
)


def load_path_config(config_path: Path) -> dict[str, Path]:
    """Read a JSON object whose values are local filesystem paths."""
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Path config must be a non-empty JSON object.")

    paths: dict[str, Path] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not isinstance(value, str) or not value.strip():
            raise ValueError("Every path config entry must contain a string name and path.")
        paths[name] = Path(value)
    return paths


def check_paths(paths: Mapping[str, Path]) -> dict[str, dict[str, object]]:
    """Return existence and type information without modifying any path."""
    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "is_dir": path.is_dir(),
            "is_file": path.is_file(),
        }
        for name, path in paths.items()
    }


def package_versions() -> dict[str, str]:
    """Report installed versions for the agreed data-analysis stack."""
    versions: dict[str, str] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "MISSING"
    return versions


def build_report(config_path: Path) -> dict[str, object]:
    """Build a machine-readable health report."""
    path_results = check_paths(load_path_config(config_path))
    packages = package_versions()
    return {
        "ok": all(item["exists"] for item in path_results.values())
        and all(version != "MISSING" for version in packages.values()),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "platform": platform.platform(),
        },
        "packages": packages,
        "paths": path_results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paths.local.json"),
        help="JSON file containing local external paths.",
    )
    args = parser.parse_args(argv)
    report = build_report(args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


"""Create the SLP S0 structural inventory and annotation-boundary audit."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from topper_perception.healthcheck import load_path_config
from topper_perception.io.slp_inventory import (
    ANNOTATION_COLUMNS,
    INVENTORY_COLUMNS,
    audit_slp_annotations,
    canonical_summary_json,
    inventory_slp_dataset,
    resolve_slp_root,
    summarise_slp_inventory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "paths.local.json",
    )
    parser.add_argument(
        "--inventory-output",
        type=Path,
        default=Path("data/processed/slp/slp_modality_inventory_v0.1.csv"),
    )
    parser.add_argument(
        "--annotation-output",
        type=Path,
        default=Path("data/processed/slp/slp_annotation_inventory_v0.1.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/reports/slp_inventory_summary_v0.1.json"),
    )
    return parser


def _write_csv(rows: list[dict[str, object]], columns: tuple[str, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = load_path_config(args.config)
    if "slp_data" not in paths:
        raise KeyError(f"slp_data is missing from path config: {args.config}")

    slp_root = resolve_slp_root(paths["slp_data"])
    inventory_rows = list(inventory_slp_dataset(slp_root))
    annotation_rows = list(audit_slp_annotations(slp_root))
    summary = summarise_slp_inventory(
        inventory_rows,
        annotation_rows,
        slp_root=slp_root,
    )
    inventory_output = _project_path(args.inventory_output)
    annotation_output = _project_path(args.annotation_output)
    summary_output = _project_path(args.summary_output)
    summary.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "inventory_output": str(inventory_output),
            "annotation_output": str(annotation_output),
        }
    )

    _write_csv([row.as_dict() for row in inventory_rows], INVENTORY_COLUMNS, inventory_output)
    _write_csv([row.as_dict() for row in annotation_rows], ANNOTATION_COLUMNS, annotation_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(canonical_summary_json(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    has_errors = bool(summary["group_error_counts"] or summary["annotation_error_counts"])
    return 2 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

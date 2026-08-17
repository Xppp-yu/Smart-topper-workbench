"""Create the R1/P1 structural inventory for every PoPu Tactilus JSON record."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt

from topper_perception.healthcheck import load_path_config
from topper_perception.io.popu_inventory import (
    INVENTORY_COLUMNS,
    POPU_POSTURES,
    inventory_tactilus_dataset,
    resolve_tactilus_root,
    summarise_inventory,
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
        default=Path("data/processed/popu/popu_tactilus_inventory_v0.1.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/reports/popu_tactilus_inventory_summary_v0.1.json"),
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("outputs/figures/popu_tactilus_label_distribution_v0.1.png"),
    )
    parser.add_argument(
        "--include-sha256",
        action="store_true",
        help="Add per-file SHA-256 values; this makes the pass more I/O intensive.",
    )
    return parser


def _write_inventory(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_label_distribution(summary: dict[str, object], output_path: Path) -> None:
    posture_counts = dict(summary["posture_counts"])
    labels = [posture for posture in POPU_POSTURES if posture in posture_counts]
    labels.extend(sorted(label for label in posture_counts if label not in labels))
    counts = [int(posture_counts[label]) for label in labels]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    bars = ax.bar(labels, counts, color="#6C5CE7")
    ax.set_title("PoPu Tactilus inventory: records by posture label")
    ax.set_xlabel("Posture label from source JSON")
    ax.set_ylabel("JSON record count")
    ax.bar_label(bars, padding=3)
    ax.set_ylim(0, max(counts, default=1) * 1.15)
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = load_path_config(args.config)
    if "popu_data" not in paths:
        raise KeyError(f"popu_data is missing from path config: {args.config}")

    inventory_output = _project_path(args.inventory_output)
    summary_output = _project_path(args.summary_output)
    figure_output = _project_path(args.figure_output)
    tactilus_root = resolve_tactilus_root(paths["popu_data"])

    # Rows are compact metadata only. Source JSON files are read one at a time
    # inside the generator; no dataset-wide pressure-matrix array exists.
    inventory_rows = list(inventory_tactilus_dataset(
        tactilus_root,
        include_sha256=args.include_sha256,
    ))
    rows = [row.as_dict() for row in inventory_rows]
    summary = summarise_inventory(inventory_rows, tactilus_root=tactilus_root)
    summary.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "inventory_output": str(inventory_output),
            "figure_output": str(figure_output),
            "include_sha256": args.include_sha256,
        }
    )

    _write_inventory(rows, inventory_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_label_distribution(summary, figure_output)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary["error_counts"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

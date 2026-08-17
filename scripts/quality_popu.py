"""Run P2/R2 PoPu Tactilus record-quality gate and review galleries."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from topper_perception.io.popu import POPU_POSTURES, load_tactilus_record
from topper_perception.quality.popu import (
    QUALITY_COLUMNS,
    RecordMetrics,
    assess_quality,
    build_quality_summary,
    compute_record_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory-input",
        type=Path,
        default=Path("data/processed/popu/popu_tactilus_inventory_v0.1.csv"),
    )
    parser.add_argument(
        "--quality-output",
        type=Path,
        default=Path("outputs/metrics/popu_tactilus_quality_results_v0.1.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/reports/popu_tactilus_quality_summary_v0.1.json"),
    )
    parser.add_argument(
        "--gallery-output",
        type=Path,
        default=Path("outputs/figures/popu_posture_gallery_v0.1.png"),
    )
    parser.add_argument(
        "--abnormal-output",
        type=Path,
        default=Path("outputs/figures/popu_abnormal_samples_v0.1.png"),
    )
    parser.add_argument(
        "--robust-z-threshold",
        type=float,
        default=4.5,
        help="Provisional per-posture WARN threshold; requires visual review.",
    )
    return parser


def _read_inventory(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(records: Sequence[RecordMetrics], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUALITY_COLUMNS)
        writer.writeheader()
        writer.writerows(record.as_dict() for record in records)


def _selected_record(rows: Sequence[dict[str, object]], posture: str, category: str) -> dict[str, object]:
    candidates = [
        row
        for row in rows
        if row["posture"] == posture and row["quality_status"] in ("ACCEPT", "WARN")
    ]
    if category == "typical":
        return min(candidates, key=lambda row: float(row["maximum_robust_z"]))
    if category == "low signal":
        return min(candidates, key=lambda row: float(row["median_total_signal"]))
    return max(candidates, key=lambda row: float(row["median_total_signal"]))


def _load_selected_frame(row: dict[str, object]) -> np.ndarray:
    frames = load_tactilus_record(Path(str(row["source_file"])))
    return frames[int(row["representative_frame_index"])].values


def _render_gallery(records: Sequence[RecordMetrics], output_path: Path) -> None:
    rows = [record.as_dict() for record in records]
    categories = ("typical", "low signal", "high signal")
    selected = [
        (posture, category, _selected_record(rows, posture, category))
        for posture in POPU_POSTURES
        for category in categories
    ]
    arrays = [_load_selected_frame(row) for _, _, row in selected]
    vmax = max(float(np.max(array)) for array in arrays)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(5, 3, figsize=(10, 16), constrained_layout=True)
    for ax, (posture, category, row), values in zip(axes.flat, selected, arrays, strict=True):
        image = ax.imshow(values, cmap="magma", origin="upper", interpolation="nearest", vmin=0, vmax=vmax)
        ax.set_title(f"{posture} | {category}\nsubject={row['subject_id']} | v={row['variation']}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=list(axes.flat), fraction=0.025, pad=0.02, label="Raw sensor value (dataset units)")
    fig.suptitle("PoPu Tactilus P2 quality gallery | representative frames | common scale", fontsize=14)
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def _render_abnormal(records: Sequence[RecordMetrics], output_path: Path) -> None:
    rows = [record.as_dict() for record in records]
    flagged = [row for row in rows if row["quality_status"] in ("WARN", "REJECT")]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not flagged:
        fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
        ax.axis("off")
        ax.text(0.5, 0.5, "No WARN or REJECT records under the current provisional rules.", ha="center", va="center")
        fig.savefig(output_path, dpi=180, facecolor="white")
        plt.close(fig)
        return

    selected = sorted(flagged, key=lambda row: float(row["maximum_robust_z"] or 0), reverse=True)[:12]
    columns = min(3, len(selected))
    rows_count = int(np.ceil(len(selected) / columns))
    fig, axes = plt.subplots(rows_count, columns, figsize=(3.4 * columns, 4.4 * rows_count), constrained_layout=True, squeeze=False)
    for ax, row in zip(axes.flat, selected, strict=False):
        if row["quality_status"] == "REJECT":
            ax.axis("off")
            ax.set_title(f"REJECT\n{row['sample_id']}", fontsize=8)
            continue
        values = _load_selected_frame(row)
        ax.imshow(values, cmap="magma", origin="upper", interpolation="nearest")
        ax.set_title(f"WARN | {row['posture']} | subject={row['subject_id']}\n{row['quality_reasons']}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[len(selected):]:
        ax.axis("off")
    fig.suptitle("PoPu Tactilus P2 candidates for human quality review", fontsize=13)
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.robust_z_threshold <= 0:
        raise ValueError("robust-z-threshold must be positive")

    inventory_input = _project_path(args.inventory_input)
    quality_output = _project_path(args.quality_output)
    summary_output = _project_path(args.summary_output)
    gallery_output = _project_path(args.gallery_output)
    abnormal_output = _project_path(args.abnormal_output)

    inventory_rows = _read_inventory(inventory_input)
    # The source reader processes one JSON record at a time. Only compact
    # record metrics remain in memory after each source file has been closed.
    computed = [compute_record_metrics(row) for row in inventory_rows]
    assessed, profiles = assess_quality(computed, robust_z_threshold=args.robust_z_threshold)
    summary = build_quality_summary(
        assessed,
        profiles=profiles,
        robust_z_threshold=args.robust_z_threshold,
    )
    summary.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "inventory_input": str(inventory_input),
            "quality_output": str(quality_output),
            "gallery_output": str(gallery_output),
            "abnormal_output": str(abnormal_output),
        }
    )

    _write_csv(assessed, quality_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _render_gallery(assessed, gallery_output)
    _render_abnormal(assessed, abnormal_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["quality_status_counts"].get("REJECT", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

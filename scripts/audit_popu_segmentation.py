"""Audit whether PoPu COCO segmentation labels can supervise Tactilus samples."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt

from topper_perception.labels.popu import BODY_PART_CATEGORIES, audit_segmentation_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_COLUMNS = (
    "annotation_file", "subject_id", "posture", "variation", "image_count",
    "image_id", "image_file_name", "canvas_height", "canvas_width",
    "annotation_count", "annotation_category_ids",
    "category_name_mismatch_count", "annotation_image_reference_error_count",
    "annotation_bbox_error_count", "annotation_category_error_count",
    "candidate_tactilus_record_count", "candidate_tactilus_records",
    "invalid_polygon_point_count", "structural_errors", "alignment_status",
    "supervision_boundary",
)


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paths.local.json"))
    parser.add_argument("--audit-output", type=Path, default=Path("outputs/metrics/popu_segmentation_alignment_audit_v0.1.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("outputs/reports/popu_segmentation_alignment_summary_v0.1.json"))
    parser.add_argument("--figure-output", type=Path, default=Path("outputs/figures/popu_segmentation_alignment_v0.1.png"))
    return parser


def _load_paths(config_path: Path) -> tuple[Path, Path]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_root = Path(config["popu_data"]).expanduser()
    return data_root / "segmentation_data", data_root / "tactilus_data"


def _write_csv(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_figure(status_counts: Counter[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels, values = zip(*sorted(status_counts.items()), strict=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bars = ax.bar(labels, values, color="#4c78a8")
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("COCO annotation files")
    ax.set_title("PoPu segmentation-to-Tactilus pairing audit")
    ax.tick_params(axis="x", rotation=16)
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = _project_path(args.config)
    segmentation_root, tactilus_root = _load_paths(config_path)
    audit_output = _project_path(args.audit_output)
    summary_output = _project_path(args.summary_output)
    figure_output = _project_path(args.figure_output)

    files = sorted(segmentation_root.glob("*/*_annotations.coco.json"))
    rows: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []
    for annotation_path in files:
        try:
            rows.append(audit_segmentation_file(annotation_path, tactilus_root))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            rejected.append({"annotation_file": str(annotation_path), "reason": type(exc).__name__})

    status_counts: Counter[str] = Counter(str(row["alignment_status"]) for row in rows)
    _write_csv(rows, audit_output)
    _write_figure(status_counts, figure_output)
    summary = {
        "dataset": "PoPu",
        "stage": "P3.2/R3.2",
        "purpose": "Audit structural validity and Tactilus pairing cardinality before region-supervised training.",
        "category_contract": BODY_PART_CATEGORIES,
        "segmentation_files_discovered": len(files),
        "files_audited": len(rows),
        "read_rejects": rejected,
        "alignment_status_counts": dict(sorted(status_counts.items())),
        "candidate_record_count_distribution": dict(sorted(Counter(str(row["candidate_tactilus_record_count"]) for row in rows).items())),
        "supervision_decision": (
            "HOLD frame-level body-part training unless audit results establish a documented "
            "one-to-one record/snapshot pairing rule. COCO regions may still be used for "
            "structural review and candidate-label research with this limitation attached."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "audit_output": str(audit_output),
        "figure_output": str(figure_output),
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not rejected else 2


if __name__ == "__main__":
    raise SystemExit(main())

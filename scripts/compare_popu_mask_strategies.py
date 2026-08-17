"""Run P3.1: compare explicit PoPu contact-mask candidates before features."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from topper_perception.geometry.mask_strategies import MASK_STRATEGIES, build_strategy_mask, jaccard, mask_summary
from topper_perception.io.popu import load_tactilus_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_COLUMNS = (
    "sample_id", "subject_id", "posture", "variation", "p2_quality_status", "strategy",
    "frame_count", "usable_frame_count", "usable_frame_fraction", "median_mask_fraction",
    "median_bbox_area_fraction", "median_component_count", "mean_consecutive_mask_iou",
    "mean_consecutive_cop_shift", "comparison_status", "comparison_reason",
)


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-input", type=Path, default=Path("outputs/metrics/popu_tactilus_quality_results_v0.1.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/popu_mask_strategy_comparison_v0.1.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("outputs/reports/popu_mask_strategy_comparison_summary_v0.1.json"))
    parser.add_argument("--figure-output", type=Path, default=Path("outputs/figures/popu_mask_strategy_stability_v0.1.png"))
    parser.add_argument("--positive-percentile", type=float, default=50.0)
    parser.add_argument("--minimum-raw-threshold", type=float, default=1.0)
    parser.add_argument("--minimum-component-cells", type=int, default=3)
    parser.add_argument("--minimum-component-fraction-of-largest", type=float, default=0.02)
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _evaluate_record(row: dict[str, str], args: argparse.Namespace, strategy: str) -> dict[str, object]:
    result: dict[str, object] = {
        "sample_id": row["sample_id"], "subject_id": row["subject_id"], "posture": row["posture"],
        "variation": row["variation"], "p2_quality_status": row["quality_status"], "strategy": strategy,
        **{column: "" for column in RESULT_COLUMNS[6:]},
    }
    if row["quality_status"] not in {"ACCEPT", "WARN"}:
        result.update({"comparison_status": "EXCLUDED", "comparison_reason": "not_in_p2_eligible_population"})
        return result
    try:
        frames = load_tactilus_record(Path(row["source_file"]))
        masks: list[np.ndarray] = []
        summaries: list[dict[str, float | int]] = []
        for frame in frames:
            mask, _ = build_strategy_mask(
                frame.values, strategy=strategy,
                positive_percentile=args.positive_percentile,
                minimum_raw_threshold=args.minimum_raw_threshold,
                minimum_component_cells=args.minimum_component_cells,
                minimum_component_fraction_of_largest=args.minimum_component_fraction_of_largest,
            )
            masks.append(mask)
            summaries.append(mask_summary(mask, frame.values))
        usable = [item for item in summaries if int(item["mask_cell_count"]) > 0]
        ious = [jaccard(left, right) for left, right in zip(masks, masks[1:], strict=False)]
        cop_shifts = [
            float(np.hypot(float(right["cop_row_fraction"]) - float(left["cop_row_fraction"]), float(right["cop_column_fraction"]) - float(left["cop_column_fraction"])))
            for left, right in zip(summaries, summaries[1:], strict=False)
            if np.isfinite(float(left["cop_row_fraction"])) and np.isfinite(float(right["cop_row_fraction"]))
        ]
        result.update({
            "frame_count": len(frames), "usable_frame_count": len(usable), "usable_frame_fraction": len(usable) / len(frames),
            "median_mask_fraction": float(np.median([float(item["mask_fraction"]) for item in usable])) if usable else 0.0,
            "median_bbox_area_fraction": float(np.median([float(item["bbox_area_fraction"]) for item in usable])) if usable else 0.0,
            "median_component_count": float(np.median([float(item["component_count"]) for item in usable])) if usable else 0.0,
            "mean_consecutive_mask_iou": float(np.nanmean(ious)) if any(np.isfinite(ious)) else "",
            "mean_consecutive_cop_shift": float(np.mean(cop_shifts)) if cop_shifts else "",
            "comparison_status": "OK" if usable else "WARN", "comparison_reason": "" if usable else "all_frames_empty_mask",
        })
    except (OSError, ValueError, IndexError) as exc:
        result.update({"comparison_status": "REJECT", "comparison_reason": f"source_or_mask_error:{type(exc).__name__}"})
    return result


def _write_csv(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_figure(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels, values = [], []
    for strategy in MASK_STRATEGIES:
        selected = [float(row["mean_consecutive_mask_iou"]) for row in rows if row["strategy"] == strategy and row["comparison_status"] == "OK" and row["mean_consecutive_mask_iou"] != ""]
        labels.append(strategy); values.append(float(np.median(selected)) if selected else 0.0)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bars = ax.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Median consecutive-frame mask IoU")
    ax.set_title("P3.1 candidate stability (not anatomical accuracy)")
    ax.tick_params(axis="x", rotation=12)
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    quality_input, output = _project_path(args.quality_input), _project_path(args.output)
    summary_output, figure_output = _project_path(args.summary_output), _project_path(args.figure_output)
    input_rows = _read_csv(quality_input)
    rows = [_evaluate_record(row, args, strategy) for row in input_rows for strategy in MASK_STRATEGIES]
    _write_csv(rows, output); _write_figure(rows, figure_output)
    by_strategy = {}
    for strategy in MASK_STRATEGIES:
        selected = [row for row in rows if row["strategy"] == strategy and row["comparison_status"] == "OK"]
        ious = [float(row["mean_consecutive_mask_iou"]) for row in selected if row["mean_consecutive_mask_iou"] != ""]
        by_strategy[strategy] = {"ok_records": len(selected), "median_consecutive_mask_iou": float(np.median(ious)) if ious else None}
    summary = {
        "dataset": "PoPu", "stage": "P3.1/R3.1", "candidate_strategies": list(MASK_STRATEGIES),
        "record_status_counts": dict(sorted(Counter(str(row["comparison_status"]) for row in rows).items())),
        "by_strategy": by_strategy,
        "decision_boundary": "Stability metrics rank candidates only. Freeze no mask rule until representative overlays and failure cases are reviewed.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "quality_input": str(quality_input), "output": str(output), "figure_output": str(figure_output),
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not any(row["comparison_status"] == "REJECT" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())

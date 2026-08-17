"""Run P3/R3 contact-mask and geometry extraction for eligible PoPu records."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from topper_perception.geometry.popu import (
    GEOMETRY_COLUMNS,
    describe_geometry,
    summarise_geometry,
)
from topper_perception.geometry.mask_strategies import MASK_STRATEGIES
from topper_perception.io.popu import POPU_POSTURES, load_tactilus_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality-input",
        type=Path,
        default=Path("outputs/metrics/popu_tactilus_quality_results_v0.1.csv"),
    )
    parser.add_argument(
        "--strategy",
        choices=MASK_STRATEGIES,
        default="relative_filtered",
        help="Named contact-mask strategy; keep the default for v0.1 reproducibility.",
    )
    parser.add_argument(
        "--geometry-output",
        type=Path,
        default=Path("outputs/metrics/popu_geometry_results_v0.1.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/reports/popu_geometry_summary_v0.1.json"),
    )
    parser.add_argument(
        "--mask-figure-output",
        type=Path,
        default=Path("outputs/figures/popu_mask_overlay_v0.1.png"),
    )
    parser.add_argument(
        "--geometry-figure-output",
        type=Path,
        default=Path("outputs/figures/popu_geometry_overlay_v0.1.png"),
    )
    parser.add_argument("--positive-percentile", type=float, default=50.0)
    parser.add_argument("--minimum-raw-threshold", type=float, default=1.0)
    parser.add_argument("--minimum-component-cells", type=int, default=3)
    parser.add_argument("--minimum-component-fraction-of-largest", type=float, default=0.02)
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _geometry_row(quality_row: dict[str, str], args: argparse.Namespace) -> dict[str, object]:
    base: dict[str, object] = {
        "sample_id": quality_row["sample_id"],
        "source_file": quality_row["source_file"],
        "subject_id": quality_row["subject_id"],
        "posture": quality_row["posture"],
        "variation": quality_row["variation"],
        "p2_quality_status": quality_row["quality_status"],
        "representative_frame_index": quality_row["representative_frame_index"],
        "mask_strategy": args.strategy,
        **{column: "" for column in GEOMETRY_COLUMNS[8:]},
    }
    if quality_row["quality_status"] not in ("ACCEPT", "WARN"):
        base.update({"geometry_status": "EXCLUDED", "geometry_reason": "not_in_p2_eligible_population"})
        return base
    try:
        frames = load_tactilus_record(Path(quality_row["source_file"]))
        values = frames[int(quality_row["representative_frame_index"])].values
        geometry, _ = describe_geometry(
            values,
            strategy=args.strategy,
            positive_percentile=args.positive_percentile,
            minimum_raw_threshold=args.minimum_raw_threshold,
            minimum_component_cells=args.minimum_component_cells,
            minimum_component_fraction_of_largest=args.minimum_component_fraction_of_largest,
        )
        return {**base, **geometry}
    except (OSError, ValueError, IndexError) as exc:
        base.update({"geometry_status": "REJECT", "geometry_reason": f"source_or_geometry_error:{type(exc).__name__}"})
        return base


def _write_csv(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GEOMETRY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _typical_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for posture in POPU_POSTURES:
        candidates = [row for row in rows if row["posture"] == posture and row["geometry_status"] == "OK" and row["p2_quality_status"] == "ACCEPT"]
        selected.append(min(candidates, key=lambda row: abs(float(row["principal_axis_anisotropy"]) - 0.5)))
    return selected


def _load_values_and_mask(row: dict[str, object], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    frames = load_tactilus_record(Path(str(row["source_file"])))
    values = frames[int(row["representative_frame_index"])].values
    _, mask = describe_geometry(
        values,
        strategy=args.strategy,
        positive_percentile=args.positive_percentile,
        minimum_raw_threshold=args.minimum_raw_threshold,
        minimum_component_cells=args.minimum_component_cells,
        minimum_component_fraction_of_largest=args.minimum_component_fraction_of_largest,
    )
    return values, mask


def _render_overlays(rows: Sequence[dict[str, object]], args: argparse.Namespace, *, mode: str, output_path: Path) -> None:
    selected = _typical_rows(rows)
    loaded = [(_load_values_and_mask(row, args), row) for row in selected]
    vmax = max(float(values.max()) for ((values, _), _) in loaded)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(selected), figsize=(3.2 * len(selected), 7), constrained_layout=True)
    for ax, ((values, mask), row) in zip(axes, loaded, strict=True):
        image = ax.imshow(values, cmap="magma", origin="upper", interpolation="nearest", vmin=0, vmax=vmax)
        if mask.any():
            ax.contour(mask, levels=[0.5], colors=["#00d4ff"], linewidths=1.5)
        if mode == "geometry" and row["geometry_status"] == "OK":
            left, top = float(row["bbox_column_min"]), float(row["bbox_row_min"])
            width, height = float(row["bbox_width"]), float(row["bbox_height"])
            ax.add_patch(Rectangle((left - 0.5, top - 0.5), width, height, fill=False, edgecolor="#00d4ff", linewidth=1.2))
            ax.scatter([float(row["centroid_column"])], [float(row["centroid_row"])], c="#00d4ff", s=24, marker="x", label="centroid")
            ax.scatter([float(row["cop_column"])], [float(row["cop_row"])], c="#9cff00", s=22, marker="+", label="CoP")
            angle = np.radians(float(row["principal_axis_degrees"]))
            length = min(values.shape) * 0.28
            dc, dr = np.cos(angle) * length, np.sin(angle) * length
            ax.plot([float(row["cop_column"]) - dc, float(row["cop_column"]) + dc], [float(row["cop_row"]) - dr, float(row["cop_row"]) + dr], color="#ff4d6d", linewidth=1.2)
        ax.set_title(f"{row['posture']}\nsubject={row['subject_id']} | v={row['variation']}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(image, ax=list(axes), fraction=0.03, pad=0.02, label="Raw sensor value (dataset units)")
    fig.suptitle(f"PoPu P3 {mode} overlay | representative ACCEPT records | common scale", fontsize=14)
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        not 0 < args.positive_percentile < 100
        or args.minimum_component_cells < 1
        or not 0 <= args.minimum_component_fraction_of_largest <= 1
    ):
        raise ValueError("Invalid mask thresholds.")
    quality_input = _project_path(args.quality_input)
    geometry_output = _project_path(args.geometry_output)
    summary_output = _project_path(args.summary_output)
    mask_output = _project_path(args.mask_figure_output)
    overlay_output = _project_path(args.geometry_figure_output)

    rows = [_geometry_row(row, args) for row in _read_csv(quality_input)]
    eligible = [row for row in rows if row["geometry_status"] != "EXCLUDED"]
    summary = summarise_geometry(eligible)
    summary.update(
        {
            "dataset": "PoPu",
            "input_quality_policy": "P2 ACCEPT and WARN retained; EXCLUDED records stay outside geometry extraction",
            "mask_rule": {
                "strategy": args.strategy,
                "positive_percentile": args.positive_percentile,
                "minimum_raw_threshold": args.minimum_raw_threshold,
                "minimum_component_cells": args.minimum_component_cells,
                "minimum_component_fraction_of_largest": args.minimum_component_fraction_of_largest,
                "boundary": "A relative raw-signal mask, not a calibrated body-contact or anatomical boundary.",
            },
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "quality_input": str(quality_input),
            "geometry_output": str(geometry_output),
            "mask_figure_output": str(mask_output),
            "geometry_figure_output": str(overlay_output),
        }
    )
    _write_csv(rows, geometry_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _render_overlays(rows, args, mode="mask", output_path=mask_output)
    _render_overlays(rows, args, mode="geometry", output_path=overlay_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status_counts"].get("REJECT", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

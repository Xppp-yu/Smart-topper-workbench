"""Run P3.1: compare explicit PoPu contact-mask candidates before features.

Results rank candidate strategies on temporal stability only. They never
auto-freeze a mask rule: a human must review the representative overlays and
failure cases before any Geometry input rule is chosen.
"""

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

from topper_perception.geometry.mask_strategies import (
    MASK_STRATEGIES,
    build_strategy_mask,
    consecutive_bbox_stability,
    jaccard,
    mask_summary,
)
from topper_perception.io.popu import load_tactilus_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/experiments/popu_mask_strategy_comparison_v0.1.json")
RESULT_COLUMNS = (
    "sample_id", "subject_id", "posture", "variation", "p2_quality_status", "strategy",
    "source_file", "frame_count", "usable_frame_count", "usable_frame_fraction",
    "median_mask_fraction", "median_bbox_area_fraction", "median_component_count",
    "mean_consecutive_mask_iou", "mean_consecutive_cop_shift",
    "mean_consecutive_bbox_iou", "mean_bbox_center_shift",
    "median_bbox_width", "median_bbox_height", "bbox_width_iqr", "bbox_height_iqr",
    "comparison_status", "comparison_reason",
)


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--quality-input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--stability-figure-output", type=Path, default=None)
    parser.add_argument("--overlay-figure-output", type=Path, default=None)
    parser.add_argument("--positive-percentile", type=float, default=None)
    parser.add_argument("--minimum-raw-threshold", type=float, default=None)
    parser.add_argument("--minimum-component-cells", type=int, default=None)
    parser.add_argument("--minimum-component-fraction-of-largest", type=float, default=None)
    parser.add_argument("--accept-count", type=int, default=None)
    parser.add_argument("--warn-count", type=int, default=None)
    parser.add_argument("--divergence-count", type=int, default=None)
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument("--divergence-bbox-fraction-spread", type=float, default=None)
    return parser


def _resolve_params(args: argparse.Namespace) -> dict:
    config = _load_config(_project_path(args.config))
    params: dict = {}
    params.update(config.get("paths", {}))
    params.update(config.get("mask_rule", {}))
    params.update(config.get("overlay_selection", {}))
    params["decision_boundary"] = config.get("decision_boundary", "")
    overrides = {
        "quality_input": args.quality_input,
        "output": args.output,
        "summary_output": args.summary_output,
        "stability_figure_output": args.stability_figure_output,
        "overlay_figure_output": args.overlay_figure_output,
        "positive_percentile": args.positive_percentile,
        "minimum_raw_threshold": args.minimum_raw_threshold,
        "minimum_component_cells": args.minimum_component_cells,
        "minimum_component_fraction_of_largest": args.minimum_component_fraction_of_largest,
        "accept_count": args.accept_count,
        "warn_count": args.warn_count,
        "divergence_count": args.divergence_count,
        "frame_index": args.frame_index,
        "divergence_bbox_fraction_spread": args.divergence_bbox_fraction_spread,
    }
    for key, value in overrides.items():
        if value is not None:
            params[key] = value
    return params


def _mask_params(params: dict) -> dict:
    return {
        "positive_percentile": float(params["positive_percentile"]),
        "minimum_raw_threshold": float(params["minimum_raw_threshold"]),
        "minimum_component_cells": int(params["minimum_component_cells"]),
        "minimum_component_fraction_of_largest": float(params["minimum_component_fraction_of_largest"]),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _blank_if_nan(value: float) -> float | str:
    return "" if not np.isfinite(value) else float(value)


def _evaluate_record(
    row: dict[str, str], mask_kwargs: dict, strategy: str
) -> dict[str, object]:
    result: dict[str, object] = {
        "sample_id": row["sample_id"], "subject_id": row["subject_id"], "posture": row["posture"],
        "variation": row["variation"], "p2_quality_status": row["quality_status"], "strategy": strategy,
        "source_file": row.get("source_file", ""),
    }
    for column in RESULT_COLUMNS[7:]:
        result[column] = ""
    if row["quality_status"] not in {"ACCEPT", "WARN"}:
        result.update({"comparison_status": "EXCLUDED", "comparison_reason": "not_in_p2_eligible_population"})
        return result
    try:
        frames = load_tactilus_record(Path(row["source_file"]))
        masks: list[np.ndarray] = []
        summaries: list[dict[str, float | int]] = []
        for frame in frames:
            mask, _ = build_strategy_mask(frame.values, strategy=strategy, **mask_kwargs)
            masks.append(mask)
            summaries.append(mask_summary(mask, frame.values))
        usable = [item for item in summaries if int(item["mask_cell_count"]) > 0]
        ious = [jaccard(left, right) for left, right in zip(masks, masks[1:], strict=False)]
        cop_shifts = [
            float(np.hypot(
                float(right["cop_row_fraction"]) - float(left["cop_row_fraction"]),
                float(right["cop_column_fraction"]) - float(left["cop_column_fraction"]),
            ))
            for left, right in zip(summaries, summaries[1:], strict=False)
            if np.isfinite(float(left["cop_row_fraction"])) and np.isfinite(float(right["cop_row_fraction"]))
        ]
        bbox = consecutive_bbox_stability(masks)
        result.update({
            "frame_count": len(frames),
            "usable_frame_count": len(usable),
            "usable_frame_fraction": len(usable) / len(frames),
            "median_mask_fraction": float(np.median([float(item["mask_fraction"]) for item in usable])) if usable else 0.0,
            "median_bbox_area_fraction": float(np.median([float(item["bbox_area_fraction"]) for item in usable])) if usable else 0.0,
            "median_component_count": float(np.median([float(item["component_count"]) for item in usable])) if usable else 0.0,
            "mean_consecutive_mask_iou": float(np.nanmean(ious)) if any(np.isfinite(ious)) else "",
            "mean_consecutive_cop_shift": float(np.mean(cop_shifts)) if cop_shifts else "",
            "mean_consecutive_bbox_iou": _blank_if_nan(bbox["mean_consecutive_bbox_iou"]),
            "mean_bbox_center_shift": _blank_if_nan(bbox["mean_bbox_center_shift"]),
            "median_bbox_width": _blank_if_nan(bbox["median_bbox_width"]),
            "median_bbox_height": _blank_if_nan(bbox["median_bbox_height"]),
            "bbox_width_iqr": _blank_if_nan(bbox["bbox_width_iqr"]),
            "bbox_height_iqr": _blank_if_nan(bbox["bbox_height_iqr"]),
            "comparison_status": "OK" if usable else "WARN",
            "comparison_reason": "" if usable else "all_frames_empty_mask",
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


def _group_by_sample(rows: Sequence[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sample_id"]), []).append(row)
    return grouped


def _bbox_fraction_spread(strategy_rows: Sequence[dict[str, object]]) -> float:
    values = [
        float(row["median_bbox_area_fraction"])
        for row in strategy_rows
        if row["median_bbox_area_fraction"] != ""
    ]
    return (max(values) - min(values)) if values else 0.0


def _select_representative_samples(
    rows: Sequence[dict[str, object]], overlay: dict
) -> dict[str, list[str]]:
    """Pick deterministic representative samples, never choosing a winner."""
    grouped = _group_by_sample(rows)
    accept: list[str] = []
    warn: list[str] = []
    divergence: list[str] = []
    spread_threshold = float(overlay["divergence_bbox_fraction_spread"])
    for sample_id in sorted(grouped):
        strategy_rows = grouped[sample_id]
        statuses = {str(row["comparison_status"]) for row in strategy_rows}
        p2 = {str(row["p2_quality_status"]) for row in strategy_rows}
        if "WARN" in p2:
            warn.append(sample_id)
            continue
        partial_failure = "OK" in statuses and any(status != "OK" for status in statuses)
        if partial_failure or _bbox_fraction_spread(strategy_rows) > spread_threshold:
            divergence.append(sample_id)
            continue
        if all(status == "OK" for status in statuses):
            accept.append(sample_id)
    return {
        "accept": accept[: int(overlay["accept_count"])],
        "warn": warn[: int(overlay["warn_count"])],
        "divergence": divergence[: int(overlay["divergence_count"])],
    }


def _resolve_selected_samples(
    rows: Sequence[dict[str, object]], selection: dict[str, list[str]]
) -> list[tuple[str, dict[str, object]]]:
    by_sample = {str(row["sample_id"]): row for row in rows}
    samples: list[tuple[str, dict[str, object]]] = []
    seen: set[str] = set()
    for category in ("accept", "warn", "divergence"):
        for sample_id in selection[category]:
            if sample_id in seen:
                continue
            seen.add(sample_id)
            row = by_sample.get(sample_id)
            if row is not None:
                samples.append((category, row))
    return samples


def _render_overlays(
    samples: Sequence[tuple[str, dict[str, object]]],
    mask_kwargs: dict,
    overlay: dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_index = int(overlay["frame_index"])
    loaded: list[tuple[str, dict[str, object], object, list[np.ndarray]]] = []
    for category, row in samples:
        try:
            frames = load_tactilus_record(Path(row["source_file"]))
            frame = frames[min(frame_index, len(frames) - 1)]
        except (OSError, ValueError, IndexError):
            continue
        masks = [build_strategy_mask(frame.values, strategy=s, **mask_kwargs)[0] for s in MASK_STRATEGIES]
        loaded.append((category, row, frame, masks))
    if not loaded:
        return
    fig, axes = plt.subplots(
        len(loaded), len(MASK_STRATEGIES),
        figsize=(4.2 * len(MASK_STRATEGIES), 5.2 * len(loaded)),
        constrained_layout=True, squeeze=False,
    )
    vmin = min(float(np.min(frame.values)) for _, _, frame, _ in loaded)
    vmax = max(float(np.max(frame.values)) for _, _, frame, _ in loaded)
    for row_index, (category, row, frame, masks) in enumerate(loaded):
        for column_index, strategy in enumerate(MASK_STRATEGIES):
            ax = axes[row_index][column_index]
            ax.imshow(frame.values, cmap="magma", origin="upper", interpolation="nearest", aspect="equal", vmin=vmin, vmax=vmax)
            mask = masks[column_index]
            if mask.any():
                ax.contour(mask.astype(np.uint8), levels=[0.5], colors=["#00ffcc"], linewidths=1.1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"strategy={strategy}", fontsize=9)
        axes[row_index][0].set_ylabel(
            f"{category}\n"
            f"sample_id={row['sample_id']}\n"
            f"subject={row['subject_id']} | posture={row['posture']}\n"
            f"variation={row['variation']} | frame={frame.snapshot_key}",
            fontsize=8,
        )
    fig.suptitle("P3.1 mask-strategy side-by-side overlays (cyan = mask boundary; not anatomical truth)", fontsize=12)
    fig.savefig(output_path, dpi=160, facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    params = _resolve_params(args)
    mask_kwargs = _mask_params(params)
    quality_input = _project_path(Path(params["quality_input"]))
    output = _project_path(Path(params["output"]))
    summary_output = _project_path(Path(params["summary_output"]))
    stability_figure = _project_path(Path(params["stability_figure_output"]))
    overlay_figure = _project_path(Path(params["overlay_figure_output"]))

    input_rows = _read_csv(quality_input)
    rows = [_evaluate_record(row, mask_kwargs, strategy) for row in input_rows for strategy in MASK_STRATEGIES]
    _write_csv(rows, output)
    _write_figure(rows, stability_figure)
    selection = _select_representative_samples(rows, params)
    samples = _resolve_selected_samples(rows, selection)
    _render_overlays(samples, mask_kwargs, params, overlay_figure)

    by_strategy = {}
    for strategy in MASK_STRATEGIES:
        selected = [row for row in rows if row["strategy"] == strategy and row["comparison_status"] == "OK"]
        ious = [float(row["mean_consecutive_mask_iou"]) for row in selected if row["mean_consecutive_mask_iou"] != ""]
        bbox_ious = [float(row["mean_consecutive_bbox_iou"]) for row in selected if row["mean_consecutive_bbox_iou"] != ""]
        by_strategy[strategy] = {
            "ok_records": len(selected),
            "median_consecutive_mask_iou": float(np.median(ious)) if ious else None,
            "median_consecutive_bbox_iou": float(np.median(bbox_ious)) if bbox_ious else None,
        }
    summary = {
        "dataset": "PoPu", "stage": "P3.1/R3.1", "candidate_strategies": list(MASK_STRATEGIES),
        "record_status_counts": dict(sorted(Counter(str(row["comparison_status"]) for row in rows).items())),
        "by_strategy": by_strategy,
        "representative_overlay_selection": selection,
        "decision_boundary": params.get("decision_boundary", ""),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(_project_path(args.config)),
        "quality_input": str(quality_input), "output": str(output),
        "stability_figure_output": str(stability_figure), "overlay_figure_output": str(overlay_figure),
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not any(row["comparison_status"] == "REJECT" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())

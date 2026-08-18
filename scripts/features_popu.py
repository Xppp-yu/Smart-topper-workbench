"""Run P4a/A3 per-snapshot label-free feature extraction for PoPu Tactilus.

One row is written per fixed-posture pressure snapshot.  Label and metadata
columns are kept strictly separate from the numeric feature columns; the
``others.json`` records are written to an exclusion manifest and never expanded
into the feature table.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from topper_perception.features.popu import (
    DATASET_ID,
    MASK_RULE_VERSION,
    METADATA_COLUMNS,
    extract_row,
    feature_column_names,
)
from topper_perception.geometry.popu import describe_geometry
from topper_perception.io.popu import POPU_POSTURES, load_tactilus_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXCLUDED_COLUMNS = (
    "sample_id",
    "source_relative_path",
    "subject_id",
    "posture",
    "variation",
    "quality_status",
    "exclusion_reason",
)


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/popu_features_p4a_v0.1.json"),
        help="P4a feature schema and rule config.",
    )
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config(config_path: Path) -> dict[str, Any]:
    config = _read_json(_project_path(config_path))
    mask_rule = dict(config["mask_rule"])
    mask_rule.pop("source", None)
    mask_rule.pop("boundary", None)
    config["mask_rule"] = mask_rule
    return config


def _index_by_sample_id(rows: Sequence[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["sample_id"]: row for row in rows}


def _exclusion_reason(inventory: dict[str, str], quality: dict[str, str]) -> str:
    if str(inventory.get("posture", "")) not in POPU_POSTURES:
        return "missing_fixed_posture_label"
    return f"quality_status={quality.get('quality_status', '')}"


def _write_rows(rows: Sequence[dict[str, object]], fieldnames: Sequence[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _representative_accept_records(
    eligible: Sequence[dict[str, str]], quality_by_id: dict[str, dict[str, str]]
) -> dict[str, dict[str, str]]:
    """Pick the most typical ACCEPT record per posture using P2's robust z."""
    selected: dict[str, dict[str, str]] = {}
    for posture in POPU_POSTURES:
        candidates = [
            inventory
            for inventory in eligible
            if inventory["posture"] == posture
            and quality_by_id[inventory["sample_id"]]["quality_status"] == "ACCEPT"
        ]
        if not candidates:
            continue
        chosen = min(
            candidates,
            key=lambda row: float(quality_by_id[row["sample_id"]]["maximum_robust_z"] or 0),
        )
        selected[posture] = chosen
    return selected


def _render_overview(
    representative: dict[str, dict[str, str]],
    quality_by_id: dict[str, dict[str, str]],
    mask_rule: dict[str, Any],
    posture_counts: dict[str, int],
    cohort_by_posture: dict[str, dict[str, int]],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 1])
    heat_grid = grid[0].subgridspec(1, max(len(representative), 1))
    heat_axes = [fig.add_subplot(heat_grid[0, i]) for i in range(len(representative))]
    bar_ax = fig.add_subplot(grid[1])

    arrays: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    titles: list[str] = []
    for posture, inventory in sorted(representative.items()):
        quality = quality_by_id[inventory["sample_id"]]
        frames = load_tactilus_record(Path(quality["source_file"]))
        frame = frames[int(quality["representative_frame_index"])]
        values = frame.values
        _, mask = describe_geometry(
            np.asarray(values, dtype=np.float64),
            **mask_rule,
        )
        arrays.append(values)
        masks.append(mask)
        titles.append(f"{posture}\nsubject={inventory['subject_id']}")

    vmax = max(float(np.max(array)) for array in arrays)
    for ax, values, mask, title in zip(heat_axes, arrays, masks, titles, strict=True):
        ax.imshow(values, cmap="magma", origin="upper", interpolation="nearest", vmin=0, vmax=vmax)
        if mask.any():
            ax.contour(mask, levels=[0.5], colors=["#00d4ff"], linewidths=1.2)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    postures = list(posture_counts.keys())
    primary = [cohort_by_posture.get(p, {}).get("primary", 0) for p in postures]
    warn = [cohort_by_posture.get(p, {}).get("warn", 0) for p in postures]
    x = np.arange(len(postures))
    bar_ax.bar(x, primary, label="primary (ACCEPT)", color="#4c78a8")
    bar_ax.bar(x, warn, bottom=primary, label="warn", color="#f58518")
    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels(postures)
    bar_ax.set_ylabel("snapshot rows")
    bar_ax.set_title("Feature table rows per posture | primary (ACCEPT) vs warn")
    bar_ax.legend()

    fig.suptitle(f"PoPu P4a feature overview | {DATASET_ID} | {MASK_RULE_VERSION}", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    mask_rule: dict[str, Any] = config["mask_rule"]
    grid = config["grid"]
    row_bands = int(grid["row_bands"])
    column_bands = int(grid["column_bands"])
    schema_version = str(config["feature_schema_version"])

    inventory = _read_csv(_project_path(Path(config["input_inventory"])))
    quality = _read_csv(_project_path(Path(config["input_quality_results"])))
    inventory_by_id = _index_by_sample_id(inventory)
    quality_by_id = _index_by_sample_id(quality)

    feature_table_path = _project_path(Path(config["outputs"]["feature_table"]))
    excluded_path = _project_path(Path(config["outputs"]["excluded_manifest"]))
    cohort_keys_path = _project_path(Path(config["outputs"]["primary_cohort_keys"]))
    summary_path = _project_path(Path(config["outputs"]["summary"]))
    distribution_path = _project_path(Path(config["outputs"]["distribution"]))
    figure_path = _project_path(Path(config["outputs"]["overview_figure"]))

    eligible_statuses = set(config["eligible_quality_statuses"])
    eligible: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    for inventory_row in inventory:
        sample_id = inventory_row["sample_id"]
        quality_row = quality_by_id.get(sample_id, {})
        if (
            inventory_row["posture"] in POPU_POSTURES
            and quality_row.get("quality_status") in eligible_statuses
        ):
            eligible.append(inventory_row)
        else:
            excluded.append(inventory_row)

    fieldnames = list(METADATA_COLUMNS) + list(feature_column_names(row_bands, column_bands))
    feature_columns = list(feature_column_names(row_bands, column_bands))

    posture_counts: Counter[str] = Counter()
    cohort_by_posture: dict[str, Counter[str]] = defaultdict(Counter)
    subject_posture_counts: Counter[tuple[str, str]] = Counter()
    feature_status_counts: Counter[str] = Counter()
    quality_status_counts: Counter[str] = Counter()

    primary_keys: list[dict[str, object]] = []

    feature_table_path.parent.mkdir(parents=True, exist_ok=True)
    with feature_table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for inventory_row in eligible:
            quality_row = quality_by_id[inventory_row["sample_id"]]
            quality_status = quality_row["quality_status"]
            source_file = Path(quality_row["source_file"])
            try:
                frames = load_tactilus_record(source_file)
            except (OSError, ValueError) as exc:
                # A record that fails to re-read cannot produce rows; log nothing
                # silently, but count it so the summary is truthful.
                feature_status_counts["REJECT"] += 1
                continue
            for snapshot_index, frame in enumerate(frames):
                row = extract_row(
                    frame,
                    source_relative_path=inventory_row["source_relative_path"],
                    snapshot_index=snapshot_index,
                    quality_status=quality_status,
                    mask_rule=mask_rule,
                    row_bands=row_bands,
                    column_bands=column_bands,
                    schema_version=schema_version,
                )
                writer.writerow(row)
                posture = row["posture"]
                posture_counts[posture] += 1
                cohort_by_posture[posture][row["cohort"]] += 1
                subject_posture_counts[(row["subject_id"], posture)] += 1
                feature_status_counts[row["feature_status"]] += 1
                quality_status_counts[quality_status] += 1
                if row["cohort"] == "primary":
                    primary_keys.append(
                        {
                            "sample_id": row["sample_id"],
                            "subject_id": row["subject_id"],
                            "posture": row["posture"],
                            "variation": row["variation"],
                            "snapshot_index": row["snapshot_index"],
                        }
                    )

    excluded_rows = [
        {
            "sample_id": inventory_row["sample_id"],
            "source_relative_path": inventory_row["source_relative_path"],
            "subject_id": inventory_row["subject_id"],
            "posture": inventory_row["posture"],
            "variation": inventory_row["variation"],
            "quality_status": quality_by_id.get(inventory_row["sample_id"], {}).get("quality_status", ""),
            "exclusion_reason": _exclusion_reason(
                inventory_row, quality_by_id.get(inventory_row["sample_id"], {})
            ),
        }
        for inventory_row in excluded
    ]
    _write_rows(excluded_rows, EXCLUDED_COLUMNS, excluded_path)

    _write_rows(
        primary_keys,
        ("sample_id", "subject_id", "posture", "variation", "snapshot_index"),
        cohort_keys_path,
    )

    distribution_rows = [
        {
            "subject_id": subject,
            "posture": posture,
            "snapshot_count": count,
        }
        for (subject, posture), count in sorted(subject_posture_counts.items())
    ]
    _write_rows(distribution_rows, ("subject_id", "posture", "snapshot_count"), distribution_path)

    representative = _representative_accept_records(eligible, quality_by_id)
    summary: dict[str, Any] = {
        "dataset": "PoPu",
        "sensor_layer": "tactilus",
        "stage": "P4a/A3",
        "feature_schema_version": schema_version,
        "mask_rule": {**mask_rule, "source": "popu_geometry_frozen_v0.2.json"},
        "grid": grid,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "records_processed": len(inventory),
        "eligible_records": len(eligible),
        "excluded_records": len(excluded),
        "snapshot_rows_total": sum(posture_counts.values()),
        "feature_count": len(feature_columns),
        "label_and_metadata_columns": list(METADATA_COLUMNS),
        "feature_columns": feature_columns,
        "cohort_definition": "primary = fixed posture and quality_status=ACCEPT",
        "cohort_snapshot_counts": {
            "primary": sum(c["primary"] for c in cohort_by_posture.values()),
            "warn": sum(c["warn"] for c in cohort_by_posture.values()),
        },
        "quality_status_snapshot_counts": dict(sorted(quality_status_counts.items())),
        "feature_status_counts": dict(sorted(feature_status_counts.items())),
        "posture_snapshot_counts": dict(sorted(posture_counts.items())),
        "subject_count": len({subject for subject, _ in subject_posture_counts}),
        "excluded_manifest": str(excluded_path),
        "primary_cohort_keys": str(cohort_keys_path),
        "note": (
            "others.json records (60) lack a fixed posture label and are EXCLUDED; "
            "their 35247 snapshots are not expanded into the feature table."
        ),
    }
    _write_json(summary, summary_path)

    if representative:
        _render_overview(
            representative,
            quality_by_id,
            mask_rule,
            dict(posture_counts),
            {p: dict(c) for p, c in cohort_by_posture.items()},
            figure_path,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

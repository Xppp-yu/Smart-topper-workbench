"""Run SLP Body Axis and Bounding-Box Geometry (A08) on real SLP data.

This script:
1. Loads A05 canonical samples and the A07 anomaly CSV.
2. Computes body axes (shoulder, hip, longitudinal), body center,
   orientation, and bounding box for every usable J0 frame.
3. Produces: geometry CSV, QA summary, error-cases CSV, and small overlay visualizations.
4. Respects the A06 frozen split (no data modification).

Design contract:
- J0 only (RGB space); J1 remains downstream.
- Deterministic: same inputs → same outputs.
- No silent imputation: missing key joints → explicit reject/uncertain.
- No region ground truth generated.
- A06 split is read-only.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Add project root to path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from topper_perception.io.slp_body_geometry import (
    ADAPTER_VERSION,
    DEFAULT_TASK_ID,
    DEFAULT_GENERATOR,
    JOINT_NAMES,
    RGB_HEIGHT,
    RGB_WIDTH,
    BodyGeometryResult,
    compute_body_geometry,
    geometry_result_to_csv_row,
    GEOMETRY_CSV_COLUMNS,
    geometry_schema_dict,
)
from topper_perception.io.slp_subject_split import load_canonical_samples_from_csv
from topper_perception.io.slp_joint_eda import (
    JointCoords,
    joints_to_coords,
    load_subject_joints_rgb,
)

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[1]

# Default paths.
DEFAULT_A05_CSV = PROJECT_ROOT_PATH / "data/processed/slp/slp_canonical_samples_v0.1.csv"
DEFAULT_A06_JSON = PROJECT_ROOT_PATH / "data/processed/slp/slp_subject_split_v0.1.json"
DEFAULT_A07_ANOMALY_CSV = PROJECT_ROOT_PATH / "outputs/analysis/slp_joint_eda_v0.1/slp_joint_anomalies_v0.1.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT_PATH / "outputs/reports/slp_body_geometry_v0.1"
DEFAULT_CSV_DIR = PROJECT_ROOT_PATH / "outputs/analysis/slp_body_geometry_v0.1"


# ---------------------------------------------------------------------------
# Anomaly lookup
# ---------------------------------------------------------------------------

def load_anomaly_flags(anomaly_csv_path: Path) -> tuple[set[str], set[str]]:
    """Load A07 anomaly CSV and return sets of sample_ids with extreme jumps and anomalous bone lengths."""
    extreme_jump_ids: set[str] = set()
    anomalous_length_ids: set[str] = set()

    if not anomaly_csv_path.is_file():
        return extreme_jump_ids, anomalous_length_ids

    with anomaly_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row.get("sample_id", "").strip()
            anomaly_type = row.get("anomaly_type", "").strip()
            if not sample_id:
                continue
            if anomaly_type == "extreme_frame_jump":
                extreme_jump_ids.add(sample_id)
            elif anomaly_type == "anomalous_bone_length":
                anomalous_length_ids.add(sample_id)

    return extreme_jump_ids, anomalous_length_ids


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slp-root",
        type=Path,
        required=True,
        help="SLP data root (e.g. E:/TeamProjects/datasets/smart-topper/SLP2022/SLP).",
    )
    parser.add_argument(
        "--a05-csv",
        type=Path,
        default=DEFAULT_A05_CSV,
        help="A05 canonical samples CSV.",
    )
    parser.add_argument(
        "--a06-json",
        type=Path,
        default=DEFAULT_A06_JSON,
        help="A06 subject split manifest JSON (used only for reporting, not modification).",
    )
    parser.add_argument(
        "--a07-anomaly-csv",
        type=Path,
        default=DEFAULT_A07_ANOMALY_CSV,
        help="A07 anomaly cases CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for report outputs.",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=DEFAULT_CSV_DIR,
        help="Directory for CSV geometry outputs.",
    )
    parser.add_argument(
        "--task-id",
        default=DEFAULT_TASK_ID,
    )
    return parser


# ---------------------------------------------------------------------------
# Statistics aggregation
# ---------------------------------------------------------------------------

def aggregate_statistics(results: list[BodyGeometryResult]) -> dict:
    """Aggregate statistics across all results."""
    n = len(results)

    if n == 0:
        return {}

    status_counts: dict[str, int] = {}
    axis_status_counts: dict[str, int] = {}
    bbox_status_counts: dict[str, int] = {}
    orient_status_counts: dict[str, int] = {}
    overall_conf_sum = 0.0
    orient_conf_sum = 0.0
    visible_sum = 0
    occluded_sum = 0
    oob_sum = 0
    missing_sum = 0
    flip_count = 0
    face_up_count = 0
    extreme_jump_count = 0
    anomalous_length_count = 0

    for r in results:
        status_counts[r.overall_status] = status_counts.get(r.overall_status, 0) + 1
        axis_status_counts[r.axis_status] = axis_status_counts.get(r.axis_status, 0) + 1
        bbox_status_counts[r.bbox_status] = bbox_status_counts.get(r.bbox_status, 0) + 1
        orient_status_counts[r.orientation_status] = orient_status_counts.get(r.orientation_status, 0) + 1
        overall_conf_sum += r.overall_confidence
        orient_conf_sum += r.orientation_confidence
        visible_sum += r.visible_joints
        occluded_sum += r.occluded_joints
        oob_sum += r.out_of_bounds_joints
        missing_sum += r.missing_joints
        if r.left_right_flip_detected:
            flip_count += 1
        if r.face_up_detected:
            face_up_count += 1
        if r.extreme_frame_jump:
            extreme_jump_count += 1
        if r.anomalous_bone_length:
            anomalous_length_count += 1

    # Per-sample average.
    avg_overall_conf = overall_conf_sum / n
    avg_orient_conf = orient_conf_sum / n

    # Error code aggregation.
    error_code_counts: dict[str, int] = {}
    for r in results:
        for ec in r.error_codes:
            error_code_counts[ec] = error_code_counts.get(ec, 0) + 1

    return {
        "total_frames": n,
        "accept_count": status_counts.get("accept", 0),
        "uncertain_count": status_counts.get("uncertain", 0),
        "reject_count": status_counts.get("reject", 0),
        "status_distribution": status_counts,
        "axis_status_distribution": axis_status_counts,
        "bbox_status_distribution": bbox_status_counts,
        "orientation_status_distribution": orient_status_counts,
        "avg_overall_confidence": avg_overall_conf,
        "avg_orientation_confidence": avg_orient_conf,
        "total_visible_joints": visible_sum,
        "total_occluded_joints": occluded_sum,
        "total_out_of_bounds_joints": oob_sum,
        "total_missing_joints": missing_sum,
        "frames_with_left_right_flip": flip_count,
        "frames_with_face_up_detected": face_up_count,
        "frames_with_extreme_frame_jump": extreme_jump_count,
        "frames_with_anomalous_bone_length": anomalous_length_count,
        "error_code_distribution": dict(sorted(error_code_counts.items(), key=lambda x: -x[1])),
    }


# ---------------------------------------------------------------------------
# Visualization helpers (matplotlib-based overlays)
# ---------------------------------------------------------------------------

def write_overlay(
    sample_id: str,
    rgb_path: Path,
    result: BodyGeometryResult,
    output_dir: Path,
) -> Path | None:
    """Write a small overlay PNG with RGB image, joints, axes, and bbox."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    if not rgb_path.is_file():
        return None

    img = cv2.imread(str(rgb_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    canvas = img.copy()

    # Draw bounding box.
    bb = result.bbox
    if bb.valid:
        x1 = int(max(0, min(w - 1, bb.x_min)))
        y1 = int(max(0, min(h - 1, bb.y_min)))
        x2 = int(max(0, min(w - 1, bb.x_max)))
        y2 = int(max(0, min(h - 1, bb.y_max)))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw axes.
    for axis_name, axis in [
        ("shoulder", result.shoulder_axis),
        ("hip", result.hip_axis),
        ("longitudinal", result.longitudinal_axis),
    ]:
        if axis.direction_valid and axis.start.valid and axis.end.valid:
            color = {"shoulder": (255, 0, 0), "hip": (0, 255, 255), "longitudinal": (0, 200, 255)}[axis_name]
            pt1 = (int(axis.start.x), int(axis.start.y))
            pt2 = (int(axis.end.x), int(axis.end.y))
            cv2.line(canvas, pt1, pt2, color, 3)
            cv2.circle(canvas, pt1, 5, color, -1)
            cv2.circle(canvas, pt2, 5, color, -1)

    # Draw body center.
    if result.body_center.valid:
        cx = int(result.body_center.x)
        cy = int(result.body_center.y)
        cv2.circle(canvas, (cx, cy), 6, (255, 255, 0), -1)

    # Draw per-joint points (color-coded by visibility).
    for jv_dict in result.per_joint_validity:
        jx = float(jv_dict["x"])
        jy = float(jv_dict["y"])
        if not jv_dict["is_visible"]:
            continue
        if jv_dict["is_out_of_bounds"]:
            color = (200, 200, 100)
        else:
            color = (0, 255, 0)
        cv2.circle(canvas, (int(jx), int(jy)), 4, color, -1)

    # Add status label.
    label = f"{result.overall_status.upper()} | conf={result.overall_confidence:.2f} | {sample_id}"
    cv2.putText(canvas, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(canvas, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    output_path = output_dir / f"{sample_id}_overlay.png"
    cv2.imwrite(str(output_path), canvas)
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Validate inputs.
    if not args.a05_csv.is_file():
        print(f"ERROR: A05 canonical samples CSV not found: {args.a05_csv}")
        return 1

    # Load A07 anomaly flags.
    extreme_jump_ids, anomalous_length_ids = load_anomaly_flags(args.a07_anomaly_csv)
    print(f"Loaded A07 anomaly flags: {len(extreme_jump_ids)} extreme jumps, "
          f"{len(anomalous_length_ids)} anomalous bone lengths")

    # Load A05 canonical samples.
    print(f"Loading A05 canonical samples from {args.a05_csv}")
    canonical_samples = load_canonical_samples_from_csv(args.a05_csv)
    usable_samples = [s for s in canonical_samples if str(s.get("quarantine", "")).strip().lower() not in ("true", "1")]
    quarantined_samples = [s for s in canonical_samples if str(s.get("quarantine", "")).strip().lower() in ("true", "1")]
    print(f"  Total samples: {len(canonical_samples)}, "
          f"Usable: {len(usable_samples)}, Quarantined: {len(quarantined_samples)}")

    # Output directories.
    output_dir = Path(args.output_dir)
    csv_dir = Path(args.csv_dir)
    overlay_dir = output_dir / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # Process each usable sample.
    results: list[BodyGeometryResult] = []
    error_results: list[BodyGeometryResult] = []
    slp_root = Path(args.slp_root)

    print(f"Processing {len(usable_samples)} usable frames ...")
    processed = 0
    skipped = 0

    for row in usable_samples:
        sample_id = str(row["sample_id"])
        setting = str(row["setting"])
        subject_id = str(row["subject_id"])
        cover = str(row["cover_condition"])
        frame_index = int(row["frame_index"])

        # Load J0 RGB joints.
        subject_dir = slp_root / setting / subject_id
        joints_arr = load_subject_joints_rgb(subject_dir, cover=cover, frame_index=frame_index)
        if joints_arr is None:
            skipped += 1
            continue

        joints = joints_to_coords(joints_arr)

        # Check anomaly flags.
        has_extreme_jump = sample_id in extreme_jump_ids
        has_anomalous_length = sample_id in anomalous_length_ids

        # Compute geometry.
        result = compute_body_geometry(
            sample_id=sample_id,
            joints=joints,
            task_id=args.task_id,
            extreme_frame_jump=has_extreme_jump,
            anomalous_bone_length=has_anomalous_length,
        )
        results.append(result)

        if result.overall_status in ("uncertain", "reject"):
            error_results.append(result)

        processed += 1
        if processed % 2000 == 0:
            print(f"  Processed {processed}/{len(usable_samples)} frames ...")

    print(f"  Done: {processed} processed, {skipped} skipped")

    # Write geometry CSV.
    geometry_csv_path = csv_dir / "slp_body_geometry_v0.1.csv"
    with geometry_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GEOMETRY_CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(geometry_result_to_csv_row(result))
    print(f"Geometry CSV written: {geometry_csv_path}")

    # Write error cases CSV.
    error_csv_path = csv_dir / "slp_body_geometry_error_cases_v0.1.csv"
    with error_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GEOMETRY_CSV_COLUMNS)
        writer.writeheader()
        for result in error_results:
            writer.writerow(geometry_result_to_csv_row(result))
    print(f"Error cases CSV written: {error_csv_path} ({len(error_results)} error frames)")

    # Aggregate statistics.
    stats = aggregate_statistics(results)

    # Write QA summary JSON.
    now = datetime.now(timezone.utc).isoformat()
    summary = {
        "task_id": args.task_id,
        "adapter_version": ADAPTER_VERSION,
        "generator": DEFAULT_GENERATOR,
        "created_at": now,
        "slp_root": str(slp_root.resolve()),
        "a05_csv": str(args.a05_csv.resolve()),
        "a06_json": str(args.a06_json.resolve()),
        "a07_anomaly_csv": str(args.a07_anomaly_csv.resolve()),
        "total_usable_frames_processed": len(results),
        "total_quarantined_frames": len(quarantined_samples),
        "frames_skipped_no_joints": skipped,
        "statistics": stats,
    }
    summary_path = output_dir / "slp_body_geometry_summary_v0.1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"QA summary written: {summary_path}")

    # Write geometry schema JSON.
    schema_path = csv_dir / "slp_body_geometry_v0.1.schema.json"
    schema_path.write_text(json.dumps(geometry_schema_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Schema written: {schema_path}")

    # Write overlays for a small sample (normal, uncertain, reject).
    print("Writing overlays ...")
    overlay_manifest: list[dict] = []

    def pick_samples(label: str, status_filter: str | None = None, count: int = 5) -> list[BodyGeometryResult]:
        if status_filter:
            candidates = [r for r in results if r.overall_status == status_filter]
        else:
            candidates = results
        return candidates[:count]

    # Normal (accept) samples.
    for result in pick_samples("accept", "accept", 3):
        sample_id = result.sample_id
        row = next((s for s in usable_samples if s["sample_id"] == sample_id), None)
        if row:
            rgb_path = slp_root / str(row.get("rgb_uri", ""))
            op = write_overlay(sample_id, rgb_path, result, overlay_dir)
            if op:
                overlay_manifest.append({"sample_id": sample_id, "overlay": op.name, "status": "accept"})

    # Uncertain samples.
    for result in pick_samples("uncertain", "uncertain", 3):
        sample_id = result.sample_id
        row = next((s for s in usable_samples if s["sample_id"] == sample_id), None)
        if row:
            rgb_path = slp_root / str(row.get("rgb_uri", ""))
            op = write_overlay(sample_id, rgb_path, result, overlay_dir)
            if op:
                overlay_manifest.append({"sample_id": sample_id, "overlay": op.name, "status": "uncertain"})

    # Reject samples.
    for result in pick_samples("reject", "reject", 3):
        sample_id = result.sample_id
        row = next((s for s in usable_samples if s["sample_id"] == sample_id), None)
        if row:
            rgb_path = slp_root / str(row.get("rgb_uri", ""))
            op = write_overlay(sample_id, rgb_path, result, overlay_dir)
            if op:
                overlay_manifest.append({"sample_id": sample_id, "overlay": op.name, "status": "reject"})

    # SimLab samples.
    simlab_results = [r for r in results if any(
        s["sample_id"] == r.sample_id and s["setting"] == "simLab"
        for s in usable_samples
    )]
    for result in simlab_results[:3]:
        sample_id = result.sample_id
        row = next((s for s in usable_samples if s["sample_id"] == sample_id), None)
        if row:
            rgb_path = slp_root / str(row.get("rgb_uri", ""))
            op = write_overlay(sample_id, rgb_path, result, overlay_dir)
            if op:
                overlay_manifest.append({"sample_id": sample_id, "overlay": op.name, "status": result.overall_status})

    # Write overlay manifest.
    manifest_path = output_dir / "overlay_manifest_v0.1.json"
    manifest_path.write_text(json.dumps(overlay_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Overlay manifest written: {manifest_path} ({len(overlay_manifest)} overlays)")

    # Print summary.
    print()
    print("=" * 60)
    print(f"  A08 Body Geometry — Summary")
    print("=" * 60)
    print(f"  Processed frames : {len(results)}")
    print(f"  Skipped frames   : {skipped}")
    print(f"  ACCEPT           : {stats.get('accept_count', 0)}")
    print(f"  UNCERTAIN        : {stats.get('uncertain_count', 0)}")
    print(f"  REJECT           : {stats.get('reject_count', 0)}")
    print(f"  Avg confidence   : {stats.get('avg_overall_confidence', 0):.4f}")
    print(f"  Left/right flip  : {stats.get('frames_with_left_right_flip', 0)}")
    print(f"  Face-up detected : {stats.get('frames_with_face_up_detected', 0)}")
    print(f"  Extreme jump     : {stats.get('frames_with_extreme_frame_jump', 0)}")
    print(f"  Anomalous length : {stats.get('frames_with_anomalous_bone_length', 0)}")
    print(f"  Error cases CSV  : {error_csv_path} ({len(error_results)} rows)")
    print(f"  Overlays written : {len(overlay_manifest)}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
